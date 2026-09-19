"""HeronLoom — local web dashboard.

A web UI for pipeline.py / reload.py / run_store.py. Read-only listing
calls run_store.py directly; mutating or interactive actions run as a
real subprocess of `reload.py` / `pipeline.py` (see webapp/runs_api.py
and webapp/pty_bridge.py).

Usage:
    python dashboard.py                  # http://127.0.0.1:8000
    python dashboard.py --port 8080
    python dashboard.py --host 0.0.0.0   # exposes on the network, no auth layer

Run from the project root, same as reload.py.
"""
from __future__ import annotations

from _bootstrap import ensure_venv

ensure_venv()

import argparse
import asyncio
import contextlib
import copy
import json
import threading
import webbrowser
from pathlib import Path
from urllib.parse import urlparse

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import run_store
from utils.logger import get_logger
from webapp import docs_api, runs_api

# Fixed name, not __name__ — dashboard.py is a real entry point (see
# get_logger()'s docstring). Uvicorn's own access/error logging stays on
# its own console-only path (_timestamped_log_config below).
logger = get_logger("dashboard")
from webapp.pty_bridge import PTY_SUPPORTED, PTY_UNAVAILABLE_REASON, PtySession

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = "config/config.yaml"


# DNS-rebinding / drive-by CSRF protection: binding to 127.0.0.1 alone
# doesn't stop a page's JS from firing a request here, and WebSocket
# handshakes aren't covered by CORS. Plain ASGI middleware rather than
# BaseHTTPMiddleware, so it also covers the WebSocket scope.

ALLOWED_ORIGIN_HOSTNAMES = {"localhost", "127.0.0.1", "::1"}  # extended in main() by --allow-origin


def _origin_allowed(origin: str | None) -> bool:
    """No Origin header (curl, direct navigation) is treated as trusted."""
    if origin is None:
        return True
    hostname = urlparse(origin).hostname
    return hostname is not None and hostname in ALLOWED_ORIGIN_HOSTNAMES


def _is_cross_site_fetch(sec_fetch_site: str | None) -> bool:
    """Absent on older browsers, so this is defense in depth, not required."""
    return sec_fetch_site == "cross-site"


class OriginCheckMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)

        headers = dict(scope.get("headers") or [])
        origin_header = headers.get(b"origin")
        origin = origin_header.decode("latin-1") if origin_header else None
        sec_fetch_site_header = headers.get(b"sec-fetch-site")
        sec_fetch_site = sec_fetch_site_header.decode("latin-1") if sec_fetch_site_header else None

        # No GET exemption: DNS rebinding lets attacker JS read a GET
        # response too, since Origin/Host still match the rebound page.
        rejected = _is_cross_site_fetch(sec_fetch_site) or not _origin_allowed(origin)

        if rejected:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            else:
                await send({
                    "type": "http.response.start", "status": 403,
                    "headers": [(b"content-type", b"text/plain")],
                })
                await send({"type": "http.response.body", "body": b"cross-origin request rejected"})
            return

        await self.app(scope, receive, send)


# Server-owned watcher pushed over /ws/runs instead of per-tab polling.
# Polls rather than watching the filesystem since run state can be
# written by processes this server has no handle on (a separate
# reload.py/pipeline.py invocation, another dashboard instance).

RUNS_CHECK_INTERVAL_SECONDS = 2.0


def _runs_snapshot() -> dict:
    """Row counts come from parquet metadata only, so this stays cheap
    regardless of run count. Keyed by run_id with named fields so
    _diff_snapshots() can report which field changed."""
    active = runs_api.list_active_rows()
    trashed = runs_api.list_trash_rows()

    def _fields(r: dict) -> dict:
        return {
            "status": r["status"], "stage": r["stage"], "posts": r["posts"],
            "edges": r["edges"], "saved_at": r["saved_at"], "active": r["active"],
            "has_render": r["has_render"], "has_output": r["has_output"],
            "locked": bool(r["locked_by"]),
        }

    return {
        "active": {r["run_id"]: _fields(r) for r in active},
        "trashed": {r["run_id"] for r in trashed},
    }


def _diff_snapshots(old: dict | None, new: dict) -> list[str]:
    """Human-readable lines describing what changed, for the watcher's log
    only — never sent to a client."""
    if old is None:
        return ["initial snapshot"]

    lines: list[str] = []
    old_active, new_active = old["active"], new["active"]

    for run_id in sorted(new_active.keys() - old_active.keys()):
        lines.append(f"{run_id}: new active run ({new_active[run_id]['status']})")
    for run_id in sorted(old_active.keys() - new_active.keys()):
        lines.append(f"{run_id}: left the active table")
    for run_id in sorted(new_active.keys() & old_active.keys()):
        old_fields, new_fields = old_active[run_id], new_active[run_id]
        if old_fields == new_fields:
            continue
        changed = ", ".join(
            f"{k} {old_fields[k]!r}\u2192{new_fields[k]!r}"
            for k in old_fields if old_fields[k] != new_fields[k]
        )
        lines.append(f"{run_id}: {changed}")

    for run_id in sorted(new["trashed"] - old["trashed"]):
        lines.append(f"{run_id}: moved to trash")
    for run_id in sorted(old["trashed"] - new["trashed"]):
        lines.append(f"{run_id}: left trash (restored or purged)")

    return lines or ["snapshot differs but no field-level change resolved — inspect manually"]


class RunsBroadcaster:
    """Owns connected /ws/runs clients and the background watch task.
    notify_changed() pushes right after a local mutation; _watch_loop() is
    a periodic fallback for changes this process has no event for."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._last_snapshot: dict | None = None
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._watch_loop())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task

    def register(self, ws: WebSocket) -> None:
        self._clients.add(ws)

    def unregister(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def notify_changed(self) -> None:
        """Push 'runs-changed' now and refresh the stored snapshot so the
        next tick doesn't rebroadcast the same change."""
        with contextlib.suppress(Exception):
            self._last_snapshot = await run_in_threadpool(_runs_snapshot)
        await self._broadcast({"type": "runs-changed"})

    async def _watch_loop(self) -> None:
        while True:
            try:
                snapshot = await run_in_threadpool(_runs_snapshot)
                if snapshot != self._last_snapshot:
                    for line in _diff_snapshots(self._last_snapshot, snapshot):
                        logger.info("runs watcher: %s", line)
                    self._last_snapshot = snapshot
                    await self._broadcast({"type": "runs-changed"})
            except Exception:
                # A bad read costs this tick only — next tick retries.
                logger.exception("runs watcher: failed to compute run-state snapshot")
            await asyncio.sleep(RUNS_CHECK_INTERVAL_SECONDS)

    async def _broadcast(self, message: dict) -> None:
        dead = []
        for ws in self._clients:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)


runs_broadcaster = RunsBroadcaster()


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    runs_broadcaster.start()
    try:
        yield
    finally:
        await runs_broadcaster.stop()


app = FastAPI(title="HeronLoom Dashboard", lifespan=_lifespan)
app.add_middleware(OriginCheckMiddleware)
app.mount("/static", StaticFiles(directory=str(REPO_ROOT / "static")), name="static")
app.mount("/assets", StaticFiles(directory=str(REPO_ROOT / "assets")), name="assets")
templates = Jinja2Templates(directory=str(REPO_ROOT / "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "dashboard.html", {
        "pty_supported": PTY_SUPPORTED,
        "pty_unavailable_reason": PTY_UNAVAILABLE_REASON,
        "run_mode_choices": sorted(runs_api.RUN_MODE_CHOICES),
        "content_type_choices": sorted(runs_api.CONTENT_TYPE_CHOICES),
        "stage_choices": run_store.STAGE_ORDER,
        "doc_groups": docs_api.grouped_docs(),
    })


# ─── DOCS ─────────────────────────────────────────────────────────────────
# Read-only rendering of README.md + docs/*.md, same allow-list spirit as
# runs_api.py's other lookups (see docs_api.DOCS). Two views on the same
# render_doc(): a fragment for the topbar's "Guide" button (opens inline in
# #detail-panel-body, no new tab — see _doc_partial.html) and a full page
# for the "Docs" dropdown's entries (each opens /docs/{id} in a new tab —
# see docs_page.html). A missing/unregistered doc is a clean 404 either way,
# never a 500 — a doc can be listed in docs_api.DOCS before its file exists.

@app.get("/partials/docs/{doc_id}", response_class=HTMLResponse)
async def partial_doc(request: Request, doc_id: str):
    try:
        doc = await run_in_threadpool(docs_api.render_doc, doc_id, REPO_ROOT)
    except docs_api.DocNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    return templates.TemplateResponse(request, "_doc_partial.html", {"doc": doc})


@app.get("/docs/{doc_id}", response_class=HTMLResponse)
async def doc_page(request: Request, doc_id: str):
    try:
        doc = await run_in_threadpool(docs_api.render_doc, doc_id, REPO_ROOT)
    except docs_api.DocNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    return templates.TemplateResponse(request, "docs_page.html", {
        "doc": doc, "groups": docs_api.grouped_docs(),
    })


@app.get("/partials/runs/active", response_class=HTMLResponse)
async def partial_active_runs(request: Request):
    rows = await run_in_threadpool(runs_api.list_active_rows)
    return templates.TemplateResponse(request, "_runs_table.html", {
        "rows": rows, "trashed": False, "pty_supported": PTY_SUPPORTED,
        "stage_choices": run_store.STAGE_ORDER,
        "run_mode_choices": sorted(runs_api.RUN_MODE_CHOICES),
    })


@app.get("/partials/runs/trash", response_class=HTMLResponse)
async def partial_trash_runs(request: Request):
    rows = await run_in_threadpool(runs_api.list_trash_rows)
    return templates.TemplateResponse(request, "_runs_table.html", {
        "rows": rows, "trashed": True, "pty_supported": PTY_SUPPORTED,
    })


# `trashed` is a query param, not part of the path — it only changes
# where on disk get_run_detail() looks.
@app.get("/partials/runs/{run_id}/detail", response_class=HTMLResponse)
async def partial_run_detail(request: Request, run_id: str, trashed: bool = False):
    try:
        detail = await run_in_threadpool(runs_api.get_run_detail, run_id, CONFIG_PATH, trashed)
    except runs_api.InvalidRunId as exc:
        raise HTTPException(400, str(exc)) from exc
    if detail is None:
        raise HTTPException(404, f"no run found with id '{run_id}'")
    return templates.TemplateResponse(request, "_run_detail.html", {"run": detail, "pty_supported": PTY_SUPPORTED})


# ─── SEARCH HISTORY / REPORT VIEWER ──────────────────────────────────────────
# Read-only, both routes go straight through runs_api.py's own filename
# validation (SEARCH_FILENAME_RE) — never touch run_store or the filesystem
# here. The history list is an HTML partial like every other list in this
# app; the record itself comes back as JSON because the dashboard renders
# its Markdown client-side (dashboard.js's renderMarkdown()) rather than
# through a server-side Markdown dependency.

@app.get("/partials/runs/{run_id}/searches", response_class=HTMLResponse)
async def partial_search_history(request: Request, run_id: str):
    try:
        rows = await run_in_threadpool(runs_api.list_searches, run_id)
    except runs_api.InvalidRunId as exc:
        raise HTTPException(400, str(exc)) from exc
    return templates.TemplateResponse(request, "_search_history.html", {"rows": rows, "run_id": run_id})


@app.get("/api/runs/{run_id}/searches/{filename}")
async def api_get_search_record(run_id: str, filename: str):
    try:
        record = await run_in_threadpool(runs_api.get_search_record_parsed, run_id, filename)
    except (runs_api.InvalidRunId, runs_api.InvalidSearchFilename) as exc:
        raise HTTPException(400, str(exc)) from exc
    if record is None:
        raise HTTPException(404, f"no saved search '{filename}' for run '{run_id}'")
    return record


@app.delete("/api/runs/{run_id}/searches/{filename}")
async def api_delete_search_record(run_id: str, filename: str):
    try:
        deleted = await run_in_threadpool(runs_api.delete_search, run_id, filename)
    except (runs_api.InvalidRunId, runs_api.InvalidSearchFilename) as exc:
        raise HTTPException(400, str(exc)) from exc
    if not deleted:
        raise HTTPException(404, f"no saved search '{filename}' for run '{run_id}'")
    return {"deleted": True}


@app.get("/api/browse")
async def api_browse(path: str | None = None):
    """Still gated by OriginCheckMiddleware — this exposes the process's
    filesystem tree, exactly what a DNS-rebinding attacker would want."""
    try:
        return await run_in_threadpool(runs_api.browse_directory, path, REPO_ROOT)
    except runs_api.BrowseError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.websocket("/ws/runs")
async def ws_runs(ws: WebSocket):
    """Push-only: RunsBroadcaster is the only writer. The receive loop just
    keeps something awaiting receive() so Starlette raises
    WebSocketDisconnect promptly on close."""
    await ws.accept()
    runs_broadcaster.register(ws)
    # Catches up on anything that changed before this socket connected.
    with contextlib.suppress(Exception):
        await ws.send_json({"type": "runs-changed"})
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        runs_broadcaster.unregister(ws)


@app.get("/runs/{run_id}/render.html")
async def get_render_html(run_id: str):
    try:
        path = runs_api.render_html_path(run_id)
    except runs_api.InvalidRunId as exc:
        raise HTTPException(400, str(exc)) from exc
    if not path.exists():
        raise HTTPException(404, f"'{run_id}' has no render.html yet — use 'Generate 3D view' first.")
    return FileResponse(path, media_type="text/html")


# trash/restore/purge: subprocess of reload.py, errors turned into HTTP
# 400. HX-Trigger refreshes this tab; notify_changed() pushes to every
# other connected /ws/runs client.

def _run_or_400(fn, *args) -> None:
    try:
        rc, out, err = fn(*args)
    except runs_api.InvalidRunId as exc:
        raise HTTPException(400, str(exc)) from exc
    if rc != 0:
        raise HTTPException(400, (err or out or "action failed").strip().splitlines()[-1])


@app.post("/api/runs/{run_id}/trash")
async def api_trash(run_id: str):
    await run_in_threadpool(_run_or_400, runs_api.trash, run_id, REPO_ROOT)
    await runs_broadcaster.notify_changed()
    return Response(status_code=204, headers={"HX-Trigger": "runs-changed"})


@app.post("/api/trash/{run_id}/restore")
async def api_restore(run_id: str):
    await run_in_threadpool(_run_or_400, runs_api.restore, run_id, REPO_ROOT)
    await runs_broadcaster.notify_changed()
    return Response(status_code=204, headers={"HX-Trigger": "runs-changed"})


@app.post("/api/trash/{run_id}/purge")
async def api_purge(run_id: str):
    await run_in_threadpool(_run_or_400, runs_api.purge, run_id, REPO_ROOT)
    await runs_broadcaster.notify_changed()
    return Response(status_code=204, headers={"HX-Trigger": "runs-changed"})


@app.post("/api/trash/purge-all")
async def api_purge_all():
    await run_in_threadpool(_run_or_400, runs_api.purge_all, REPO_ROOT)
    await runs_broadcaster.notify_changed()
    return Response(status_code=204, headers={"HX-Trigger": "runs-changed"})


def _build_argv(msg: dict) -> list[str]:
    """Translate a launch request into an argv list, using only the
    allow-listed builders in runs_api.py — the browser never supplies a
    raw command."""
    action = msg.get("action")

    if action == "new_run":
        return runs_api.build_new_run_argv(
            run_mode=msg.get("run_mode") or None,
            content_type=msg.get("content_type") or None,
            input_dir=msg.get("input_dir") or None,
            set_k=msg.get("set_k"),
        )

    run_id = msg.get("run_id")
    if not run_id:
        raise ValueError("run_id is required")

    if action == "resume":
        return runs_api.build_resume_argv(
            run_id, run_mode=msg.get("run_mode") or None,
            resume_from=msg.get("resume_from") or None,
        )
    if action == "relabel":
        return runs_api.build_relabel_argv(run_id, force_skip_llm=bool(msg.get("force_skip_llm")))
    if action == "show_render":
        return runs_api.build_show_render_argv(run_id)
    if action == "sync_models":
        return runs_api.build_sync_models_argv(run_id)
    if action == "search":
        return runs_api.build_search_argv(run_id, msg.get("query"), budget=msg.get("budget"))

    raise ValueError(f"unknown action '{action}'")


@app.websocket("/ws/terminal")
async def ws_terminal(ws: WebSocket):
    await ws.accept()

    if not PTY_SUPPORTED:
        await ws.send_json({"type": "error",
                             "message": "Interactive sessions need a POSIX pty and aren't "
                                        "available on this platform (try WSL on Windows)."})
        await ws.close(code=1011)
        return

    try:
        launch_msg = await ws.receive_json()
        argv = _build_argv(launch_msg)
    except (ValueError, runs_api.InvalidRunId) as exc:
        await ws.send_json({"type": "error", "message": str(exc)})
        await ws.close(code=1008)
        return
    except (WebSocketDisconnect, json.JSONDecodeError):
        return

    session = PtySession(argv=argv, cwd=str(REPO_ROOT))
    try:
        session.start()
    except RuntimeError as exc:
        await ws.send_json({"type": "error", "message": str(exc)})
        await ws.close(code=1011)
        return

    await ws.send_json({"type": "started", "argv": argv})

    out_queue: asyncio.Queue[str | None] = asyncio.Queue()

    def on_output(data: bytes) -> None:
        out_queue.put_nowait(data.decode("utf-8", errors="replace"))

    async def pump_output() -> None:
        while True:
            text = await out_queue.get()
            if text is None:
                return
            with contextlib.suppress(Exception):
                await ws.send_json({"type": "stdout", "data": text})

    async def pump_input() -> None:
        while True:
            try:
                msg = await ws.receive_json()
            except WebSocketDisconnect:
                # Ordinary end of a terminal session, not an error.
                return
            kind = msg.get("type")
            if kind == "stdin":
                session.write(msg.get("data", "").encode())
            elif kind == "resize":
                session.resize(int(msg.get("rows", 30)), int(msg.get("cols", 120)))

    output_task = asyncio.create_task(pump_output())
    attach_task = asyncio.create_task(session.attach(on_output))
    input_task = asyncio.create_task(pump_input())

    done, pending = await asyncio.wait({attach_task, input_task}, return_when=asyncio.FIRST_COMPLETED)

    # asyncio.wait() only requests cancellation; input_task's exception
    # must still be consumed explicitly below, or it warns on GC.
    if attach_task in done:
        exit_code = attach_task.result()
        await runs_broadcaster.notify_changed()  # notifies other tabs
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "exit", "code": exit_code})
    else:
        # Browser disconnected before the child finished — don't orphan it.
        session.close()
        with contextlib.suppress(Exception):
            await attach_task
        with contextlib.suppress(asyncio.CancelledError):
            input_task.exception()  # retrieve it so it's never "unread"

    for t in pending:
        t.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    out_queue.put_nowait(None)
    with contextlib.suppress(Exception):
        await output_task

    session.close()
    with contextlib.suppress(Exception):
        await ws.close()


def _parse_allowed_origin_hostname(value: str) -> str:
    """Accepts a bare hostname or a full origin URL; only the hostname is
    kept, since that's all _origin_allowed() compares against."""
    if "://" in value:
        hostname = urlparse(value).hostname
        if not hostname:
            raise argparse.ArgumentTypeError(f"'{value}' is not a valid origin or hostname")
        return hostname
    return value


def _timestamped_log_config() -> dict:
    """Same as uvicorn's default LOGGING_CONFIG, with %(asctime)s
    prepended — its default formatters carry no timestamp at all."""
    config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    for name in ("default", "access"):
        config["formatters"][name]["fmt"] = "%(asctime)s " + config["formatters"][name]["fmt"]
        config["formatters"][name]["datefmt"] = "%H:%M:%S"
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="HeronLoom — local web dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1, localhost-only).")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="Don't open a browser tab automatically.")
    parser.add_argument(
        "--allow-origin", action="append", default=[], metavar="HOST",
        type=_parse_allowed_origin_hostname,
        help="Additional hostname allowed past the Origin check for WebSockets and "
             "POST actions (e.g. --allow-origin 192.168.1.5). Required if you also pass "
             "--host to serve this on your network: the browser's Origin will be that "
             "address, not 'localhost', and every WebSocket / mutating request is "
             "rejected as cross-origin until its exact hostname is allow-listed here — "
             "including the dashboard's own address, which is never inferred "
             "automatically. Repeatable for more than one hostname.",
    )
    args = parser.parse_args()

    if args.allow_origin:
        ALLOWED_ORIGIN_HOSTNAMES.update(args.allow_origin)

    if not PTY_SUPPORTED:
        print(f"[Dashboard] NOTE: {PTY_UNAVAILABLE_REASON} The run table, and "
              f"trash/restore/purge, still work regardless — only new runs / resume / "
              f"relabel / show need the terminal panel.")

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"[Dashboard] WARNING: binding to {args.host} exposes this dashboard — "
              f"including the ability to launch pipeline runs — to anyone who can reach "
              f"it on your network. There is no authentication layer. Prefer 127.0.0.1 "
              f"unless you understand the exposure.")
        if args.host not in ALLOWED_ORIGIN_HOSTNAMES:
            print(f"[Dashboard] NOTE: the run table will still load, but every WebSocket "
                  f"and every trash/restore/purge/new-run/resume action will be rejected "
                  f"as cross-origin until you also pass --allow-origin {args.host} (or "
                  f"whichever hostname/IP you'll actually browse to).")

    url = f"http://{args.host}:{args.port}"
    print(f"[Dashboard] serving at {url}")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info",
                log_config=_timestamped_log_config())


if __name__ == "__main__":
    main()