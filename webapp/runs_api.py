"""Bridges the dashboard to the existing run_store.py / reload.py code.

Read-only listing calls run_store.py in-process. Everything else — trash/
restore/purge (which sys.exit() on a bad run_id, fatal in a long-lived
server) and anything that can hit an input() prompt — runs as a real
subprocess of reload.py/pipeline.py instead, via pty_bridge.py.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

import reload as reload_cli  # the existing CLI, reused — not reimplemented
import run_store

RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}(?:-[0-9a-f]{4})?$")
RUN_MODE_CHOICES = {"full", "clustering_only"}
CONTENT_TYPE_CHOICES = {"social_media", "document"}
STAGE_CHOICES = set(run_store.STAGE_ORDER)


class InvalidRunId(ValueError):
    pass


def validate_run_id(run_id: str) -> str:
    """Rejects anything that doesn't match run_store._new_run_id()'s shape,
    so a hostile value (e.g. starting with '-') never reaches argparse."""
    if not RUN_ID_RE.match(run_id):
        raise InvalidRunId(f"'{run_id}' is not a valid run id")
    return run_id


def list_active_rows() -> list[dict]:
    return _rows(run_store.list_runs(), trashed=False)


def list_trash_rows() -> list[dict]:
    return _rows(run_store.list_trashed_runs(), trashed=True)


def _rows(runs: list[dict], trashed: bool) -> list[dict]:
    """Goes through reload.py's _run_summary() and run_store.effective_status()
    so the dashboard can't drift from what the CLI shows."""
    out = []
    for r in runs:
        base_dir = (run_store.trash_root() / r["run_id"]) if trashed else None
        summary = reload_cli._run_summary(r, base_dir=base_dir)
        status = r.get("status") if trashed else run_store.effective_status(r)
        out.append({
            "run_id": r["run_id"],
            "project_name": r.get("project_name") or "—",
            "stage": r.get("stage") or "—",
            "status": status or "—",
            "run_mode": r.get("run_mode") or "—",
            "active": bool(r.get("active")),
            "posts": summary["posts"],
            "edges": summary["edges"],
            "saved_at": summary["saved_at"] or r.get("started_at"),
            "locked_by": _lock_summary(r["run_id"]) if not trashed else None,
            "has_render": False if trashed else has_render(r["run_id"]),
            "has_output": False if trashed else output_ready(r["run_id"]),
        })
    return out


def _lock_summary(run_id: str) -> dict | None:
    if not run_store.lock_is_live(run_id):
        return None  # missing, or a stale lock file — not actually running
    return run_store.lock_holder(run_id)


def get_run_or_none(run_id: str) -> dict | None:
    run = run_store.load_run(run_id)
    if run is not None:
        run["run_id"] = run_id
    return run


# Top-level config_snapshot.yaml keys that come entirely from
# config/advanced.yaml, not config.yaml — the merged snapshot on disk
# doesn't remember which file a key came from, so this is how the display
# tells them apart. Keep in sync with run_store.load_config().
_ADVANCED_CONFIG_KEYS = {
    "embedding", "umap_cluster", "pacmap_layout",
    "clustering", "adr", "hdbscan", "edges", "fa2", "render",
}

# nova/adept live in both files: config.yaml contributes only `model_slot`,
# config/advanced.yaml contributes the rest, merged into one dict per key —
# so unlike _ADVANCED_CONFIG_KEYS these need filtering key-by-key, not
# dropping whole. `naming` has no advanced.yaml counterpart.
_SPLIT_SECTION_BASIC_KEYS = {"nova": {"model_slot"}, "adept": {"model_slot"}}

# Sections folded into one "Module routing" group in the structured view
# instead of three one-row groups of their own — see _config_groups().
_ROUTING_KEYS = ("nova", "adept", "naming")


def _basic_config(config_yaml: str | None) -> dict | None:
    """The config snapshot with config/advanced.yaml material filtered out,
    leaving only what was chosen via config.yaml. None if nothing to show."""
    if not config_yaml:
        return None
    try:
        cfg = yaml.safe_load(config_yaml)
    except yaml.YAMLError:
        return None
    if not isinstance(cfg, dict):
        return None

    basic: dict = {}
    for key, value in cfg.items():
        if key in _ADVANCED_CONFIG_KEYS:
            continue
        if key in _SPLIT_SECTION_BASIC_KEYS and isinstance(value, dict):
            kept = {k: v for k, v in value.items() if k in _SPLIT_SECTION_BASIC_KEYS[key]}
            if kept:
                basic[key] = kept
        else:
            basic[key] = value
    return basic or None


def _basic_config_yaml(cfg: dict | None) -> str | None:
    """The filtered dict back as YAML text, for the raw/copy view."""
    return yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False) if cfg else None


def _label(key: str) -> str:
    """'max_retries' -> 'Max retries' — used for both group and row labels
    so the structured view never needs a hardcoded name per config key."""
    return key.replace("_", " ").strip().capitalize()


def _format_value(value) -> str:
    """Nested dicts render as flow-style YAML so they still fit one kv-grid row."""
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, dict):
        return yaml.safe_dump(value, default_flow_style=True, sort_keys=False).strip()
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) if value else "—"
    return str(value)


def _config_groups(cfg: dict | None) -> list[dict]:
    """One display group per top-level dict key, top-level scalars under
    "General", nova/adept/naming folded into "Module routing". api_key is
    never shown — it's a credential and doesn't belong in config.yaml."""
    if not cfg:
        return []

    groups: list[dict] = []
    general_rows: list[dict] = []
    routing_rows: list[dict] = []

    for key, value in cfg.items():
        if key in _ROUTING_KEYS and isinstance(value, dict):
            slot = value.get("model_slot")
            if slot:
                routing_rows.append({"label": _label(key), "value": slot})
            continue

        if not isinstance(value, dict):
            general_rows.append({"label": _label(key), "value": _format_value(value)})
            continue

        # A flat group of its own, or one subgroup per child if its values
        # are themselves dicts (models.heavy/light).
        flat_rows = []
        for sub_key, sub_value in value.items():
            if sub_key == "api_key":
                continue
            if isinstance(sub_value, dict):
                child_rows = [
                    {"label": _label(k2), "value": _format_value(v2)}
                    for k2, v2 in sub_value.items() if k2 != "api_key"
                ]
                if child_rows:
                    groups.append({"label": f"{_label(key)} · {_label(sub_key)}", "rows": child_rows})
            else:
                flat_rows.append({"label": _label(sub_key), "value": _format_value(sub_value)})
        if flat_rows:
            groups.append({"label": _label(key), "rows": flat_rows})

    result: list[dict] = []
    if general_rows:
        result.append({"label": "General", "rows": general_rows})
    result.extend(groups)
    if routing_rows:
        result.append({"label": "Module routing", "rows": routing_rows})
    return result


def get_run_detail(run_id: str, config_path: str, trashed: bool = False) -> dict | None:
    """Input path goes through reload.py's own _resolve_cli_args(), so this
    can't show a different input than `resume` would replay. The config
    snapshot is read as raw text rather than through
    run_store.load_config_snapshot(), which sys.exit()s on a missing
    snapshot — fine for `resume`, wrong for a read-only display."""
    validate_run_id(run_id)
    run = run_store.load_trashed_run(run_id) if trashed else run_store.load_run(run_id)
    if run is None:
        return None
    run["run_id"] = run_id

    status = run.get("status") if trashed else run_store.effective_status(run)

    base_dir = (run_store.trash_root() / run_id) if trashed else None
    summary = reload_cli._run_summary(run, base_dir=base_dir)

    cli_args_recorded = bool(run.get("cli_args"))
    resolved = reload_cli._resolve_cli_args(run)
    input_dir = resolved.get("input")

    snapshot_path = (run_store.trash_root() / run_id / "config_snapshot.yaml") if trashed \
        else run_store.config_snapshot_path(run_id)
    config_yaml = None
    if snapshot_path.exists():
        try:
            config_yaml = snapshot_path.read_text(encoding="utf-8")
        except OSError:
            config_yaml = None

    basic_config = _basic_config(config_yaml)

    return {
        "run_id": run_id,
        "trashed": trashed,
        "project_name": run.get("project_name") or "—",
        "status": status or "—",
        "run_mode": run.get("run_mode") or "—",
        "stage": run.get("stage") or "—",
        "active": (not trashed) and run_store.get_active_run_id() == run_id,
        "locked_by": _lock_summary(run_id) if not trashed else None,
        "posts": summary["posts"],
        "edges": summary["edges"],
        "started_at": run.get("started_at"),
        "saved_at": summary["saved_at"],
        "relabeled_at": run.get("relabeled_at"),
        "input_dir": input_dir or "—",
        "input_is_recorded": cli_args_recorded,
        "config_groups": _config_groups(basic_config),
        "config_yaml": _basic_config_yaml(basic_config),
        "config_snapshot_exists": snapshot_path.exists(),
        "has_output": False if trashed else output_ready(run_id),
    }


def render_html_path(run_id: str) -> Path:
    return run_store.run_dir(validate_run_id(run_id)) / "render.html"


def has_render(run_id: str) -> bool:
    return render_html_path(run_id).exists()


def output_ready(run_id: str) -> bool:
    return (run_store.output_dir(validate_run_id(run_id)) / "clusters.parquet").exists()


# ─── SEARCH HISTORY ──────────────────────────────────────────────────────────
# Read-only access to what search.py._save_search_record() writes to
# <run_dir>/searches/*.md — the dashboard's "past questions" list and its
# Markdown report viewer both go through here, never through run_store
# directly, so a filename always gets the same path-traversal check.

# Exactly the shape _save_search_record() produces: "%Y%m%d-%H%M%S-<slug>.md",
# slug being lowercase alnum/hyphen, 1-40 chars. Anchoring both ends means a
# value like "../../etc/passwd" or an absolute path never reaches disk.
SEARCH_FILENAME_RE = re.compile(r"^\d{8}-\d{6}-[a-z0-9-]{1,40}\.md$")

# Pulls the two header lines _save_search_record() always writes first
# ("**Asked:** <iso timestamp>" / "**Question:** <query>") without reading
# the rest of a potentially long report.
_SEARCH_HEADER_RE = re.compile(
    r"\*\*Asked:\*\*\s*(?P<asked>.+?)\s*\n\*\*Question:\*\*\s*(?P<question>.+?)\s*\n"
)


class InvalidSearchFilename(ValueError):
    pass


def validate_search_filename(filename: str) -> str:
    """Rejects anything that isn't exactly a _save_search_record() filename."""
    if not SEARCH_FILENAME_RE.match(filename):
        raise InvalidSearchFilename(f"'{filename}' is not a valid search record filename")
    return filename


def _searches_dir(run_id: str) -> Path:
    return run_store.run_dir(validate_run_id(run_id)) / "searches"


def list_searches(run_id: str) -> list[dict]:
    """One row per saved search for this run, newest first.

    The filename's leading timestamp sorts lexically in the same order as
    the searches were made, so no parsing is needed to order the list —
    only to display it. Only the header of each file is read (never the
    full report), so this stays cheap no matter how many searches a run
    has accumulated or how long any one report is.

    Returns:
        [] if the run has no searches/ directory yet (never raises for
        that — a run with no searches is a normal, common case).
    """
    out_dir = _searches_dir(run_id)
    if not out_dir.is_dir():
        return []

    rows = []
    for path in out_dir.glob("*.md"):
        if not SEARCH_FILENAME_RE.match(path.name):
            continue  # ignore anything not written by _save_search_record()
        try:
            head = path.read_text(encoding="utf-8")[:2000]
        except OSError:
            continue
        m = _SEARCH_HEADER_RE.search(head)
        rows.append({
            "filename": path.name,
            "question": m.group("question") if m else path.stem,
            "asked_at": m.group("asked") if m else None,
        })
    rows.sort(key=lambda r: r["filename"], reverse=True)
    return rows


def get_search_record(run_id: str, filename: str) -> str | None:
    """Raw Markdown text of one saved search — report, then its "## Sources"
    appendix, already linkified by search.py (see linkify_citations /
    render_citation_appendix). None if the run or the file doesn't exist;
    a bad filename raises InvalidSearchFilename rather than silently
    returning None, so the dashboard can tell "wrong shape" from "not
    found there" while a nonexistent-but-valid-shaped file still reads as
    a clean 404 either way.
    """
    path = _searches_dir(run_id) / validate_search_filename(filename)
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


# Matches exactly what _save_search_record() puts between the header and
# the report: "...\n**Question:** <query>\n\n---\n\n<report>\n".
_SEARCH_BODY_SPLIT_RE = re.compile(r"\n\n---\n\n")


def get_search_record_parsed(run_id: str, filename: str) -> dict | None:
    """Same record as get_search_record(), split into what the dashboard's
    report viewer actually needs: the question/timestamp for its own panel
    header, and the Markdown body to hand to the client's renderer —
    everything _save_search_record() wrote after its "---" separator,
    already linkified by search.py.

    None under the same conditions as get_search_record() (run or file
    doesn't exist). A record whose separator can't be found (hand-edited,
    or not written by _save_search_record() at all) still comes back as a
    dict, with `body` falling back to the whole file, rather than dropping
    the record entirely.
    """
    text = get_search_record(run_id, filename)
    if text is None:
        return None
    header = _SEARCH_HEADER_RE.search(text)
    parts = _SEARCH_BODY_SPLIT_RE.split(text, maxsplit=1)
    body = parts[1] if len(parts) == 2 else text
    return {
        "filename": filename,
        "question": header.group("question") if header else None,
        "asked_at": header.group("asked") if header else None,
        "body": body.rstrip("\n"),
    }


def delete_search(run_id: str, filename: str) -> bool:
    """Delete one saved search record from <run_dir>/searches/.

    Same filename validation as get_search_record() — a malformed filename
    raises InvalidSearchFilename rather than silently no-op'ing. Returns
    True if a file was actually removed, False if it was already gone
    (both are a fine outcome for a "delete this" action; only a genuine
    OSError during removal propagates).
    """
    path = _searches_dir(run_id) / validate_search_filename(filename)
    if not path.is_file():
        return False
    path.unlink()
    return True


# trash/restore/purge sys.exit() on error paths in run_store.py; running
# them as a subprocess keeps a bad run_id from taking the server down.
def _run_reload_cli(args: list[str], repo_root: Path) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "reload.py", *args],
            cwd=str(repo_root), capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        return 1, exc.stdout or "", f"'reload.py {' '.join(args)}' timed out after 30s"
    return proc.returncode, proc.stdout, proc.stderr


def trash(run_id: str, repo_root: Path) -> tuple[int, str, str]:
    return _run_reload_cli(["trash", validate_run_id(run_id)], repo_root)


def restore(run_id: str, repo_root: Path) -> tuple[int, str, str]:
    return _run_reload_cli(["restore", validate_run_id(run_id)], repo_root)


def purge(run_id: str, repo_root: Path) -> tuple[int, str, str]:
    return _run_reload_cli(["purge", validate_run_id(run_id)], repo_root)


def purge_all(repo_root: Path) -> tuple[int, str, str]:
    return _run_reload_cli(["purge", "--all"], repo_root)


# Duplicated from pipeline.py's _FILE_READERS rather than imported —
# importing pipeline.py pulls in the whole embedding/clustering stack.
# Keep in sync if pipeline.py's readers change.
_SUPPORTED_INPUT_EXTENSIONS = {".json", ".csv", ".tsv", ".txt", ".md", ".pdf"}


class BrowseError(ValueError):
    """Raised by browse_directory() for anything that stops a folder-picker
    request from being served as-is (missing path, not a directory, no
    read permission) — same shape as InvalidRunId, caught in dashboard.py
    and turned into a clean 400."""


def browse_directory(path: str | None, repo_root: Path) -> dict:
    """One level of a server-side folder picker. --input always takes a
    folder, never a file, so subdirectories are navigable and supported
    files are listed for visibility only. No path is off-limits — same as
    the free-text field this replaces. Defaults to data/, falling back to
    repo_root."""
    if path:
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = (repo_root / target).resolve()
    else:
        data_root = repo_root / "data"
        target = data_root if data_root.is_dir() else repo_root

    if not target.exists():
        raise BrowseError(f"'{target}' does not exist.")
    if not target.is_dir():
        raise BrowseError(f"'{target}' is not a folder.")

    try:
        children = list(target.iterdir())
    except PermissionError as exc:
        raise BrowseError(f"Permission denied reading '{target}'.") from exc

    dirs = sorted(
        (p.name for p in children if p.is_dir() and not p.name.startswith(".")),
        key=str.lower,
    )

    supported_files = sorted(
        (p for p in children if p.is_file() and p.suffix.lower() in _SUPPORTED_INPUT_EXTENSIONS),
        key=lambda p: p.name.lower(),
    )
    files = []
    for p in supported_files:
        try:
            size = p.stat().st_size
        except OSError:
            # Gone or unreadable between iterdir() and stat() — list it without a size.
            size = None
        files.append({"name": p.name, "size": size})

    parent = target.parent
    at_filesystem_root = parent == target

    return {
        "path": str(target),
        "parent": None if at_filesystem_root else str(parent),
        "dirs": [{"name": name, "path": str(target / name)} for name in dirs],
        "files": files,
        "file_count": len(files),
    }


def validate_input_dir(path: str) -> str:
    """Rejects a value that could be read as another flag by argparse."""
    if not path or path.startswith("-"):
        raise ValueError(f"invalid input directory '{path}'")
    return path


def build_new_run_argv(*, run_mode: str | None,
                        input_dir: str | None, set_k: int | None,
                        content_type: str | None = None,
                        fast_k: bool = False) -> list[str]:
    """fast_k mirrors pipeline.py's --k-fast; a custom set_k always wins
    regardless of checkbox state (the elif below decides it)."""
    # No --render: pipeline.py's own default ("threejs") is what we want.
    # No --no-browser: pipeline.py doesn't define that flag.
    argv = [sys.executable, "pipeline.py"]
    if run_mode:
        if run_mode not in RUN_MODE_CHOICES:
            raise ValueError(f"invalid run_mode '{run_mode}'")
        argv += ["--run-mode", run_mode]
    if content_type:
        if content_type not in CONTENT_TYPE_CHOICES:
            raise ValueError(f"invalid content_type '{content_type}'")
        argv += ["--content-type", content_type]
    if input_dir:
        argv += ["--input", validate_input_dir(input_dir)]
    if set_k is not None:
        if set_k < 1:
            raise ValueError(f"invalid set_k '{set_k}'")
        argv += ["--set-k", str(int(set_k))]
    elif fast_k:
        argv.append("--k-fast")
    # Always fresh: this button starts a brand-new run and never proposes
    # resuming whatever is currently active — that's what Resume is for.
    argv.append("--fresh")
    return argv


def build_resume_argv(run_id: str, *, run_mode: str | None = None,
                       resume_from: str | None = None) -> list[str]:
    # No --no-browser: `resume`'s subparser doesn't define it.
    argv = [sys.executable, "reload.py", "resume", validate_run_id(run_id),
            "--render", "threejs"]
    if resume_from:
        if resume_from not in STAGE_CHOICES:
            raise ValueError(f"invalid resume_from '{resume_from}'")
        argv += ["--resume-from", resume_from]
    if run_mode:
        if run_mode not in RUN_MODE_CHOICES:
            raise ValueError(f"invalid run_mode '{run_mode}'")
        argv += ["--run-mode", run_mode]
    return argv


def build_relabel_argv(run_id: str, *, force_skip_llm: bool = False) -> list[str]:
    argv = [sys.executable, "reload.py", "relabel", validate_run_id(run_id)]
    if force_skip_llm:
        argv.append("--force-skip-llm")
    return argv


def build_show_render_argv(run_id: str) -> list[str]:
    return [sys.executable, "reload.py", "show", validate_run_id(run_id),
            "--render", "threejs", "--no-browser"]


def build_sync_models_argv(run_id: str) -> list[str]:
    return [sys.executable, "reload.py", "sync-models", validate_run_id(run_id)]


SEARCH_BUDGET_MIN = 100
SEARCH_BUDGET_MAX = 6000


def build_search_argv(run_id: str, query: str, budget: int | None = None) -> list[str]:
    if not query or not query.strip():
        raise ValueError("query is required")
    argv = [sys.executable, "reload.py", "search", validate_run_id(run_id), query]
    if budget is not None:
        try:
            budget = int(budget)
        except (TypeError, ValueError):
            raise ValueError("budget must be an integer") from None
        if not (SEARCH_BUDGET_MIN <= budget <= SEARCH_BUDGET_MAX):
            raise ValueError(f"budget must be between {SEARCH_BUDGET_MIN} and {SEARCH_BUDGET_MAX}")
        argv += ["--budget", str(budget)]
    return argv


ACTION_BUILDERS = {
    "resume": build_resume_argv,
    "relabel": build_relabel_argv,
    "show_render": build_show_render_argv,
    "sync_models": build_sync_models_argv,
}