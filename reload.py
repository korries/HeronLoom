"""Browse and act on past HeronLoom pipeline runs.

Every run pipeline.py has ever produced lives in its own folder under
``runs/``. This is a subcommand CLI (see `reload.py --help` / `reload.py
<command> --help`), one verb per action:

  list      list every active run (add --trash to list trashed ones)
  show      render a run's saved state — read-only, no recompute
  open      open a run's already-generated render.html in the browser —
            no data load, no recompute (see `show` to make/refresh one)
  resume    continue a run from its last checkpointed stage, or a chosen
            one via --resume-from — works even on a run marked complete
  restart   full re-run as a brand-new run_id
  relabel      redo naming (cluster labels) only, in place — no recompute
               of clustering / Nova / ADEPT / edges / layout
  sync-models  patch a run's frozen config_snapshot.yaml with the live
               config.yaml's model routing ONLY (models: heavy/light;
               nova/adept/naming's model_slot; nova/adept's
               parallel_clusters) — every other snapshot key (algorithm
               tuning, storage.processed_dir, ...) is untouched
  trash        move a run to runs/_trash/ (reversible)
  restore      move a run back out of runs/_trash/
  purge        permanently delete a trashed run (or --all of them)

`resume` and `restart` replay the run's originally recorded CLI invocation
by default; pass --input / --run-mode / --limit / --set-k / --k-range /
--use-k-cache / --k-fast / --skip-naming / --force-nova to
override any of those for just this launch — cli_args itself is never
modified. `resume` also
accepts --resume-from to target a specific checkpoint stage instead of
the furthest one on disk (e.g. to add Nova/ADEPT to a clustering_only
run), and works even on a run marked "complete". `restart` has no
--resume-from: it always starts a brand-new run_id from stage 1.

`resume` always reads config_snapshot.yaml, frozen at launch, never the
live config.yaml — editing config.yaml has no effect on an existing run
by itself. Use `sync-models` to pull just the model routing into an
existing run's snapshot, then `resume --resume-from STAGE` to apply it
from that stage onward; already-checkpointed stages are not recomputed.

Examples
--------
::

    python reload.py list
    python reload.py list --trash
    python reload.py show 20260821-181530-a3f2
    python reload.py show 20260821-181530-a3f2 --render threejs
    python reload.py open 20260821-181530-a3f2
    python reload.py resume 20260821-181530-a3f2
    python reload.py resume 20260821-181530-a3f2 --run-mode full
    python reload.py resume 20260821-181530-a3f2 --resume-from naming --run-mode full
    python reload.py restart 20260821-181530-a3f2 --run-mode full --limit 500
    python reload.py relabel 20260821-181530-a3f2
    python reload.py relabel 20260821-181530-a3f2 --force-skip-llm
    python reload.py sync-models 20260821-181530-a3f2
    python reload.py sync-models 20260821-181530-a3f2 --dry-run
    python reload.py sync-models 20260821-181530-a3f2 && python reload.py resume 20260821-181530-a3f2 --resume-from nova
    python reload.py trash 20260821-181530-a3f2
    python reload.py restore 20260821-181530-a3f2
    python reload.py purge 20260821-181530-a3f2
    python reload.py purge --all
"""

from _bootstrap import ensure_venv

ensure_venv()

import argparse
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd

import run_store
from utils.logger import get_logger

# Explicit name, not __name__: reload.py is a real entry point, not an
# imported module — see get_logger()'s docstring.
logger = get_logger("reload")

CONFIG_PATH = "config/config.yaml"


# Defaults for reconstructing a pipeline.py invocation for runs that
# predate cli_args tracking (see action_resume / action_restart).
_DEFAULT_CLI_ARGS = {
    "input": "data/raw/",
    "limit": None, "resume_from": None, "run_mode": None,
    "use_k_cache": False, "set_k": None, "k_range": None,
    "k_fast": False,
    "skip_naming": False, "force_nova": False,
    "fresh": False, "yes": False, "debug": False,
}


def _parquet_row_count(path: Path):
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return "?"


def _run_summary(run: dict, base_dir: Path | None = None) -> dict:
    """Row counts + saved_at for a run's most relevant parquet pair, without
    loading them: output/ if complete, else the latest stage checkpoint.

    base_dir overrides where to look on disk. Needed for trashed runs,
    whose files live under runs/_trash/<run_id>/ rather than runs/<run_id>/
    (run_store's own path helpers only know about the latter)."""
    info = {"posts": None, "edges": None, "saved_at": None}
    root = base_dir if base_dir is not None else run_store.run_dir(run["run_id"])

    if run["status"] == "complete":
        d = root / "output"
        posts_path, edges_path = d / "clusters.parquet", d / "edges.parquet"
    elif run.get("stage") in run_store.STAGE_ORDER:
        d = root / "stages" / run["stage"]
        posts_path, edges_path = d / "posts.parquet", d / "edges.parquet"
    else:
        return info

    if posts_path.exists():
        info["posts"]    = _parquet_row_count(posts_path)
        info["saved_at"] = datetime.fromtimestamp(posts_path.stat().st_mtime).isoformat(timespec="seconds")
    if edges_path.exists():
        info["edges"] = _parquet_row_count(edges_path)
    return info


def find_run(run_id: str) -> dict:
    """Return the run manifest for run_id, or exit with a clear message."""
    run = run_store.load_run(run_id)
    if run is None:
        logger.error("no run found with id '%s' — use `reload.py list` to see available runs", run_id)
        sys.exit(1)
    run["run_id"] = run_id
    return run


def print_table(runs: list[dict], trashed: bool = False) -> None:
    """Print a fixed-width table of every discovered run (active runs by
    default, or trashed ones if trashed=True)."""
    if not runs:
        where = "runs/_trash/" if trashed else "runs/"
        print(f"[Reload] No runs found under {where}.")
        return

    headers = ["run_id", "project", "stage", "status", "posts", "edges", "saved_at"]
    rows = []
    for r in runs:
        base_dir = (run_store.trash_root() / r["run_id"]) if trashed else None
        summary  = _run_summary(r, base_dir=base_dir)
        run_id   = r["run_id"] + (" *" if r.get("active") else "")
        # A trashed run's lock is never meaningful (see runs_api._rows()'s
        # own `if not trashed` gate for the same reason) — only compute
        # the crashed/in_progress distinction for the active table, so
        # `reload.py list` and the dashboard can never disagree about
        # what a given run's status means.
        status = r.get("status") if trashed else run_store.effective_status(r)
        rows.append([
            run_id,
            r.get("project_name") or "-",
            r.get("stage") or "-",
            status or "-",
            str(summary["posts"]) if summary["posts"] is not None else "-",
            str(summary["edges"]) if summary["edges"] is not None else "-",
            summary["saved_at"] or r.get("started_at") or "-",
        ])

    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    line   = "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=False))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=False)))

    print()
    if trashed:
        print("Actions:")
        print("  python reload.py restore <run_id>   move back out of trash")
        print("  python reload.py purge <run_id>     permanently delete")
        print("  python reload.py purge --all        empty the trash entirely")
    else:
        print("'*' marks the active run.")
        print()
        print("Actions:")
        print("  python reload.py show <run_id>       render saved state, no recompute")
        print("  python reload.py open <run_id>       open existing render.html, no recompute")
        print("  python reload.py search [run_id] \"q\" ask a question, defaults to the active run")
        print("  python reload.py resume <run_id>     continue from checkpoint, or --resume-from a stage")
        print("  python reload.py restart <run_id>    full re-run (new run_id)")
        print("  python reload.py relabel <run_id>    redo naming (labels) only, in place")
        print("  python reload.py sync-models <run_id>  patch snapshot's model routing from config.yaml")
        print("  python reload.py trash <run_id>      move to runs/_trash/ (reversible)")
        print("  python reload.py list --trash        list trashed runs")
        print("  python reload.py purge <run_id>      permanently delete a trashed run")


def action_show(run: dict, config_path: str, render: str, no_browser: bool = False) -> None:
    """Load the run's saved output and optionally open the renderer. Read-only:
    writes nothing, never touches active.json."""
    d = run_store.output_dir(run["run_id"])
    posts_path, edges_path = d / "clusters.parquet", d / "edges.parquet"

    if not posts_path.exists():
        logger.error(
            "'%s' has no output/ — this run never finished (status: %s). "
            "Use `python reload.py resume %s` instead.",
            run["run_id"], run.get("status", "?"), run["run_id"],
        )
        sys.exit(1)

    posts_df = pd.read_parquet(posts_path)
    edges_df = pd.read_parquet(edges_path) if edges_path.exists() else pd.DataFrame()
    logger.info("loaded '%s': %d posts  %d edges", run["run_id"], len(posts_df), len(edges_df))

    if render == "none":
        print("[Reload] Pass --render threejs to open the 3-D viewer for this run.")
        return

    sys.path.insert(0, str(Path(__file__).parent / "src"))
    import render as render_module

    # Prefer this run's own frozen config (matches what it was actually
    # rendered with) over the live config.yaml; fall back if it predates
    # config-snapshot tracking.
    snapshot = run_store.config_snapshot_path(run["run_id"])
    effective_config_path = str(snapshot) if snapshot.exists() else config_path
    # Isolated in its own try/except so a render failure can't wipe out the
    # posts/edges count already logged above — `show` is this command's own
    # purpose, so unlike relabel's background refresh, this is fatal.
    try:
        render_module.run(posts_df, edges_df, config_path=effective_config_path, open_browser=not no_browser)
    except Exception:
        logger.exception("'%s' loaded but the 3-D render failed", run["run_id"])
        sys.exit(1)


def action_search(run: dict, query: str, config_path: str, run_mode: str,
                   no_decompose: bool, budget: int | None = None) -> None:
    """Ask a question against a run's saved output. Read-only: writes nothing
    besides --dry-run's own output files, never touches active.json."""
    d = run_store.output_dir(run["run_id"])
    if not (d / "clusters.parquet").exists():
        logger.error(
            "'%s' has no output/ — this run never finished (status: %s). "
            "Use `python reload.py resume %s` instead.",
            run["run_id"], run.get("status", "?"), run["run_id"],
        )
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).parent / "src"))
    import search

    cfg, output_dir = search.resolve_run_cfg(run["run_id"], config_path)
    search_kwargs = {}
    if budget is not None:
        search_kwargs["budget"] = budget
    search.run_search(cfg, query, output_dir=output_dir, run_mode_override=run_mode,
                       no_decompose=no_decompose, **search_kwargs)


def action_open(run: dict) -> None:
    """Open a run's already-generated render.html in the browser as-is —
    no parquet load, no call into the render module. Distinct from `show`,
    which always (re)generates render.html from output/ before opening it."""
    render_html = run_store.run_dir(run["run_id"]) / "render.html"
    if not render_html.exists():
        logger.error(
            "'%s' has no render.html yet — run `reload.py show %s --render "
            "threejs` to generate one.", run["run_id"], run["run_id"],
        )
        sys.exit(1)
    webbrowser.open(render_html.as_uri())
    print(f"[Reload] opened '{render_html}' in your browser.")


def _resolve_cli_args(run: dict) -> dict:
    """The run's own recorded CLI invocation, or pipeline.py's defaults for
    a run that predates cli_args tracking — with a clear warning either
    way, since anything not explicitly overridden on this reload.py call
    falls back to whichever of these two was used."""
    cli_args = run.get("cli_args")
    if cli_args:
        return cli_args
    logger.warning(
        "no recorded CLI invocation for '%s' (predates cli_args tracking) — "
        "falling back to pipeline.py's defaults (--input=%s) for anything not "
        "explicitly overridden below; pass --input etc. if that's wrong",
        run["run_id"], _DEFAULT_CLI_ARGS["input"],
    )
    return dict(_DEFAULT_CLI_ARGS)


def action_resume(run: dict, render: str, config_path: str, overrides: dict) -> int:
    """Continue a run from its last checkpointed stage. Replays the run's
    original CLI args by default; `overrides` (--input/--run-mode/
    --resume-from/etc. on this call) take precedence for this launch only
    and never touch the run's own recorded cli_args.

    A run with no completed stage resumes under its own run_id from stage
    1, instead of being replaced by a new run. A --run-mode change past
    the "edges"/"fa2" checkpoint forces the resume point back to "naming"
    automatically (pipeline.py can only apply a mode change from there
    on — see run_init's run_mode × resume_from check).
    """
    explicit_stage = overrides.get("resume_from")
    requested_mode = overrides.get("run_mode")
    mode_changed   = requested_mode is not None and requested_mode != run.get("run_mode")

    stage = explicit_stage or run_store.last_completed_stage(run["run_id"])
    auto_forced = explicit_stage is None and mode_changed and stage in ("edges", "fa2")
    if auto_forced:
        stage = "naming"

    # A "complete" run has nothing left to auto-resume into; an explicit
    # --resume-from or a --run-mode change (handled above) is what allows
    # redoing from an earlier checkpoint.
    if run["status"] == "complete" and explicit_stage is None and not auto_forced:
        logger.error(
            "'%s' is already complete — use `restart` for a full re-run, "
            "`resume %s --resume-from STAGE` to redo from a specific "
            "checkpoint onward, `resume %s --run-mode MODE` to switch mode, "
            "or `show` to view it",
            run["run_id"], run["run_id"], run["run_id"],
        )
        sys.exit(1)

    # pipeline.py's own --resume-from resolution always targets the active run.
    run_store.set_active_run_id(run["run_id"])

    cli_args = _resolve_cli_args(run)
    # --yes: the user already confirmed intent by naming this run_id
    # explicitly. No-op for a checkpointed stage; for a stage-less run it
    # lets pipeline.py restart from the top non-interactively.
    override = {"resume_from": stage, "fresh": False, "yes": True, **overrides}
    cmd = _rebuild_command(cli_args, override=override)
    # Always pass --render explicitly, even "none" — never rely on
    # pipeline.py's own default for the subprocess.
    cmd += ["--render", render]

    if stage is None:
        logger.info("resuming '%s' from the top (no checkpoint reached yet): %s", run["run_id"], " ".join(cmd))
    elif explicit_stage:
        logger.info("resuming '%s' from stage '%s' (forced via --resume-from): %s", run["run_id"], stage, " ".join(cmd))
    elif auto_forced:
        logger.info("resuming '%s' from stage '%s' (run_mode changed '%s' → '%s'): %s",
                     run["run_id"], stage, run.get("run_mode"), requested_mode, " ".join(cmd))
    else:
        logger.info("resuming '%s' from stage '%s': %s", run["run_id"], stage, " ".join(cmd))
    # check=False + explicit returncode propagation, not check=False and
    # silence: a cron job calling `reload.py resume` needs its own exit
    # code to reflect whatever pipeline.py actually did underneath.
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def action_restart(run: dict, render: str, config_path: str, overrides: dict) -> int:
    """Full re-run as a brand-new run, using the run's original CLI flags
    by default (see _resolve_cli_args), with `overrides` taking precedence
    for this launch only. The run being restarted is left completely
    untouched."""
    cli_args = _resolve_cli_args(run)

    # --fresh guarantees a brand-new run_id no matter what's currently
    # active, and never resumes.
    override = {"resume_from": None, "fresh": True, **overrides}
    cmd = _rebuild_command(cli_args, override=override)
    # Always explicit — see the matching comment in action_resume.
    cmd += ["--render", render]

    logger.info("restarting '%s' from scratch (new run): %s", run["run_id"], " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def _patch_labels_everywhere(run_id: str, patch: pd.DataFrame) -> int:
    """Overwrite cluster_label/cluster_topic (joined on 'id') in every
    persisted posts.parquet from 'naming' onward, plus output/ if the run
    completed. 'adr' is deliberately left untouched — it's the state right
    before naming ever ran, which is what makes it safe to relabel from
    again. Nothing else (cluster_id, embeddings, Nova/ADEPT columns,
    edges, FA2 coordinates) is read or modified."""
    root = run_store.run_dir(run_id)
    targets = [root / "stages" / s / "posts.parquet" for s in run_store.STAGE_ORDER[1:]]
    targets += [root / "output" / "clusters.parquet", root / "output" / "clusters_render.parquet"]

    n = 0
    for path in targets:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = (df.drop(columns=["cluster_label", "cluster_topic"], errors="ignore")
                .merge(patch, on="id", how="left"))
        df.to_parquet(path, index=False)
        n += 1
    return n


def action_relabel(run: dict, force_skip_llm: bool) -> None:
    """Redo Naming (stage 4) only: reloads the 'adr' checkpoint, re-labels
    clusters, and patches cluster_label/cluster_topic into every persisted
    copy of posts_df. Clustering, Nova/ADEPT, edges, and FA2 are never
    touched or recomputed. Typical use: re-run after the LLM was
    unavailable and c-TF-IDF fallback labels were used the first time."""
    run_id = run["run_id"]

    # Exits if this run predates config-snapshot tracking — same rule
    # `resume`/`show` apply to their own config reads.
    cfg         = run_store.load_config_snapshot(run_id)
    config_path = str(run_store.config_snapshot_path(run_id))

    # Exits if this run never reached the 'adr' stage.
    adr_posts, _ = run_store.load_stage_checkpoint(cfg, "adr")

    # naming.py pulls in sklearn/bertopic/stopwordsiso — kept out of every
    # other command's import path (same reasoning as action_show's lazy
    # `import render`).
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    import naming as naming_module

    run_store.acquire_lock(run_id)
    try:
        logger.info("relabeling '%s' from its 'adr' checkpoint (%d posts)...", run_id, len(adr_posts))
        relabeled = naming_module.run(
            adr_posts, config_path,
            force_skip_llm=(True if force_skip_llm else None),
        )
        patch     = relabeled[["id", "cluster_label", "cluster_topic"]]
        n_patched = _patch_labels_everywhere(run_id, patch)
        run_store.update_run(run_id, relabeled_at=datetime.now().isoformat(timespec="seconds"))
    finally:
        run_store.release_lock(run_id)

    print(f"[Reload] '{run_id}' relabeled — {n_patched} file(s) updated "
          f"(naming checkpoint onward, and output/ where present). "
          f"Clustering, Nova/ADEPT, edges and layout untouched.")

    # Refreshes the 3-D view only if one already exists (render_html) and
    # the run reached output/ — never invents a render for a run that
    # never had one. open_browser is always False: this is a background
    # refresh, not a user asking to see the viewer (that's `show --render threejs`).
    render_html = run_store.run_dir(run_id) / "render.html"
    out_dir = run_store.output_dir(run_id)
    posts_path, edges_path = out_dir / "clusters.parquet", out_dir / "edges.parquet"
    if render_html.exists() and posts_path.exists():
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        import render as render_module
        posts_df = pd.read_parquet(posts_path)
        edges_df = pd.read_parquet(edges_path) if edges_path.exists() else pd.DataFrame()
        # Non-fatal: relabeling already succeeded and printed above — a
        # failed view refresh shouldn't make `relabel` look like it failed.
        try:
            render_module.run(posts_df, edges_df, config_path=config_path, open_browser=False)
            print(f"[Reload] render.html regenerated with the new labels "
                  f"(not opened — see '{render_html}', or Open 3D view in the dashboard).")
        except Exception:
            logger.warning(
                "'%s' relabeled successfully, but refreshing render.html failed — "
                "run `python reload.py show %s --render threejs` to regenerate it manually",
                run_id, run_id,
            )


# Scope is hard-coded, not flag-selected: `models:` (owned entirely by
# config.yaml) plus the `model_slot` key of nova/adept/naming, plus
# nova/adept's `parallel_clusters` — also config.yaml-only, and tied to
# model_slot in practice (how many clusters you can run concurrently
# depends on whether model_slot points at a fast API model or a local
# one; see config.yaml's own comment on parallel_clusters). Every other
# nova/adept/naming key comes from advanced.yaml and must never be
# touched here. advanced.yaml carries no `model_slot` / `parallel_clusters`
# under nova/adept and no `models:`/`naming:` block at all, so none of
# these paths can ever clobber an advanced.yaml-owned key.
_MODEL_ROUTING_SECTIONS = ("nova", "adept", "naming")
_EXTRA_ROUTING_KEYS = {
    "nova":  ("parallel_clusters",),
    "adept": ("parallel_clusters",),
    # naming has no parallel_clusters key.
}


def _routing_summary(cfg: dict) -> dict:
    """Compact, printable view of the model-routing slice, for the change
    log below only — the actual patch copies the full models.heavy/
    models.light dicts (url, timeout, api_key, ...), not just the model
    name shown here."""
    models = cfg.get("models") or {}
    summary = {f"models.{slot}": (models.get(slot) or {}).get("model") for slot in ("heavy", "light")}
    for section in _MODEL_ROUTING_SECTIONS:
        section_cfg = cfg.get(section) or {}
        summary[f"{section}.model_slot"] = section_cfg.get("model_slot")
        for key in _EXTRA_ROUTING_KEYS.get(section, ()):
            summary[f"{section}.{key}"] = section_cfg.get(key)
    return summary


def _validate_model_routing(cfg: dict, source: str) -> None:
    """Exit with a clear message if `cfg` (loaded from `source`) is missing
    any piece sync-models needs, rather than silently writing a
    half-patched snapshot."""
    missing = [k for k in ("models.heavy", "models.light") if k.split(".")[1] not in (cfg.get("models") or {})]
    missing += [f"{s}.model_slot" for s in _MODEL_ROUTING_SECTIONS if not (cfg.get(s) or {}).get("model_slot")]
    if missing:
        logger.error(
            "'%s' is missing expected model-routing key(s): %s — sync-models "
            "needs models.heavy / models.light / nova.model_slot / "
            "adept.model_slot / naming.model_slot all present",
            source, ", ".join(missing),
        )
        sys.exit(1)


def _patch_model_routing(snapshot: dict, live_cfg: dict) -> dict:
    """Return a copy of `snapshot` with exclusively the model-routing slice
    overwritten from `live_cfg`. Every other key — algorithm tuning,
    storage.processed_dir, any nova/adept key besides model_slot and
    parallel_clusters — is carried over untouched (shallow copy: sections
    we don't patch keep their original nested dicts by reference, which
    is safe since we never mutate them in place)."""
    patched = dict(snapshot)
    patched["models"] = live_cfg.get("models")
    for section in _MODEL_ROUTING_SECTIONS:
        current = dict(snapshot.get(section) or {})
        live_section = live_cfg[section]
        current["model_slot"] = live_section["model_slot"]
        for key in _EXTRA_ROUTING_KEYS.get(section, ()):
            if key in live_section:
                current[key] = live_section[key]
        patched[section] = current
    return patched


def action_sync_models(run: dict, config_path: str, dry_run: bool) -> None:
    """Patch a run's frozen config_snapshot.yaml in place with the live
    config.yaml's model routing ONLY — see the module comment above for
    exact scope. Safe with respect to advanced.yaml (no overlapping keys),
    but NOT retroactive: stages already checkpointed were computed under
    the old routing and keep their results as-is. Follow up with
    `resume <run_id> --resume-from STAGE` to actually apply the new
    routing to stages not yet (re)computed."""
    run_id = run["run_id"]

    live_cfg = run_store.load_config(config_path)
    _validate_model_routing(live_cfg, config_path)

    snapshot = run_store.load_config_snapshot(run_id)  # exits if this run has no snapshot
    old_view, new_view = _routing_summary(snapshot), _routing_summary(live_cfg)

    changed = {k: (old_view[k], new_view[k]) for k in new_view if old_view[k] != new_view[k]}
    if not changed:
        print(f"[Reload] '{run_id}' already matches {config_path}'s model routing — nothing to do.")
        return

    print(f"[Reload] model routing change for '{run_id}':")
    for key, (old, new) in changed.items():
        print(f"  {key}: {old!r} -> {new!r}")

    if dry_run:
        print("[Reload] --dry-run: config_snapshot.yaml left untouched.")
        return

    run_store.acquire_lock(run_id)  # refuses if a resume/restart is actively running on this run_id
    try:
        run_store.save_config_snapshot(run_id, _patch_model_routing(snapshot, live_cfg))
        run_store.update_run(run_id, models_synced_at=datetime.now().isoformat(timespec="seconds"))
    finally:
        run_store.release_lock(run_id)

    print(f"[Reload] '{run_id}' config_snapshot.yaml updated — model routing only, nothing else "
          f"touched. Already-checkpointed stages ran under the OLD routing and are not "
          f"recomputed; run `python reload.py resume {run_id} --resume-from STAGE` to apply "
          f"the new routing from a given stage onward.")


def action_trash(run_id: str) -> None:
    """Move a run to runs/_trash/. Reversible via `restore` until purged."""
    dst = run_store.trash_run(run_id)
    print(f"[Reload] '{run_id}' moved to {dst} — "
          f"`reload.py restore {run_id}` restores it; "
          f"`reload.py purge {run_id}` deletes it permanently.")


def action_restore(run_id: str) -> None:
    """Move a run back out of runs/_trash/. Reverses `trash`."""
    dst = run_store.restore_run(run_id)
    print(f"[Reload] '{run_id}' restored → {dst}")


def action_purge(run_id: str | None, purge_all: bool) -> None:
    """Permanently delete one trashed run, or all of them if purge_all."""
    if purge_all:
        n = run_store.purge_trash()
        print(f"[Reload] permanently deleted {n} run(s) from runs/_trash/.")
    else:
        run_store.purge_trash(run_id)
        print(f"[Reload] '{run_id}' permanently deleted.")


def _rebuild_command(cli_args: dict, override: dict) -> list[str]:
    """Reconstruct a pipeline.py CLI invocation from a saved vars(args) dict
    (or from _DEFAULT_CLI_ARGS for a run that predates cli_args tracking),
    with `override` applied on top."""
    args = {**cli_args, **override}

    cmd = [sys.executable, "pipeline.py",
           "--input",  args["input"]]

    if args.get("limit")       is not None: cmd += ["--limit",       str(args["limit"])]
    if args.get("resume_from"):             cmd += ["--resume-from", args["resume_from"]]
    if args.get("run_mode"):                cmd += ["--run-mode",    args["run_mode"]]
    if args.get("use_k_cache"):             cmd.append("--use-k-cache")
    if args.get("set_k")       is not None: cmd += ["--set-k",       str(args["set_k"])]
    if args.get("k_range"):                 cmd += ["--k-range",     str(args["k_range"][0]), str(args["k_range"][1])]
    if args.get("k_fast"):                  cmd.append("--k-fast")
    if args.get("skip_naming"):             cmd.append("--skip-naming")
    if args.get("force_nova"):              cmd.append("--force-nova")
    if args.get("fresh"):                   cmd.append("--fresh")
    if args.get("yes"):                     cmd.append("--yes")
    if args.get("debug"):                   cmd.append("--debug")

    return cmd


def _add_override_args(p: argparse.ArgumentParser) -> None:
    """Flags shared by `resume`/`restart` to override the recorded CLI
    invocation for this launch only — mirrors pipeline.py's own flag names
    and choices. Anything left None/False falls through to whatever
    _resolve_cli_args() returns (the run's recorded value)."""
    p.add_argument("--input", default=None,
                    help="Override the raw input directory for this launch only.")
    p.add_argument("--run-mode", choices=["full", "clustering_only"], default=None, dest="run_mode",
                    help="Override run_mode for this launch only.")
    p.add_argument("--limit", type=int, default=None,
                    help="Override the post limit for this launch only.")
    p.add_argument("--set-k", type=int, default=None, dest="set_k",
                    help="Override --set-k for this launch only.")
    p.add_argument("--k-range", type=int, nargs=2, default=None, dest="k_range", metavar=("MIN", "MAX"),
                    help="Override --k-range for this launch only.")
    p.add_argument("--k-fast", action="store_true", dest="k_fast",
                    help="Force --k-fast on for this launch (can't unset a recorded one).")
    p.add_argument("--use-k-cache", action="store_true", dest="use_k_cache",
                    help="Force --use-k-cache on for this launch (can't unset a recorded one).")
    p.add_argument("--skip-naming", action="store_true", dest="skip_naming",
                    help="Force --skip-naming on for this launch (can't unset a recorded one).")
    p.add_argument("--force-nova", action="store_true", dest="force_nova",
                    help="Force --force-nova on for this launch (can't unset a recorded one).")


def _collect_overrides(args: argparse.Namespace) -> dict:
    """Only the fields actually passed on this reload.py invocation, so
    anything not mentioned here falls back to the run's own recorded value
    untouched (see _rebuild_command). `resume_from` only exists on the
    `resume` subcommand's args (not `restart`'s) — getattr covers both
    without needing to know which subcommand called this."""
    overrides = {}
    if getattr(args, "resume_from", None) is not None:
        overrides["resume_from"] = args.resume_from
    if args.input     is not None: overrides["input"]    = args.input
    if args.run_mode  is not None: overrides["run_mode"] = args.run_mode
    if args.limit     is not None: overrides["limit"]    = args.limit
    if args.set_k     is not None: overrides["set_k"]    = args.set_k
    if args.k_range   is not None: overrides["k_range"]  = args.k_range
    if args.k_fast:                overrides["k_fast"]     = True
    if args.use_k_cache:           overrides["use_k_cache"] = True
    if args.skip_naming:           overrides["skip_naming"] = True
    if args.force_nova:            overrides["force_nova"]  = True
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reload.py",
        description="Browse and act on past HeronLoom pipeline runs.",
    )

    sub = parser.add_subparsers(dest="command", required=False, metavar="COMMAND")

    p_list = sub.add_parser("list", help="List every run found.")
    p_list.add_argument("--trash", action="store_true",
                         help="List runs currently in runs/_trash/ instead of active runs.")

    p_show = sub.add_parser("show", help="Render a run's saved state (no recompute).")
    p_show.add_argument("run_id", metavar="RUN_ID")
    p_show.add_argument("--render", choices=["none", "threejs"], default="none",
                         help="Open the Three.js viewer after loading (default: none).")
    p_show.add_argument("--no-browser", action="store_true", dest="no_browser",
                         help="With --render threejs: write render.html but don't open a "
                              "browser tab for it (used by the dashboard, which serves the "
                              "file itself instead). No effect if --render is left at 'none'.")

    p_open = sub.add_parser("open", help="Open a run's existing render.html in the browser (no recompute).")
    p_open.add_argument("run_id", metavar="RUN_ID")

    p_search = sub.add_parser("search", help="Ask a question against a run's saved output (no recompute).")
    p_search.add_argument("run_id", metavar="RUN_ID", nargs="?", default=None,
                           help="Run id to search. Omit to use the active run (see `reload.py list`).")
    p_search.add_argument("query", metavar="QUERY", help="The question or topic to search")
    p_search.add_argument("--run-mode", choices=["full", "clustering_only"], default=None,
                           help="Force full/clustering_only instead of auto-detecting from the "
                                "run's saved output.")
    p_search.add_argument("--no-decompose", action="store_true", dest="no_decompose",
                           help="Skip LLM query decomposition; retrieve on the original query only.")
    p_search.add_argument("--budget", type=int, default=None,
                           help="Total post budget across all retrieved groups (default: search.py's "
                                "own default, currently 3000). Components are never split to fit it.")

    p_resume = sub.add_parser("resume", help="Continue a run from checkpoint, or a chosen stage via --resume-from.")
    p_resume.add_argument("run_id", metavar="RUN_ID")
    p_resume.add_argument("--render", choices=["none", "threejs"], default="threejs",
                           help="Write render.html after the run (default: threejs). Never opens "
                                "a browser tab — this re-runs pipeline.py under the hood, which "
                                "also runs unattended (cron/automation); pass --render none to "
                                "skip it.")
    p_resume.add_argument("--resume-from", choices=run_store.STAGE_ORDER, default=None, dest="resume_from",
                           help="Resume from this checkpoint stage instead of the furthest one "
                                "found on disk — e.g. `--resume-from naming --run-mode full` to "
                                "add Nova/ADEPT to a run that finished as clustering_only. Works "
                                "even if the run's status is already 'complete'. Not available on "
                                "`restart`, which always starts a brand-new run from stage 1.")
    _add_override_args(p_resume)

    p_restart = sub.add_parser("restart", help="Full re-run as a brand-new run (new run_id).")
    p_restart.add_argument("run_id", metavar="RUN_ID")
    p_restart.add_argument("--render", choices=["none", "threejs"], default="threejs",
                            help="Write render.html after the run (default: threejs). Never opens "
                                 "a browser tab, same reasoning as `resume` above. Pass --render "
                                 "none to skip it.")
    _add_override_args(p_restart)

    p_relabel = sub.add_parser("relabel", help="Redo naming (cluster labels) only, in place.")
    p_relabel.add_argument("run_id", metavar="RUN_ID")
    p_relabel.add_argument("--force-skip-llm", action="store_true", dest="force_skip_llm",
                            help="Skip the LLM even if available; c-TF-IDF keywords only "
                                 "(mirrors naming.py's own fallback). Default: use the LLM "
                                 "if reachable, same auto-detection as a normal run.")

    p_sync = sub.add_parser(
        "sync-models",
        help="Patch a run's config_snapshot.yaml with the live config.yaml's model "
             "routing only (models: heavy/light; nova/adept/naming model_slot; "
             "nova/adept parallel_clusters).",
    )
    p_sync.add_argument("run_id", metavar="RUN_ID")
    p_sync.add_argument("--dry-run", action="store_true", dest="dry_run",
                         help="Print what would change without writing anything.")

    p_trash = sub.add_parser("trash", help="Move a run to runs/_trash/ (reversible).")
    p_trash.add_argument("run_id", metavar="RUN_ID")

    p_restore = sub.add_parser("restore", help="Move a run back out of runs/_trash/.")
    p_restore.add_argument("run_id", metavar="RUN_ID")

    p_purge = sub.add_parser("purge", help="Permanently delete one trashed run, or empty the trash with --all.")
    p_purge.add_argument("run_id", metavar="RUN_ID", nargs="?", default=None)
    p_purge.add_argument("--all", action="store_true", help="Permanently empty runs/_trash/.")

    return parser


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate action."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        # No verb given: show the same full help as --help (git/docker/kubectl
        # convention), not argparse's terse "required: COMMAND" error. Exit 1
        # (not 0) since this is still a missing-argument case, not a genuine
        # --help request — scripts that call reload.py with no command by
        # mistake still see it as a failure.
        parser.print_help()
        sys.exit(1)

    if args.command == "list":
        runs = run_store.list_trashed_runs() if args.trash else run_store.list_runs()
        print_table(runs, trashed=args.trash)

    elif args.command == "show":
        action_show(find_run(args.run_id), CONFIG_PATH, args.render, args.no_browser)

    elif args.command == "open":
        action_open(find_run(args.run_id))

    elif args.command == "search":
        run_id = args.run_id or run_store.get_active_run_id()
        if not run_id:
            logger.error("no run id given and no active run set — run `python reload.py list` "
                          "first, or pass a run id: `python reload.py search RUN_ID \"question\"`")
            sys.exit(1)
        action_search(find_run(run_id), args.query, CONFIG_PATH, args.run_mode,
                      args.no_decompose, budget=args.budget)

    elif args.command == "resume":
        sys.exit(action_resume(find_run(args.run_id), args.render, CONFIG_PATH, _collect_overrides(args)))

    elif args.command == "restart":
        sys.exit(action_restart(find_run(args.run_id), args.render, CONFIG_PATH, _collect_overrides(args)))

    elif args.command == "relabel":
        action_relabel(find_run(args.run_id), args.force_skip_llm)

    elif args.command == "sync-models":
        action_sync_models(find_run(args.run_id), CONFIG_PATH, args.dry_run)

    elif args.command == "trash":
        action_trash(args.run_id)

    elif args.command == "restore":
        action_restore(args.run_id)

    elif args.command == "purge":
        if args.all and args.run_id:
            parser.error("purge: pass a RUN_ID or --all, not both")
        if not args.all and not args.run_id:
            parser.error("purge: pass a RUN_ID or --all")
        action_purge(args.run_id, purge_all=args.all)


if __name__ == "__main__":
    main()