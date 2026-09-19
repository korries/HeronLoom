"""Run storage and lifecycle management.

One run is a self-contained folder under ``runs/``, created once at launch.
It is never renamed or copied, and stays in place for its whole lifetime
unless explicitly moved to ``_trash/`` (see below)::

    runs/
    ├── <run_id>/
    │   ├── run.json                status, run_mode, cli_args, started_at, finished_at, stage,
    │   │                           relabeled_at (set by `reload.py relabel`, if ever used)
    │   ├── config_snapshot.yaml    config frozen at launch; resume reads this,
    │   │                           never the live config.yaml (see resolve_run_for_resume)
    │   ├── <ambient files>         stage output written via cfg["storage"]["processed_dir"]
    │   │                           (k_cache.pkl, cluster_stats.parquet, nova_assignments.parquet, ...)
    │   ├── stages/
    │   │   ├── adr/{posts.parquet, meta.json}
    │   │   ├── naming/{posts.parquet, meta.json}
    │   │   ├── nova/{posts.parquet, edges.parquet, meta.json}
    │   │   ├── adept/{posts.parquet, edges.parquet, meta.json}
    │   │   ├── edges/{posts.parquet, edges.parquet, meta.json}
    │   │   └── fa2/{posts.parquet, edges.parquet, meta.json}
    │   └── output/                  present only once status == "complete"
    │       └── clusters.parquet, clusters_render.parquet, edges.parquet
    ├── _trash/<run_id>/             soft-deleted runs (see trash_run / purge_trash)
    └── active.json                  {"run_id": "..."}, the single mutable pointer
"""

import contextlib
import json
import os
import shutil
import socket
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import filelock
import pandas as pd
import yaml

from utils.logger import get_logger

logger = get_logger(__name__)


# config.yaml is merged on top of advanced.yaml, winning on overlapping
# keys. reload.py never loads config itself — it only reads from the
# fixed config/config.yaml path, same as every other entry point.

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override onto base, merging nested dicts key by
    key instead of replacing them wholesale."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str) -> dict:
    """Load config_path, merged over advanced.yaml if present.

    advanced.yaml is resolved next to config_path. Missing advanced.yaml is
    not an error — config_path is used on its own.
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    advanced_path = Path(config_path).parent / "advanced.yaml"
    if advanced_path.exists():
        with open(advanced_path, encoding="utf-8") as f:
            advanced_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(advanced_cfg, cfg)

    return cfg


# Shared with pipeline.py's --resume-from validation.
STAGE_ORDER = ["adr", "naming", "nova", "adept", "edges", "fa2"]

RUNS_ROOT = Path("runs")


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON via a temp file + rename, so a crash mid-write never
    leaves a partial file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _read_json(path: Path) -> dict | None:
    """Return the parsed JSON at path, or None if missing or corrupted.

    Corruption is never raised as an exception — e.g. list_runs() surfaces
    it as an "unreadable" run entry instead of crashing; other callers
    just treat None as "nothing there".
    """
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _new_run_id() -> str:
    """Timestamp + short random suffix — sortable, human-readable, never reused."""
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


def run_dir(run_id: str) -> Path:
    return RUNS_ROOT / run_id


def stages_dir(run_id: str) -> Path:
    return run_dir(run_id) / "stages"


def stage_dir(cfg: dict, stage: str) -> Path:
    return Path(cfg["storage"]["processed_dir"]) / "stages" / stage


def output_dir(run_id: str) -> Path:
    return run_dir(run_id) / "output"


def run_json_path(run_id: str) -> Path:
    return run_dir(run_id) / "run.json"


def config_snapshot_path(run_id: str) -> Path:
    return run_dir(run_id) / "config_snapshot.yaml"


def trash_root() -> Path:
    return RUNS_ROOT / "_trash"


def _run_id_from_cfg(cfg: dict) -> str | None:
    """Recover the run_id from a resolved cfg's storage.processed_dir.

    Assumes cfg came from create_run() or resolve_run_for_resume(), which
    always set processed_dir to runs/<run_id>.
    """
    try:
        return Path(cfg["storage"]["processed_dir"]).name
    except (KeyError, TypeError):
        return None


def _active_path() -> Path:
    return RUNS_ROOT / "active.json"


def get_active_run_id() -> str | None:
    data = _read_json(_active_path())
    return data.get("run_id") if data else None


def set_active_run_id(run_id: str | None) -> None:
    """Repoint active.json. Just a pointer — never destroys anything."""
    _atomic_write_json(_active_path(), {"run_id": run_id})


# active.json records which run would be resumed, not whether a process
# is still working on it — the OS-level lock decides that (see below).
def lock_path(run_id: str) -> Path:
    """Informational pid/host/locked_at for display only. Never read to
    decide whether a run is active; see lock_is_live()."""
    return run_dir(run_id) / "pipeline.lock"


def _os_lock_path(run_id: str) -> Path:
    """Path `filelock` locks. Separate from lock_path() so the
    informational JSON write never contends with the OS lock's own fd."""
    return run_dir(run_id) / "pipeline.lock.os"


# FileLock objects held by this process, keyed by run_id. Must stay open
# for the run's duration — an OS lock is only in effect while its fd is.
_held_locks: dict[str, filelock.FileLock] = {}


def lock_holder(run_id: str) -> dict | None:
    """Informational pid/host/locked_at for `run_id`'s lock, or None if
    unlocked. Not a liveness check — see lock_is_live()."""
    return _read_json(lock_path(run_id))


def lock_is_live(run_id: str) -> bool:
    """True if another process currently holds `run_id`'s lock.

    Backed by an OS-level advisory lock (fcntl.flock on POSIX,
    msvcrt.locking on Windows, via `filelock`), released by the kernel on
    process exit regardless of cause — clean exit, crash, kill -9, power
    loss.

    Single source of truth for effective_status(), check_not_locked(),
    and runs_api._lock_summary().
    """
    if run_id in _held_locks:
        return False
    fl = filelock.FileLock(str(_os_lock_path(run_id)), timeout=0)
    try:
        fl.acquire(timeout=0)
    except filelock.Timeout:
        return True
    fl.release()
    return False


def effective_status(run: dict) -> str | None:
    """run["status"], except "in_progress" with no live lock is reported
    as "crashed". Read-only — run.json is never modified here.

    Race: a run checked between writing "in_progress" and acquiring the
    lock is misreported as "crashed" for one tick; self-corrects within
    RUNS_CHECK_INTERVAL_SECONDS.
    """
    status = run.get("status")
    if status != "in_progress":
        return status
    return status if lock_is_live(run["run_id"]) else "crashed"


def check_not_locked(run_id: str) -> None:
    """Exit if `run_id` is locked by another process."""
    if not lock_is_live(run_id):
        return
    holder = lock_holder(run_id)
    detail = (
        f"pid {holder['pid']} on {holder.get('host', '?')} since {holder.get('locked_at', '?')}"
        if holder else "another process"
    )
    logger.error("'%s' is already locked by %s.", run_id, detail)
    sys.exit(1)


def acquire_lock(run_id: str) -> None:
    """Claim `run_id`'s lock for this process. Pair with release_lock()
    in a try/finally."""
    check_not_locked(run_id)
    os_lock_path = _os_lock_path(run_id)
    os_lock_path.parent.mkdir(parents=True, exist_ok=True)
    fl = filelock.FileLock(str(os_lock_path), timeout=0)
    try:
        fl.acquire(timeout=0)
    except filelock.Timeout:
        check_not_locked(run_id)  # race with another acquirer: reports and exits
        return
    _held_locks[run_id] = fl
    _atomic_write_json(lock_path(run_id), {
        "pid": os.getpid(), "host": socket.gethostname(),
        "locked_at": datetime.now().isoformat(timespec="seconds"),
    })


def release_lock(run_id: str) -> None:
    """Release `run_id`'s lock. Safe to call even if never acquired."""
    fl = _held_locks.pop(run_id, None)
    if fl is not None:
        with contextlib.suppress(Exception):
            fl.release()
    with contextlib.suppress(FileNotFoundError):
        _os_lock_path(run_id).unlink()
    with contextlib.suppress(FileNotFoundError):
        lock_path(run_id).unlink()


def load_run(run_id: str) -> dict | None:
    """Return this run's manifest, or None if it has none / is unreadable."""
    return _read_json(run_json_path(run_id))


def _save_run(run_id: str, data: dict) -> None:
    _atomic_write_json(run_json_path(run_id), data)


def update_run(run_id: str, **fields) -> dict:
    """Merge `fields` into run.json and write it back atomically."""
    data = load_run(run_id) or {"run_id": run_id}
    data.update(fields)
    _save_run(run_id, data)
    return data


def _snapshot_config(run_id: str, cfg: dict) -> None:
    path = config_snapshot_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    tmp.replace(path)


def save_config_snapshot(run_id: str, cfg: dict) -> None:
    """Overwrite an existing run's config_snapshot.yaml (vs. freezing it once
    at create_run() time). No validation or merging — the caller
    (reload.py's `sync-models`) must start from load_config_snapshot()'s
    own output and mutate only the keys it means to touch."""
    _snapshot_config(run_id, cfg)


def load_config_snapshot(run_id: str) -> dict:
    """Load the config exactly as it was when this run started.

    Exits rather than falling back to the live config.yaml — a resume must
    never mix configs from different points in time.
    """
    path = config_snapshot_path(run_id)
    if not path.exists():
        logger.error(
            "no config snapshot for run '%s' at %s — cannot safely resume "
            "(this run predates config-snapshot tracking)", run_id, path,
        )
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_run(config_path: str, run_mode: str | None, content_type: str | None = None) -> tuple[str, dict]:
    """Allocate a new run: id, folder, frozen config snapshot, active pointer.

    `content_type`, unlike `run_mode`, is baked into `cfg` here before the
    snapshot is frozen rather than replayed as a runtime override on every
    launch/resume. It's read once, at embedding time, and never again —
    changing it mid-run would desync already-computed embeddings from
    whatever Nova/ADEPT would do with the new value, so there's no
    "override for this invocation" concept for it the way `run_mode` has
    one. None leaves whatever config.yaml already has untouched.

    Returns:
        (run_id, cfg). cfg is loaded via load_config() with
        storage.processed_dir repointed at this run's folder, so any file a
        pipeline stage writes there lands inside the run automatically.
    """
    cfg = load_config(config_path)

    run_id = _new_run_id()
    d = run_dir(run_id)
    (d / "stages").mkdir(parents=True, exist_ok=True)

    cfg.setdefault("storage", {})
    cfg["storage"]["processed_dir"] = str(d)

    if content_type is not None:
        cfg["content_type"] = content_type

    now = datetime.now().isoformat(timespec="seconds")
    _save_run(run_id, {
        "run_id":       run_id,
        "project_name": cfg.get("project_name", "run"),
        "status":       "in_progress",
        "stage":        "started",
        # Initial value only — pipeline.py's run_init() refreshes this on
        # every launch to the mode actually in force. It's a human-facing
        # summary only; see each stage checkpoint's own meta.json for the
        # run_mode that actually produced it.
        "run_mode":     run_mode or cfg.get("run_mode", "full"),
        "cli_args":     None,
        "started_at":   now,
        "finished_at":  None,
    })
    _snapshot_config(run_id, cfg)
    set_active_run_id(run_id)
    logger.info("new run created: '%s'", run_id)
    return run_id, cfg


def resolve_run_for_resume(run_id: str) -> dict:
    """cfg for continuing an existing run — always from its frozen snapshot."""
    cfg = load_config_snapshot(run_id)
    cfg.setdefault("storage", {})
    cfg["storage"]["processed_dir"] = str(run_dir(run_id))
    return cfg


def last_completed_stage(run_id: str) -> str | None:
    """Scan stages/ on disk for the furthest completed checkpoint.

    Reads the filesystem rather than trusting run.json's cached "stage"
    field, so a crash between writing a stage's parquet and updating
    run.json can never strand a resume behind a stale pointer.
    """
    d = stages_dir(run_id)
    if not d.exists():
        return None
    found = None
    for stage in STAGE_ORDER:
        if (d / stage / "posts.parquet").exists():
            found = stage
    return found


_LAST_SHARED_STAGE = "naming"
_STAGE_OUTPUT_PREFIXES = ("nova_", "adept_", "pool_explanations")


def mode_conflict(
    cfg: dict, run_mode: str, previous_run_mode: str | None,
) -> tuple[list[str], list[Path]]:
    """Checkpoints after naming that were produced under another run_mode.

    A stage's run_mode is read from its meta.json, falling back to
    `previous_run_mode` when the meta is missing or has none.

    Returns:
        (other_modes, paths). Both empty when nothing conflicts. Otherwise
        `paths` lists everything after naming, latest first: output/, the
        stage directories, then the nova_*/adept_* files in the run root.
    """
    root  = Path(cfg["storage"]["processed_dir"])
    after = STAGE_ORDER[STAGE_ORDER.index(_LAST_SHARED_STAGE) + 1:]
    later = [s for s in after if stage_dir(cfg, s).exists()]

    produced_under = {
        s: (load_stage_meta(cfg, s) or {}).get("run_mode") or previous_run_mode for s in later
    }
    other_modes = sorted({m for m in produced_under.values() if m not in (None, run_mode)})
    if not other_modes:
        return [], []

    paths = [root / "output"] if (root / "output").exists() else []
    paths += [stage_dir(cfg, s) for s in reversed(later)]
    paths += sorted(
        p for p in root.iterdir() if p.is_file() and p.name.startswith(_STAGE_OUTPUT_PREFIXES)
    )
    return other_modes, paths


def _remove_with_retry(path: Path, attempts: int = 5, delay: float = 0.5) -> None:
    """Delete a file or directory tree, retrying on PermissionError (files
    briefly held open by OneDrive or an antivirus on Windows)."""
    for attempt in range(1, attempts + 1):
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == attempts:
                raise
            time.sleep(delay)


def delete_paths(paths: list[Path]) -> None:
    """Permanently delete `paths` in order. Exits (code 1) on the first failure."""
    for path in paths:
        try:
            _remove_with_retry(path)
        except OSError as e:
            logger.error(
                "could not delete %s (%s: %s) — close anything that has files "
                "under it open (Explorer, a viewer, another Python process) "
                "and relaunch.", path, type(e).__name__, e,
            )
            sys.exit(1)


def mark_in_progress(run_id: str, **fields) -> None:
    """Flag the run in_progress, with its stage pointer set to the last
    checkpoint on disk. `fields` are merged into run.json as well."""
    update_run(
        run_id, status="in_progress", finished_at=None,
        stage=last_completed_stage(run_id) or "started", **fields,
    )


def resolve_or_create_run(
    config_path: str,
    run_mode: str | None,
    auto_yes: bool = False,
    fresh: bool = False,
    explicit_resume_stage: str | None = None,
    content_type: str | None = None,
) -> tuple[str, dict, str | None]:
    """Decide which run a `pipeline.py` invocation targets.

    Never destroys anything: a fresh start only allocates a new run_id and
    repoints active.json. An active run with no completed stage yet is
    resumed under its own run_id rather than replaced.

    `content_type` only ever reaches create_run() below — every resume path
    (explicit --resume-from, or "Y" at either prompt) keeps the frozen
    snapshot's value as-is; see create_run()'s docstring for why it isn't a
    per-invocation override the way run_mode is.

    Returns:
        (run_id, cfg, resume_stage). resume_stage is the stage to reload a
        checkpoint from, or None to start from stage 1.
    """
    if explicit_resume_stage is not None:
        run_id = get_active_run_id()
        if run_id is None or load_run(run_id) is None:
            logger.error(
                "--resume-from '%s' given but there is no active run to resume — "
                "use `python reload.py --list` and `--resume <run_id>` to target "
                "one explicitly.", explicit_resume_stage,
            )
            sys.exit(1)
        check_not_locked(run_id)
        logger.info("resuming active run '%s' from '%s' (--resume-from)", run_id, explicit_resume_stage)
        return run_id, resolve_run_for_resume(run_id), explicit_resume_stage

    if fresh:
        run_id, cfg = create_run(config_path, run_mode, content_type)
        logger.info("--fresh: new run '%s' — any in-progress run is left untouched", run_id)
        return run_id, cfg, None

    active_id = get_active_run_id()
    active    = load_run(active_id) if active_id else None

    if active is None or active.get("status") != "in_progress":
        run_id, cfg = create_run(config_path, run_mode, content_type)
        return run_id, cfg, None

    # Refuse if it's still running elsewhere.
    check_not_locked(active_id)

    stage = last_completed_stage(active_id)
    if stage is None:
        # No checkpoint yet — resume under the same run_id, restarting from
        # stage 1, instead of minting a new one.
        if auto_yes:
            logger.info("-y: restarting '%s' from stage 1 (no checkpoint yet)", active_id)
            return active_id, resolve_run_for_resume(active_id), None

        print(f"\n[Pipeline] Found an in-progress run: '{active_id}' (no checkpoint reached yet).")
        print( "[Pipeline]   Y — resume it, restarting from Embedding")
        print( "[Pipeline]   N — leave it untouched and start a new run")
        answer = _prompt_choice("[Pipeline] Resume? [Y/n]: ", {"y", "n"}, "y")
        if answer == "y":
            return active_id, resolve_run_for_resume(active_id), None

        run_id, cfg = create_run(config_path, run_mode, content_type)
        return run_id, cfg, None

    if auto_yes:
        logger.info("-y: resuming active run '%s' from '%s'", active_id, stage)
        return active_id, resolve_run_for_resume(active_id), stage

    print(f"\n[Pipeline] Found an in-progress run: '{active_id}' (last checkpoint: '{stage}').")
    print( "[Pipeline]   Y — resume from that checkpoint")
    print( "[Pipeline]   N — leave it untouched and start a new run")
    answer = _prompt_choice("[Pipeline] Resume? [Y/n]: ", {"y", "n"}, "y")
    if answer == "y":
        return active_id, resolve_run_for_resume(active_id), stage

    run_id, cfg = create_run(config_path, run_mode, content_type)
    return run_id, cfg, None


def safe_input(prompt: str) -> str | None:
    """input() hardened for CLI use — the one implementation every
    interactive prompt in this project goes through (pipeline.py, nova.py,
    adept.py).

    Returns None instead of raising or hanging when there's no interactive
    terminal: sys.stdin.isatty() is checked before reading, so a
    non-interactive context (cron, CI, a piped invocation) is caught up
    front. EOFError and KeyboardInterrupt also return None.

    Does not strip/lowercase — callers own their own default, since "no
    answer" doesn't mean the same thing at every call site.
    """
    if not sys.stdin.isatty():
        logger.warning("no interactive terminal attached — prompt unanswered")
        return None
    try:
        return input(prompt)
    except EOFError:
        logger.warning("stdin closed — prompt unanswered")
        return None
    except KeyboardInterrupt:
        print()  # move past the ^C echo before the next log line
        logger.warning("interrupted by user — prompt unanswered")
        return None


def _prompt_choice(prompt: str, valid: set[str], default: str) -> str:
    """Prompt until the user enters one of `valid` (case-insensitive) or hits
    Enter for `default`. Anything else is rejected and re-asked. Falls back
    to `default` immediately if there is no interactive terminal to answer
    from (see safe_input) instead of hanging or crashing."""
    valid_str = "/".join(sorted(valid))
    while True:
        raw = safe_input(prompt)
        if raw is None:
            return default
        answer = raw.strip().lower()
        if answer == "":
            return default
        if answer in valid:
            return answer
        print(f"[Pipeline] '{answer}' is not a valid choice — enter one of: {valid_str}.")


def confirm_yes_no(prompt: str) -> bool:
    """Prompt until the user enters y or n (case-insensitive). Anything else,
    including an empty line, is rejected and re-asked. No interactive
    terminal, EOF and Ctrl-C all count as n."""
    while True:
        raw = safe_input(prompt)
        if raw is None:
            return False
        answer = raw.strip().lower()
        if answer == "y":
            return True
        if answer == "n":
            return False
        print(f"[Pipeline] '{answer}' is not a valid choice — enter one of: n/y.")


def save_stage_checkpoint(
    cfg: dict, stage: str, posts_df: pd.DataFrame, edges_df: pd.DataFrame = None,
    run_mode: str | None = None,
) -> None:
    """Persist posts_df (and edges_df if given) under stages/<stage>/, then
    advance run.json's "stage" pointer.

    Args:
        run_mode: The effective run_mode ("full" / "clustering_only") in
            force for this checkpoint. Recorded in meta.json so a later
            resume can tell which mode actually produced it, even if the
            run's creation-time run_mode has since changed via
            --run-mode. See run_init()'s run_mode × resume_from
            consistency check.
    """
    sdir = stage_dir(cfg, stage)
    sdir.mkdir(parents=True, exist_ok=True)

    posts_df.to_parquet(sdir / "posts.parquet", index=False)
    if edges_df is not None:
        edges_df.to_parquet(sdir / "edges.parquet", index=False)

    # Written only after both parquet files are flushed, so run.json's
    # "stage" pointer never points at incomplete data.
    _atomic_write_json(sdir / "meta.json", {
        "stage":    stage,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "n_posts":  len(posts_df),
        "n_edges":  len(edges_df) if edges_df is not None else None,
        "run_mode": run_mode,
    })

    run_id = _run_id_from_cfg(cfg)
    if run_id:
        update_run(run_id, stage=stage)
    logger.info("checkpoint saved @ '%s' (run_mode=%s) → %s", stage, run_mode, sdir)


def load_stage_meta(cfg: dict, stage: str) -> dict | None:
    """Read a stage's meta.json sidecar (stage, saved_at, n_posts, n_edges,
    run_mode) without touching the parquet files.

    Returns None if the stage has no checkpoint, or its meta.json is
    missing or unreadable — callers must treat that as unknown provenance.
    """
    return _read_json(stage_dir(cfg, stage) / "meta.json")


def load_stage_checkpoint(cfg: dict, stage: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reload posts_df (and edges_df if present) for a given stage.

    Raises:
        SystemExit: if the checkpoint parquet for this stage does not exist.
    """
    sdir = stage_dir(cfg, stage)
    posts_path = sdir / "posts.parquet"
    if not posts_path.exists():
        logger.error(
            "no checkpoint for stage %r: %s — run at least up to that stage first", stage, posts_path
        )
        sys.exit(1)
    posts_df   = pd.read_parquet(posts_path)
    edges_path = sdir / "edges.parquet"
    edges_df   = pd.read_parquet(edges_path) if edges_path.exists() else None
    return posts_df, edges_df


def save_state(posts_df: pd.DataFrame, edges_df: pd.DataFrame, cfg: dict) -> None:
    """Persist the final post table and edge list under this run's output/,
    and mark the run complete.

    Writes clusters.parquet (full table, incl. embedding columns) and
    clusters_render.parquet (same, without embedding columns, for
    render.py), plus edges.parquet. Filenames are fixed regardless of
    cfg, so reload.py can find them without config.yaml.
    """
    run_id  = _run_id_from_cfg(cfg)
    out_dir = Path(cfg["storage"]["processed_dir"]) / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    posts_df.to_parquet(out_dir / "clusters.parquet", index=False)

    render_cols = [c for c in posts_df.columns if c not in ("embedding_raw", "embedding_lda")]
    posts_df[render_cols].to_parquet(out_dir / "clusters_render.parquet", index=False)

    edges_df.to_parquet(out_dir / "edges.parquet", index=False)

    if run_id:
        update_run(
            run_id, status="complete", stage="complete",
            finished_at=datetime.now().isoformat(timespec="seconds"),
        )
    logger.info("state saved → %s", out_dir)


def save_checkpoint(
    cfg: dict, stage: str, complete: bool = False, cli_args: dict | None = None
) -> None:
    """Update run.json's stage/status (and cli_args, if given)."""
    run_id = _run_id_from_cfg(cfg)
    if run_id is None:
        return
    fields = {"stage": stage, "status": "complete" if complete else "in_progress"}
    if complete:
        fields["finished_at"] = datetime.now().isoformat(timespec="seconds")
    if cli_args is not None:
        fields["cli_args"] = cli_args
    update_run(run_id, **fields)


def list_runs() -> list[dict]:
    """Every run under runs/ (excluding _trash), sorted by started_at.

    Reads only run.json per run, never a parquet file, so this stays fast
    regardless of run count. A run with a missing or corrupt run.json is
    still listed, with status "unreadable".
    """
    if not RUNS_ROOT.exists():
        return []

    active_id = get_active_run_id()
    runs = []
    for child in sorted(RUNS_ROOT.iterdir()):
        if not child.is_dir() or child.name == "_trash":
            continue
        run_id = child.name
        data = load_run(run_id)
        if data is None:
            data = {
                "run_id": run_id, "project_name": "?", "stage": None,
                "status": "unreadable", "run_mode": None, "cli_args": None,
                "started_at": None, "finished_at": None,
            }
        data["active"] = (run_id == active_id)
        runs.append(data)

    runs.sort(key=lambda r: r.get("started_at") or "")
    return runs


def trash_run(run_id: str) -> Path:
    """Move runs/<run_id>/ → runs/_trash/<run_id>/. Reversible (it's just a move);
    also clears active.json if this run was the active one."""
    src = run_dir(run_id)
    if not src.exists():
        logger.error("no run '%s' to trash", run_id)
        sys.exit(1)

    # list_trashed_runs() shows run.json's raw "status", not
    # effective_status() — a trashed run has no process left to check a
    # lock against, so this is the one place a crashed run's status gets
    # corrected on disk.
    run = load_run(run_id)
    if run is not None:
        run["run_id"] = run_id
        resolved = effective_status(run)
        if resolved != run.get("status"):
            update_run(run_id, status=resolved)

    dst_root = trash_root()
    dst_root.mkdir(parents=True, exist_ok=True)
    dst = dst_root / run_id
    if dst.exists():
        # Unlikely (run_ids carry a random suffix), but never overwrite an
        # existing trashed run.
        dst = dst_root / f"{run_id}-{uuid.uuid4().hex[:4]}"

    shutil.move(str(src), str(dst))

    if get_active_run_id() == run_id:
        set_active_run_id(None)
        logger.warning("'%s' was the active run — active.json cleared", run_id)

    logger.info("'%s' moved to trash → %s", run_id, dst)
    return dst


def load_trashed_run(run_id: str) -> dict | None:
    """Like load_run(), but reads a run's manifest from _trash/ instead of
    runs/ — trashed runs are never visible to load_run()/run_json_path()."""
    return _read_json(trash_root() / run_id / "run.json")


def list_trashed_runs() -> list[dict]:
    """Every run currently sitting in runs/_trash/, same shape as
    list_runs() minus "active" (a trashed run is never the active one)."""
    root = trash_root()
    if not root.exists():
        return []

    runs = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        run_id = child.name
        data = load_trashed_run(run_id)
        if data is None:
            data = {
                "run_id": run_id, "project_name": "?", "stage": None,
                "status": "unreadable", "run_mode": None, "cli_args": None,
                "started_at": None, "finished_at": None,
            }
        data["run_id"] = run_id
        data["active"] = False
        runs.append(data)

    runs.sort(key=lambda r: r.get("started_at") or "")
    return runs


def restore_run(run_id: str) -> Path:
    """Move runs/_trash/<run_id>/ back to runs/<run_id>/. Reverses trash_run().

    Never overwrites: refuses if a run with this id already exists outside
    the trash (shouldn't normally happen, since run_ids carry a random
    suffix — see _new_run_id / trash_run's own collision handling)."""
    src = trash_root() / run_id
    if not src.exists():
        logger.error("'%s' not found in trash — use `reload.py list --trash` to see what's there", run_id)
        sys.exit(1)

    dst = run_dir(run_id)
    if dst.exists():
        logger.error(
            "cannot restore '%s' — a run with this id already exists at %s; "
            "resolve the conflict manually before retrying", run_id, dst,
        )
        sys.exit(1)

    shutil.move(str(src), str(dst))
    logger.info("'%s' restored from trash → %s", run_id, dst)
    return dst


def purge_trash(run_id: str | None = None) -> int:
    """Permanently delete one trashed run (run_id given) or every trashed
    run (run_id=None). Returns the number of runs deleted."""
    root = trash_root()
    if run_id is not None:
        target = root / run_id
        if not target.exists():
            logger.error("'%s' not found in trash", run_id)
            sys.exit(1)
        shutil.rmtree(target)
        logger.info("permanently deleted '%s'", run_id)
        return 1

    if not root.exists():
        return 0
    n = 0
    for child in sorted(root.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
            n += 1
    logger.info("permanently deleted %d run(s) from trash", n)
    return n