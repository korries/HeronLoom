"""
pipeline.py — HeronLoom orchestrator

Raw posts in, a laid-out 3-D graph out. Adaptive Discriminant Refinement
(ADR / TopiCLEAR) always runs as part of clustering — there's no flag to
disable it.

    python pipeline.py
    python pipeline.py --limit 1000

Full flag reference (including --resume-from / --run-mode / --fresh
semantics): ``python pipeline.py --help``. Stage breakdown, input format,
and resume/recovery details: see README.md.
"""

from _bootstrap import ensure_venv

ensure_venv()

import argparse
import hashlib
import json
import logging
import sys
import textwrap
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from threadpoolctl import threadpool_limits

load_dotenv()

from utils.logger import get_logger, set_console_level, setup_logger

setup_logger()
# Fixed name, not __name__ — this can run as "__main__" directly or as a
# subprocess; see get_logger()'s docstring in utils/logger.py.
logger = get_logger("pipeline")

CONFIG_PATH = "config/config.yaml"

# Stage modules live in src/; the repository root is already sys.path[0]
# because this file is the script being run. Nothing is pip-installed on
# purpose — see docs/ARCHITECTURE.md#source-layout-and-imports. Anything that
# is not one of the four entry points (pytest, for one) must do the same.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import adept as adept_module
import adr as adr_module
import edges as edges_module
import embedding as emb_module
import fa2_layout as fa2_module
import k_estimator_gmm as k_estimator_module
import k_estimator_hdbscan as hdbscan_module
import naming as naming_module
import nova as nova_module
import run_store

_BOX_WIDTH = 74  # fixed inner width so every banner in the run lines up


def _box(lines: list[str], title: str = "") -> str:
    """Render *lines* inside a fixed-width Unicode box.

    Long lines are word-wrapped onto extra rows rather than overflowing
    the border.
    """
    inner      = _BOX_WIDTH - 2
    text_width = inner - 2  # "│ " / " │" padding

    wrapped: list[str] = []
    for line in lines:
        if not line:
            wrapped.append("")
        else:
            wrapped.extend(textwrap.wrap(line, text_width) or [""])

    rows = ["┌" + "─" * inner + "┐"]
    if title:
        rows += ["│" + title.center(inner) + "│", "├" + "─" * inner + "┤"]
    rows += [f"│ {line:<{inner - 1}}│" for line in wrapped]
    rows.append("└" + "─" * inner + "┘")
    return "\n".join(rows)


def _prompt_gate_override(msg_lines: list[str], title: str) -> bool:
    """Show a gate-failure banner and ask, once, whether to proceed anyway.

    Fails closed: a non-interactive context, EOFError, or Ctrl-C all come
    back from safe_input() as None, treated here as "quit" — same as
    pressing Q. Independent of --yes/-y, which only covers the separate
    resume-an-active-run prompt in resolve_or_create_run().
    """
    print("\n" + _box(msg_lines, title=title))
    print("  C — continue to the next stage anyway (not recommended)")
    print("  Q — abort (stop here; rerun the pipeline to retry the failed clusters)")

    while True:
        raw = run_store.safe_input("Choice [c/Q]: ")
        if raw is None:
            logger.warning("no answer available — defaulting to quit (fail-closed)")
            return False

        choice = raw.strip().lower()
        if choice in ("", "q"):
            return False
        if choice == "c":
            return True

        print(f"[Pipeline] '{choice}' is not a valid choice — please enter C or Q.")


def _load_embeddings(posts_df: pd.DataFrame) -> np.ndarray:
    """Extract the ``embedding_raw`` column into a contiguous float32 ndarray.

    Handles ``list``/``tuple`` and ``np.ndarray`` storage formats.
    """
    emb_list: list[np.ndarray] = []
    for emb in posts_df["embedding_raw"].values:
        if isinstance(emb, (list, tuple)):
            emb = np.array(emb, dtype=np.float32)
        emb_list.append(emb)
    return np.stack(emb_list).astype(np.float32)


def _run_gmm_bic_sweep(
    posts_df: pd.DataFrame,
    cfg: dict,
    k_min: int,
    k_max: int,
) -> tuple[int, np.ndarray]:
    """Run the GMM-BIC sweep directly over [k_min, k_max], HDBSCAN pre-run skipped.

    Shared by ``--k-range``, ``--set-k`` (k_min == k_max == N), and the
    ``hdbscan`` method — all three need real ``seed_labels`` for ADR, not
    just a bare k value.

    Returns:
        Tuple ``(k_optimal, seed_labels)`` — also persisted to ``k_cache.pkl``.
    """
    clust_cfg = cfg.get("clustering", {})
    alpha  = clust_cfg.get("bic_penalty_alpha", k_estimator_module._DEFAULT_ALPHA)
    n_jobs = clust_cfg.get("gmm_n_jobs", -1)
    soft_zca_eps: float | None = None
    if clust_cfg.get("gmm_input_space") == "soft_zca":
        soft_zca_eps = float(clust_cfg.get("gmm_zca_eps", 0.01))

    embeddings = _load_embeddings(posts_df)
    with threadpool_limits(limits=1, user_api="blas"):
        k_optimal, _, seed_labels = k_estimator_module.find_optimal_k(
            embeddings, k_min, k_max,
            alpha=alpha, cfg_n_jobs=n_jobs,
            soft_zca_eps=soft_zca_eps,
        )
    k_estimator_module._save_cache(k_optimal, cfg, seed_labels=seed_labels)
    return k_optimal, seed_labels


def _apply_field_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Rename incoming columns according to field_mapping in config.

    No validation against canonical names — any key becomes a real rename.

    Args:
        df: Raw records as loaded from JSON.
        mapping: ``field_mapping`` section from config.yaml. Keys are
            canonical names (id, content, title, timestamp, engagement).
            Values are the actual field names in the incoming JSON, or null
            to keep the default.

    Returns:
        DataFrame with columns renamed to canonical names.
    """
    rename = {}
    for canonical, incoming in (mapping or {}).items():
        if incoming and incoming != canonical and incoming in df.columns:
            rename[incoming] = canonical
    if rename:
        df = df.rename(columns=rename)
        logger.info("field_mapping applied: %s", rename)
    return df


# First match wins, case-insensitive.
_TITLE_ALIASES: list[str] = [
    "title", "headline", "subject",
]
_BODY_ALIASES: list[str] = [
    "content", "body", "text", "selftext", "message",
]
_ENGAGEMENT_ALIASES: list[str] = [
    "engagement", "score",
]
_TIMESTAMP_ALIASES: list[str] = [
    "timestamp", "created_at", "published_at", "posted_at",
]


def _match_alias(cols: list[str], aliases: list[str]) -> str | None:
    """Return the first column (case-insensitive) that matches an alias list."""
    col_lower = {c.lower(): c for c in cols}
    for alias in aliases:
        if alias.lower() in col_lower:
            return col_lower[alias.lower()]
    return None


def _detect_and_normalize_fields(df: pd.DataFrame, file_tag: str) -> pd.DataFrame:
    """Auto-detect semantic fields and build the canonical ``content`` column.

    Original columns (``title``, ``body``, etc.) are kept untouched so the
    renderer can still display them separately. Only columns matching
    ``_TIMESTAMP_ALIASES`` are auto-detected for epoch-to-ISO8601
    conversion — other names (e.g. Reddit's ``created``) need
    ``field_mapping`` first.

    Args:
        df: Records as loaded from one file, after field_mapping rename.
        file_tag: File name, used for logging only.

    Returns:
        Same frame with ``content``, ``engagement``, ``timestamp`` columns
        guaranteed to exist (or left absent if truly not found — handled
        downstream).
    """
    df    = df.copy()
    cols  = df.columns.tolist()
    notes = []

    if "content" not in cols:
        title_col = _match_alias(cols, _TITLE_ALIASES)
        body_col  = _match_alias(cols, _BODY_ALIASES)

        if title_col and body_col:
            title_ser = df[title_col].fillna("").astype(str).str.strip()
            body_ser  = df[body_col].fillna("").astype(str).str.strip()
            # Separator only when both parts are non-empty.
            df["content"] = title_ser.where(body_ser == "", title_ser + "\n\n" + body_ser)
            df["content"] = df["content"].where(title_ser != "", body_ser)
            notes.append(f"content ← {title_col!r} + {body_col!r}")
        elif title_col:
            df["content"] = df[title_col].fillna("").astype(str).str.strip()
            notes.append(f"content ← {title_col!r}")
        elif body_col:
            df["content"] = df[body_col].fillna("").astype(str).str.strip()
            notes.append(f"content ← {body_col!r}")

    if "engagement" not in df.columns:
        eng_col = _match_alias([c for c in df.columns if c != "content"], _ENGAGEMENT_ALIASES)
        if eng_col:
            df = df.rename(columns={eng_col: "engagement"})
            notes.append(f"engagement ← {eng_col!r}")

    if "timestamp" not in df.columns:
        ts_col = _match_alias(
            [c for c in df.columns if c not in ("content", "engagement")],
            _TIMESTAMP_ALIASES,
        )
        if ts_col:
            sample = df[ts_col].dropna().iloc[0] if not df[ts_col].dropna().empty else None
            if sample is not None and isinstance(sample, (float, int, np.integer, np.floating)):
                df[ts_col] = pd.to_datetime(df[ts_col], unit="s", utc=True).dt.strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            df = df.rename(columns={ts_col: "timestamp"})
            notes.append(f"timestamp ← {ts_col!r}")
        else:
            logger.info(
                "%s: no timestamp/date column detected among %s — "
                "temporal ordering will be disabled for this file",
                file_tag, [c for c in df.columns if c not in ("content", "engagement")],
            )
    else:
        logger.info("%s: timestamp ← already present as 'timestamp' (no rename needed)", file_tag)

    if notes:
        logger.info("%s: columns mapped → %s", file_tag, ", ".join(notes))

    return df


def _normalize_engagement_per_file(
    df: pd.DataFrame,
    file_tag: str,
) -> pd.DataFrame:
    """Normalize the engagement column of a single file to [1, 10] via percentile rank.

    Values map onto [1, 10] via ``1 + 9 * rank``. Documents without a value
    get NaN here, filled with the global median in a later pass once all
    files are concatenated.

    Args:
        df: Records from a single JSON file; must contain an ``engagement``
            column (may be NaN).
        file_tag: File name used only for logging.

    Returns:
        Input frame with ``engagement`` replaced by the normalised [1, 10]
        score (float). Original raw values are dropped.
    """
    df = df.copy()
    col = df["engagement"].apply(pd.to_numeric, errors="coerce")
    valid = col.dropna()

    if len(valid) == 0:
        df["engagement"] = np.nan
        logger.debug("%s: no engagement values — will use global median", file_tag)
        return df

    e_min = float(valid.min())
    e_max = float(valid.max())

    if e_max == e_min:
        # No variance to rank against — ceiling for all valid values.
        df["engagement"] = col.apply(lambda v: 10.0 if pd.notna(v) else np.nan)
        logger.debug("%s: engagement constant (%.0f) → all at ceiling 10.0", file_tag, e_max)
    else:
        rank = valid.rank(pct=True)
        scaled = 1.0 + 9.0 * rank
        df["engagement"] = np.nan
        df.loc[scaled.index, "engagement"] = scaled
        logger.debug("%s: engagement percentile-rank normalised [1–10] (raw min=%.0f, max=%.0f, n=%d)",
                     file_tag, e_min, e_max, len(valid))
    return df


def _content_hash_id(content: str) -> str:
    """Deterministic id derived from content (SHA-1, first 16 hex chars).

    Same content always produces the same id, so re-ingesting the same
    source data is reproducible and naturally de-duplicates.
    """
    return hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]


def _read_json_file(path: Path) -> pd.DataFrame:
    """Load one JSON file (list of records or a single record)."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return pd.DataFrame(data if isinstance(data, list) else [data])


def _read_csv_file(path: Path) -> pd.DataFrame:
    """Load one CSV/TSV file. Delimiter auto-detected from extension."""
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep, encoding="utf-8")


def _read_text_file(path: Path) -> pd.DataFrame:
    """Load one plain-text file as a single post (content = whole file)."""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return pd.DataFrame([{"content": text}])


def _read_pdf_file(path: Path) -> pd.DataFrame:
    """Load one PDF file as a single post (content = extracted text, all pages)."""
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    return pd.DataFrame([{"content": text}])


_FILE_READERS = {
    ".json": _read_json_file,
    ".csv":  _read_csv_file,
    ".tsv":  _read_csv_file,
    ".txt":  _read_text_file,
    ".md":   _read_text_file,
    ".pdf":  _read_pdf_file,
}


def _ensure_unique_ids(posts_df: pd.DataFrame, file_tags: list) -> pd.DataFrame:
    """Guarantee globally unique, non-null ``id`` values across all input files.

    A missing id is generated deterministically from content (see
    ``_content_hash_id``). A duplicate id already present in the source
    data is not auto-resolved: the user sees which id(s) collide and in
    which file(s), then confirms (y/n) whether to continue with a numeric
    suffix or abort to fix the source data.
    """
    posts_df = posts_df.copy()
    if "id" not in posts_df.columns:
        posts_df["id"] = None
    posts_df["id"] = posts_df["id"].astype(object)

    missing_mask = posts_df["id"].isna() | (posts_df["id"].astype(str).str.strip() == "")
    n_missing = int(missing_mask.sum())
    if n_missing > 0:
        generated = posts_df.loc[missing_mask, "content"].astype(str).map(_content_hash_id)
        posts_df.loc[missing_mask, "id"] = generated
        logger.info("%d posts had no id — generated from content hash", n_missing)

    posts_df["id"] = posts_df["id"].astype(str)

    # Source ids only — hash-generated duplicates are expected de-duplication.
    source_id_mask = ~missing_mask.values
    dup_ids = posts_df.loc[source_id_mask, "id"][
        posts_df.loc[source_id_mask, "id"].duplicated(keep=False)
    ].unique().tolist()

    if dup_ids:
        logger.warning("duplicate id(s) found in source data:")
        for dup in dup_ids:
            rows = posts_df.index[(posts_df["id"] == dup) & source_id_mask].tolist()
            tags = sorted({file_tags[i] for i in rows})
            logger.warning("  id=%r appears %dx in: %s", dup, len(rows), ", ".join(tags))
        raw = run_store.safe_input(
            f"[Pipeline] {len(dup_ids)} duplicate id(s) found. "
            f"Continue and auto-suffix duplicates? [y/n]: "
        )
        answer = (raw or "").strip().lower()
        if answer != "y":
            raise ValueError(
                f"Aborted by user — fix duplicate id(s) in source data: {dup_ids}"
            )
        seen: dict = {}
        new_ids = posts_df["id"].tolist()
        for i, pid in enumerate(new_ids):
            if pid in dup_ids:
                seen[pid] = seen.get(pid, 0) + 1
                if seen[pid] > 1:
                    new_ids[i] = f"{pid}__dup{seen[pid] - 1}"
        posts_df["id"] = new_ids
        logger.info("duplicates auto-suffixed — continuing")

    return posts_df


def load_raw_posts(
    input_dir: str,
    cfg: dict = None,
    limit: int = None,
) -> pd.DataFrame:
    """Load and validate all supported files from *input_dir*.

    Supported formats (by extension): ``.json``, ``.csv``/``.tsv``,
    ``.txt``/``.md`` (whole file = one post), ``.pdf`` (extracted text = one
    post). ``timestamp``/``engagement`` are optional and degrade gracefully
    to NaN. ``id`` is optional too, generated from content hash when
    missing; real duplicates in source data trigger a y/n confirmation.

    Args:
        input_dir: Directory containing one or more supported files.
        cfg: Pipeline configuration (field_mapping, mode). When None, only
            the config-driven rename is skipped — alias auto-detection and
            engagement normalisation still run.
        limit: Truncate to the first `limit` rows.

    Returns:
        Validated post records with at least ``id`` and ``content`` columns.

    Raises:
        ValueError: If the directory does not exist, contains no supported
            files, is missing the required ``content`` column, or the user
            declines to continue past a duplicate-id warning.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory not found: {input_dir}")

    mapping = (cfg or {}).get("field_mapping") or {}

    frames: list[pd.DataFrame] = []
    file_tags: list[str] = []
    paths = sorted(p for p in input_path.iterdir() if p.suffix.lower() in _FILE_READERS)
    for path in paths:
        reader = _FILE_READERS[path.suffix.lower()]
        file_df = reader(path)
        file_df = _apply_field_mapping(file_df, mapping)
        file_df = _detect_and_normalize_fields(file_df, path.name)

        if "engagement" not in file_df.columns:
            file_df["engagement"] = np.nan

        file_df = _normalize_engagement_per_file(file_df, path.name)
        frames.append(file_df)
        file_tags.extend([path.name] * len(file_df))

    if not frames:
        supported = ", ".join(sorted(_FILE_READERS))
        raise ValueError(f"No supported files found in {input_dir} (supported: {supported})")

    posts_df = pd.concat(frames, ignore_index=True)
    logger.info("%d posts loaded from %s", len(posts_df), input_dir)

    required = ["content"]
    missing  = [c for c in required if c not in posts_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    posts_df = _ensure_unique_ids(posts_df, file_tags)

    if "timestamp" not in posts_df.columns:
        posts_df["timestamp"] = pd.NaT
        logger.warning("timestamp absent — temporal edges and timeline ordering disabled")

    # Any format (ISO with offset, Z, epoch, naive) → tz-naive UTC.
    ts = pd.to_datetime(posts_df["timestamp"], errors="coerce", utc=True)
    posts_df["timestamp"] = ts.dt.tz_localize(None)

    n_missing = posts_df["engagement"].isna().sum()
    if n_missing > 0:
        global_median = float(posts_df["engagement"].median())
        if np.isnan(global_median):
            global_median = 5.5  # no engagement anywhere in the corpus
        posts_df["engagement"] = posts_df["engagement"].fillna(global_median)
        logger.info("%d posts without engagement → global median %.2f", n_missing, global_median)

    if limit:
        posts_df = posts_df.head(limit)
        logger.info("truncated to %d posts", limit)

    return posts_df


def save_state(posts_df: pd.DataFrame, edges_df: pd.DataFrame, cfg: dict) -> None:
    """Thin re-export of ``run_store.save_state``."""
    run_store.save_state(posts_df, edges_df, cfg)


def _try_reuse_nova_outputs(cfg: dict, posts_df: pd.DataFrame):
    """Reuse a completed, VALIDATED Nova run on disk instead of recomputing it.

    ``save_nova_trees()`` archives (renames) ``nova_checkpoint.json`` once
    every cluster is processed AND validated — that's the only signal a
    validated run already exists. A live (unarchived) checkpoint next to
    the parquet outputs means the last attempt didn't pass the gate (or
    crashed); that case falls through to a normal Nova run instead of
    silently reusing unvalidated output.

    Args:
        cfg: Pipeline configuration.
        posts_df: Current post DataFrame to merge Nova outputs into.

    Returns:
        Tuple ``(posts_df, nova_edges, reused)``. ``reused`` is False if no
        validated output was found on disk — caller should fall back to a
        normal Nova run.
    """
    out_dir         = Path(cfg["storage"]["processed_dir"])
    assign_path     = out_dir / "nova_assignments.parquet"
    edges_path      = out_dir / "nova_edges.parquet"
    live_checkpoint = out_dir / "nova_checkpoint.json"
    if not (assign_path.exists() and edges_path.exists()):
        return posts_df, None, False
    if live_checkpoint.exists():
        logger.info(
            "nova_assignments.parquet found but nova_checkpoint.json is still live "
            "(last Nova attempt did not pass validation) — recomputing "
            "instead of reusing."
        )
        return posts_df, None, False

    nova_posts = pd.read_parquet(assign_path)
    nova_edges = pd.read_parquet(edges_path)

    nova_cols = [c for c in nova_posts.columns if c != "id"]
    posts_df  = posts_df.drop(columns=[c for c in nova_cols if c in posts_df.columns])
    posts_df  = posts_df.merge(nova_posts, on="id", how="left")

    logger.info(
        "reusing existing Nova output on disk (%s, %s) — skipping recompute. "
        "Pass --force-nova to ignore these files and recompute from scratch.",
        assign_path.name, edges_path.name,
    )
    return posts_df, nova_edges, True


STAGE_ORDER                     = run_store.STAGE_ORDER
_save_stage_checkpoint           = run_store.save_stage_checkpoint
_load_stage_checkpoint           = run_store.load_stage_checkpoint
_save_checkpoint                 = run_store.save_checkpoint


def _prompt_llm_absent(cfg: dict) -> str:
    """Interactive prompt when the LLM drops during naming. Returns 'skip_llm' or 'ok'.

    Naming's LLM slot is independent from Nova's/ADEPT's (see config.yaml
    model_slot per module) — this only decides naming's own fallback to
    c-TF-IDF. Nova and ADEPT detect and retry their own LLM failures.
    """
    from _llm import check_llm_available, get_llm_cfg
    while True:
        print("\n" + _box(
            ["Y — skip the naming LLM (use c-TF-IDF labels only)",
             "R — retry"],
            title="LLM UNAVAILABLE — NAMING",
        ))
        raw = run_store.safe_input("Choice [Y/r]: ")
        # No terminal / EOF / Ctrl-C: same as pressing Enter, skip the LLM.
        answer = (raw or "").strip().lower()
        if answer in ("", "y"):
            return "skip_llm"
        if answer == "r":
            if check_llm_available(get_llm_cfg(cfg, "naming")):
                return "ok"
            print("[Pipeline] LLM still unavailable.")


def run_init(
    run_id:       str,
    cfg:          dict,
    input_dir:    str,
    limit:        int  = None,
    resume_from:  str  = None,
    run_mode:     str  = None,
    use_k_cache:  bool = False,
    set_k:        int  = None,
    skip_naming:  bool = False,
    k_range:      tuple = None,
    k_fast:       bool = False,
    force_nova:   bool = False,
    cli_args:     dict = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full pipeline, start to finish.

    Args:
        run_id: This run's id, used to locate its frozen config snapshot.
        cfg: Already-resolved pipeline config, not a raw
            ``yaml.safe_load(config_path)`` (see ``run_store.create_run``).
        input_dir: Directory of raw input files (see ``load_raw_posts``).
        limit: Truncate input to the first N posts.
        resume_from: One of ``STAGE_ORDER`` — see ``--resume-from`` in
            ``main()`` for what "nova"/"adept" do differently.
        run_mode: "full" or "clustering_only" — overrides ``config.yaml``.
            ``clustering_only`` skips Nova and ADEPT entirely. Checkpoints
            after ``naming`` produced under the other mode are deleted after
            a y/n confirmation, and the run resumes from ``naming``.
        use_k_cache: Load ``k_optimal`` from ``k_cache.pkl`` instead of
            running stage 2.
        set_k: Force k through the GMM-BIC sweep at a single value;
            produces real ``seed_labels`` for ADR, unlike a bare k assignment.
        skip_naming: Skip stage 4 (Naming); ``cluster_label`` set to ``None``.
        force_nova: Ignore existing Nova output on disk and recompute from
            scratch (see ``_try_reuse_nova_outputs``).
        k_range: Debug: bypass HDBSCAN pre-run; sweep GMM-BIC over
            [k_range[0], k_range[1]].
        k_fast: Force k_estimation_method to "hdbscan" for this run only;
            only takes effect when K is auto-estimated.
        cli_args: Invocation to record in run.json, once run_mode
            confirmation (if any) has passed.

    Returns:
        Tuple ``(posts_df, edges_df)`` — final state persisted to disk.
    """
    t0 = datetime.now()

    # Read from this run's frozen config snapshot, not the live config.yaml
    # file, so a mid-run edit can't mix old and new settings across stages.
    effective_config_path = str(run_store.config_snapshot_path(run_id))

    effective_run_mode = run_mode or cfg.get("run_mode", "full")
    if effective_run_mode not in ("full", "clustering_only"):
        raise ValueError(f"run_mode must be 'full' or 'clustering_only', got {effective_run_mode!r}")

    previous_run_mode = (run_store.load_run(run_id) or {}).get("run_mode")

    mode_paths: list[Path] = []
    mode_names: list[str] = []
    if resume_from is not None:
        other_modes, mode_paths = run_store.mode_conflict(cfg, effective_run_mode, previous_run_mode)
        if mode_paths:
            root = Path(cfg["storage"]["processed_dir"])
            mode_names = [p.relative_to(root).as_posix() for p in mode_paths]
            logger.warning(
                "checkpoints after 'naming' were produced under run_mode=%s, this run is '%s'.",
                "/".join(other_modes), effective_run_mode,
            )
            logger.warning(
                "continuing will permanently delete all stages after 'naming': %s",
                ", ".join(mode_names),
            )
            if not run_store.confirm_yes_no("[Pipeline] Delete them and continue? [y/n]: "):
                logger.warning("aborted — nothing was deleted or changed.")
                sys.exit(1)

    if (not mode_paths and resume_from in (None, "adr", "naming")
            and previous_run_mode not in (None, effective_run_mode)):
        logger.warning(
            "run_mode changed for '%s': last run as '%s', now running as "
            "'%s'. Safe — nothing Nova/ADEPT-specific has been computed "
            "yet. Pass --run-mode %s explicitly if unintentional.",
            run_id, previous_run_mode, effective_run_mode, previous_run_mode,
        )

    # content_type drives Nova's ordering/evidence_types AND its posts/documents
    # wording, and ADEPT's posts/documents wording only (no ordering concept
    # there — see adept.py's _build_prompt_params). Shown in the startup panel
    # unconditionally, alongside run_mode, since it's a top-level run setting —
    # even though it only actually takes effect on Nova/ADEPT under "full".
    print("\n" + _box(
        [f"run_id       : {run_id}",
         f"project      : {cfg.get('project_name', 'run')}",
         f"input        : {input_dir}",
         f"run_mode     : {effective_run_mode}",
         f"content_type : {cfg.get('content_type', 'social_media')}"],
        title="HERONLOOM PIPELINE",
    ) + "\n")

    out_dir    = Path(cfg["storage"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "cluster_stats.parquet"

    if resume_from is not None and resume_from not in STAGE_ORDER:
        raise ValueError(f"resume_from must be one of {STAGE_ORDER}, got {resume_from!r}")

    done_idx = STAGE_ORDER.index(resume_from) + 1 if resume_from else 0
    posts_df  = None
    edges_df  = None
    cluster_stats = None

    if mode_paths:
        done_idx = min(done_idx, STAGE_ORDER.index("naming") + 1)
    elif effective_run_mode == "clustering_only" and resume_from in ("nova", "adept"):
        logger.warning(
            "resumed from '%s' checkpoint but run_mode=clustering_only — the "
            "already-validated Nova%s data in that checkpoint will be discarded "
            "(clustering_only never keeps Nova/ADEPT edges); clustering + naming "
            "are reused as-is, Edges/FA2 recompute with semantic + temporal "
            "links only.",
            resume_from, "" if resume_from == "nova" else "/ADEPT",
        )
        done_idx = STAGE_ORDER.index("naming") + 1
    elif resume_from in ("edges", "fa2"):
        checkpoint_meta = run_store.load_stage_meta(cfg, resume_from)
        checkpoint_run_mode = checkpoint_meta.get("run_mode") if checkpoint_meta else None

        if checkpoint_run_mode is None and run_mode is not None:
            logger.warning(
                "'%s' checkpoint predates run_mode provenance tracking — "
                "cannot verify consistency with --run-mode %s. Proceeding; "
                "if this checkpoint was produced under clustering_only, its "
                "edges will not have Nova/ADEPT links regardless of "
                "--run-mode.", resume_from, effective_run_mode,
            )

    if resume_from:
        load_stage = STAGE_ORDER[done_idx - 1]
        posts_df, edges_df = _load_stage_checkpoint(cfg, load_stage)

        if mode_paths:
            run_store.delete_paths(mode_paths)
            logger.info("deleted: %s", ", ".join(mode_names))

        _SKIPPED_LABEL = {
            "adr":    "1–3",
            "naming": "1–4",
            "nova":   "1–5a (Nova)",
            "adept":  "1–5b (Nova+ADEPT)",
            "edges":  "1–5 (Nova+ADEPT+Edges)",
            "fa2":    "1–6",
        }
        logger.info("RESUME @ '%s' — %d posts loaded, skipping stages %s",
                    load_stage, len(posts_df), _SKIPPED_LABEL[load_stage])

        if stats_path.exists():
            cluster_stats = pd.read_parquet(stats_path)
            logger.info("cluster_stats loaded (%d clusters)", len(cluster_stats))
        else:
            logger.warning("cluster_stats.parquet not found — rebuilding minimal version")
            cluster_stats = (
                posts_df.groupby("cluster_id")
                .agg(size=("id", "count"))
                .reset_index()
            )

        if skip_naming and resume_from in ("adr", "naming"):
            logger.info("RESUME + --skip-naming: label columns set to None")
            posts_df["cluster_label"] = None

    # run.json's run_mode is display-only; refresh it so it never goes stale
    # (the actual logic reads meta.json).
    run_fields = {"run_mode": effective_run_mode}
    if cli_args is not None:
        run_fields["cli_args"] = cli_args
    run_store.mark_in_progress(run_id, **run_fields)

    if done_idx < 1:
        posts_df = load_raw_posts(input_dir, cfg=cfg, limit=limit)

        print("\n[1/7] Embedding...")
        embedding_pipeline = emb_module.EmbeddingPipeline(effective_config_path)
        posts_df           = embedding_pipeline.process_dataframe(posts_df)

        # Separate sidecar, raw text (no instruct_prompt) for the search
        # step's document-side index; same pipeline instance so it reuses
        # the Ollama connection, cache, and retry logic above.
        print("\n[1/7] Embedding — retrieval store...")
        embedding_pipeline.process_dataframe_retrieval(posts_df)

        clustering_method = "hdbscan" if k_fast else cfg.get("clustering", {}).get("k_estimation_method", "hdbscan")

        if k_fast and (set_k is not None or use_k_cache or k_range is not None):
            bypass_flag = "--set-k" if set_k is not None else ("--use-k-cache" if use_k_cache else "--k-range")
            logger.warning(
                "--k-fast has no effect here — %s already "
                "bypasses automatic K estimation entirely.",
                bypass_flag,
            )

        seed_labels = None

        if set_k is not None:
            print(f"\n[2/7] K forced to {set_k} (--set-k) — running GMM-BIC at k={set_k} for ADR seed labels")
            k_optimal, seed_labels = _run_gmm_bic_sweep(posts_df, cfg, set_k, set_k)
        elif use_k_cache:
            k_optimal, seed_labels = k_estimator_module._load_cache(cfg)
            print(f"\n[2/7] K loaded from cache: {k_optimal} (--use-k-cache)")
        elif k_range is not None:
            k_min, k_max = int(k_range[0]), int(k_range[1])
            print(f"\n[2/7] K estimation — GMM-BIC sweep (--k-range {k_min} {k_max}, HDBSCAN pre-run skipped)")
            k_optimal, seed_labels = _run_gmm_bic_sweep(posts_df, cfg, k_min, k_max)
        else:
            print(f"\n[2/7] K estimation — method: {clustering_method.upper()}")
            if clustering_method == "hdbscan":
                k_pivot = hdbscan_module.run(posts_df, effective_config_path)
                print(f"[2/7] HDBSCAN k={k_pivot} — running GMM-BIC at k={k_pivot} for ADR seed labels")
                k_optimal, seed_labels = _run_gmm_bic_sweep(posts_df, cfg, k_pivot, k_pivot)
            elif clustering_method == "gmm_bic":
                with threadpool_limits(limits=1, user_api="blas"):
                    k_optimal, seed_labels = k_estimator_module.run(posts_df, effective_config_path)
                k_estimator_module._save_cache(k_optimal, cfg, seed_labels=seed_labels)
            else:
                raise ValueError(f"Unknown k_estimation_method: {clustering_method}")

        print(f"\n[3/7] ADR — TopiCLEAR refinement (k={k_optimal})...")
        embeddings = _load_embeddings(posts_df)
        topiclear_model, labels_final, _ = adr_module.run_adr(
            embeddings, cfg["adr"], k_override=k_optimal, seed_labels=seed_labels
        )
        posts_df["cluster_id"] = hdbscan_module.cluster_label_to_id_array(labels_final)

        embeddings_for_transform = adr_module.maybe_apply_soft_zca(embeddings, cfg["adr"])
        embeddings_lda = adr_module.transform_lda(topiclear_model, embeddings_for_transform).astype(np.float32)
        posts_df["embedding_lda"]   = list(embeddings_lda)
        logger.info("LDA space: %dd — all posts transformed", embeddings_lda.shape[1])

        adr_module.save_models(topiclear_model, cfg)

        stats_rows = []
        for cid in posts_df["cluster_id"].dropna().unique():
            mask     = posts_df["cluster_id"] == cid
            centroid = embeddings_lda[mask.values].mean(axis=0).tolist()
            stats_rows.append({"cluster_id": cid, "size": int(mask.sum()), "centroid": centroid})
        cluster_stats = pd.DataFrame(stats_rows)

        cluster_stats.to_parquet(stats_path, index=False)

        if len(cluster_stats) > 0:
            min_row = cluster_stats.loc[cluster_stats["size"].idxmin()]
            max_row = cluster_stats.loc[cluster_stats["size"].idxmax()]
            logger.info("smallest cluster: %s (%d posts)", min_row['cluster_id'], min_row['size'])
            logger.info("largest cluster:  %s (%d posts)", max_row['cluster_id'], max_row['size'])

        _save_stage_checkpoint(cfg, "adr", posts_df, run_mode=effective_run_mode)

    _llm_absent = False
    if done_idx < 2:
        if skip_naming:
            print("\n[4/7] naming skipped (--skip-naming)")
            posts_df["cluster_label"] = None
        else:
            from _llm import check_llm_available, get_llm_cfg
            _llm_absent = not check_llm_available(get_llm_cfg(cfg, "naming"))
            if _llm_absent:
                decision = _prompt_llm_absent(cfg)
                _llm_absent = decision != "ok"

            force_skip = _llm_absent

            print("\n[4/7] Naming — cluster labelling...")
            posts_df = naming_module.run(posts_df, effective_config_path, force_skip_llm=force_skip)

        _save_stage_checkpoint(cfg, "naming", posts_df, run_mode=effective_run_mode)

    if effective_run_mode == "clustering_only" and done_idx < 5:
        print("\n[Pipeline] run_mode=clustering_only: Nova/ADEPT skipped")

        for col in ("nova_role", "nova_parent", "nova_depth",
                    "nova_symbol", "nova_force_score", "subtopic_id", "subtopic_label"):
            posts_df[col] = None
        for col in ("adept_role", "adept_symbol", "eng_norm"):
            posts_df[col] = None

        print("\n[5/7] Edges (semantic + temporal only)...")
        edges_df = edges_module.run_full(
            posts_df,
            effective_config_path,
            nova_edges=pd.DataFrame(columns=["source", "target", "type", "force"]),
            adept_edges=pd.DataFrame(columns=["source", "target", "type", "force"]),
        )
        _save_stage_checkpoint(cfg, "edges", posts_df, edges_df, run_mode=effective_run_mode)

        print("\n[6/7] ForceAtlas2 — 3-D layout...")
        posts_df = fa2_module.run(posts_df, edges_df, config_path=effective_config_path)
        _save_stage_checkpoint(cfg, "fa2", posts_df, edges_df, run_mode=effective_run_mode)

        print("\n[7/7] Saving state...")
        save_state(posts_df, edges_df, cfg)
        _save_checkpoint(cfg, stage="complete", complete=True)

        elapsed = timedelta(seconds=round((datetime.now() - t0).total_seconds()))
        print("\n" + _box(
            [f"posts    : {len(posts_df)}",
             f"clusters : {posts_df['cluster_id'].nunique()}",
             f"edges    : {len(edges_df)}  (Nova/ADEPT skipped)",
             f"elapsed  : {elapsed}"],
            title="PIPELINE COMPLETE — CLUSTERING ONLY",
        ) + "\n")
        return posts_df, edges_df

    # full mode only — clustering_only already returned above.
    nova_edges = None
    if done_idx < 3:
        reused = False
        if not force_nova:
            posts_df, nova_edges, reused = _try_reuse_nova_outputs(cfg, posts_df)

        if reused:
            print("[5/7] Nova — reusing existing nova_assignments/nova_edges.parquet "
                  "(skip recompute; pass --force-nova to redo it)")
        else:
            print("\n[5/7] Nova — intra-cluster subtopic trees...")
            nova_result = nova_module.build_nova_trees(posts_df, cfg)
            posts_df, nova_edges = nova_result.posts_df, nova_result.nova_edges
            nova_metadata        = nova_result.all_metadata
            nova_failed_clusters = nova_result.failed_clusters
            nova_validated       = nova_result.nova_validated
            nova_module.save_nova_trees(
                posts_df, nova_edges, cfg,
                all_metadata=nova_metadata, archive_when_done=nova_validated,
            )
            n_clusters_total     = int(posts_df["cluster_id"].nunique())
            n_clusters_failed    = len(nova_failed_clusters)
            n_clusters_completed = n_clusters_total - n_clusters_failed

            print(f"[5/7] Nova: {len(nova_edges)} edges | "
                  f"{n_clusters_completed}/{n_clusters_total} cluster(s) completed"
                  + (f", {n_clusters_failed} failed" if n_clusters_failed else ""))
            logger.debug("nova failed cluster ids: %s", nova_failed_clusters)

            # Soft gate: stops here by default, but asks once whether to
            # proceed with unvalidated clusters left as orphans. Fails
            # closed outside a live terminal; see _prompt_gate_override.
            if not nova_validated:
                msg_lines = [
                    f"{n_clusters_completed}/{n_clusters_total} cluster(s) completed, "
                    f"{n_clusters_failed} not validated by Nova.",
                    "The LLM failed after all retries on these clusters and they",
                    "were skipped rather than accepted with a fallback.",
                    "",
                    "Continuing sends the failed clusters into ADEPT/Edges as",
                    "orphans (no subtopic tree, no nova_role) instead of a fully",
                    "validated Nova pass. NOT RECOMMENDED unless you have reviewed",
                    "why each cluster failed.",
                    "",
                    "If Nova/ADEPT is not required for this run, clustering-only",
                    "mode can be enabled instead to skip this step entirely.",
                ]
                if _prompt_gate_override(msg_lines, title="NOVA NOT VALIDATED"):
                    logger.warning(
                        "user chose to continue past %d unvalidated cluster(s) — "
                        "proceeding to ADEPT with those clusters left as orphans.",
                        n_clusters_failed,
                    )
                else:
                    logger.error(
                        "nova_validated=False (%d/%d cluster(s) failed) — stopping before ADEPT.",
                        n_clusters_failed, n_clusters_total,
                    )
                    logger.debug(
                        "rerun resumes the failed clusters automatically; to skip "
                        "Nova/ADEPT entirely instead, pass --run-mode clustering_only "
                        "(or set run_mode: clustering_only in config.yaml).",
                    )
                    sys.exit(1)

        _save_stage_checkpoint(cfg, "nova", posts_df, nova_edges, run_mode=effective_run_mode)

    elif resume_from == "nova":
        # edges_df here is nova_edges — each checkpoint stores its own (posts, edges) pair.
        print("\n[5/7] Nova skipped — resumed from 'nova' checkpoint "
              f"({len(edges_df) if edges_df is not None else 0} nova edges reloaded)")
        nova_edges = edges_df if edges_df is not None else pd.DataFrame(
            columns=["source", "target", "type", "force"]
        )

    adept_edges = None
    if done_idx < 4:
        print("\n[5/7] ADEPT — event pooling...")
        # No degraded "skip the LLM" mode: the Nova gate above already exited on failure.
        adept_result = adept_module.build_adept_events(posts_df, nova_edges, cfg)
        posts_df, adept_edges = adept_result.posts_df, adept_result.adept_edges
        adept_metadata        = adept_result.all_metadata
        adept_failed_pools    = adept_result.failed_clusters
        adept_validated       = adept_result.adept_validated
        adept_module.save_adept_events(
            posts_df, adept_edges, cfg,
            all_metadata=adept_metadata, archive_when_done=adept_validated,
        )
        n_pools_total     = int(posts_df["cluster_id"].nunique())
        n_pools_failed    = len(adept_failed_pools)
        n_pools_completed = n_pools_total - n_pools_failed

        print(f"[5/7] ADEPT: {len(adept_edges)} edges | "
              f"{n_pools_completed}/{n_pools_total} pool(s) completed"
              + (f", {n_pools_failed} failed" if n_pools_failed else ""))
        logger.debug("adept failed pool ids: %s", adept_failed_pools)

        # Soft gate: same mechanism as the Nova gate above. Fails closed
        # automatically outside a live terminal; see _prompt_gate_override.
        if not adept_validated:
            msg_lines = [
                f"{n_pools_completed}/{n_pools_total} pool(s) completed, "
                f"{n_pools_failed} not validated by ADEPT.",
                "The LLM failed after all retries on these pools and they",
                "were skipped rather than accepted with a fallback.",
                "",
                "Continuing sends the failed pools into Edges without ADEPT",
                "event pooling for those posts. NOT RECOMMENDED unless you",
                "have reviewed why each pool failed.",
                "",
                "If Nova/ADEPT is not required for this run, clustering-only",
                "mode can be enabled instead to skip this step entirely.",
            ]
            if _prompt_gate_override(msg_lines, title="ADEPT NOT VALIDATED"):
                logger.warning(
                    "user chose to continue past %d unvalidated pool(s) — "
                    "proceeding to Edges with those pools left unpooled.",
                    n_pools_failed,
                )
            else:
                logger.error(
                    "adept_validated=False (%d/%d pool(s) failed) — stopping before Edges.",
                    n_pools_failed, n_pools_total,
                )
                logger.debug(
                    "rerun resumes the failed pools automatically; to skip "
                    "Nova/ADEPT entirely instead, pass --run-mode clustering_only "
                    "(or set run_mode: clustering_only in config.yaml).",
                )
                sys.exit(1)

        _save_stage_checkpoint(cfg, "adept", posts_df, adept_edges, run_mode=effective_run_mode)

    elif resume_from == "adept":
        # edges_df here is adept_edges — each checkpoint stores its own (posts, edges) pair.
        print("\n[5/7] ADEPT skipped — resumed from 'adept' checkpoint "
              f"({len(edges_df) if edges_df is not None else 0} adept edges reloaded)")
        adept_edges = edges_df if edges_df is not None else pd.DataFrame(
            columns=["source", "target", "type", "force"]
        )

    if done_idx < 5:
        print("\n[5/7] Edges — semantic, temporal, and ADEPT grafts...")
        edges_df = edges_module.run_full(
            posts_df, effective_config_path,
            nova_edges=nova_edges,
            adept_edges=adept_edges,
        )

        _save_stage_checkpoint(cfg, "edges", posts_df, edges_df, run_mode=effective_run_mode)

    if done_idx < 6:
        print("\n[6/7] ForceAtlas2 — 3-D layout...")
        posts_df = fa2_module.run(posts_df, edges_df, config_path=effective_config_path)

        _save_stage_checkpoint(cfg, "fa2", posts_df, edges_df, run_mode=effective_run_mode)

    print("\n[7/7] Saving state...")
    save_state(posts_df, edges_df, cfg)
    _save_checkpoint(cfg, stage="complete", complete=True)

    elapsed  = timedelta(seconds=round((datetime.now() - t0).total_seconds()))

    # From the persisted ADR model, not the local embeddings_lda (absent when resuming).
    try:
        _adr_refiner = adr_module.load_models(cfg)
        lda_dims = _adr_refiner.rotation_.shape[1] if _adr_refiner is not None else "?"
    except Exception:
        logger.warning("Could not read LDA dims from persisted ADR model", exc_info=True)
        lda_dims = "?"

    print("\n" + _box(
        [f"posts    : {len(posts_df)}",
         f"clusters : {posts_df['cluster_id'].nunique()}",
         f"edges    : {len(edges_df)}",
         f"LDA dims : {lda_dims}d",
         f"elapsed  : {elapsed}"],
        title="PIPELINE COMPLETE",
    ) + "\n")

    if _llm_absent:
        print(_box(
            ["This run completed without the LLM — cluster labels use",
             "keyword-based (c-TF-IDF) naming only.",
             "",
             "Once the LLM is available, the naming step can be redone",
             "without re-running clustering."],
            title="WARNING",
        ) + "\n")
        logger.debug(
            "advanced: redo naming with the AI model from the ADR checkpoint via "
            "`python pipeline.py --resume-from adr`.",
        )

    return posts_df, edges_df


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate run function."""
    parser = argparse.ArgumentParser(
        description="HeronLoom — pipeline orchestrator"
    )
    parser.add_argument("--input",  default="data/raw/",    help="Raw input directory (default: data/raw/).")
    parser.add_argument("--limit",  type=int, default=None, help="Truncate input to the first N posts (for a quick smoke test).")
    parser.add_argument(
        "--resume-from", choices=["adr", "naming", "nova", "adept", "edges", "fa2"],
        default=None, dest="resume_from",
        help="Force resume from a specific stage, regardless of checkpoint state. "
             "'nova' reloads posts + nova_edges exactly as they stood right after "
             "Nova finished and jumps into ADEPT — no Nova recompute, no "
             "--force-nova needed. 'adept' reloads posts + adept_edges exactly as "
             "they stood right after ADEPT finished and jumps straight into Edges "
             "— no Nova or ADEPT recompute.",
    )
    parser.add_argument(
        "--run-mode", choices=["full", "clustering_only"],
        default=None, dest="run_mode",
        help="Override config.yaml's run_mode for this run. 'full' = clustering + "
             "naming + Nova + ADEPT + edges + 3-D layout. 'clustering_only' = the "
             "same minus Nova/ADEPT; edges and the 3-D layout still run.",
    )
    parser.add_argument(
        "--content-type", choices=["social_media", "document"],
        default=None, dest="content_type",
        help="Override config.yaml's content_type for this new run only — has no "
             "effect on --resume-from or a resumed active run, since it's baked "
             "into the frozen config snapshot at creation, not replayed per "
             "invocation like --run-mode. 'social_media' = chronological Nova "
             "ordering. 'document' = thematic Nova ordering.",
    )
    parser.add_argument(
        "--use-k-cache", action="store_true", dest="use_k_cache",
        help="Load k_optimal from k_cache.pkl; skip stage 2.",
    )
    parser.add_argument(
        "--set-k", type=int, default=None, dest="set_k",
        help="Force k=N through the GMM-BIC sweep (same as --k-range N N); "
             "persist seed_labels to k_cache.pkl.",
    )
    parser.add_argument(
        "--k-range", type=int, nargs=2, default=None, dest="k_range",
        metavar=("MIN", "MAX"),
        help="Debug: bypass HDBSCAN pre-run; sweep GMM-BIC over [MIN, MAX].",
    )
    parser.add_argument(
        "--k-fast", action="store_true", dest="k_fast",
        help="NOT RECOMMENDED. Skip the GMM-BIC sweep; use HDBSCAN's own "
             "cluster count as k directly (fast, less refined). Same as "
             "config.yaml's clustering.k_estimation_method: hdbscan.",
    )
    parser.add_argument(
        "--skip-naming", action="store_true", dest="skip_naming",
        help="Skip stage 4 (Naming); label column set to None.",
    )
    parser.add_argument(
        "--force-nova", action="store_true", dest="force_nova",
        help="Ignore existing nova_assignments.parquet/nova_edges.parquet on disk and "
             "recompute Nova from scratch, even though they would otherwise be reused.",
    )
    parser.add_argument(
        "--render", choices=["none", "threejs"],
        default="threejs",
        help="3-D render after the run. Runs automatically (default: threejs); "
             "pass 'none' to skip it. Writes render.html only — never opens a "
             "browser tab (this command runs as part of automation/cron too; "
             "use `reload.py show <run_id> --render threejs` afterward to view it).",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Always start a brand-new run with no prompt and never resume, "
             "regardless of an in-progress active run. That run is left "
             "completely untouched — still listed and resumable via "
             "`reload.py list` (automation / cron use).",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", dest="yes",
        help="Auto-confirm the resume prompt instead of asking interactively: "
             "resumes an in-progress active run automatically. Does not "
             "cover the confirmation asked before a run_mode change deletes "
             "checkpoints.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Show DEBUG-level logs in the console for this run "
             "(the log file already captures DEBUG+ regardless).",
    )
    args = parser.parse_args()

    if args.debug:
        set_console_level(logging.DEBUG)

    run_id, cfg, resume_stage = run_store.resolve_or_create_run(
        CONFIG_PATH, args.run_mode,
        auto_yes=args.yes, fresh=args.fresh,
        explicit_resume_stage=args.resume_from,
        content_type=args.content_type,
    )

    # Refuses if another live process already holds this run.
    run_store.acquire_lock(run_id)
    try:
        posts_df, edges_df = run_init(
            run_id, cfg, args.input,
            limit=args.limit,
            resume_from=resume_stage,
            run_mode=args.run_mode,
            use_k_cache=args.use_k_cache,
            set_k=args.set_k,
            skip_naming=args.skip_naming,
            k_range=args.k_range,
            k_fast=args.k_fast,
            force_nova=args.force_nova,
            # So `reload.py --restart` can replay this exact invocation.
            cli_args=vars(args),
        )
    finally:
        run_store.release_lock(run_id)

    if args.render != "none":
        import render as render_module
        # Never opens a browser — this also runs unattended (cron); viewing
        # is a separate step via `reload.py show <run_id> --render threejs`.
        #
        # Isolated in its own try/except: "complete" (written above, inside
        # run_init()) covers the data in output/, not the 3-D view. A
        # rendering failure must not make a successful run look like it
        # needs redoing — but must not fail silently either, so it's logged
        # loudly and this process still exits non-zero for cron to notice.
        try:
            render_module.run(posts_df, edges_df,
                               config_path=str(run_store.config_snapshot_path(run_id)),
                               open_browser=False)
        except Exception:
            logger.exception(
                "'%s' finished and its data is safe (status: complete), but the "
                "3-D render step itself failed — retry just the render with "
                "`python reload.py show %s --render threejs`",
                run_id, run_id,
            )
            sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # run_store.acquire_lock()/release_lock() inside main() already ran
        # its own try/finally by the time this is reached, so the lock is
        # released and whatever checkpoint exists on disk is intact — this
        # is purely about a clean message instead of a raw traceback dump.
        print("\n[Pipeline] stopped by user request (Ctrl-C) — resume with "
              "`python reload.py resume <run_id>` (or `--resume-from <stage>`).")
        sys.exit(130)  # 130 = standard exit code for SIGINT