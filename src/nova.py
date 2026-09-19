"""Step 5a — NOVA: intra-cluster subtopic trees via LLM. Entry point: run()."""

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from run_store import load_config, safe_input
from utils.logger import get_logger

logger = get_logger(__name__)

class NovaResult(NamedTuple):
    """Return value of `build_nova_trees`.

    nova_validated is the flag callers must gate ADEPT on — it is True
    only if every cluster was processed AND none of them failed or were
    abandoned. all_clusters_done alone is not sufficient: it is True once
    every cluster has been processed (success OR failure), so it stays
    True even when clusters were skipped after exhausting retries.
    """
    posts_df: pd.DataFrame
    nova_edges: pd.DataFrame
    llm_abandoned: bool       # True if the user quit (Q) instead of retrying/ignoring
                              # a failed cluster. Does NOT mean Nova ran without the
                              # LLM — there is no such mode; it only forces
                              # nova_validated=False.
    all_metadata: list        # subtopic metadata dicts
    all_clusters_done: bool   # every cluster was processed, success or failure
    failed_clusters: list     # sorted cluster ids where the LLM exhausted all
                              # retries and got auto-skipped (empty if none)
    nova_validated: bool      # gate ADEPT on this, not on all_clusters_done alone


def _nova_cfg(cfg: dict) -> dict:
    from _llm import get_llm_cfg
    defaults = {
        "min_posts_per_subtopic": 2,
        "force_nova_edge":        1,  # fixed default; not exposed in config.yaml — tune via fa2.edge_weights.nova instead
        "max_text_chars":         500,
        "max_retries":            3,
        "checkpoint_enabled":     True,
        "parallel_clusters":      1,  # 1 = sequential (default) — enables the interactive Y/I/Q retry prompt
    }
    ncfg = {**defaults, **get_llm_cfg(cfg, "nova")}
    return ncfg


def _log_subtopic_summary(cluster_id, subtopics: list[dict]) -> None:
    """Log one compact line per subtopic (title, post count) at INFO — visible
    in console and file — plus the full detail (reasoning, sentiment arc) at
    DEBUG — file only, since the console handler runs at INFO.

    logger-only, no print(): keeps a single logging channel for the whole
    project instead of mixing print() (console-only, invisible to the log
    file) with logger (console+file). Used in both sequential and parallel
    mode so the two paths can't drift into different formats.
    """
    for st in subtopics:
        title     = st.get("title", "Unknown title")
        reasoning = st.get("reasoning", "No explanation provided")
        arc       = st.get("sentiment_arc", "")
        n_posts   = len(st.get("post_ids", []))

        logger.info("[%s] %r (%d posts)", st.get("id"), title, n_posts)
        logger.debug(
            "cluster=%s subtopic=%s title=%r reasoning=%r arc=%r posts=%d",
            cluster_id, st.get("id"), title, reasoning, arc, n_posts,
        )


_SYSTEM_PROMPT = """You are an analyst mapping {domain_description}.
You receive {n_posts} {unit_word} from a cluster labeled: "{cluster_label}" — read every one before grouping.
These {unit_word} will be organized into {ordering_description}.

CRITICAL RULES — STRICT ENFORCEMENT :
- NO BROAD CATEGORIZATION: Do NOT group {unit_word} just because they share a location, institution (e.g., "university", "hospital"), broad topic (e.g., "AI", "climate"), or isolated keywords. There MUST be a clear {connection_type} between the specific {unit_word}.
- EVIDENCE-BASED REASONING: Your 'reasoning' must be factual, explicit, observable connections ({evidence_types}). No inference, no interpretation, no invented thematic links.
- OBSERVED SENTIMENT ARC: Describe the actual, observable evolution of the conversation (e.g., "Rumor → Statistical Rebuttal → Conspiracy Claim"). Do not fabricate a narrative; the arc must be directly grounded in the text sequence.

YOUR TASK:

Group {unit_word} into {group_label}.

FOR EACH SUB-TOPIC YOU MUST PROVIDE:
- title: factual, specific title (max 10 words). Good: "Ukraine Starobelsk School Attack with Alleged Starlink Guidance". Bad: "War News".
- label: A precise lowercase label with underscores. Example: "ukraine_starobelsk_starlink".
- reasoning: reasoning: WHY did you group these specific {unit_word}? (1-3 sentences). Explain the common narrative thread and the explicit evidence linking them.
- sentiment_arc: The emotional/narrative journey (use "→" separators). Example: "Shock → Allegations of Complicity → Media Blackout".
- post_ids: List of ALL post_ids in this subtopic, {order_instruction}.

RULES:
- Each sub-topic must be SPECIFIC and COHERENT.
- A {unit_singular} belongs to ONLY ONE sub-topic.
- Minimum {min_size} {unit_word} per sub-topic.
- {unit_word} without a clear, explicit narrative thread MUST BE EXCLUDED.
- Return ONLY valid JSON (no markdown, no backticks):
{{
 "subtopics": [
  {{
   "id": "s1",
   "title": "Engaging Title",
   "label": "technical_label",
   "reasoning": "Why these {unit_word} form a coherent narrative based on explicit evidence...",
   "sentiment_arc": "Stage 1 → Stage 2 → Stage 3",
   "post_ids": ["post_id_1", "post_id_2"]
  }}
 ]
}}
"""

_USER_TEMPLATE = """Cluster: "{cluster_label}"

{unit_word_cap} to analyze ({n_posts} total){timestamp_note}:
{posts_block}

Group these {unit_word} into precise sub-topics. Return only JSON."""


def _build_prompt_params(cfg: dict) -> dict:
    """Derive all prompt-wording params from `content_type` alone.

    `content_type` ("social_media" | "document") is the single source of
    truth for both the vocabulary (posts vs documents) and the ordering
    strategy — there is no separate `nova_ordering` config key anymore.
    social_media is always chronological, document is always thematic;
    neither depends on whether timestamps happen to be present in the data.

    `evidence_types` also branches on `content_type`: social_media
    explicitly lists "exact dates" as valid grouping evidence (safe, since
    ordering is already chronological there); document omits it so the LLM
    is never told dates are a grouping signal. The `date="..."` attribute on
    each post (see _build_posts_block) is still shown when available for
    both content types — only the explicit instruction to use it as
    evidence differs.

    `domain_description` completes _SYSTEM_PROMPT's opening sentence ("You
    are an analyst mapping ...") — social_media keeps the original
    cascade framing, document swaps it for a non-temporal, argument-based
    framing consistent with `nova_ordering="thematic"`.
    """
    content_type = cfg.get("content_type", "social_media")

    if content_type == "social_media":
        unit_word     = "posts"
        unit_singular = "post"
        nova_ordering = "timeline"
        evidence_types = "shared full names, exact dates, specific claims"
        domain_description = "information cascades in social media"
    else:  # document
        unit_word     = "documents"
        unit_singular = "document"
        nova_ordering = "thematic"
        evidence_types = "shared full names, specific claims, verifiable facts"
        domain_description = "argumentative and thematic structure across documents"

    if nova_ordering == "thematic":
        ordering_description = "logical/argumentative threads showing how ideas connect"
        connection_type      = "logical or argumentative connection"
        group_label          = "THEMATIC GROUPS"
        order_instruction    = "ordered by logical progression as you see fit"
    else:  # timeline
        ordering_description = "CHRONOLOGICAL TIMELINES showing how narratives evolve over time"
        connection_type      = "chronological cause-and-effect or evolving debate"
        group_label          = "CHRONOLOGICAL TIMELINES"
        order_instruction    = "ordered chronologically"

    return {
        "unit_word":             unit_word,
        "unit_singular":         unit_singular,
        "unit_word_cap":         unit_word.capitalize(),
        "nova_ordering":         nova_ordering,
        "evidence_types":        evidence_types,
        "domain_description":    domain_description,
        "ordering_description":  ordering_description,
        "connection_type":       connection_type,
        "group_label":           group_label,
        "order_instruction":     order_instruction,
    }


def _call_llm_robust(
    cluster_label: str,
    cluster_df: pd.DataFrame,
    ncfg: dict,
    prompt_params: dict,
    cluster_id: str = "",
) -> list[dict] | None:
    """Call the LLM with exponential-backoff retries; return validated subtopic list or None."""
    max_retries = ncfg.get("max_retries", 3)
    max_chars   = ncfg.get("max_text_chars", 280)
    valid_ids   = set(cluster_df["id"].astype(str).tolist())

    n_posts = len(cluster_df)
    posts_block, has_timestamps = _build_posts_block(
        cluster_df, max_chars=max_chars, unit_singular=prompt_params["unit_singular"],
    )
    timestamp_note = ", each carrying a `date` attribute in ISO 8601 format when known" if has_timestamps else ""

    system = _SYSTEM_PROMPT.format(
        cluster_label=cluster_label,
        min_size=ncfg["min_posts_per_subtopic"],
        n_posts=n_posts,
        **prompt_params,
    )
    user = _USER_TEMPLATE.format(
        cluster_label=cluster_label,
        posts_block=posts_block,
        n_posts=n_posts,
        timestamp_note=timestamp_note,
        **prompt_params,
    )
    from _llm import raw_request

    for attempt in range(1, max_retries + 1):
        fail_reason = None
        try:
            # 1. HTTP call via shared _llm.raw_request
            raw = raw_request(ncfg, system, user)

            # 2. JSON extraction + parsing
            json_str = _extract_json_from_text(raw)
            if json_str is None:
                fail_reason = f"could not extract JSON (response: {raw[:200]})"
                raise ValueError(fail_reason)

            parsed = json.loads(json_str)

            # 3. Structure repair + validation
            parsed = _repair_subtopics(parsed, valid_ids)
            if not isinstance(parsed.get("subtopics"), list):
                fail_reason = f"invalid JSON structure after repair: {str(parsed)[:200]}"
                raise ValueError(fail_reason)

            # 4. Business validation (min_size, valid ids…)
            subtopics = _validate_subtopics(parsed, valid_ids, ncfg["min_posts_per_subtopic"])
            if not subtopics:
                proposed = parsed.get("subtopics", [])

                if not proposed:
                    # LLM found no groupable subtopics — a valid result, not an error.
                    logger.info("cluster=%s — no groupable subtopics", cluster_id or cluster_label)
                    return []

                # LLM proposed subtopics but all were filtered out (invalid IDs
                # or below min_size) — worth retrying.
                sizes = [len(st.get("post_ids", [])) for st in proposed]
                fail_reason = (
                    f"0 valid subtopics after validation — LLM proposed {len(proposed)} subtopic(s), "
                    f"post-ID-matching sizes: {sizes} (need \u2265{ncfg['min_posts_per_subtopic']} each)"
                )
                raise ValueError(fail_reason)

            return subtopics

        except Exception as e:
            reason = fail_reason or f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                wait = 2 ** (attempt - 1)
                logger.warning("attempt %d/%d failed — %s — retrying in %ds", attempt, max_retries, reason, wait)
                time.sleep(wait)
            else:
                logger.warning("attempt %d/%d failed — %s — giving up", attempt, max_retries, reason)

    return None


def _truncate_clean(text: str, max_chars: int) -> str:
    """Truncate at a word boundary and append an ellipsis to mark the cut."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space]
    return cut.rstrip() + "…"


def _escape_content(text: str) -> str:
    """Replace angle brackets with lookalike characters so post content can't spoof a `<post>` tag boundary."""
    return text.replace("<", "‹").replace(">", "›")


def _build_posts_block(cluster_df: pd.DataFrame, max_chars: int = 280,
                        unit_singular: str = "post") -> tuple[str, bool]:
    """Render posts as `<{unit_singular} id="..." date="...">text</{unit_singular}>`
    blocks for the LLM prompt (tag name follows content_type, same as adept.py).

    Each post uses an explicit closing tag so multi-line content is never
    mistaken for a post boundary. The `date` attribute is set per post and
    omitted when the timestamp is missing; timestamps are only included at
    all when the cluster has usable dates, formatted as ISO 8601.

    Args:
        cluster_df: posts belonging to this cluster.
        max_chars: per-post text truncation limit.
        unit_singular: tag name — "post" or "document", from prompt_params.

    Returns:
        (block, has_timestamps) — has_timestamps lets the caller adapt the
        surrounding prompt wording only when there's something to mention.
    """
    ts_series = (
        pd.to_datetime(cluster_df["timestamp"], errors="coerce")
        if "timestamp" in cluster_df.columns else None
    )
    has_timestamps = ts_series is not None and ts_series.notna().any()

    blocks = []
    for idx, row in cluster_df.iterrows():
        raw_text = str(row.get("content", "")).strip()
        text = _escape_content(_truncate_clean(raw_text, max_chars))

        date_attr = ""
        if has_timestamps:
            ts = ts_series.loc[idx]
            if pd.notna(ts):
                date_attr = f' date="{ts.strftime("%Y-%m-%dT%H:%M")}"'

        blocks.append(f'<{unit_singular} id="{row["id"]}"{date_attr}>\n{text}\n</{unit_singular}>')
    return "\n\n".join(blocks), has_timestamps


def _extract_json_from_text(text: str) -> str | None:
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    if text.startswith("{"):
        return text

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)

    return None


def _repair_subtopics(raw_json: dict, valid_ids: set) -> dict:
    # Normalise root key — LLM sometimes uses topics / groups / clusters
    for alt_key in ("topics", "groups", "clusters", "sub_topics"):
        if alt_key in raw_json and "subtopics" not in raw_json:
            raw_json["subtopics"] = raw_json.pop(alt_key)
            break

    if "subtopics" not in raw_json:
        return raw_json

    repaired = []
    for st in raw_json["subtopics"]:
        if not isinstance(st, dict):
            continue

        # Normalise post_ids key — LLM sometimes uses posts / ids / items
        for alt_key in ("posts", "ids", "post_id", "items"):
            if alt_key in st and "post_ids" not in st:
                st["post_ids"] = st.pop(alt_key)
                break

        post_ids = st.get("post_ids", [])
        if not isinstance(post_ids, list):
            post_ids = [post_ids] if post_ids else []

        # Stringify and strip stray brackets
        cleaned = [str(pid).strip().strip("[]") for pid in post_ids]

        # Strict match first
        valid_cleaned = [p for p in cleaned if p in valid_ids]

        # Partial match fallback when nothing matches exactly
        if not valid_cleaned and cleaned:
            for pid in cleaned:
                for vid in valid_ids:
                    if pid in vid or vid in pid:
                        valid_cleaned.append(vid)
                        break

        st["post_ids"] = list(dict.fromkeys(valid_cleaned))

        # Fill missing title/reasoning/sentiment_arc with safe defaults
        if "title" not in st:
            st["title"] = st.get("label", "Untitled Timeline")
        if "reasoning" not in st:
            st["reasoning"] = "Posts grouped by semantic similarity."
        if "sentiment_arc" not in st:
            st["sentiment_arc"] = "Narrative evolution not specified."

        repaired.append(st)

    raw_json["subtopics"] = repaired
    return raw_json


def _validate_subtopics(
    raw_result: dict,
    valid_ids: set,
    min_size: int,
) -> list[dict]:
    """Filter and deduplicate subtopics; discard any with fewer than min_size valid post IDs.

    Each post ID is assigned to at most one subtopic (first-seen wins across subtopics).
    """
    if not raw_result or "subtopics" not in raw_result:
        return []

    seen_ids = set()
    validated = []

    for st in raw_result["subtopics"]:
        if not isinstance(st, dict):
            continue

        post_ids = st.get("post_ids", [])
        if not isinstance(post_ids, list):
            continue

        clean_ids = []
        for pid in post_ids:
            pid = str(pid)
            if pid in valid_ids and pid not in seen_ids:
                clean_ids.append(pid)
                seen_ids.add(pid)

        if len(clean_ids) < min_size:
            continue

        validated.append({
            "id":             st.get("id", f"s{len(validated)+1}"),
            "title":          str(st.get("title", "Untitled")),
            "label":          str(st.get("label", "subtopic")),
            "reasoning":      str(st.get("reasoning", "")),
            "sentiment_arc":  str(st.get("sentiment_arc", "")),
            "post_ids":       clean_ids,
        })

    return validated


def _resolve_chrono_order(
    post_ids: list[str],
    id_to_row: dict[str, "pd.Series"],
) -> list[tuple[str, "pd.Series"]]:
    """Sort dated posts chronologically while leaving undated posts in place.

    Posts with a valid timestamp are sorted ascending among themselves and
    reinserted into the exact set of positions (slots) they originally
    occupied in `post_ids` — the LLM's own ordering. Posts with no usable
    timestamp are never moved and never assigned a synthetic date (no more
    1970-01-01 fallback): they keep the position the LLM gave them, so an
    undated post can never accidentally become the "pioneer" just because
    epoch time sorts first.
    """
    original = []
    for pid in post_ids:
        row = id_to_row.get(pid)
        if row is None:
            continue
        ts = pd.to_datetime(row.get("timestamp"), errors="coerce")
        original.append((pid, row, ts))

    if not original:
        return []

    dated_slots = [i for i, (_, _, ts) in enumerate(original) if pd.notna(ts)]
    dated_sorted = sorted((original[i] for i in dated_slots), key=lambda x: x[2])

    resolved = list(original)
    for slot, item in zip(dated_slots, dated_sorted, strict=False):
        resolved[slot] = item

    return [(pid, row) for pid, row, _ in resolved]


def _build_nova_for_cluster(
    cluster_df: pd.DataFrame,
    subtopics: list[dict],
    ncfg: dict,
    cluster_id: str = "",
    nova_ordering: str = "timeline",
) -> tuple[dict[str, dict], list[dict]]:
    """Build Nova role assignments and edge list for one cluster.

    Args:
        cluster_df: posts belonging to this cluster.
        subtopics: validated subtopic list from _validate_subtopics.
        ncfg: Nova config dict (force_nova_edge, …).
        cluster_id: used to scope subtopic IDs as "<cluster_id>__<st_id>".
        nova_ordering: "timeline" sorts by timestamp; "thematic" preserves LLM order.
            In "timeline" mode, only posts that actually have a valid
            timestamp are reordered — a post with no date is never assigned
            a fake one; it simply stays at the position the LLM originally
            gave it in post_ids (safety net, see _resolve_chrono_order).

    Returns:
        Tuple of (assignments dict keyed by post_id, edges list of dicts).
    """
    assignments: dict[str, dict] = {}
    edges: list[dict] = []

    base_force = float(ncfg.get("force_nova_edge", 1))
    id_to_row = {str(row["id"]): row for _, row in cluster_df.iterrows()}

    has_timestamps = (
        "timestamp" in cluster_df.columns
        and cluster_df["timestamp"].notna().any()
    )
    use_chrono = (nova_ordering == "timeline") and has_timestamps

    for st in subtopics:
        post_ids = st["post_ids"]

        if use_chrono:
            ordered = _resolve_chrono_order(post_ids, id_to_row)
            if not ordered:
                continue
        else:
            # Thematic mode — preserve LLM-returned order in post_ids
            ordered = []
            for pid in post_ids:
                row = id_to_row.get(pid)
                if row is None:
                    continue
                ordered.append((pid, row))
            if not ordered:
                continue

        # First in ordered list = pioneer (chronological or logical depending on mode)
        nova_id, nova_row = ordered[0]
        scoped_stid = f"{cluster_id}__{st['id']}" if cluster_id else st["id"]
        assignments[nova_id] = {
            "nova_role":          "pioneer",
            "nova_parent":        None,
            "nova_depth":         0,
            "nova_symbol":        "⭐",
            "nova_force_score":   1.0,
            "subtopic_id":        scoped_stid,
            "subtopic_label":     st["label"],
        }

        # Remaining members form a chain in the retained order, all with the
        # same offspring force regardless of rank. Visual spacing along the
        # chain is left entirely to FA2 starting from each node's own PaCMAP
        # position: two posts that start close together stay close, two that
        # start far apart stay far apart.
        for rank, (pid, _row) in enumerate(ordered[1:], start=1):
            force = base_force
            parent_id, _ = ordered[rank - 1]

            assignments[pid] = {
                "nova_role":          "offspring",
                "nova_parent":        parent_id,
                "nova_depth":         rank,
                "nova_symbol":        " ",
                "nova_force_score":   force,
                "subtopic_id":        scoped_stid,
                "subtopic_label":     st["label"],
            }

            edges.append({
                "source":   parent_id,
                "target":   pid,
                "type":       "nova",
                "force":    force,
                "nova_role":   "offspring",
            })

    return assignments, edges


def _process_cluster_worker(
    cluster_id,
    group: pd.DataFrame,
    ncfg: dict,
    prompt_params: dict,
    nova_ordering: str,
    cluster_label: str,
) -> dict:
    """Do all the work for one cluster and return a plain result dict.

    Deliberately side-effect-free (no shared dict/list mutation, no
    _save_checkpoint, no input()) so it can be safely run from any thread.
    Only used by the parallel branch of build_nova_trees — the sequential
    branch duplicates this logic inline instead of calling it, so the two
    paths must be kept in sync by hand if either changes.
    """
    n = len(group)
    if n < 2:
        return {"cluster_id": cluster_id, "status": "too_small", "n": n}

    subtopics = _call_llm_robust(cluster_label, group, ncfg, prompt_params,
                                  cluster_id=str(cluster_id))
    if subtopics is None:
        return {"cluster_id": cluster_id, "status": "failed", "n": n}

    assignments, edges = _build_nova_for_cluster(
        group, subtopics, ncfg, cluster_id=str(cluster_id), nova_ordering=nova_ordering,
    )
    metadata = []
    for st in subtopics:
        metadata.append({
            "subtopic_id":    f"{cluster_id}__{st['id']}",
            "cluster_id":     cluster_id,
            "cluster_label":  cluster_label,
            "title":          st["title"],
            "label":          st["label"],
            "reasoning":      st["reasoning"],
            "sentiment_arc":  st["sentiment_arc"],
            "post_count":     len(st["post_ids"]),
        })
    return {
        "cluster_id":   cluster_id,
        "status":       "ok",
        "assignments":  assignments,
        "edges":        edges,
        "metadata":     metadata,
        "subtopics":    subtopics,
        "n":            n,
    }


def _get_checkpoint_path(cfg: dict) -> Path:
    out_dir = Path(cfg["storage"]["processed_dir"])
    return out_dir / "nova_checkpoint.json"


def _save_checkpoint(
    assignments: dict[str, dict],
    edges: list[dict],
    metadata: list[dict],
    completed_clusters: set[str],
    cfg: dict,
    llm_raw_order: dict[str, list] | None = None,
):
    """Save Nova progress to nova_checkpoint.json.

    llm_raw_order is a debug/audit trail only: {scoped_subtopic_id: [post_ids
    in the order the LLM originally returned them]}, captured BEFORE
    _build_nova_for_cluster re-sorts by real timestamp. It is JSON-only —
    never written to the final parquet outputs (see save_nova_trees).
    """
    ncfg = _nova_cfg(cfg)
    if not ncfg.get("checkpoint_enabled", True):
        return

    cp_path = _get_checkpoint_path(cfg)
    cp_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "completed_clusters": list(completed_clusters),
        "assignments": assignments,
        "edges": edges,
        "metadata": metadata,
        "llm_raw_order": llm_raw_order or {},
        "timestamp": time.time(),
    }

    with open(cp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    logger.info("checkpoint saved: %d clusters processed", len(completed_clusters))


def _load_checkpoint(cfg: dict) -> tuple[dict, list, list, set, dict] | None:
    """Load Nova checkpoint if it exists.

    Returns:
        (assignments, edges, metadata, completed_clusters, llm_raw_order),
        or None if no checkpoint found.
    """
    cp_path = _get_checkpoint_path(cfg)
    if not cp_path.exists():
        return None

    try:
        with open(cp_path, encoding="utf-8") as f:
            checkpoint = json.load(f)

        assignments        = checkpoint.get("assignments", {})
        edges              = checkpoint.get("edges", [])
        metadata           = checkpoint.get("metadata", [])
        completed_clusters = set(checkpoint.get("completed_clusters", []))
        llm_raw_order       = checkpoint.get("llm_raw_order", {})

        logger.info("checkpoint loaded: %d clusters already processed", len(completed_clusters))
        return assignments, edges, metadata, completed_clusters, llm_raw_order

    except Exception as e:
        logger.warning("failed to load checkpoint: %s — ignored", e)
        return None


def _archive_checkpoint(cfg: dict):
    """Archive (rename) the finished checkpoint instead of deleting it.

    Moves nova_checkpoint.json into a checkpoints_archive/ subfolder with a
    timestamped filename, so the llm_raw_order audit trail (and the rest of
    the run's checkpoint state) is preserved for later inspection, while the
    active checkpoint path is cleared so a future run starts fresh.
    """
    cp_path = _get_checkpoint_path(cfg)
    if not cp_path.exists():
        return

    archive_dir = cp_path.parent / "checkpoints_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    dest = archive_dir / f"nova_checkpoint_{ts}.json"
    cp_path.rename(dest)
    logger.info("checkpoint archived: %s", dest)


def _log_parallelism(ncfg: dict, n_remaining: int) -> int:
    """Log the parallelism mode and the number of clusters to process.

    Returns the worker count (>= 1).
    """
    workers = max(1, int(ncfg.get("parallel_clusters", 1) or 1))

    if workers > 1:
        url = str(ncfg.get("url", ""))
        if "localhost" in url or "127.0.0.1" in url:
            logger.warning(
                "nova.parallel_clusters=%d but the resolved endpoint (%s) looks local — "
                "a single local model instance usually can't serve concurrent requests any "
                "faster and this may cause contention/OOM. Consider parallel_clusters: 1 "
                "for local models.",
                workers, url,
            )
        logger.info(
            "Nova: parallel_clusters=%d (PARALLEL) — processing %d cluster(s), up to %d at "
            "once. The interactive Y/I/Q retry prompt is disabled in this mode; clusters "
            "that exhaust all retries are auto-skipped and reported at the end (rerun the "
            "pipeline to retry them — the checkpoint will resume only on those).",
            workers, n_remaining, workers,
        )
    else:
        logger.notice(
            "Nova: parallel_clusters=1 (sequential) — processing %d cluster(s), one at a time.",
            n_remaining,
        )
        logger.notice(
            "Local models use sequential processing. API models can process multiple "
            "clusters in parallel (e.g. 20–50)."
        )
        logger.notice(
            "See CONFIGURATION.md > \"Run mode and module routing\" for details and how "
            "to change this setting."
        )
    return workers


def build_nova_trees(
    posts_df: pd.DataFrame,
    cfg: dict,
) -> NovaResult:
    """Build Nova subtopic trees for all clusters.

    Args:
        posts_df: full post DataFrame with at least cluster_id and content columns.
        cfg: pipeline configuration dict.

    Returns:
        NovaResult — see class docstring for field semantics.
    """
    ncfg = _nova_cfg(cfg)

    # nova_ordering is fully determined by content_type: social_media is
    # always chronological, document is always thematic. No config override,
    # no dependency on whether the corpus actually has timestamps — see
    # _build_prompt_params docstring. The per-post fallback for individual
    # undated posts within a chronological cluster is handled locally in
    # _build_nova_for_cluster (it keeps the post at its LLM-assigned
    # position rather than forcing a fake date).
    content_type = cfg.get("content_type", "social_media")
    prompt_params = _build_prompt_params(cfg)
    nova_ordering = prompt_params["nova_ordering"]

    logger.info("content_type=%s  nova_ordering=%s", content_type, nova_ordering)

    posts_df = posts_df.copy()
    for col in ("nova_role", "nova_parent", "nova_depth",
                "nova_symbol", "nova_force_score", "subtopic_id", "subtopic_label"):
        posts_df[col] = None

    # Resume from checkpoint if one exists
    checkpoint_data = _load_checkpoint(cfg)
    if checkpoint_data:
        all_assignments, all_edges, all_metadata, completed_clusters, all_llm_order = checkpoint_data
        logger.info("resuming from checkpoint")
    else:
        all_assignments: dict[str, dict] = {}
        all_edges: list[dict] = []
        all_metadata: list[dict] = []
        completed_clusters: set[str] = set()
        all_llm_order: dict[str, list] = {}

    clustered = posts_df[posts_df["cluster_id"].notna()]
    cluster_ids = clustered["cluster_id"].unique()

    remaining_clusters = [cid for cid in cluster_ids if str(cid) not in completed_clusters]

    logger.info("total clusters: %d | already done: %d | remaining: %d | model: %s | max_retries: %d",
                len(cluster_ids), len(completed_clusters), len(remaining_clusters),
                ncfg['model'], ncfg.get('max_retries', 3))

    llm_abandoned = False
    auto_skip_failures = False  # set once user picks "ignore" — no more prompts for the rest of this run

    # Clusters where the LLM genuinely failed (all retries exhausted) and got
    # auto-skipped/ignored rather than successfully classified. Distinct from
    # "too_small" clusters, which are an intentional design skip, not a
    # failure. Any entry here means nova did NOT validate that cluster, and
    # callers (see run()/pipeline.py) must not treat the run as clean.
    failed_clusters: set[str] = set()

    parallel_workers = _log_parallelism(ncfg, len(remaining_clusters))

    if not remaining_clusters:
        logger.info("all clusters already processed — skipping")
    elif parallel_workers <= 1:
        clustering_only_tip_shown = False  # print the --run-mode clustering_only tip once, not per failure
        for cidx, cluster_id in enumerate(remaining_clusters, start=1):
            group = clustered[clustered["cluster_id"] == cluster_id].copy()
            n = len(group)

            cluster_label = str(cluster_id)
            if "cluster_label" in group.columns and group["cluster_label"].notna().any():
                cluster_label = str(group["cluster_label"].iloc[0])

            logger.info("[%d/%d] cluster=%s | %d posts | %r",
                        cidx, len(remaining_clusters), cluster_id, n, cluster_label)

            if n < 2:
                logger.warning("cluster=%s too small (%d post) — sending to ADEPT as orphan", cluster_id, n)
                completed_clusters.add(str(cluster_id))
                continue

            subtopics = _call_llm_robust(cluster_label, group, ncfg, prompt_params,
                                          cluster_id=str(cluster_id))

            if subtopics is None:
                if auto_skip_failures:
                    logger.warning("cluster=%s failed after %d attempts — auto-skipped (ignore-all active)",
                                   cluster_id, ncfg.get('max_retries', 3))
                    # NOT added to completed_clusters on purpose: a failed
                    # cluster must stay eligible for retry on the next run.
                    # completed_clusters means "successfully done", not
                    # "attempted" — see failed_clusters for the latter.
                    failed_clusters.add(str(cluster_id))
                    _save_checkpoint(all_assignments, all_edges, all_metadata, completed_clusters, cfg,
                                  llm_raw_order=all_llm_order)
                    continue

                # All retries exhausted — prompt the user to decide. This loop
                # only exits via "I" (ignore this cluster) or "Q" (abort);
                # "Y"/Enter retries the same cluster and loops back. There is
                # no "keep going without the LLM" option: ADEPT never starts
                # on an unvalidated Nova pass (mandatory gate, see
                # pipeline.py), so a degraded continue is not something the
                # pipeline can actually do — quitting is the only way out.
                while True:
                    print(f"\n[Nova] cluster={cluster_id} failed after {ncfg.get('max_retries', 3)} attempts — retries exhausted.")
                    print("  Y — retry this cluster (retries again on failure; does not auto-skip)")
                    print("  I — ignore and continue (skip this cluster; auto-skip any further failures for the rest of this run)")
                    print("  Q — abort (this cluster and any remaining ones stay unvalidated;")
                    print("      rerun the pipeline to resume from here)")
                    if not clustering_only_tip_shown:
                        print("[Nova] Note: to skip Nova/ADEPT entirely and avoid this prompt,")
                        print("       restart with --run-mode clustering_only.")
                        clustering_only_tip_shown = True
                    raw = safe_input("Choice [Y/i/q]: ")
                    # No terminal / stdin closed / Ctrl-C: fall back to "q",
                    # the same safe exit already used when a human quits —
                    # never fall back to "i" (ignore) or a silent retry.
                    choice = raw.strip().lower() if raw is not None else "q"

                    if choice == "q":
                        n_remaining = len(remaining_clusters) - cidx + 1
                        logger.warning("user quit — %d cluster(s) left unprocessed (including "
                                       "this one); rerun the pipeline to resume exactly where "
                                       "this left off",
                                       n_remaining)
                        for skip_cid in remaining_clusters[cidx - 1:]:
                            # Not added to completed_clusters — see comment above.
                            failed_clusters.add(str(skip_cid))
                        _save_checkpoint(all_assignments, all_edges, all_metadata, completed_clusters, cfg,
                                  llm_raw_order=all_llm_order)
                        llm_abandoned = True  # → nova_validated=False; pipeline.py's mandatory
                                               # gate stops the run before ADEPT and explains resume.
                        break

                    elif choice == "i":
                        logger.warning("user chose ignore — cluster=%s skipped, auto-skip enabled for remaining failures this run",
                                       cluster_id)
                        # Not added to completed_clusters — see comment above.
                        failed_clusters.add(str(cluster_id))
                        _save_checkpoint(all_assignments, all_edges, all_metadata, completed_clusters, cfg,
                                  llm_raw_order=all_llm_order)
                        auto_skip_failures = True
                        break

                    elif choice in ("", "y"):
                        # Enter or "Y": retry the same cluster, looping back
                        # to the prompt above on further failure.
                        logger.info("retrying cluster=%s...", cluster_id)
                        subtopics = _call_llm_robust(cluster_label, group, ncfg, prompt_params,
                                                      cluster_id=str(cluster_id))
                        if subtopics is not None:
                            # Success on retry — break out of the prompt loop
                            # and fall through to normal processing below.
                            break
                        logger.warning("cluster=%s still failing — asking again (Y/I/Q)", cluster_id)

                    else:
                        # Anything else is not a valid choice — reprompt
                        # instead of silently treating it as a retry.
                        print(f"[Nova] '{choice}' is not a valid choice — please enter Y, I, or Q.")

                if llm_abandoned:
                    break
                if subtopics is None:
                    # Only reachable via the "I" (ignore) branch above.
                    continue

            n_classified = sum(len(st["post_ids"]) for st in subtopics)
            n_orphans = n - n_classified
            logger.info("→ %d subtopics | %d classified | %d → ADEPT",
                        len(subtopics), n_classified, n_orphans)
            _log_subtopic_summary(cluster_id, subtopics)

            assignments, edges = _build_nova_for_cluster(
                group, subtopics, ncfg,
                cluster_id=str(cluster_id),
                nova_ordering=nova_ordering,
            )
            all_assignments.update(assignments)
            all_edges.extend(edges)

            for st in subtopics:
                scoped_stid = f"{cluster_id}__{st['id']}"
                all_metadata.append({
                    "subtopic_id":    scoped_stid,
                    "cluster_id":     cluster_id,
                    "cluster_label":  cluster_label,
                    "title":          st["title"],
                    "label":          st["label"],
                    "reasoning":      st["reasoning"],
                    "sentiment_arc":  st["sentiment_arc"],
                    "post_count":     len(st["post_ids"]),
                })
                # Raw LLM-returned post_ids order, captured BEFORE
                # _build_nova_for_cluster re-sorts by real timestamp.
                # JSON checkpoint only — never fed into all_metadata/parquet.
                all_llm_order[scoped_stid] = st["post_ids"]

            completed_clusters.add(str(cluster_id))
            _save_checkpoint(all_assignments, all_edges, all_metadata, completed_clusters, cfg,
                              llm_raw_order=all_llm_order)

    else:
        # Parallel mode: N clusters in flight at once via threads
        # HTTP calls are I/O-bound, so threads (not processes) are enough —
        # each _call_llm_robust/raw_request call is independent (no shared
        # requests.Session), so concurrent HTTP is safe by construction.
        # What's NOT safe by construction is the shared state below
        # (all_assignments/all_edges/all_metadata/completed_clusters) and the
        # checkpoint file — both are protected by state_lock so results are
        # merged and saved one cluster at a time, never interleaved.
        state_lock = threading.Lock()
        done_count = 0

        def _handle_result(result: dict):
            nonlocal done_count
            cid_str = str(result["cluster_id"])
            with state_lock:
                if result["status"] == "too_small":
                    logger.warning("cluster=%s too small (%d post) — sending to ADEPT as orphan",
                                    cid_str, result["n"])
                    completed_clusters.add(cid_str)

                elif result["status"] == "failed":
                    logger.warning(
                        "cluster=%s failed after %d attempts — auto-skipped "
                        "(parallel mode: no interactive prompt)",
                        cid_str, ncfg.get("max_retries", 3),
                    )
                    # Not added to completed_clusters — see comment in the
                    # sequential branch above: a failed cluster must stay
                    # eligible for retry on the next run, otherwise
                    # remaining_clusters silently drops it forever.
                    failed_clusters.add(cid_str)

                else:  # "ok"
                    subtopics = result["subtopics"]
                    n_classified = sum(len(st["post_ids"]) for st in subtopics)
                    logger.info("cluster=%s → %d subtopics | %d classified | %d → ADEPT",
                                cid_str, len(subtopics), n_classified, result["n"] - n_classified)
                    _log_subtopic_summary(cid_str, subtopics)

                    all_assignments.update(result["assignments"])
                    all_edges.extend(result["edges"])
                    all_metadata.extend(result["metadata"])
                    for st in result["subtopics"]:
                        scoped_stid = f"{cid_str}__{st['id']}"
                        all_llm_order[scoped_stid] = st["post_ids"]
                    completed_clusters.add(cid_str)

                _save_checkpoint(all_assignments, all_edges, all_metadata, completed_clusters, cfg,
                                  llm_raw_order=all_llm_order)
                done_count += 1
                logger.info("[parallel %d/%d clusters done] cluster=%s",
                            done_count, len(remaining_clusters), cid_str)

        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {}
            for cluster_id in remaining_clusters:
                group = clustered[clustered["cluster_id"] == cluster_id].copy()
                cluster_label = str(cluster_id)
                if "cluster_label" in group.columns and group["cluster_label"].notna().any():
                    cluster_label = str(group["cluster_label"].iloc[0])

                fut = executor.submit(
                    _process_cluster_worker, cluster_id, group, ncfg, prompt_params,
                    nova_ordering, cluster_label,
                )
                futures[fut] = cluster_id

            for fut in as_completed(futures):
                cluster_id = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    # Should not normally happen — _call_llm_robust already
                    # swallows its own exceptions — but guard anyway so one
                    # cluster crashing a thread can't take down the whole run.
                    logger.error("cluster=%s raised unexpectedly: %s: %s",
                                 cluster_id, type(e).__name__, e)
                    result = {"cluster_id": cluster_id, "status": "failed", "n": 0}
                _handle_result(result)

        if failed_clusters:
            logger.warning(
                "nova parallel mode: %d cluster(s) failed and were auto-skipped "
                "— rerun the pipeline to retry them (checkpoint resumes only on those).",
                len(failed_clusters),
            )
            logger.debug("nova parallel mode failed cluster ids: %s", sorted(failed_clusters))

    # Apply assignments to posts_df
    for pid, info in all_assignments.items():
        mask = posts_df["id"].astype(str) == str(pid)
        posts_df.loc[mask, "nova_role"]         = info.get("nova_role")
        posts_df.loc[mask, "nova_parent"]       = info.get("nova_parent")
        posts_df.loc[mask, "nova_depth"]        = info.get("nova_depth", 0)
        posts_df.loc[mask, "nova_symbol"]       = info.get("nova_symbol", " ")
        posts_df.loc[mask, "nova_force_score"]  = info.get("nova_force_score", 0.0)
        posts_df.loc[mask, "subtopic_id"]       = info.get("subtopic_id")
        posts_df.loc[mask, "subtopic_label"]    = info.get("subtopic_label")

    nova_edges = pd.DataFrame(all_edges) if all_edges else pd.DataFrame(
        columns=["source", "target", "type", "force", "nova_role"]
    )

    role_counts  = posts_df["nova_role"].value_counts(dropna=True).to_dict()
    orphan_count = posts_df["nova_role"].isna().sum()
    logger.info(
        "nova complete — roles: %s | orphans: %d | nova_edges: %d | subtopic_metadata: %d",
        role_counts, orphan_count, len(nova_edges), len(all_metadata),
    )

    # completed_clusters only ever gains an entry on genuine success (or a
    # legitimate too-small skip) — failed/ignored/abandoned clusters are
    # deliberately kept OUT of it (see the three "Not added to
    # completed_clusters" comments above) so they remain in
    # remaining_clusters and get retried automatically on the next run,
    # instead of being silently dropped forever.
    all_clusters_done = len(completed_clusters) == len(cluster_ids)

    # Kept as an explicit, separate check (rather than relying solely on
    # all_clusters_done) so the intent is self-documenting at call sites and
    # this stays correct even if completed_clusters' bookkeeping changes
    # later. Callers must gate ADEPT on this, not on all_clusters_done alone.
    nova_validated = all_clusters_done and not failed_clusters and not llm_abandoned

    if failed_clusters:
        logger.warning(
            "nova: %d/%d cluster(s) were NOT validated (LLM failed and got "
            "auto-skipped) — nova_validated=False",
            len(failed_clusters), len(cluster_ids),
        )
        logger.debug("nova failed cluster ids: %s", sorted(failed_clusters))

    return NovaResult(
        posts_df=posts_df,
        nova_edges=nova_edges,
        llm_abandoned=llm_abandoned,
        all_metadata=all_metadata,
        all_clusters_done=all_clusters_done,
        failed_clusters=sorted(failed_clusters),
        nova_validated=nova_validated,
    )


def save_nova_trees(
    posts_df: pd.DataFrame,
    nova_edges: pd.DataFrame,
    cfg: dict,
    all_metadata: list | None = None,
    archive_when_done: bool = False,
):
    """Save Nova assignments, edges, and subtopic metadata to parquet.

    If all_metadata is None, it is read from the checkpoint file instead.
    If archive_when_done is True, the checkpoint is archived after
    nova_metadata.parquet has been written.
    """
    out_dir = Path(cfg["storage"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    nova_cols = ["id", "cluster_id", "nova_role", "nova_parent", "nova_depth",
                 "nova_symbol", "nova_force_score",
                 "subtopic_id", "subtopic_label"]
    available = [c for c in nova_cols if c in posts_df.columns]
    posts_df[available].to_parquet(out_dir / "nova_assignments.parquet", index=False)
    nova_edges.to_parquet(out_dir / "nova_edges.parquet", index=False)

    if all_metadata is None:
        checkpoint_data = _load_checkpoint(cfg)
        all_metadata = checkpoint_data[2] if checkpoint_data else []

    if all_metadata:
        nova_metadata = pd.DataFrame(all_metadata)
        nova_metadata.to_parquet(out_dir / "nova_metadata.parquet", index=False)
        logger.info("saved → nova_assignments + nova_edges + nova_metadata.parquet")
    else:
        logger.info("saved → nova_assignments.parquet + nova_edges.parquet")

    if archive_when_done:
        _archive_checkpoint(cfg)


def run(
    posts_df: pd.DataFrame,
    config_path: str = "config.yaml",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and save Nova trees for the full corpus."""
    cfg = load_config(config_path)
    result = build_nova_trees(posts_df, cfg)
    posts_df, nova_edges, all_metadata, nova_validated = (
        result.posts_df, result.nova_edges, result.all_metadata, result.nova_validated
    )
    # Only archive (and clear) the checkpoint on a clean, fully-validated run —
    # archiving after a failed run would wipe the resume state and force a
    # full reprocess of already-successful clusters on the next attempt.
    save_nova_trees(posts_df, nova_edges, cfg, all_metadata=all_metadata, archive_when_done=nova_validated)
    return posts_df, nova_edges