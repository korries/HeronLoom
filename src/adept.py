"""Step 5b — ADEPT: groups NOVA orphans into semantic pools via Density Peak Clustering + LLM. Entry point: run()."""

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from run_store import safe_input
from utils.logger import get_logger

logger = get_logger(__name__)


class AdeptResult(NamedTuple):
    """Return value of `build_adept_events`.

    adept_validated is the flag callers must gate Edges on — it is True
    only if every cluster was processed AND none of them failed or were
    abandoned. all_clusters_done alone is not sufficient: it is True once
    every cluster has been processed (success OR failure), so it stays
    True even when clusters were skipped after exhausting retries.
    """
    posts_df: pd.DataFrame
    adept_edges: pd.DataFrame
    all_metadata: list        # per-cluster metadata dicts
    llm_abandoned: bool       # True if the user quit (Q) instead of retrying/ignoring
                              # a failed cluster. Does NOT mean ADEPT ran without the
                              # LLM — there is no such mode; it only forces
                              # adept_validated=False.
    all_clusters_done: bool   # every cluster was processed, success or failure
    failed_clusters: list     # sorted cluster ids where the LLM exhausted all
                              # retries and got auto-skipped (empty if none)
    adept_validated: bool     # gate Edges on this, not on all_clusters_done alone


def _adept_cfg(cfg: dict) -> dict:
    from _llm import get_llm_cfg
    defaults = {
        "parallel_clusters":  1,     # 1 = sequential — enables the interactive Y/I/Q retry prompt
        "max_retries":        3,
        "checkpoint_enabled": True,
    }
    return {**defaults, **get_llm_cfg(cfg, "adept")}


def _euclidean_matrix(embs: np.ndarray) -> np.ndarray:
    diff = embs[:, None, :] - embs[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)


def _load_embeddings_df(df: pd.DataFrame, prefer: str = "raw") -> np.ndarray:
    """Load embeddings from df into a contiguous float32 array (prefer raw or lda).

    Raises:
        ValueError: column missing, contains None, or format unrecognised.
    """
    import pickle
    col = "embedding_raw" if prefer == "raw" else "embedding_lda"
    if col not in df.columns:
        raise ValueError(f"Column '{col}' missing (available: {list(df.columns)})")
    vals = df[col].values
    if vals[0] is None:
        raise ValueError(f"Column '{col}' contains None values")
    if isinstance(vals[0], np.ndarray):
        return np.stack(vals).astype(np.float32)
    if isinstance(vals[0], bytes):
        return np.stack([pickle.loads(v) for v in vals]).astype(np.float32)
    if isinstance(vals[0], (list, tuple)):
        return np.array([np.array(v) for v in vals], dtype=np.float32)
    raise ValueError(f"Unrecognised embedding format in '{col}': {type(vals[0])}")


def _truncate_clean(text: str, max_chars: int) -> str:
    """Truncate at a word boundary and append an ellipsis to mark the cut.

    Mirrors nova.py's _truncate_clean — same technique, reimplemented
    locally since it's a private per-module helper, not a shared import.
    """
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space]
    return cut.rstrip() + "…"


def _escape_content(text: str) -> str:
    """Replace angle brackets with lookalike characters so post content can't
    spoof a <post> tag boundary. Mirrors nova.py's _escape_content."""
    return text.replace("<", "‹").replace(">", "›")


def _build_prompt_params(cfg: dict) -> dict:
    """Derive ADEPT's prompt-wording params from `content_type`, the same
    single source of truth nova.py's own `_build_prompt_params` uses.

    Only the vocabulary (posts vs documents) needs deriving here — unlike
    Nova, ADEPT has no chronological/thematic ordering to branch on (see
    pipeline.py's comment above the content_type log line, which currently
    overstates this: ADEPT reads content_type for wording only).
    """
    content_type = cfg.get("content_type", "social_media")

    if content_type == "social_media":
        unit_word     = "posts"
        unit_singular = "post"
    else:  # document
        unit_word     = "documents"
        unit_singular = "document"

    return {
        "unit_word":     unit_word,
        "unit_singular": unit_singular,
    }


def _build_pool_explanation_prompt(
    cluster_label: str,
    prompt_params: dict,
    pools: list[dict],
    posts_df: pd.DataFrame,
) -> str:
    unit_word     = prompt_params["unit_word"]
    unit_singular = prompt_params["unit_singular"]

    groups_text = []
    for i, pool in enumerate(pools, 1):
        hub_id     = pool["hub_id"]
        member_ids = pool["member_ids"]

        hub_post    = posts_df[posts_df["id"] == hub_id].iloc[0]
        hub_content = _escape_content(_truncate_clean(str(hub_post["content"]).strip(), 300))

        candidates_text = []
        for mid in member_ids[:5]:
            member_post = posts_df[posts_df["id"] == mid]
            if not member_post.empty:
                member_content = _escape_content(
                    _truncate_clean(str(member_post.iloc[0]["content"]).strip(), 200)
                )
                candidates_text.append(f'  <{unit_singular} id="{mid}">{member_content}</{unit_singular}>')

        groups_text.append(
            f"GROUP {i}:\n"
            f'REFERENCE: <{unit_singular} id="{hub_id}">{hub_content}</{unit_singular}>\n'
            f"CANDIDATES:\n"
            f"{chr(10).join(candidates_text)}\n"
        )

    n_groups = len(pools)

    return f"""<role>
You are reviewing groups of {unit_word} that were automatically pre-grouped by
topic similarity. Each group has one REFERENCE {unit_singular} that anchors the
topic, and a set of CANDIDATE {unit_word} proposed as belonging to that same
topic — the candidates have not been verified yet.
</role>

<context>
Topic area: {cluster_label}
</context>

<groups>
{chr(10).join(groups_text)}
</groups>

<task>
For EACH group shown above, return one entry using its exact GROUP number
(1, 2, 3... exactly as labeled — do not renumber, skip, or start at 0) as
"group_id", plus:

1. **excluded_candidates**: which CANDIDATEs don't genuinely belong to this topic? Decide this FIRST, before reasoning/title below.
   - Judge each CANDIDATE against the REFERENCE and the rest of the group together — the REFERENCE is a starting point, not a strict rule
   - Exclude a CANDIDATE only if it has no real thematic connection to the group; list its id
   - The REFERENCE can NEVER be excluded
   - If every candidate fits, return an empty list

2. **reasoning**: ONE simple sentence — why are the kept {unit_word} about the same topic? (using only the {unit_word} you kept)
   - Max 20 words, plain language
   - Do NOT justify excluded_candidates here — that belongs above, not in reasoning
   - Do NOT list ids or quote the {unit_word} — just name the shared theme

3. **title**: a concise, self-explanatory title (max 8 words)
   - Structure: Entity + State/Condition
   - Good: "AI Data Centers Water Scarcity"
   - Good: "Will Smith Pasta Benchmark Meme"
   - Good: "Pope Francis Religious AI Opposition"
   - Bad: "Content About AI"
   - Bad: "Discussion of Technology Concerns"
</task>

<output_format>
Return ONLY valid JSON, with exactly {n_groups} entries in "groups" — one per
GROUP shown above, "group_id" matching its number:
{{
  "groups": [
    {{
      "group_id": 1,
      "excluded_candidates": ["candidate_id_1", "candidate_id_2"],
      "reasoning": "Explanation of the shared theme...",
      "title": "Entity State Title"
    }}
  ]
}}
</output_format>"""


def _extract_json_from_text(text: str) -> str | None:
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text).strip()
    if text.startswith("{"):
        return text
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else None


def _call_llm_robust_adept(
    cluster_label: str,
    prompt_params: dict,
    pools: list[dict],
    posts_df: pd.DataFrame,
    ncfg: dict,
    cluster_id: str = "",
) -> tuple[list[dict] | None, str | None]:
    """Call the LLM for pool explanations with retry, classifying each failed
    attempt as either:
      - "network": the HTTP call itself raised (connection error, timeout,
        non-2xx status, ...) — the endpoint looks unreachable.
      - "format": the HTTP call succeeded but the response could not be
        turned into valid pool explanations (bad/missing JSON).

    Retries up to ncfg["max_retries"] times regardless of failure kind
    (exponential backoff between attempts). err_kind is informational only —
    build_adept_events prompts identically (retry/ignore/quit) whichever
    kind the last attempt failed with.

    Returns:
        (explanations, error_kind). error_kind is None on success, else the
        kind of the *last* failed attempt ("network" or "format").
    """
    from _llm import raw_request

    max_retries = ncfg.get("max_retries", 3)
    prompt      = _build_pool_explanation_prompt(cluster_label, prompt_params, pools, posts_df)

    last_kind: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = raw_request(ncfg, "", prompt)
        except Exception as e:
            last_kind = "network"
            logger.warning("cluster=%s attempt %d/%d — network/HTTP error: %s: %s",
                            cluster_id, attempt, max_retries, type(e).__name__, e)
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))
            continue

        try:
            json_str = _extract_json_from_text(raw)
            if json_str is None:
                raise ValueError(f"could not extract JSON (response: {raw[:200]})")
            parsed = json.loads(json_str)
            explanations = parsed.get("groups", [])
            if not isinstance(explanations, list):
                raise ValueError(f"'groups' is not a list: {str(parsed)[:200]}")

            # Structural check: every group_id must fall inside 1..len(pools).
            # Without this, an LLM that numbers from 0 (or skips/duplicates
            # a number outside range) passes silently through
            # _match_explanations_to_pools and produces a wrong-pool
            # misattribution with no exception raised anywhere — the one
            # failure mode that doesn't degrade cleanly. Catching it here
            # reuses the existing retry loop; no new mechanism needed.
            n_pools = len(pools)
            bad_ids = [
                exp.get("group_id") for exp in explanations
                if isinstance(exp, dict) and (
                    not isinstance(exp.get("group_id"), int)
                    or not (1 <= exp["group_id"] <= n_pools)
                )
            ]
            if bad_ids:
                raise ValueError(
                    f"group_id(s) out of expected range 1..{n_pools}: {bad_ids}"
                )

            return explanations, None
        except Exception as e:
            last_kind = "format"
            logger.warning("cluster=%s attempt %d/%d — malformed response: %s: %s",
                            cluster_id, attempt, max_retries, type(e).__name__, e)
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))
            continue

    logger.warning("cluster=%s — pool explanations failed after %d attempts (last failure: %s)",
                    cluster_id, max_retries, last_kind)
    return None, last_kind


def _match_explanations_to_pools(pools: list[dict], explanations: list[dict]) -> list[dict]:
    """Align LLM-returned group explanations to `pools` by their own
    "group_id" (1-indexed, matching the "GROUP N" labels sent in the prompt)
    instead of by raw list position.

    The LLM's JSON array order/count is not guaranteed to match the pools
    sent (a skipped, reordered, or duplicated entry is common with generated
    JSON) — trusting position alone silently attaches one pool's title to a
    different hub. Matching by the id the LLM itself echoed back is robust
    to all of that: a genuinely missing pool just gets {}, instead of
    misattributing a neighbour's title.

    Returns:
        A list the same length as `pools`; slot i holds the matched
        explanation dict for pools[i], or {} if none matched.
    """
    if not explanations:
        return [{} for _ in pools]

    by_id: dict[int, dict] = {}
    for exp in explanations:
        if not isinstance(exp, dict):
            continue
        try:
            gid = int(exp.get("group_id"))
        except (TypeError, ValueError):
            continue  # missing/non-numeric group_id — can't place it safely, skip
        if gid not in by_id:  # first entry for a given group_id wins; later duplicates ignored
            by_id[gid] = exp

    return [by_id.get(i, {}) for i in range(1, len(pools) + 1)]


def _apply_pool_exclusions(
    pools: list[dict],
    explanations: list[dict],
    updated_group: pd.DataFrame,
    adept_edges: pd.DataFrame,
    cluster_id,
    cluster_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Drop members the LLM flagged as not belonging to their group.

    For each pool, any id listed under explanations[i]["excluded_candidates"]
    is removed from pool["member_ids"]/["labels"], its adept_role/adept_symbol/
    eng_norm are reset to None in updated_group (it goes back to being an
    unassigned/orphan post), and its hub→member "adept_spoke" edge is
    dropped. The HUB (REFERENCE in the prompt) itself is never excluded —
    it's the reference anchor candidates are checked against, not a strict,
    sole definer of the group's exact subject (see the prompt's
    "excluded_candidates" instructions). Unknown or hub ids in
    "excluded_candidates" are ignored rather than trusted blindly.

    Returns:
        (updated_group, adept_edges, exclusion_log) — exclusion_log is a
        list of {cluster_id, cluster_label, hub_id, pool_title, excluded_id}.
    """
    exclusion_log: list[dict] = []
    if not explanations:
        return updated_group, adept_edges, exclusion_log

    drop_mask = pd.Series(False, index=adept_edges.index) if len(adept_edges) else None
    matched = _match_explanations_to_pools(pools, explanations)

    for i, pool in enumerate(pools):
        exp = matched[i]
        raw_excluded = exp.get("excluded_candidates", [])
        if not isinstance(raw_excluded, list) or not raw_excluded:
            continue

        hub_id     = pool["hub_id"]
        member_set = set(pool["member_ids"])
        pool_title = exp.get("title", "Unknown title")

        for entry in raw_excluded:
            # excluded_candidates is a plain list of ids; tolerate a stray
            # {"candidate_id": ...} (or the pre-rename "post_id") shape too
            # in case the model wraps it anyway.
            if isinstance(entry, dict):
                excluded_id = str(entry.get("candidate_id") or entry.get("post_id") or "").strip()
            else:
                excluded_id = str(entry).strip()

            if not excluded_id or excluded_id == hub_id or excluded_id not in member_set:
                continue  # hub can't be excluded; unknown/already-removed ids are ignored

            pool["member_ids"].remove(excluded_id)
            pool["labels"].pop(excluded_id, None)
            member_set.discard(excluded_id)

            mask = updated_group["id"] == excluded_id
            for col in ("adept_role", "adept_symbol", "eng_norm"):
                updated_group.loc[mask, col] = None

            if drop_mask is not None:
                drop_mask = drop_mask | (
                    (adept_edges["type"] == "adept_spoke")
                    & (adept_edges["source"] == hub_id)
                    & (adept_edges["target"] == excluded_id)
                )

            exclusion_log.append({
                "cluster_id":    cluster_id,
                "cluster_label": cluster_label,
                "hub_id":        hub_id,
                "pool_title":    pool_title,
                "excluded_id":   excluded_id,
            })

    if drop_mask is not None and drop_mask.any():
        adept_edges = adept_edges[~drop_mask].reset_index(drop=True)

    return updated_group, adept_edges, exclusion_log


def _log_pool_summary(
    cluster_id,
    cluster_label: str,
    pools: list[dict],
    explanations: list[dict],
    exclusion_log: list[dict],
) -> None:
    """Log one compact line per pool (title, member count) at INFO, plus
    full detail (reasoning, excluded ids) at DEBUG.

    logger-only, no print() — keeps a single logging channel instead of
    mixing print() (console-only) with logger (console+file). Mirrors the
    per-subtopic summary logged by nova.py's _log_subtopic_summary.
    """
    excl_by_hub: dict[str, list[dict]] = {}
    for rec in exclusion_log:
        excl_by_hub.setdefault(rec["hub_id"], []).append(rec)

    matched = _match_explanations_to_pools(pools, explanations or [])
    for i, pool in enumerate(pools):
        exp       = matched[i]
        title     = exp.get("title", "Unknown title")
        reasoning = exp.get("reasoning", "No explanation provided")
        hub_id    = pool["hub_id"]
        excluded  = excl_by_hub.get(hub_id, [])
        excluded_ids = [rec["excluded_id"] for rec in excluded]

        logger.info("pool[%s] %r (%d members)", hub_id, title, len(pool["member_ids"]))
        logger.debug(
            "cluster=%s pool hub=%s title=%r reasoning=%r excluded=%r",
            cluster_id, hub_id, title, reasoning, excluded_ids,
        )


def _compute_dc(dist_mat: np.ndarray, percentile: float = 2.0) -> float:
    """Cut-off distance dc so each point neighbours ~percentile% of others."""
    n   = dist_mat.shape[0]
    idx = np.triu_indices(n, k=1)
    return float(np.percentile(dist_mat[idx], percentile))


def _dpc_rho_delta(
    dist_mat: np.ndarray, dc: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute DPC density rho, distance-to-higher-density delta, and parent index.

    Returns:
        (rho, delta, parent) each of shape (n,). parent[i] == -1 for the global density peak.
    """
    n      = dist_mat.shape[0]
    rho    = np.array([(dist_mat[i] < dc).sum() - 1 for i in range(n)], dtype=int)
    delta  = np.full(n, np.inf, dtype=np.float64)
    parent = np.full(n, -1, dtype=int)

    order = np.argsort(rho)[::-1]
    for rank, i in enumerate(order):
        higher = order[:rank]
        if len(higher) == 0:
            delta[i]  = float(dist_mat[i].max())
        else:
            dists     = dist_mat[i, higher]
            best      = int(np.argmin(dists))
            delta[i]  = float(dists[best])
            parent[i] = int(higher[best])

    return rho, delta, parent


def _membership_threshold(delta: np.ndarray, percentile: float = 50.0) -> float:
    """Max delta (distance-to-higher-density) for a point to join its parent's pool.

    percentile-th percentile of the finite delta values; np.inf (nothing
    admitted) if every delta is infinite (single point / degenerate input).
    """
    finite = delta[np.isfinite(delta)]
    return float(np.percentile(finite, percentile)) if len(finite) > 0 else np.inf


def _build_pools(
    orphans_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    cfg: dict,
) -> tuple[list[dict], set[str]]:
    """Cluster orphans into pools via Density Peak Clustering on LDA embeddings.

    Returns:
        (pools, used_ids) where pools is a list of {hub_id, member_ids, labels}
        and used_ids is the set of post IDs consumed into a pool.
    """
    if len(orphans_df) < 2:
        return [], set()

    adept_cfg             = cfg.get("adept", {})
    min_pool_size         = int(adept_cfg.get("pool_min_size", 2))
    max_pool_size         = adept_cfg.get("pool_max_size")  # None = no cap
    dc_percentile         = float(adept_cfg.get("dpc_dc_percentile", 2.0))
    membership_percentile = float(adept_cfg.get("dpc_membership_percentile", 50.0))
    gamma_percentile      = float(adept_cfg.get("dpc_gamma_percentile", 80.0))

    post_ids    = orphans_df["id"].tolist()
    engagements = orphans_df["engagement"].fillna(0).values.astype(float)

    if "timestamp" in orphans_df.columns:
        ts_raw = pd.to_datetime(orphans_df["timestamp"], errors="coerce")
        timestamps = (
            ts_raw.astype("int64").values
            if ts_raw.notna().any()
            else np.zeros(len(orphans_df), dtype="int64")
        )
    else:
        timestamps = np.zeros(len(orphans_df), dtype="int64")

    embs     = _load_embeddings_df(orphans_df, prefer="lda")
    dist_mat = _euclidean_matrix(embs)
    n        = len(post_ids)

    dc            = _compute_dc(dist_mat, percentile=dc_percentile)
    rho, delta, parent = _dpc_rho_delta(dist_mat, dc)
    mem_threshold = _membership_threshold(delta, percentile=membership_percentile)

    gamma           = rho * delta
    id_to_local     = {pid: i for i, pid in enumerate(post_ids)}
    gamma_threshold = float(np.percentile(gamma, gamma_percentile))
    # Hubs: local maxima in gamma space with at least one neighbour
    hub_indices     = [i for i in range(n) if gamma[i] >= gamma_threshold and rho[i] > 0]
    hub_set         = set(hub_indices)

    # Transitive resolution: walk each non-hub point's parent chain (its
    # nearest-higher-density neighbour, then THAT point's own nearest
    # higher-density neighbour, and so on) until either a hub is reached
    # (the point's root) or the chain breaks. Unlike the previous one-hop
    # check (`parent[j] == hub_idx`), this lets a point several hops away
    # from a hub still join its pool, as long as it belongs to the same
    # density basin. Every hop must individually satisfy delta<=mem_threshold
    # (a single weak link still breaks the chain — this preserves what
    # mem_threshold originally meant) and a point stops at the FIRST hub it
    # meets walking up, never skipping past a nearer hub to reach a farther
    # one. Points whose entire chain fails to reach any hub stay orphan,
    # same as before.
    def _resolve_root(start: int) -> int | None:
        cur = start
        while True:
            if cur in hub_set:
                return cur
            if parent[cur] == -1 or delta[cur] > mem_threshold:
                return None
            cur = parent[cur]

    members_by_hub: dict[int, list[int]] = {h: [] for h in hub_indices}
    for j in range(n):
        if j in hub_set:
            continue
        root = _resolve_root(j)
        if root is not None:
            members_by_hub[root].append(j)

    node_roles = cfg.get("render", {}).get("node_roles", {})
    max_eng    = float(engagements.max()) or 1.0
    used_ids: set[str] = set()
    pools: list[dict]  = []

    for hub_idx in sorted(hub_indices, key=lambda i: gamma[i], reverse=True):
        hub_id = post_ids[hub_idx]
        if hub_id in used_ids:
            continue

        members: list[str] = [
            post_ids[j]
            for j in members_by_hub[hub_idx]
            if post_ids[j] not in used_ids
        ]

        if len(members) < (min_pool_size - 1):
            continue

        # Cap pool size: keep only the closest candidates (smallest delta = most
        # tightly bound to the hub); the rest stay orphan (no pool at all for them).
        if max_pool_size is not None and len(members) > (int(max_pool_size) - 1):
            members = sorted(members, key=lambda pid: delta[id_to_local[pid]])[: int(max_pool_size) - 1]

        all_pool_ids = [hub_id] + members
        used_ids.update(all_pool_ids)

        pool_ts     = {pid: timestamps[id_to_local[pid]] for pid in all_pool_ids}
        has_real_ts = len(set(pool_ts.values())) > 1
        pioneer_id  = min(all_pool_ids, key=lambda p: pool_ts[p]) if has_real_ts else None
        latest_id   = max(all_pool_ids, key=lambda p: pool_ts[p]) if has_real_ts else None

        labels: dict[str, dict] = {}
        for pid in all_pool_ids:
            eng_norm = engagements[id_to_local[pid]] / max_eng
            if pid == hub_id:
                role = "hub"
            elif pioneer_id and pid == pioneer_id:
                role = "pioneer"
            elif latest_id and pid == latest_id:
                role = "latest"
            else:
                role = "member"
            labels[pid] = {
                "adept_role":   role,
                "adept_symbol": node_roles.get(role, {}).get("symbol", " "),
                "eng_norm":     eng_norm,
            }

        pools.append({"hub_id": hub_id, "member_ids": members, "labels": labels})

    return pools, used_ids


def _build_adept_edges(
    pools: list[dict],
    cfg: dict,
) -> pd.DataFrame:
    """Build hub→member (spoke) edges for all pools.

    Args:
        pools: pool list as returned by `_build_pools`.
        cfg: full pipeline config.

    Returns:
        DataFrame with columns [source, target, type, force].
    """
    adept_cfg   = cfg.get("adept", {})
    force_spoke = adept_cfg.get("force_spoke", 0.5)

    if int(adept_cfg.get("apply_forces", 1)) == 0:
        force_spoke = 0.0

    rows = []

    for pool in pools:
        for member_id in pool["member_ids"]:
            rows.append({"source": pool["hub_id"], "target": member_id,
                         "type": "adept_spoke", "force": force_spoke})

    edges_df = (pd.DataFrame(rows) if rows
                else pd.DataFrame(columns=["source", "target", "type", "force"]))

    if len(edges_df) > 0:
        logger.debug("Edges created: %s", edges_df["type"].value_counts().to_dict())

    return edges_df


def build_adept_for_cluster(
    cluster_df: pd.DataFrame,
    nova_edges: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[dict]]:
    """Run ADEPT on a single cluster and return annotated posts + edges.

    nova_edges is unused — kept for API compatibility with the pipeline.

    Returns:
        (cluster_df with adept_role/adept_symbol/eng_norm, adept_edges, metadata dict, pools list).
    """
    cluster_df  = cluster_df.copy()
    min_orphans = int(cfg.get("adept", {}).get("min_orphans_for_event", 2))
    orphans_df  = cluster_df[cluster_df["nova_role"].isna()].copy()

    if len(orphans_df) < min_orphans:
        return cluster_df, pd.DataFrame(columns=["source", "target", "type", "force"]), {
            "n_orphans": len(orphans_df), "n_pools": 0
        }, []

    logger.debug("Cluster: %d posts, %d orphans", len(cluster_df), len(orphans_df))

    pools, used_ids = _build_pools(orphans_df, cluster_df, cfg)
    logger.debug("%d pools formed", len(pools))

    for col in ("adept_role", "adept_symbol", "eng_norm"):
        if col not in cluster_df.columns:
            cluster_df[col] = None

    for pool in pools:
        for pid, info in pool["labels"].items():
            mask = cluster_df["id"] == pid
            cluster_df.loc[mask, "adept_role"]   = info["adept_role"]
            cluster_df.loc[mask, "adept_symbol"]  = info["adept_symbol"]
            cluster_df.loc[mask, "eng_norm"]       = info["eng_norm"]

    adept_edges = _build_adept_edges(pools, cfg)

    return cluster_df, adept_edges, {
        "n_orphans": len(orphans_df),
        "n_pools":   len(pools),
        "n_edges":   len(adept_edges),
    }, pools


def _process_adept_cluster_worker(
    cluster_id,
    group: pd.DataFrame,
    nova_edges: pd.DataFrame,
    cfg: dict,
    skip_llm: bool,
    prompt_params: dict,
) -> dict:
    """Do all the work for one cluster and return a plain result dict.

    Deliberately side-effect-free (no posts_df mutation, no shared list
    append, no input()) so it can be safely run from any thread. The caller
    merges the result under a lock (mirrors the pattern used in
    nova.py). A failed cluster (network or malformed response, after
    max_retries) returns llm_status="failed" with no partial content — the
    caller must not mark it completed. The interactive retry/ignore/quit
    prompt only exists in sequential mode (see build_adept_events), exactly
    like Nova disables its Y/I/Q prompt in parallel mode.
    """
    updated_group, adept_edges, metadata, pools_local = build_adept_for_cluster(
        group, nova_edges, cfg
    )
    metadata["cluster_id"]   = cluster_id
    metadata["cluster_size"] = len(group)

    pool_explanations: list[dict] = []
    exclusion_log: list[dict] = []
    llm_status = "skipped"

    if pools_local and not skip_llm:
        cluster_label = str(cluster_id)
        if "cluster_label" in group.columns and group["cluster_label"].notna().any():
            cluster_label = str(group["cluster_label"].iloc[0])

        ncfg = _adept_cfg(cfg)
        explanations, err_kind = _call_llm_robust_adept(
            cluster_label, prompt_params, pools_local, group, ncfg, cluster_id=str(cluster_id)
        )

        if explanations is None:
            return {"cluster_id": cluster_id, "llm_status": "failed"}

        llm_status = "ok"
        updated_group, adept_edges, exclusion_log = _apply_pool_exclusions(
            pools_local, explanations, updated_group, adept_edges, cluster_id, cluster_label
        )
        _log_pool_summary(cluster_id, cluster_label, pools_local, explanations, exclusion_log)

        matched = _match_explanations_to_pools(pools_local, explanations)
        for i, pool in enumerate(pools_local):
            exp = matched[i]
            pool_explanations.append({
                "hub_id":         pool["hub_id"],
                "cluster_id":     cluster_id,
                "cluster_label":  cluster_label,
                "pool_title":     exp.get("title", "Unknown title"),
                "pool_reasoning": exp.get("reasoning", "No explanation provided"),
                "member_count":   len(pool["member_ids"]),
                "excluded_count": sum(1 for r in exclusion_log if r["hub_id"] == pool["hub_id"]),
            })

    # Collect per-post column updates instead of writing into posts_df directly.
    # Excluded posts were reset to None above, which needs no explicit update
    # here since posts_df already defaults every adept_* column to None.
    col_updates: dict[str, dict] = {}
    for col in ("adept_role", "adept_symbol", "eng_norm"):
        if col in updated_group.columns:
            sub = updated_group[updated_group[col].notna()]
            for _, row in sub.iterrows():
                col_updates.setdefault(row["id"], {})[col] = row[col]

    return {
        "cluster_id":        cluster_id,
        "col_updates":       col_updates,
        "adept_edges":       adept_edges,
        "metadata":          metadata,
        "pool_explanations": pool_explanations,
        "exclusions":        exclusion_log,
        "llm_status":        llm_status,
    }


# Checkpointing mirrors nova.py's design, adapted to ADEPT's data shape
# (col_updates instead of assignments; adept_edges/pool_explanations/
# exclusions lists instead of a single edges list).
def _get_adept_checkpoint_path(cfg: dict) -> Path:
    out_dir = Path(cfg["storage"]["processed_dir"])
    return out_dir / "adept_checkpoint.json"


def _save_adept_checkpoint(
    col_updates: dict[str, dict],
    adept_edges: list[dict],
    metadata: list[dict],
    pool_explanations: list[dict],
    exclusions: list[dict],
    completed_clusters: set[str],
    cfg: dict,
) -> None:
    ncfg = _adept_cfg(cfg)
    if not ncfg.get("checkpoint_enabled", True):
        return

    cp_path = _get_adept_checkpoint_path(cfg)
    cp_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "completed_clusters": list(completed_clusters),
        "col_updates":        col_updates,
        "adept_edges":        adept_edges,
        "metadata":           metadata,
        "pool_explanations":  pool_explanations,
        "exclusions":         exclusions,
        "timestamp":          time.time(),
    }

    with open(cp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2, default=str)

    logger.info("adept checkpoint saved: %d clusters processed", len(completed_clusters))


def _load_adept_checkpoint(
    cfg: dict,
) -> tuple[dict, list, list, list, list, set] | None:
    """Load ADEPT checkpoint if it exists.

    Returns:
        (col_updates, adept_edges, metadata, pool_explanations, exclusions,
         completed_clusters), or None if no checkpoint found.
    """
    cp_path = _get_adept_checkpoint_path(cfg)
    if not cp_path.exists():
        return None

    try:
        with open(cp_path, encoding="utf-8") as f:
            checkpoint = json.load(f)

        col_updates        = checkpoint.get("col_updates", {})
        adept_edges        = checkpoint.get("adept_edges", [])
        metadata           = checkpoint.get("metadata", [])
        pool_explanations  = checkpoint.get("pool_explanations", [])
        exclusions         = checkpoint.get("exclusions", [])
        completed_clusters = set(checkpoint.get("completed_clusters", []))

        logger.info("adept checkpoint loaded: %d clusters already processed", len(completed_clusters))
        return col_updates, adept_edges, metadata, pool_explanations, exclusions, completed_clusters

    except Exception as e:
        logger.warning("failed to load adept checkpoint: %s — ignored", e)
        return None


def _finalize_adept_checkpoint(cfg: dict) -> None:
    """Archive (rename) the finished checkpoint instead of deleting it.

    Mirrors nova.py's _archive_checkpoint exactly: moves
    adept_checkpoint.json into a checkpoints_archive/ subfolder (same
    processed_dir, same subfolder name) with a timestamped filename, so it's
    preserved for later analysis while the active checkpoint path is
    cleared for the next run.
    """
    cp_path = _get_adept_checkpoint_path(cfg)
    if not cp_path.exists():
        return

    archive_dir = cp_path.parent / "checkpoints_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    dest = archive_dir / f"adept_checkpoint_{ts}.json"
    cp_path.rename(dest)
    logger.info("adept checkpoint archived: %s", dest)


def _prompt_adept_failure(cluster_id, err_kind: str) -> str:
    """Blocking prompt shown once a cluster's pool-explanation call has
    exhausted all retries. Sequential mode only — parallel mode auto-skips
    instead (see _process_adept_cluster_worker). Mirrors nova.py's
    Y/I/Q prompt exactly, so a failed pool behaves the same way as a failed
    Nova cluster: no silent skip, only an explicit retry/ignore/quit choice.

    There is no "keep going without the LLM" option: Edges never starts on
    an unvalidated ADEPT pass (mandatory gate, see pipeline.py), so quitting
    is the only way out once retry/ignore are exhausted.

    Returns:
        "retry", "ignore", or "quit".
    """
    while True:
        print(f"\n[ADEPT] cluster={cluster_id} failed after all retries ({err_kind}).")
        print("  Y — retry this cluster (retries again on failure; does not auto-skip)")
        print("  I — ignore and continue (skip this cluster; auto-skip any further failures for the rest of this run)")
        print("  Q — abort (this cluster and any remaining ones stay unvalidated;")
        print("      rerun the pipeline to resume from here)")
        raw = safe_input("Choice [Y/i/q]: ")
        # No terminal / stdin closed / Ctrl-C: fall back to "q", the same
        # safe exit already used when a human quits — never "i" (ignore)
        # or a silent retry.
        choice = raw.strip().lower() if raw is not None else "q"
        if choice == "i":
            return "ignore"
        if choice == "q":
            return "quit"
        if choice in ("", "y"):
            return "retry"
        # Anything else is not a valid choice — reprompt instead of
        # silently treating it as a retry.
        print(f"[ADEPT] '{choice}' is not a valid choice — please enter Y, I, or Q.")


def _log_parallelism(ncfg: dict, n_remaining: int) -> int:
    """Log the parallelism mode and the number of clusters to process.

    Returns the worker count (>= 1).
    """
    workers = max(1, int(ncfg.get("parallel_clusters", 1) or 1))

    if workers > 1:
        url = str(ncfg.get("url", ""))
        if "localhost" in url or "127.0.0.1" in url:
            logger.warning(
                "adept.parallel_clusters=%d but the resolved endpoint (%s) looks local — "
                "a single local model instance usually can't serve concurrent requests any "
                "faster and this may cause contention/OOM. Consider parallel_clusters: 1 "
                "for local models.",
                workers, url,
            )
        logger.info(
            "adept running in PARALLEL mode: up to %d clusters at once — the interactive "
            "retry-or-skip prompt is disabled in this mode; clusters that exhaust all "
            "retries are auto-skipped and reported at the end (rerun the pipeline to retry "
            "them — the checkpoint resumes only on those).",
            workers,
        )
    else:
        logger.notice(
            "Adept: parallel_clusters=1 (sequential) — processing %d "
            "clusters (some may still be skipped for lack of orphans).",
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


def build_adept_events(
    posts_df: pd.DataFrame,
    nova_edges: pd.DataFrame,
    cfg: dict,
    skip_llm: bool = False,
) -> AdeptResult:
    """Run ADEPT across all clusters; skip_llm=True builds pools without LLM explanations.

    Same contract as nova.py's build_nova_trees: a cluster whose LLM
    call fails after all retries is NOT marked completed — it stays
    eligible for retry on the next run instead of being silently accepted
    with a fallback title. Resumes from an adept_checkpoint.json if one
    exists (checkpoint_enabled in config); the checkpoint is only archived
    once the whole pass is clean (see adept_validated below).

    Pool-explanation failures are retried up to adept.max_retries times
    (exponential backoff). Once exhausted:
      - sequential mode pauses and asks retry / ignore-this-cluster /
        quit-now (same Y/I/Q prompt as Nova) — see _prompt_adept_failure.
      - parallel mode auto-skips and reports every failed cluster at the
        end (interactive prompts don't work across threads).

    Returns:
        AdeptResult — see class docstring for field semantics.
    """
    posts_df = posts_df.copy()
    for col in ("adept_role", "adept_symbol", "eng_norm"):
        posts_df[col] = None

    clustered   = posts_df[posts_df["cluster_id"].notna()]
    cluster_ids = clustered["cluster_id"].unique()

    ncfg          = _adept_cfg(cfg)
    prompt_params = _build_prompt_params(cfg)

    checkpoint_data = _load_adept_checkpoint(cfg)
    if checkpoint_data:
        (all_col_updates, all_adept_edges_records, all_metadata,
         all_pool_explanations, all_exclusions, completed_clusters) = checkpoint_data
        logger.info("resuming ADEPT from checkpoint")
    else:
        all_col_updates: dict[str, dict] = {}
        all_adept_edges_records: list[dict] = []
        all_metadata: list[dict] = []
        all_pool_explanations: list[dict] = []
        all_exclusions: list[dict] = []
        completed_clusters: set[str] = set()

    # Re-apply any checkpointed column updates so a resumed run starts from
    # the right posts_df state even before touching the remaining clusters.
    for pid, updates in all_col_updates.items():
        mask = posts_df["id"] == pid
        for col, val in updates.items():
            posts_df.loc[mask, col] = val

    remaining_cluster_ids = [cid for cid in cluster_ids if str(cid) not in completed_clusters]

    logger.info("adept: total clusters: %d | already done: %d | remaining: %d | model: %s | max_retries: %d",
                len(cluster_ids), len(completed_clusters), len(remaining_cluster_ids),
                ncfg.get("model", "?"), ncfg.get("max_retries", 3))

    # Clusters where the LLM genuinely failed (all retries exhausted) and got
    # auto-skipped/ignored rather than successfully explained. A cluster here
    # is NOT in completed_clusters — see the "Not added" comments below —
    # so adept_validated=False and the pipeline gate stops before Edges.
    failed_clusters: set[str] = set()

    parallel_workers = _log_parallelism(ncfg, len(remaining_cluster_ids))

    llm_abandoned_mid_run = False  # set once the user picks "Q" (quit) — sequential mode only
    auto_skip_failures    = False  # set once user picks "ignore" — no more prompts for the rest of this run

    if not remaining_cluster_ids:
        logger.info("adept: all clusters already processed — skipping")

    elif parallel_workers <= 1:
        n_remaining = len(remaining_cluster_ids)
        for cidx, cluster_id in enumerate(remaining_cluster_ids, 1):
            group = clustered[clustered["cluster_id"] == cluster_id].copy()
            if len(group) < 3:
                completed_clusters.add(str(cluster_id))
                continue

            updated_group, adept_edges, metadata, pools_local = build_adept_for_cluster(
                group, nova_edges, cfg
            )
            metadata["cluster_id"]   = cluster_id
            metadata["cluster_size"] = len(group)

            logger.info("[%d/%d] cluster=%s | %d posts | %d orphans | %d pools",
                        cidx, n_remaining, cluster_id, len(group),
                        metadata.get("n_orphans", 0), metadata.get("n_pools", 0))

            effective_skip_llm = skip_llm or llm_abandoned_mid_run
            explanations = None

            if pools_local and not effective_skip_llm:
                cluster_label = str(cluster_id)
                if "cluster_label" in group.columns and group["cluster_label"].notna().any():
                    cluster_label = str(group["cluster_label"].iloc[0])

                explanations, err_kind = _call_llm_robust_adept(
                    cluster_label, prompt_params, pools_local, group, ncfg, cluster_id=str(cluster_id)
                )

                if explanations is None:
                    if auto_skip_failures:
                        logger.warning("cluster=%s failed after %d attempts — auto-skipped (ignore-all active)",
                                        cluster_id, ncfg.get("max_retries", 3))
                        failed_clusters.add(str(cluster_id))
                        _save_adept_checkpoint(
                            all_col_updates, all_adept_edges_records, all_metadata,
                            all_pool_explanations, all_exclusions, completed_clusters, cfg,
                        )
                        continue

                    # All retries exhausted — prompt the user to decide. This loop
                    # only exits via "ignore" (skip this cluster) or "quit" (stop
                    # the run now); "retry" loops back.
                    while True:
                        decision = _prompt_adept_failure(cluster_id, err_kind or "unknown")

                        if decision == "quit":
                            n_left = len(remaining_cluster_ids) - cidx + 1
                            logger.warning("user quit — %d cluster(s) left unprocessed (including "
                                           "this one); rerun the pipeline to resume exactly where "
                                           "this left off",
                                           n_left)
                            for skip_cid in remaining_cluster_ids[cidx - 1:]:
                                failed_clusters.add(str(skip_cid))
                            _save_adept_checkpoint(
                                all_col_updates, all_adept_edges_records, all_metadata,
                                all_pool_explanations, all_exclusions, completed_clusters, cfg,
                            )
                            llm_abandoned_mid_run = True  # → adept_validated=False; pipeline.py's
                                                           # mandatory gate stops before Edges and
                                                           # explains how to resume.
                            break

                        elif decision == "ignore":
                            logger.warning(
                                "user chose ignore — cluster=%s skipped, auto-skip enabled for "
                                "remaining failures this run", cluster_id,
                            )
                            failed_clusters.add(str(cluster_id))
                            _save_adept_checkpoint(
                                all_col_updates, all_adept_edges_records, all_metadata,
                                all_pool_explanations, all_exclusions, completed_clusters, cfg,
                            )
                            auto_skip_failures = True
                            break

                        else:  # retry
                            logger.info("retrying cluster=%s pool explanations...", cluster_id)
                            explanations, err_kind = _call_llm_robust_adept(
                                cluster_label, prompt_params, pools_local, group, ncfg, cluster_id=str(cluster_id)
                            )
                            if explanations is not None:
                                break
                            logger.warning("cluster=%s still failing — asking again (Y/I/Q)", cluster_id)

                    if llm_abandoned_mid_run:
                        break
                    if explanations is None:
                        # Only reachable via the "ignore" branch above.
                        continue

                updated_group, adept_edges, exclusion_log = _apply_pool_exclusions(
                    pools_local, explanations, updated_group, adept_edges, cluster_id, cluster_label
                )
                _log_pool_summary(cluster_id, cluster_label, pools_local, explanations, exclusion_log)

                matched = _match_explanations_to_pools(pools_local, explanations)
                for i, pool in enumerate(pools_local):
                    exp = matched[i]
                    all_pool_explanations.append({
                        "hub_id":         pool["hub_id"],
                        "cluster_id":     cluster_id,
                        "cluster_label":  cluster_label,
                        "pool_title":     exp.get("title", "Unknown title"),
                        "pool_reasoning": exp.get("reasoning", "No explanation provided"),
                        "member_count":   len(pool["member_ids"]),
                        "excluded_count": sum(1 for r in exclusion_log if r["hub_id"] == pool["hub_id"]),
                    })
                all_exclusions.extend(exclusion_log)

            # Propagate adept_ columns back into posts_df + checkpoint state.
            # Excluded posts were reset to None in _apply_pool_exclusions —
            # no explicit update needed for them, posts_df already defaults
            # every adept_* column to None.
            for col in ("adept_role", "adept_symbol", "eng_norm"):
                if col in updated_group.columns:
                    for _, row in updated_group[updated_group[col].notna()].iterrows():
                        posts_df.loc[posts_df["id"] == row["id"], col] = row[col]
                        all_col_updates.setdefault(row["id"], {})[col] = row[col]

            if len(adept_edges) > 0:
                all_adept_edges_records.extend(adept_edges.to_dict("records"))

            all_metadata.append(metadata)
            completed_clusters.add(str(cluster_id))
            _save_adept_checkpoint(
                all_col_updates, all_adept_edges_records, all_metadata,
                all_pool_explanations, all_exclusions, completed_clusters, cfg,
            )

    else:
        # Parallel mode: N clusters in flight via threads. Same design as
        # nova.py — workers are pure (no shared-state writes, no input()),
        # the main thread merges every result under state_lock, so
        # posts_df.loc[...] and the checkpoint file are never touched
        # concurrently from two threads.
        state_lock = threading.Lock()
        eligible = [cid for cid in remaining_cluster_ids
                    if len(clustered[clustered["cluster_id"] == cid]) >= 3]
        for cid in remaining_cluster_ids:
            if cid not in eligible:
                completed_clusters.add(str(cid))

        done_count = 0

        def _handle_adept_result(result: dict):
            nonlocal done_count
            cid_str = str(result["cluster_id"])
            with state_lock:
                if result["llm_status"] == "failed":
                    logger.warning(
                        "cluster=%s failed after %d attempts — auto-skipped "
                        "(parallel mode: no interactive prompt)",
                        cid_str, ncfg.get("max_retries", 3),
                    )
                    # Not added to completed_clusters — see comment in the
                    # sequential branch above: a failed cluster must stay
                    # eligible for retry on the next run.
                    failed_clusters.add(cid_str)
                else:
                    all_col_updates.update(result["col_updates"])
                    for pid, updates in result["col_updates"].items():
                        mask = posts_df["id"] == pid
                        for col, val in updates.items():
                            posts_df.loc[mask, col] = val

                    all_metadata.append(result["metadata"])
                    if len(result["adept_edges"]) > 0:
                        all_adept_edges_records.extend(result["adept_edges"].to_dict("records"))
                    all_pool_explanations.extend(result["pool_explanations"])
                    all_exclusions.extend(result["exclusions"])
                    completed_clusters.add(cid_str)

                _save_adept_checkpoint(
                    all_col_updates, all_adept_edges_records, all_metadata,
                    all_pool_explanations, all_exclusions, completed_clusters, cfg,
                )

                done_count += 1
                logger.info("[parallel %d/%d clusters done] cluster=%s | llm=%s",
                            done_count, len(eligible), cid_str, result["llm_status"])

        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {}
            for cluster_id in eligible:
                group = clustered[clustered["cluster_id"] == cluster_id].copy()
                fut = executor.submit(
                    _process_adept_cluster_worker, cluster_id, group, nova_edges, cfg, skip_llm,
                    prompt_params,
                )
                futures[fut] = cluster_id

            for fut in as_completed(futures):
                cluster_id = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    # Should not normally happen — _call_llm_robust_adept already
                    # swallows its own exceptions — but guard anyway so one
                    # cluster crashing a thread can't take down the whole run.
                    logger.error("cluster=%s raised unexpectedly: %s: %s",
                                 cluster_id, type(e).__name__, e)
                    result = {"cluster_id": cluster_id, "llm_status": "failed"}
                _handle_adept_result(result)

        if failed_clusters:
            logger.warning(
                "adept parallel mode: %d cluster(s) failed and were auto-skipped "
                "— rerun the pipeline to retry them (checkpoint resumes only on those).",
                len(failed_clusters),
            )
            logger.debug("adept parallel mode failed cluster ids: %s", sorted(failed_clusters))

    adept_edges_df = (
        pd.DataFrame(all_adept_edges_records) if all_adept_edges_records
        else pd.DataFrame(columns=["source", "target", "type", "force"])
    )

    logger.info("TOTAL: %d ADEPT edges generated", len(adept_edges_df))
    logger.info("Pool explanations: %d pools explained | %d nodes excluded",
                len(all_pool_explanations), len(all_exclusions))

    build_adept_events._last_pool_explanations = all_pool_explanations
    build_adept_events._last_exclusions        = all_exclusions

    # completed_clusters only ever gains an entry on genuine success (or a
    # legitimate too-small/no-pools skip) — failed/ignored/abandoned clusters
    # are deliberately kept OUT of it so they remain in remaining_cluster_ids
    # and get retried automatically on the next run.
    all_clusters_done = len(completed_clusters) == len(cluster_ids)

    # Kept as an explicit, separate check (rather than relying solely on
    # all_clusters_done) so the intent is self-documenting at call sites and
    # this stays correct even if completed_clusters' bookkeeping changes
    # later. Callers must gate Edges on this, not on all_clusters_done alone.
    adept_validated = all_clusters_done and not failed_clusters and not llm_abandoned_mid_run

    if failed_clusters:
        logger.warning(
            "adept: %d/%d cluster(s) were NOT validated (LLM failed and got "
            "auto-skipped) — adept_validated=False",
            len(failed_clusters), len(cluster_ids),
        )
        logger.debug("adept failed cluster ids: %s", sorted(failed_clusters))

    return AdeptResult(
        posts_df=posts_df,
        adept_edges=adept_edges_df,
        all_metadata=all_metadata,
        llm_abandoned=llm_abandoned_mid_run,
        all_clusters_done=all_clusters_done,
        failed_clusters=sorted(failed_clusters),
        adept_validated=adept_validated,
    )


def save_adept_events(
    posts_df: pd.DataFrame,
    adept_edges: pd.DataFrame,
    cfg: dict,
    all_metadata: list | None = None,
    archive_when_done: bool = False,
) -> None:
    """Write edges, post assignments, cluster metadata, pool explanations,
    and exclusions to parquet.

    If all_metadata is None, it is read from the checkpoint file instead.
    If archive_when_done is True, the checkpoint is archived after writing
    (mirrors nova.py's save_nova_trees — only archive on a clean,
    fully-validated run so a failed run's resume state survives).
    """
    out_dir = Path(cfg["storage"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    adept_edges.to_parquet(out_dir / "adept_edges.parquet", index=False)

    adept_cols = [c for c in ("id", "adept_role", "adept_symbol", "eng_norm") if c in posts_df.columns]
    if adept_cols:
        posts_df[adept_cols].to_parquet(out_dir / "adept_assignments.parquet", index=False)

    saved = ["adept_edges", "adept_assignments"]

    if all_metadata is None:
        checkpoint_data = _load_adept_checkpoint(cfg)
        all_metadata = checkpoint_data[2] if checkpoint_data else []

    if all_metadata:
        pd.DataFrame(all_metadata).to_parquet(out_dir / "adept_metadata.parquet", index=False)
        saved.append("adept_metadata")

    explanations = getattr(build_adept_events, "_last_pool_explanations", [])
    if explanations:
        pd.DataFrame(explanations).to_parquet(out_dir / "pool_explanations.parquet", index=False)
        saved.append("pool_explanations")

    exclusions = getattr(build_adept_events, "_last_exclusions", [])
    if exclusions:
        pd.DataFrame(exclusions).to_parquet(out_dir / "adept_exclusions.parquet", index=False)
        saved.append("adept_exclusions")

    logger.info("Saved: %s", " + ".join(f"{s}.parquet" for s in saved))

    if archive_when_done:
        _finalize_adept_checkpoint(cfg)


def run(
    posts_df: pd.DataFrame,
    nova_edges: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Build and save ADEPT pools for the full corpus."""
    result = build_adept_events(posts_df, nova_edges, cfg)
    # Only archive (and clear) the checkpoint on a clean, fully-validated run —
    # archiving after a failed run would wipe the resume state and force a
    # full reprocess of already-successful clusters on the next attempt.
    save_adept_events(
        result.posts_df, result.adept_edges, cfg,
        all_metadata=result.all_metadata, archive_when_done=result.adept_validated,
    )
    return result.posts_df, result.adept_edges, result.all_metadata