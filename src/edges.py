"""Step 5c — Edge computation engine."""

import os
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from run_store import load_config
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    logger.warning("psutil not installed — chunk size estimated (less precise). pip install psutil")

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    logger.warning("numba not installed — b-matching running in numpy mode (slower). pip install numba")

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    logger.warning("tqdm not installed — progress bars disabled. pip install tqdm")

def _progress(iterable, **kwargs):
    if _HAS_TQDM:
        return _tqdm(iterable, **kwargs)
    total = kwargs.get("total", None)
    desc  = kwargs.get("desc", "")
    if total and desc:
        logger.info("%s (%s items)...", desc, f"{total:,}")
    return iterable


def _log(msg: str) -> None:
    if _HAS_TQDM:
        _tqdm.write(msg)
    else:
        logger.info(msg)


def _load_embeddings_raw(posts_df: pd.DataFrame) -> np.ndarray:
    """Extract `embedding_raw` as a contiguous float32 ndarray. Supports ndarray, pickled bytes, list/tuple storage.

    Raises ValueError on missing/empty column (unlike pipeline.py's `_load_embeddings`).
    """
    if "embedding_raw" not in posts_df.columns or posts_df["embedding_raw"].isna().all():
        raise ValueError("embedding_raw column not found or empty.")
    embs = posts_df["embedding_raw"].values
    if isinstance(embs[0], np.ndarray):
        return np.stack(embs).astype(np.float32)
    if isinstance(embs[0], bytes):
        return np.stack([pickle.loads(e) for e in embs]).astype(np.float32)
    if isinstance(embs[0], (list, tuple)):
        return np.array([np.array(e) for e in embs], dtype=np.float32)
    raise ValueError(f"Unsupported embedding_raw format: {type(embs[0])}")


def compute_cosine_matrix(embeddings: np.ndarray, use_gpu: bool = True) -> np.ndarray:
    """Pairwise dot-product matrix (N×N float32): embeddings @ embeddings.T. CuPy/GPU if available, NumPy fallback.

    Equals cosine similarity only if rows are pre-normalized (not checked here).
    """
    if use_gpu:
        try:
            import cupy as cp
            emb_gpu = cp.array(embeddings, dtype=cp.float32)
            return cp.asnumpy(cp.dot(emb_gpu, emb_gpu.T))
        except ImportError:
            logger.warning('CuPy not found — falling back to CPU. pip install "cupy-cuda13x[ctk]"')
        except Exception:
            logger.warning("GPU error — falling back to CPU.")
    return np.dot(embeddings.astype(np.float32), embeddings.astype(np.float32).T)


# Tier thresholds on a [1, 10] engagement scale: 1 + ratio * 9 for ratios
# 0.236, 0.382, 0.500, 0.618, 0.786
_FIB_THRESHOLDS = (
    1.0 + 0.236 * 9,  # 3.124
    1.0 + 0.382 * 9,  # 4.438
    1.0 + 0.500 * 9,  # 5.500
    1.0 + 0.618 * 9,  # 6.562
    1.0 + 0.786 * 9,  # 8.074
)

# max_out per tier (tiers 1-6)
_TIER_MAX_OUT = (1, 2, 3, 4, 5, 5)

# max_in per tier (tiers 1-6)
# tier 6 has max_in 0 for pass 1 (cannot receive temporal_influence)
_TIER_MAX_IN_PASS1 = (1, 1, 1, 1, 3, 0)

# pass 2 quotas: uniform, configurable via config edges.temporal.max_out / max_in
_PASS2_DEFAULT_MAX_OUT = 1
_PASS2_DEFAULT_MAX_IN  = 1


def _fibonacci_tier(engagement: float) -> int:
    for tier, threshold in enumerate(_FIB_THRESHOLDS, start=1):
        if engagement < threshold:
            return tier
    return 6


def _timestamps_to_hours(posts_df: pd.DataFrame) -> np.ndarray:
    """Convert `timestamp` to hours since epoch (float32); robust to pandas' datetime64 unit (ns / us / ms)."""
    dt = pd.to_datetime(posts_df["timestamp"], errors="coerce")
    return ((dt - pd.Timestamp("1970-01-01")) / pd.Timedelta(hours=1)).values.astype(np.float32)


# Temporal window (tau, tmax): fixed, from config.yaml
# (edges.temporal_influence.* / edges.temporal.*). Force threshold is
# adaptive — see _adaptive_threshold.


def _adaptive_threshold(cosine_matrix: np.ndarray, percentile: float = 25.0, label: str = "") -> float:
    """Force threshold: percentile of strictly-positive cosine values in the upper triangle of `cosine_matrix`, computed in RAM-bounded row chunks.

    Raises:
        ValueError: no strictly positive value found.
    """
    n = cosine_matrix.shape[0]
    chunk = _chunk_size_for_n(n)

    buckets: list[np.ndarray] = []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        cos_chunk = cosine_matrix[start:end]
        for local_i, global_i in enumerate(range(start, end)):
            row = cos_chunk[local_i, global_i + 1:]
            pos = row[row > 0]
            if len(pos):
                buckets.append(pos)

    if not buckets:
        raise ValueError(
            "adaptive threshold: no positive cosine pairs found. "
            "Check embeddings."
        )
    positive = np.concatenate(buckets)
    threshold = float(np.percentile(positive, percentile))
    prefix = f"{label} — " if label else ""
    logger.info("%sadaptive threshold (p%g): %.4f", prefix, percentile, threshold)
    return threshold


def _chunk_size_for_n(
    n: int,
    bytes_per_element: int = 4,
    target_fraction: float = 0.20,
    safety_factor: int = 1,
) -> int:
    """Row-chunk size so a (chunk, n) buffer fits in `target_fraction` of available RAM.

    `safety_factor`: number of (chunk, n) buffers alive at peak (each
    non-inplace numpy op allocates a fresh one). Underestimate → OOM;
    overestimate → smaller chunks, more loop iterations.
    """
    if _HAS_PSUTIL:
        available_bytes = psutil.virtual_memory().available
    else:
        # Fallback: estimate total RAM via os, assume 40 % is available
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else 4 * 1024 ** 3
        available_bytes = int(total * 0.40)

    max_chunk_bytes = (available_bytes * target_fraction) / max(1, safety_factor)
    chunk = int(max_chunk_bytes / (n * bytes_per_element))
    return max(1, min(chunk, n))


def compute_temporal_influence_edges(
    cosine_matrix: np.ndarray,
    posts_df: pd.DataFrame,
    semantic_min: float,
    tau_hours: float,
    tmax_hours: float,
    threshold_percentile: float = 25.0,
) -> pd.DataFrame:
    """Directed temporal_influence candidate edges (pre b-matching).

    Args:
        cosine_matrix: precomputed N×N cosine similarity matrix.
        posts_df: DataFrame with id, timestamp, engagement columns.
        semantic_min: minimum cosine similarity for a candidate edge.
        tau_hours: decay constant, hours (config.yaml).
        tmax_hours: max temporal reach, hours (config.yaml).
        threshold_percentile: percentile of positive cosine values used as force threshold.

    Returns:
        DataFrame [source, target, type, force], type=temporal_influence.

    Raises:
        ValueError: propagated from `_adaptive_threshold`.
    """
    timestamps = _timestamps_to_hours(posts_df)
    engagement = posts_df["engagement"].values.astype(float)
    post_ids   = posts_df["id"].tolist()
    n          = len(post_ids)

    threshold  = _adaptive_threshold(cosine_matrix, percentile=threshold_percentile, label="Pass 1")

    # safety_factor=3: dt_chunk (reused in place) + force_chunk + mask alive at peak.
    chunk    = _chunk_size_for_n(n, safety_factor=3)
    n_chunks = (n + chunk - 1) // chunk
    logger.info("Pass 1 — chunk=%s/%s rows (RAM-adaptive, %s chunks)", f"{chunk:,}", f"{n:,}", n_chunks)

    all_sources: list = []
    all_targets: list = []
    all_forces:  list = []

    chunks_iter = _progress(
        range(0, n, chunk),
        total=n_chunks,
        desc="  Pass 1 — candidate edges",
        unit="chunk",
        leave=False,
    )
    for start in chunks_iter:
        end = min(start + chunk, n)

        # dt > 0: row post newer than column post. source=older/influencer, target=newer/influenced.
        ts_chunk   = timestamps[start:end]
        dt_chunk   = ts_chunk[:, np.newaxis] - timestamps[np.newaxis, :]
        cos_chunk  = cosine_matrix[start:end]
        eng_chunk  = engagement[start:end]

        mask = (
            (dt_chunk > 0)
            & (dt_chunk <= tmax_hours)
            & (cos_chunk > semantic_min)
            & (eng_chunk[:, np.newaxis] > engagement[np.newaxis, :])
        )

        # In-place: reuse dt_chunk's buffer for -dt/tau, clip, exp (avoids 3 extra allocations).
        np.multiply(dt_chunk, -1.0 / tau_hours, out=dt_chunk)
        np.clip(dt_chunk, -88.0, 0.0, out=dt_chunk)
        np.exp(dt_chunk, out=dt_chunk)
        force_chunk = cos_chunk * dt_chunk
        del dt_chunk
        force_chunk[~mask] = 0.0
        force_chunk[force_chunk <= threshold] = 0.0

        local_rows, local_cols = np.nonzero(force_chunk)
        if len(local_rows) == 0:
            continue

        # local_rows are chunk-relative; global index = start + local_rows
        global_rows = local_rows + start
        forces_vals = force_chunk[local_rows, local_cols]

        # row = more recent (influenced) → target ; col = older (influencer) → source
        all_sources.extend(post_ids[j] for j in local_cols)
        all_targets.extend(post_ids[i] for i in global_rows)
        all_forces.extend(float(f) for f in forces_vals)

    edges = pd.DataFrame({
        "source": all_sources,
        "target": all_targets,
        "type":   "temporal_influence",
        "force":  all_forces,
    })
    logger.info(
        "Pass 1 — %s candidate edges  (tau=%.2fh, tmax=%.2fh, threshold=%.4f)",
        f"{len(edges):,}", tau_hours, tmax_hours, threshold,
    )
    return edges


def compute_engagement_only_edges(
    cosine_matrix: np.ndarray,
    posts_df: pd.DataFrame,
    threshold: float,
    semantic_min: float,
) -> pd.DataFrame:
    """temporal_influence candidates when no timestamp is available; direction by engagement only (eng_A > eng_B)."""
    engagement = posts_df["engagement"].values.astype(float)
    post_ids   = posts_df["id"].tolist()
    n          = len(post_ids)

    chunk    = _chunk_size_for_n(n)
    n_chunks = (n + chunk - 1) // chunk
    logger.info("Pass 1 — chunk=%s/%s rows (RAM-adaptive, %s chunks)", f"{chunk:,}", f"{n:,}", n_chunks)

    all_rows: list = []
    all_cols: list = []
    all_forces: list = []

    chunks_iter = _progress(
        range(0, n, chunk),
        total=n_chunks,
        desc="  Pass 1 — candidate edges (engagement-only)",
        unit="chunk",
        leave=False,
    )
    for start in chunks_iter:
        end = min(start + chunk, n)

        cos_chunk = cosine_matrix[start:end]
        eng_chunk = engagement[start:end]

        global_rows = np.arange(start, end)
        mask_upper  = global_rows[:, np.newaxis] < np.arange(n)[np.newaxis, :]
        mask = (
            mask_upper
            & (cos_chunk > semantic_min)
            & (cos_chunk > threshold)
            & (eng_chunk[:, np.newaxis] > engagement[np.newaxis, :])
        )

        local_rows, local_cols = np.nonzero(mask)
        if len(local_rows) == 0:
            continue

        all_rows.append(local_rows + start)
        all_cols.append(local_cols)
        all_forces.append(cos_chunk[local_rows, local_cols])

    if all_rows:
        rows   = np.concatenate(all_rows)
        cols   = np.concatenate(all_cols)
        forces = np.concatenate(all_forces)
    else:
        rows = cols = forces = np.array([], dtype=np.int64)

    edges = pd.DataFrame({
        "source": [post_ids[i] for i in rows],
        "target": [post_ids[j] for j in cols],
        "type":   "temporal_influence",
        "force":  forces.astype(float),
    })
    logger.info(
        "Pass 1 — %s candidate edges (engagement-only)  (cos_min=%s, threshold=%s)",
        f"{len(edges):,}", semantic_min, threshold,
    )
    return edges


def compute_temporal_edges(
    cosine_matrix: np.ndarray,
    posts_df: pd.DataFrame,
    semantic_min: float,
    tau_hours: float,
    tmax_hours: float,
    threshold_percentile: float = 25.0,
) -> pd.DataFrame:
    """Directed temporal candidate edges (pre b-matching), same force as pass 1 but no Rule B.

    Args:
        cosine_matrix: precomputed N×N cosine similarity matrix.
        posts_df: DataFrame with id, timestamp columns.
        semantic_min: minimum cosine similarity for a candidate edge.
        tau_hours: decay constant, hours (config.yaml).
        tmax_hours: max temporal reach, hours (config.yaml).
        threshold_percentile: percentile of positive cosine values used as force threshold.

    Returns:
        DataFrame [source, target, type, force], type=temporal.

    Raises:
        ValueError: propagated from `_adaptive_threshold`.
    """
    timestamps = _timestamps_to_hours(posts_df)
    post_ids   = posts_df["id"].tolist()
    n          = len(post_ids)

    threshold  = _adaptive_threshold(cosine_matrix, percentile=threshold_percentile, label="Pass 2")

    chunk = _chunk_size_for_n(n, safety_factor=3)
    n_chunks = (n + chunk - 1) // chunk
    logger.info("Pass 2 — chunk=%s/%s rows (RAM-adaptive, %s chunks)", f"{chunk:,}", f"{n:,}", n_chunks)

    all_sources: list = []
    all_targets: list = []
    all_forces:  list = []

    chunks_iter = _progress(
        range(0, n, chunk),
        total=n_chunks,
        desc="  Pass 2 — candidate edges",
        unit="chunk",
        leave=False,
    )
    for start in chunks_iter:
        end = min(start + chunk, n)

        # dt > 0: row post newer than column post. Same source/target convention as pass 1.
        ts_chunk  = timestamps[start:end]
        dt_chunk  = ts_chunk[:, np.newaxis] - timestamps[np.newaxis, :]
        cos_chunk = cosine_matrix[start:end]

        mask = (
            (dt_chunk > 0)
            & (dt_chunk <= tmax_hours)
            & (cos_chunk > semantic_min)
        )

        # See compute_temporal_influence_edges for why this is done in place.
        np.multiply(dt_chunk, -1.0 / tau_hours, out=dt_chunk)
        np.clip(dt_chunk, -88.0, 0.0, out=dt_chunk)
        np.exp(dt_chunk, out=dt_chunk)
        force_chunk = cos_chunk * dt_chunk
        del dt_chunk
        force_chunk[~mask] = 0.0
        force_chunk[force_chunk <= threshold] = 0.0

        local_rows, local_cols = np.nonzero(force_chunk)
        if len(local_rows) == 0:
            continue

        global_rows = local_rows + start
        forces_vals = force_chunk[local_rows, local_cols]

        # row = more recent (influenced) → target ; col = older (influencer) → source
        all_sources.extend(post_ids[j] for j in local_cols)
        all_targets.extend(post_ids[i] for i in global_rows)
        all_forces.extend(float(f) for f in forces_vals)

    edges = pd.DataFrame({
        "source": all_sources,
        "target": all_targets,
        "type":   "temporal",
        "force":  all_forces,
    })
    logger.info(
        "Pass 2 — %s candidate edges  (tau=%.2fh, tmax=%.2fh, threshold=%.4f)",
        f"{len(edges):,}", tau_hours, tmax_hours, threshold,
    )
    return edges


def _bmatching_loop_numpy(
    srcs: np.ndarray,
    tgts: np.ndarray,
    cap_out: np.ndarray,
    cap_in: np.ndarray,
) -> np.ndarray:
    accepted = np.zeros(len(srcs), dtype=np.bool_)
    for i in range(len(srcs)):
        s, t = srcs[i], tgts[i]
        if cap_out[s] > 0 and cap_in[t] > 0:
            accepted[i] = True
            cap_out[s] -= 1
            cap_in[t]  -= 1
    return accepted


if _HAS_NUMBA:
    @njit(cache=True)
    def _bmatching_loop_numba(
        srcs: np.ndarray,
        tgts: np.ndarray,
        cap_out: np.ndarray,
        cap_in: np.ndarray,
    ) -> np.ndarray:
        accepted = np.zeros(len(srcs), dtype=np.bool_)
        for i in range(len(srcs)):
            s, t = srcs[i], tgts[i]
            if cap_out[s] > 0 and cap_in[t] > 0:
                accepted[i] = True
                cap_out[s] -= 1
                cap_in[t]  -= 1
        return accepted

    _bmatching_loop = _bmatching_loop_numba
else:
    _bmatching_loop = _bmatching_loop_numpy


def _encode_ids(ids_array: np.ndarray) -> tuple:
    unique_ids, encoded = np.unique(ids_array, return_inverse=True)
    return encoded.astype(np.int32), {uid: i for i, uid in enumerate(unique_ids)}, unique_ids


def _run_bmatching(
    srcs_enc: np.ndarray,
    tgts_enc: np.ndarray,
    cap_out: np.ndarray,
    cap_in: np.ndarray,
    desc: str,
    chunk_size: int = 200_000,
) -> np.ndarray:
    n = len(srcs_enc)
    accepted = np.zeros(n, dtype=np.bool_)
    if n == 0:
        return accepted

    chunk = max(1, min(n, chunk_size))
    n_chunks = (n + chunk - 1) // chunk
    bar = _progress(
        range(0, n, chunk),
        total=n_chunks,
        desc=desc,
        unit="chunk",
        leave=False,
    )
    n_accepted = 0
    for start in bar:
        end = min(start + chunk, n)
        mask = _bmatching_loop(srcs_enc[start:end], tgts_enc[start:end], cap_out, cap_in)
        accepted[start:end] = mask
        n_accepted += int(mask.sum())
        if _HAS_TQDM:
            bar.set_postfix(accepted=f"{n_accepted:,}")
    return accepted


def apply_bmatching_pass1(
    candidates: pd.DataFrame,
    posts_df: pd.DataFrame,
) -> pd.DataFrame:
    """Tiered b-matching on temporal_influence candidates. Tier-6 nodes (max engagement) can only emit."""
    if len(candidates) == 0:
        logger.info("Pass 1 — b-matching: no candidates, skipping.")
        return candidates

    eng_map  = posts_df.set_index("id")["engagement"].to_dict()
    tier_map = {pid: _fibonacci_tier(float(eng)) for pid, eng in eng_map.items()}

    sorted_edges = candidates.sort_values("force", ascending=False)

    # Encode string IDs → int32
    _log(f"[Edges] Pass 1 — b-matching: encoding {len(sorted_edges):,} candidate edges + {len(posts_df):,} nodes...")
    all_ids = np.concatenate([
        sorted_edges["source"].to_numpy(),
        sorted_edges["target"].to_numpy(),
        posts_df["id"].to_numpy(),
    ])
    _, id_to_idx, unique_ids = _encode_ids(all_ids)
    n_nodes = len(unique_ids)

    srcs_enc = sorted_edges["source"].map(id_to_idx).to_numpy(dtype=np.int32)
    tgts_enc = sorted_edges["target"].map(id_to_idx).to_numpy(dtype=np.int32)

    # Per-node initial capacities from tier lookup
    cap_out = np.zeros(n_nodes, dtype=np.int32)
    cap_in  = np.zeros(n_nodes, dtype=np.int32)
    for pid, tier in tier_map.items():
        if pid in id_to_idx:
            idx = id_to_idx[pid]
            cap_out[idx] = _TIER_MAX_OUT[tier - 1]
            cap_in[idx]  = _TIER_MAX_IN_PASS1[tier - 1]

    accepted_mask = _run_bmatching(
        srcs_enc, tgts_enc, cap_out, cap_in, desc="  Pass 1 — b-matching"
    )

    accepted = sorted_edges[accepted_mask].copy()
    backend  = "numba" if _HAS_NUMBA else "numpy"
    logger.info(
        "Pass 1 — b-matching: %s → %s edges  (tier quotas, backend=%s)",
        f"{len(candidates):,}", f"{len(accepted):,}", backend,
    )
    return accepted


def apply_bmatching_pass2(
    candidates: pd.DataFrame,
    posts_df: pd.DataFrame,
    pass1_edges: pd.DataFrame,
    max_out: int,
    max_in: int,
) -> pd.DataFrame:
    """Uniform b-matching on temporal candidates. Pass-1 nodes excluded; tier-6 pass-1 sources become target-only."""
    if len(candidates) == 0:
        logger.info("Pass 2 — b-matching: no candidates, skipping.")
        return candidates

    eng_map  = posts_df.set_index("id")["engagement"].to_dict()
    tier_map = {pid: _fibonacci_tier(float(eng)) for pid, eng in eng_map.items()}

    # Nodes touched in pass 1
    pass1_sources = set(pass1_edges["source"].tolist()) if len(pass1_edges) > 0 else set()
    pass1_targets = set(pass1_edges["target"].tolist()) if len(pass1_edges) > 0 else set()
    pass1_touched = pass1_sources | pass1_targets

    # Tier-6 pass-1 sources → eligible in pass 2 as targets only
    tier6_target_only = {
        pid for pid in pass1_touched
        if tier_map.get(pid, 1) == 6
    }

    # All non-tier-6 touched nodes → fully excluded
    fully_excluded = pass1_touched - tier6_target_only

    # Drop fully-excluded nodes; drop tier-6 target-only nodes as source
    mask_keep = (
        ~candidates["source"].isin(fully_excluded)
        & ~candidates["target"].isin(fully_excluded)
        & ~candidates["source"].isin(tier6_target_only)
    )
    filtered = candidates[mask_keep].copy()

    _log(
        f"[Edges] Pass 2 — b-matching: {len(candidates):,} candidates → {len(filtered):,} "
        f"after exclusions, sorting by force..."
    )
    sorted_edges = filtered.sort_values("force", ascending=False)

    if len(sorted_edges) == 0:
        logger.info(
            "Pass 2 — b-matching: %s candidates → %s after exclusions → 0 edges  (max_out=%s, max_in=%s)",
            f"{len(candidates):,}", f"{len(filtered):,}", max_out, max_in,
        )
        return sorted_edges

    # Encode string IDs → int32
    _log(f"[Edges] Pass 2 — b-matching: encoding {len(sorted_edges):,} candidate edges + {len(posts_df):,} nodes...")
    all_ids = np.concatenate([
        sorted_edges["source"].to_numpy(),
        sorted_edges["target"].to_numpy(),
        posts_df["id"].to_numpy(),
    ])
    _, id_to_idx, unique_ids = _encode_ids(all_ids)
    n_nodes = len(unique_ids)

    srcs_enc = sorted_edges["source"].map(id_to_idx).to_numpy(dtype=np.int32)
    tgts_enc = sorted_edges["target"].map(id_to_idx).to_numpy(dtype=np.int32)

    # Per-node initial capacities: uniform max_out/max_in; tier-6 pass-1 sources: cap_out=0
    cap_out = np.full(n_nodes, max_out, dtype=np.int32)
    cap_in  = np.full(n_nodes, max_in,  dtype=np.int32)
    for pid in tier6_target_only:
        if pid in id_to_idx:
            cap_out[id_to_idx[pid]] = 0

    accepted_mask = _run_bmatching(
        srcs_enc, tgts_enc, cap_out, cap_in, desc="  Pass 2 — b-matching"
    )

    accepted = sorted_edges[accepted_mask].copy()
    backend  = "numba" if _HAS_NUMBA else "numpy"
    logger.info(
        "Pass 2 — b-matching: %s candidates → %s after exclusions → %s edges  (max_out=%s, max_in=%s, backend=%s)",
        f"{len(candidates):,}", f"{len(filtered):,}", f"{len(accepted):,}", max_out, max_in, backend,
    )
    return accepted


def _greedy_one_to_one_match(
    rows: np.ndarray,
    cols: np.ndarray,
    forces: np.ndarray,
    post_ids: list,
) -> pd.DataFrame:
    order    = np.argsort(-forces)
    rows_s   = rows[order]
    cols_s   = cols[order]
    forces_s = forces[order]

    used: set = set()
    sources, targets, forces_out = [], [], []

    iterator = _progress(
        zip(rows_s, cols_s, forces_s, strict=False),
        total=len(rows_s),
        desc="  semantic_inter — greedy matching",
        unit="edge",
        leave=False,
    )
    for i, j, f in iterator:
        pid_i, pid_j = post_ids[i], post_ids[j]
        if pid_i not in used and pid_j not in used:
            sources.append(pid_i)
            targets.append(pid_j)
            forces_out.append(float(f))
            used.add(pid_i)
            used.add(pid_j)

    return pd.DataFrame({"source": sources, "target": targets, "force": forces_out})


def compute_semantic_inter_edges(
    cosine_matrix: np.ndarray,
    posts_df: pd.DataFrame,
    threshold: float = 0.65,
) -> pd.DataFrame:
    """Undirected inter-cluster semantic bridges (cosine > threshold); greedy 1-to-1, exempt from b-matching."""
    post_ids = posts_df["id"].tolist()
    n        = len(post_ids)

    # NaN cluster_id -> -1 (pd.factorize default); excluded by `!= -1` checks below.
    cluster_codes = pd.factorize(posts_df["cluster_id"])[0]

    chunk    = _chunk_size_for_n(n)
    n_chunks = (n + chunk - 1) // chunk
    logger.info("Semantic inter-cluster — chunk=%s/%s rows (RAM-adaptive, %s chunks)", f"{chunk:,}", f"{n:,}", n_chunks)

    all_rows: list = []
    all_cols: list = []
    all_forces: list = []

    chunks_iter = _progress(
        range(0, n, chunk),
        total=n_chunks,
        desc="  semantic_inter — candidate edges",
        unit="chunk",
        leave=False,
    )
    for start in chunks_iter:
        end = min(start + chunk, n)

        cos_chunk   = cosine_matrix[start:end]
        codes_chunk = cluster_codes[start:end]

        global_rows = np.arange(start, end)
        mask_upper  = global_rows[:, np.newaxis] < np.arange(n)[np.newaxis, :]
        mask_inter  = (
            (codes_chunk[:, np.newaxis] != cluster_codes[np.newaxis, :])
            & (codes_chunk[:, np.newaxis] != -1)
            & (cluster_codes[np.newaxis, :] != -1)
        )
        mask = mask_upper & mask_inter & (cos_chunk > threshold)

        local_rows, local_cols = np.nonzero(mask)
        if len(local_rows) == 0:
            continue

        all_rows.append(local_rows + start)
        all_cols.append(local_cols)
        all_forces.append(cos_chunk[local_rows, local_cols])

    if all_rows:
        rows   = np.concatenate(all_rows)
        cols   = np.concatenate(all_cols)
        forces = np.concatenate(all_forces)
    else:
        rows = cols = forces = np.array([], dtype=np.int64)

    logger.info("Semantic inter-cluster — %s candidate edges  (cos_min=%s)", f"{len(rows):,}", threshold)

    edges = _greedy_one_to_one_match(rows, cols, forces, post_ids)
    edges["type"] = "semantic_inter"

    logger.info("Semantic inter-cluster — %s edges accepted  (cos_min=%s)", f"{len(edges):,}", threshold)
    return edges[["source", "target", "type", "force"]]


def compute_adept_graft_edges(
    cosine_matrix: np.ndarray,
    posts_df: pd.DataFrame,
    adept_edges: pd.DataFrame,
    min_cosine: float = 0.30,
    force: float = 0.3,
) -> pd.DataFrame:
    """Directed hub → nearest same-cluster NOVA node edge, at most one per ADEPT hub
    (skipped if no same-cluster NOVA node clears min_cosine).

    Hub ids recovered from adept_spoke "source". Nearest NOVA node is
    searched within the hub's own cluster_id only, kept if cosine >= min_cosine.

    Args:
        cosine_matrix: precomputed N×N cosine similarity matrix, aligned
            with posts_df row order (position i == posts_df.iloc[i]).
        posts_df: full post DataFrame; requires "id", "cluster_id", and
            "nova_role" columns (non-null "nova_role" marks a NOVA node).
        adept_edges: ADEPT's edge output for this run; adept_spoke rows are
            used to recover the hub ids.
        min_cosine: minimum cosine similarity for a graft to be kept.
        force: force value assigned to accepted graft edges.

    Returns:
        DataFrame with columns [source, target, type, force].
    """
    empty = pd.DataFrame(columns=["source", "target", "type", "force"])

    if adept_edges is None or len(adept_edges) == 0:
        return empty

    hub_ids = adept_edges.loc[adept_edges["type"] == "adept_spoke", "source"].unique().tolist()
    if not hub_ids:
        return empty

    if "nova_role" not in posts_df.columns:
        return empty
    nova_mask = posts_df["nova_role"].notna().to_numpy()
    if not nova_mask.any():
        return empty

    if "cluster_id" not in posts_df.columns:
        logger.warning("ADEPT graft — no cluster_id column, cannot restrict grafts to the hub's own cluster; skipping")
        return empty

    post_ids      = posts_df["id"].tolist()
    cluster_arr   = posts_df["cluster_id"].to_numpy()
    id_to_pos     = {pid: i for i, pid in enumerate(post_ids)}
    hub_positions = [id_to_pos[h] for h in hub_ids if h in id_to_pos]
    if not hub_positions:
        return empty

    nova_positions = np.nonzero(nova_mask)[0]

    rows = []
    n_no_candidate = 0
    for hub_pos in hub_positions:
        same_cluster_nova = nova_positions[cluster_arr[nova_positions] == cluster_arr[hub_pos]]
        if len(same_cluster_nova) == 0:
            n_no_candidate += 1
            continue

        cos_row  = cosine_matrix[hub_pos, same_cluster_nova]
        best_j   = int(np.argmax(cos_row))
        best_cos = float(cos_row[best_j])
        if best_cos >= min_cosine:
            rows.append({
                "source": post_ids[hub_pos],
                "target": post_ids[int(same_cluster_nova[best_j])],
                "type":   "adept_graft",
                "force":  force,
            })

    edges = pd.DataFrame(rows) if rows else empty
    logger.info(
        "ADEPT graft — %s/%s hubs grafted to a same-cluster NOVA node  "
        "(cos_min=%s, %s hubs had no NOVA node in their own cluster)",
        f"{len(edges):,}", f"{len(hub_positions):,}", min_cosine, f"{n_no_candidate:,}",
    )
    return edges


# Component-based bridge pruning (post-hoc dedup of secondary edges).
#
# Skeleton = nova + adept_spoke + adept_graft, never filtered, strictly
# intra-cluster. Its connected components define the base groups.
#
# temporal_influence / temporal / semantic_inter are not cluster-constrained
# and can duplicate skeleton connectivity or bridge two components. Steps:
#   1. Drop candidates whose endpoints already share a skeleton component.
#   2. Kruskal over skeleton components: walk remaining candidates
#      priority/force-first, keep only if endpoints are still in different
#      components. Rejects both direct and transitive duplicates. See
#      prune_cross_component_bridges.
#   3. Optional: reject a bridge between two isolated singletons
#      (edges.pruning.isolation_guard).
#
# No size limit is enforced here or anywhere during edge creation — see
# _split_oversized_components below for the final, separate step that
# splits an overgrown final component after everything is built.


class _UnionFind:
    """Union-find: path compression, union by rank, per-root size tracking (used by the isolation guard, and by _split_oversized_components)."""

    __slots__ = ("parent", "rank", "size")

    def __init__(self, n: int, sizes: list = None):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = list(sizes) if sizes is not None else [1] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _compute_components(
    posts_df: pd.DataFrame,
    *skeleton_edge_frames: pd.DataFrame,
) -> dict:
    """Connected components over posts_df ids, linked only via `skeleton_edge_frames` (nova, adept_spoke, adept_graft).

    Orphans (no skeleton edge) end up as singleton components.

    Args:
        posts_df: full post DataFrame — source of the node set.
        *skeleton_edge_frames: [source, target, ...] frames, treated as
            intra-cluster. "type" column ignored — pre-filter frames to
            the desired skeleton subset before calling.

    Returns:
        dict: post id -> component id.
    """
    post_ids  = posts_df["id"].tolist()
    id_to_idx = {pid: i for i, pid in enumerate(post_ids)}
    uf        = _UnionFind(len(post_ids))

    for frame in skeleton_edge_frames:
        if frame is None or len(frame) == 0:
            continue
        for s, t in zip(frame["source"].to_numpy(), frame["target"].to_numpy(), strict=False):
            si, ti = id_to_idx.get(s), id_to_idx.get(t)
            if si is not None and ti is not None:
                uf.union(si, ti)

    return {pid: uf.find(idx) for pid, idx in id_to_idx.items()}


# Bridge type priority: lower number wins a tie.
_BRIDGE_TYPE_PRIORITY = {
    "temporal_influence": 0,
    "temporal":            1,
    "semantic_inter":      2,
}

_SECONDARY_EDGE_COLUMNS = ["source", "target", "type", "force"]


def prune_cross_component_bridges(
    pass1_edges: pd.DataFrame,
    pass2_edges: pd.DataFrame,
    sem_edges: pd.DataFrame,
    component_map: dict,
    isolation_guard: dict = None,
) -> tuple:
    """Dedup temporal_influence / temporal / semantic_inter edges against skeleton components (see module comment above).

    Kruskal over skeleton components: candidates walked
    priority/force-first, kept only if endpoints are still in different
    components. At most one bridge per component pair (priority
    temporal_influence > temporal > semantic_inter, force breaks ties).

    Args:
        pass1_edges: accepted temporal_influence edges (post b-matching).
        pass2_edges: accepted temporal edges (post b-matching).
        sem_edges: accepted semantic_inter edges (post greedy 1-to-1).
        component_map: post id -> component id, from _compute_components.
        isolation_guard: optional dict (edges.pruning.isolation_guard).
            enabled (bool, default False): reject a bridge between two
                isolated singleton components.
            dynamic (bool, default True): isolated = live union-find state
                if True, static component_map sizes if False.

    Returns:
        (pruned_pass1, pruned_pass2, pruned_sem, stats). stats: dict of
        4 sub-dicts per edge type — intra_component_removed,
        bridge_collapsed, isolation_rejected, kept.

    Raises:
        ValueError: edge references an id absent from component_map.
    """
    isolation_guard = isolation_guard or {}
    guard_enabled   = bool(isolation_guard.get("enabled", False))
    guard_dynamic   = bool(isolation_guard.get("dynamic", True))

    frames = {
        "temporal_influence": pass1_edges,
        "temporal":            pass2_edges,
        "semantic_inter":      sem_edges,
    }
    stats = {
        "intra_component_removed": {k: 0 for k in frames},
        "bridge_collapsed":        {k: 0 for k in frames},
        "isolation_rejected":      {k: 0 for k in frames},
        "kept":                    {k: 0 for k in frames},
    }

    parts = []
    for etype, frame in frames.items():
        if frame is not None and len(frame) > 0:
            tagged = frame.copy()
            tagged["_etype"] = etype
            parts.append(tagged)

    if not parts:
        return pass1_edges, pass2_edges, sem_edges, stats

    combined = pd.concat(parts, ignore_index=True)

    comp_source = combined["source"].map(component_map)
    comp_target = combined["target"].map(component_map)

    missing = comp_source.isna() | comp_target.isna()
    if missing.any():
        bad_ids = pd.unique(pd.concat([
            combined.loc[missing, "source"], combined.loc[missing, "target"],
        ]))
        raise ValueError(
            f"Bridge pruning: {int(missing.sum())} secondary edge(s) reference id(s) "
            f"absent from component_map (e.g. {list(bad_ids[:5])}). component_map must "
            "be built (via _compute_components) from the same posts_df as these edges."
        )

    intra_mask = comp_source == comp_target
    for etype in frames:
        type_mask = combined["_etype"] == etype
        stats["intra_component_removed"][etype] = int((type_mask & intra_mask).sum())

    inter = combined[~intra_mask].copy()
    inter["_comp_a"]  = comp_source[~intra_mask].to_numpy()
    inter["_comp_b"]  = comp_target[~intra_mask].to_numpy()
    inter["_priority"] = inter["_etype"].map(_BRIDGE_TYPE_PRIORITY)
    inter_sorted = inter.sort_values(["_priority", "force"], ascending=[True, False])

    # Compact component-id space + skeleton sizes for the union-find below.
    sizes_by_comp = Counter(component_map.values())
    comp_ids      = sorted(sizes_by_comp)
    comp_to_idx   = {c: i for i, c in enumerate(comp_ids)}
    base_sizes    = [sizes_by_comp[c] for c in comp_ids]

    comp_a_idx = inter_sorted["_comp_a"].map(comp_to_idx).to_numpy()
    comp_b_idx = inter_sorted["_comp_b"].map(comp_to_idx).to_numpy()
    etype_arr  = inter_sorted["_etype"].to_numpy()
    n          = len(inter_sorted)
    keep_mask  = np.zeros(n, dtype=bool)

    # Single union-find shared across all three types.
    uf = _UnionFind(len(comp_ids), sizes=base_sizes)

    for i in range(n):
        etype  = etype_arr[i]
        a, b   = comp_a_idx[i], comp_b_idx[i]
        ra, rb = uf.find(a), uf.find(b)

        if ra == rb:
            stats["bridge_collapsed"][etype] += 1
            continue

        if guard_enabled:
            isolated_a = (uf.size[ra] == 1) if guard_dynamic else (base_sizes[a] == 1)
            isolated_b = (uf.size[rb] == 1) if guard_dynamic else (base_sizes[b] == 1)
            if isolated_a and isolated_b:
                stats["isolation_rejected"][etype] += 1
                continue

        uf.union(a, b)
        keep_mask[i] = True
        stats["kept"][etype] += 1

    kept = inter_sorted[keep_mask]
    pruned = {}
    for etype in frames:
        sub = kept[kept["_etype"] == etype]
        pruned[etype] = sub[_SECONDARY_EDGE_COLUMNS].reset_index(drop=True)

    return pruned["temporal_influence"], pruned["temporal"], pruned["semantic_inter"], stats


def _bfs_pieces(nodes: set, adj: dict, skip_edge_pos) -> list:
    """Connected pieces of `nodes`/`adj`, as if the edge at `skip_edge_pos` were removed."""
    seen   = set()
    pieces = []
    for start in nodes:
        if start in seen:
            continue
        piece = {start}
        seen.add(start)
        stack = [start]
        while stack:
            cur = stack.pop()
            for nxt, pos in adj.get(cur, ()):
                if pos == skip_edge_pos or nxt in seen:
                    continue
                seen.add(nxt)
                piece.add(nxt)
                stack.append(nxt)
        pieces.append(piece)
    return pieces


# Cut preference when splitting an oversized component: lower number is cut
# first (most disposable). nova is cut only if nothing else can split it.
_CUT_PRIORITY = {
    "semantic_inter":      0,
    "temporal":            1,
    "temporal_influence":  2,
    "adept_graft":         3,
    "adept_spoke":         4,
    "nova":                5,
}


def _split_oversized_components(all_edges: pd.DataFrame, posts_df: pd.DataFrame, max_component_size: int) -> pd.DataFrame:
    """Split any final component above max_component_size by cutting one edge at a time.

    Runs once, after every edge type has been created and merged with zero
    restriction — the only place component size is ever enforced. A
    component here is exactly what a treeview BFS would recover from any of
    its nodes: every edge type, no exceptions.

    For each oversized component, candidate edges to cut are tried in
    _CUT_PRIORITY order (most disposable first), weakest force first within
    a type. Only an edge whose removal actually disconnects its two
    endpoints (a graph cut-edge) is a valid cut. The first cut that brings
    both resulting pieces to ≤ max_component_size is taken; otherwise the
    most balanced cut found is taken and the still-oversized side is
    recursed on. A component with no cut-edge at all (2-edge-connected) is
    left untouched and logged.

    Args:
        all_edges: the fully merged edge DataFrame (all 6 types).
        posts_df: full post DataFrame — source of the node set.
        max_component_size: components at/under this are left untouched.

    Returns:
        all_edges with the chosen cut edges removed.
    """
    post_ids  = posts_df["id"].tolist()
    id_to_idx = {pid: i for i, pid in enumerate(post_ids)}
    uf        = _UnionFind(len(post_ids))

    src = all_edges["source"].to_numpy()
    tgt = all_edges["target"].to_numpy()
    typ = all_edges["type"].to_numpy()
    frc = all_edges["force"].to_numpy()

    for i in range(len(all_edges)):
        si, ti = id_to_idx.get(src[i]), id_to_idx.get(tgt[i])
        if si is not None and ti is not None:
            uf.union(si, ti)

    comp_members = {}
    for i, pid in enumerate(post_ids):
        comp_members.setdefault(uf.find(i), []).append(pid)

    oversized = {c: m for c, m in comp_members.items() if len(m) > max_component_size}
    if not oversized:
        return all_edges

    edge_rows = list(zip(range(len(all_edges)), src, tgt, typ, frc, strict=False))
    drop_positions = set()

    for members in oversized.values():
        queue = [(set(members), [
            (pos, u, v, t, f) for pos, u, v, t, f in edge_rows
            if u in members and v in members
        ])]

        while queue:
            nodes, edges = queue.pop()
            if len(nodes) <= max_component_size:
                continue

            adj = {}
            for pos, u, v, _t, _f in edges:
                adj.setdefault(u, []).append((v, pos))
                adj.setdefault(v, []).append((u, pos))

            candidates = sorted(edges, key=lambda e: (_CUT_PRIORITY.get(e[3], 99), e[4]))

            best_cut, best_pieces, best_worst = None, None, None
            for pos, _u, _v, _t, _f in candidates:
                pieces = _bfs_pieces(nodes, adj, skip_edge_pos=pos)
                if len(pieces) < 2:
                    continue
                worst = max(len(p) for p in pieces)
                if worst <= max_component_size:
                    best_cut, best_pieces = pos, pieces
                    break
                if best_worst is None or worst < best_worst:
                    best_cut, best_pieces, best_worst = pos, pieces, worst

            if best_cut is None:
                logger.warning(
                    "component of %d posts could not be split (no cut-edge found) — "
                    "left over max_component_size=%d",
                    len(nodes), max_component_size,
                )
                continue

            drop_positions.add(best_cut)
            remaining = [(pos, u, v, t, f) for pos, u, v, t, f in edges if pos != best_cut]
            for piece in best_pieces:
                queue.append((piece, [e for e in remaining if e[1] in piece and e[2] in piece]))

    if not drop_positions:
        return all_edges

    kept_mask = np.ones(len(all_edges), dtype=bool)
    kept_mask[list(drop_positions)] = False
    logger.info(
        "component size cap (max_component_size=%d): %d oversized component(s), %d edge(s) cut to split them",
        max_component_size, len(oversized), len(drop_positions),
    )
    return all_edges[kept_mask].reset_index(drop=True)


def _has_timestamp(posts_df: pd.DataFrame) -> bool:
    return (
        "timestamp" in posts_df.columns
        and posts_df["timestamp"].notna().any()
    )


def _has_engagement(posts_df: pd.DataFrame) -> bool:
    return (
        "engagement" in posts_df.columns
        and posts_df["engagement"].notna().any()
    )


def run(
    posts_df: pd.DataFrame,
    cosine_matrix: np.ndarray,
    cfg: dict,
    nova_edges: pd.DataFrame = None,
    adept_edges: pd.DataFrame = None,
) -> pd.DataFrame:
    """Build all edge types, apply sequential b-matching, return merged table."""
    ecfg      = cfg["edges"]
    inf_cfg   = ecfg.get("temporal_influence", {})
    temp_cfg  = ecfg.get("temporal", {})
    inter_cfg = ecfg.get("semantic_inter", {})

    sem_min_inf   = inf_cfg.get("semantic_min",  0.80)
    sem_min_temp  = temp_cfg.get("semantic_min", 0.80)
    sem_min_inter = inter_cfg.get("threshold",   0.80)

    # Temporal window (hours): from config.yaml
    tau_hours_inf  = inf_cfg.get("tau_hours",  6.0)
    tmax_hours_inf = inf_cfg.get("tmax_hours", 48.0)
    thresh_pct_inf = inf_cfg.get("threshold_percentile", 25)

    tau_hours_temp  = temp_cfg.get("tau_hours",  6.0)
    tmax_hours_temp = temp_cfg.get("tmax_hours", 48.0)
    thresh_pct_temp = temp_cfg.get("threshold_percentile", 25)

    # Pass-2 quotas — configurable; fall back to module defaults
    pass2_max_out = temp_cfg.get("pass2_max_out", _PASS2_DEFAULT_MAX_OUT)
    pass2_max_in  = temp_cfg.get("pass2_max_in",  _PASS2_DEFAULT_MAX_IN)

    has_ts  = _has_timestamp(posts_df)
    has_eng = _has_engagement(posts_df)

    logger.info("available fields: timestamp=%s  engagement=%s",
                "yes" if has_ts else "no", "yes" if has_eng else "no")

    # Nova edges (exempt)
    if nova_edges is not None and len(nova_edges) > 0:
        std_cols = ["source", "target", "type", "force"]
        if "nova_role" in nova_edges.columns:
            std_cols = std_cols + ["nova_role"]
            logger.info("NOVA subtypes: %s", nova_edges["nova_role"].value_counts().to_dict())
        nova_std = nova_edges[std_cols].copy()
    else:
        nova_std = pd.DataFrame(columns=["source", "target", "type", "force"])

    # Adept edges (exempt)
    if adept_edges is not None and len(adept_edges) > 0:
        adept_std = adept_edges[["source", "target", "type", "force"]].copy()
    else:
        adept_std = pd.DataFrame(columns=["source", "target", "type", "force"])

    # Adept graft (exempt) — hub → nearest NOVA node. Reuses cosine_matrix computed above.
    graft_cfg        = ecfg.get("adept_graft", {})
    graft_min_cosine = graft_cfg.get("graft_min_cosine", 0.30)
    force_graft      = graft_cfg.get("force_graft", 0.3)
    if int(cfg.get("adept", {}).get("apply_forces", 1)) == 0:
        force_graft = 0.0

    graft_std = compute_adept_graft_edges(
        cosine_matrix, posts_df, adept_edges,
        min_cosine=graft_min_cosine, force=force_graft,
    )

    # Pass 1 — temporal_influence
    pass1_edges = pd.DataFrame(columns=["source", "target", "type", "force"])

    if has_ts and has_eng:
        logger.info(
            "Pass 1 — temporal_influence  (tau=%sh, tmax=%sh, thresh_pct=%s, cos_min=%s)",
            tau_hours_inf, tmax_hours_inf, thresh_pct_inf, sem_min_inf,
        )
        raw_pass1   = compute_temporal_influence_edges(
            cosine_matrix, posts_df,
            semantic_min=sem_min_inf,
            tau_hours=tau_hours_inf,
            tmax_hours=tmax_hours_inf,
            threshold_percentile=thresh_pct_inf,
        )
        pass1_edges = apply_bmatching_pass1(raw_pass1, posts_df)

    elif has_eng and not has_ts:
        logger.info(
            "Pass 1 — temporal_influence (engagement only, no timestamp)  (thresh_pct=%s, cos_min=%s)",
            thresh_pct_inf, sem_min_inf,
        )
        raw_pass1   = compute_engagement_only_edges(
            cosine_matrix, posts_df,
            threshold=_adaptive_threshold(cosine_matrix, percentile=thresh_pct_inf, label="Pass 1"),
            semantic_min=sem_min_inf,
        )
        pass1_edges = apply_bmatching_pass1(raw_pass1, posts_df)

    else:
        logger.info("Pass 1 — temporal_influence: skipped (no engagement data)")

    # Pass 2 — temporal
    pass2_edges = pd.DataFrame(columns=["source", "target", "type", "force"])

    if has_ts:
        logger.info(
            "Pass 2 — temporal  (tau=%sh, tmax=%sh, thresh_pct=%s, cos_min=%s, b-matching=max_out=%s/max_in=%s)",
            tau_hours_temp, tmax_hours_temp, thresh_pct_temp, sem_min_temp, pass2_max_out, pass2_max_in,
        )
        raw_pass2   = compute_temporal_edges(
            cosine_matrix, posts_df,
            semantic_min=sem_min_temp,
            tau_hours=tau_hours_temp,
            tmax_hours=tmax_hours_temp,
            threshold_percentile=thresh_pct_temp,
        )
        pass2_edges = apply_bmatching_pass2(
            raw_pass2, posts_df, pass1_edges,
            max_out=pass2_max_out,
            max_in=pass2_max_in,
        )
    else:
        logger.info("Pass 2 — temporal: skipped (no timestamp data)")

    # Semantic inter-cluster edges (exempt)
    logger.info("semantic inter-cluster  (cos_min=%s)", sem_min_inter)
    sem_edges = compute_semantic_inter_edges(
        cosine_matrix, posts_df, threshold=sem_min_inter,
    )

    # Component-based bridge pruning — see prune_cross_component_bridges.
    n_p1_raw  = len(pass1_edges)
    n_p2_raw  = len(pass2_edges)
    n_sem_raw = len(sem_edges)

    pruning_cfg     = ecfg.get("pruning", {})
    isolation_guard = pruning_cfg.get("isolation_guard", {})

    component_map = _compute_components(posts_df, nova_std, adept_std, graft_std)
    pass1_edges, pass2_edges, sem_edges, prune_stats = prune_cross_component_bridges(
        pass1_edges, pass2_edges, sem_edges, component_map,
        isolation_guard=isolation_guard,
    )

    n_components = len(set(component_map.values()))
    logger.info(
        "bridge pruning (isolation_guard=%s): %d skeleton components  |  "
        "temporal_influence %d→%d (dropped %d intra-comp, collapsed %d, isolated %d)  |  "
        "temporal %d→%d (dropped %d intra-comp, collapsed %d, isolated %d)  |  "
        "semantic_inter %d→%d (dropped %d intra-comp, collapsed %d, isolated %d)",
        "on" if isolation_guard.get("enabled") else "off", n_components,
        n_p1_raw, len(pass1_edges),
        prune_stats["intra_component_removed"]["temporal_influence"],
        prune_stats["bridge_collapsed"]["temporal_influence"],
        prune_stats["isolation_rejected"]["temporal_influence"],
        n_p2_raw, len(pass2_edges),
        prune_stats["intra_component_removed"]["temporal"],
        prune_stats["bridge_collapsed"]["temporal"],
        prune_stats["isolation_rejected"]["temporal"],
        n_sem_raw, len(sem_edges),
        prune_stats["intra_component_removed"]["semantic_inter"],
        prune_stats["bridge_collapsed"]["semantic_inter"],
        prune_stats["isolation_rejected"]["semantic_inter"],
    )

    # Merge all
    all_edges = pd.concat(
        [nova_std, adept_std, graft_std, pass1_edges, pass2_edges, sem_edges],
        ignore_index=True,
    )

    n_nova   = len(nova_std)
    n_adept  = len(adept_std)
    n_graft  = len(graft_std)
    n_p1     = len(pass1_edges)
    n_p2     = len(pass2_edges)
    n_sem    = len(sem_edges)
    logger.info(
        "total: %d edges  (nova=%d, adept=%d, adept_graft=%d, temporal_influence=%d, temporal=%d, semantic_inter=%d)",
        len(all_edges), n_nova, n_adept, n_graft, n_p1, n_p2, n_sem,
    )

    # Final component size cap — the only place a size limit is enforced,
    # applied to the fully built graph (every edge type, no exceptions).
    max_component_size = pruning_cfg.get("max_component_size", 500)
    all_edges = _split_oversized_components(all_edges, posts_df, max_component_size)

    return all_edges


def run_full(
    posts_df: pd.DataFrame,
    config_path: str = "config.yaml",
    use_gpu: bool = True,
    nova_edges: pd.DataFrame = None,
    adept_edges: pd.DataFrame = None,
) -> pd.DataFrame:
    """Compute all edges from scratch. Persistence is the caller's responsibility (pipeline.py saves via _save_stage_checkpoint / save_state).

    Args:
        posts_df: full post DataFrame — embedding_raw, cluster_id, timestamp, engagement.
        config_path: path to config.yaml.
        use_gpu: attempt CuPy GPU acceleration for the cosine matrix.
        nova_edges: precomputed Nova intra-cluster edges (loaded from parquet if None).
        adept_edges: precomputed Adept intra-cluster edges.

    Returns:
        Merged edge DataFrame.
    """
    cfg = load_config(config_path)

    if nova_edges is None:
        nova_path = Path(cfg["storage"]["processed_dir"]) / "nova_edges.parquet"
        if nova_path.exists():
            nova_edges = pd.read_parquet(nova_path)
            logger.info("[Edges] NOVA loaded: %s (%d edges)", nova_path, len(nova_edges))
        else:
            logger.info("[Edges] No nova_edges found — intra-cluster edges absent.")
            nova_edges = pd.DataFrame(columns=["source", "target", "type", "force"])

    embeddings = _load_embeddings_raw(posts_df)
    logger.info(
        "[Edges] Cosine matrix: %d posts × %dd (%.1fM pairs)",
        len(posts_df), embeddings.shape[1], len(posts_df) ** 2 / 2 / 1e6,
    )
    cosine_matrix = compute_cosine_matrix(embeddings, use_gpu=use_gpu)

    all_edges = run(posts_df, cosine_matrix, cfg, nova_edges=nova_edges, adept_edges=adept_edges)
    return all_edges


def compute_incremental_edges(
    new_posts_df: pd.DataFrame,
    existing_posts_df: pd.DataFrame,
    config_path: str = "config.yaml",
    nova_edges: pd.DataFrame = None,
    adept_edges: pd.DataFrame = None,
) -> pd.DataFrame:
    """Compute edges for new posts against the full corpus (new + existing).

    Args:
        new_posts_df: incoming posts with embedding_raw and cluster columns.
        existing_posts_df: previously processed posts.
        config_path: path to config.yaml.
        nova_edges: Nova intra-cluster edges to include.
        adept_edges: Adept intra-cluster edges to include.

    Returns:
        Edge DataFrame filtered to edges involving at least one new post.
    """
    cfg = load_config(config_path)

    shared_cols = [
        "embedding_raw", "embedding_lda",
        "cluster_id", "galaxy_id",
        "nova_role", "nova_parent", "nova_depth",
    ]
    for col in shared_cols:
        if col in new_posts_df.columns and col not in existing_posts_df.columns:
            existing_posts_df = existing_posts_df.copy()
            existing_posts_df[col] = None
        if col in existing_posts_df.columns and col not in new_posts_df.columns:
            new_posts_df = new_posts_df.copy()
            new_posts_df[col] = None

    all_posts     = pd.concat([new_posts_df, existing_posts_df], ignore_index=True)
    embeddings    = _load_embeddings_raw(all_posts)
    cosine_matrix = compute_cosine_matrix(embeddings)

    all_edges = run(all_posts, cosine_matrix, cfg, nova_edges=nova_edges, adept_edges=adept_edges)

    new_ids   = set(new_posts_df["id"].tolist())
    new_edges = all_edges[
        all_edges["source"].isin(new_ids) | all_edges["target"].isin(new_ids)
    ].copy()

    logger.info("[Edges] Incremental: %d edges involving new posts.", len(new_edges))
    return new_edges