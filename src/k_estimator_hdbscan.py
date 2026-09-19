"""Step 2a — K estimation via UMAP + HDBSCAN."""

import pickle

import numpy as np
import pandas as pd

from run_store import load_config
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import hdbscan
    _HDBSCAN_AVAILABLE = True
except ImportError:
    _HDBSCAN_AVAILABLE = False

try:
    import umap as _umap_lib
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False


def _resolve_clustering_dtype(cfg: dict) -> np.dtype:
    """Return the numpy dtype for clustering inputs from cfg["clustering"]["clustering_float"].

    Accepted values: "float32", "float64" (default: "float64").
    Mirrors the adr_float pattern used in adr.py.
    """
    raw = str(cfg.get("clustering", {}).get("clustering_float", "float64")).lower()
    if raw not in ("float32", "float64"):
        logger.warning("clustering_float '%s' invalid — using float64", raw)
        raw = "float64"
    return np.float32 if raw == "float32" else np.float64


def _load_embeddings(posts_df: pd.DataFrame, dtype: np.dtype = np.float32) -> np.ndarray:
    """Extract the embedding_raw column into a contiguous ndarray.

    Embeddings are always stored as float32; cast to *dtype* here for this
    module's own HDBSCAN pipeline. Handles three storage formats: np.ndarray,
    bytes (pickle), and list/tuple.
    """
    embeddings = posts_df["embedding_raw"].values
    if isinstance(embeddings[0], np.ndarray):
        return np.stack(embeddings).astype(dtype)
    if isinstance(embeddings[0], bytes):
        return np.stack([pickle.loads(e) for e in embeddings]).astype(dtype)
    if isinstance(embeddings[0], (list, tuple)):
        return np.array([np.array(e) for e in embeddings], dtype=dtype)
    raise ValueError(f"Unsupported embedding format: {type(embeddings[0])}")


def cluster_label_to_id(label: int) -> str | None:
    """Convert an integer HDBSCAN label to a string cluster ID. Returns None for noise (-1)."""
    return None if label == -1 else f"cluster_{label:04d}"


def cluster_label_to_id_array(labels: np.ndarray) -> list[str | None]:
    """Vectorized cluster_label_to_id over an integer label array."""
    return [cluster_label_to_id(int(lbl)) for lbl in labels]


from preprocessing.whitening import maybe_apply_soft_zca


def _maybe_apply_soft_zca(embeddings: np.ndarray, cfg: dict) -> np.ndarray:
    """Apply Soft-ZCA + L2-renorm if enabled in the hdbscan: config section.

    Delegates the actual flag check and transform to
    preprocessing.whitening.maybe_apply_soft_zca.
    """
    hcfg = cfg.get("hdbscan", {})
    return maybe_apply_soft_zca(embeddings, hcfg, log_prefix="HDBSCAN")


def run_umap_for_clustering(embeddings: np.ndarray, cfg: dict) -> np.ndarray:
    """Reduce embeddings to a low-dimensional space suited for density clustering.

    Reads parameters from cfg["umap_cluster"]. Falls back to raw embeddings
    if umap-learn is unavailable.

    Args:
        embeddings: input array of shape (n, d).
        cfg: full pipeline configuration (umap_cluster sub-section used).

    Returns:
        Reduced array of shape (n, n_components), same dtype as the input
        (UMAP computes in float32 internally, cast back after).
    """
    if not _UMAP_AVAILABLE:
        logger.warning("umap-learn unavailable — running HDBSCAN on raw embeddings (degraded)")
        return embeddings

    ucfg         = cfg.get("umap_cluster", {})
    n_components = ucfg.get("n_components", 5)
    n_neighbors  = ucfg.get("n_neighbors", 15)
    min_dist     = ucfg.get("min_dist", 0.0)
    metric       = ucfg.get("metric", "cosine")
    random_state = ucfg.get("random_state", 42)

    target_dtype = embeddings.dtype

    reduced = _umap_lib.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=False,
    ).fit_transform(embeddings).astype(target_dtype)

    logger.info("UMAP done  shape=%s  dtype=%s", reduced.shape, reduced.dtype)
    return reduced


def _run_hdbscan(
    embeddings: np.ndarray, cfg: dict
) -> tuple[np.ndarray, "hdbscan.HDBSCAN"]:
    """Fit HDBSCAN and return (labels, clusterer). Parameters from cfg["hdbscan"].

    Args:
        embeddings: reduced array of shape (n, d).
        cfg: full pipeline configuration (hdbscan sub-section used).

    Returns:
        Tuple of (labels, fitted clusterer).

    Raises:
        ImportError: if the hdbscan package is not installed.
    """
    if not _HDBSCAN_AVAILABLE:
        raise ImportError("pip install hdbscan")

    hcfg    = cfg["hdbscan"]
    epsilon = hcfg.get("cluster_selection_epsilon", 0.0)

    logger.info("HDBSCAN min_cluster_size=%d  min_samples=%d  metric=%s  method=%s  epsilon=%s",
                hcfg['min_cluster_size'], hcfg['min_samples'], hcfg['metric'], hcfg['method'], epsilon)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hcfg["min_cluster_size"],
        min_samples=hcfg["min_samples"],
        metric=hcfg["metric"],
        cluster_selection_method=hcfg["method"],
        cluster_selection_epsilon=epsilon,
        prediction_data=hcfg.get("prediction_data", True),
    )
    labels = clusterer.fit_predict(embeddings)
    return labels, clusterer


def run(posts_df: pd.DataFrame, config_path: str = "config.yaml") -> int:
    """Estimate k_optimal from post embeddings via UMAP + HDBSCAN.

    Args:
        posts_df: must contain an ``embedding_raw`` column produced by Step 1.
        config_path: path to ``config.yaml``.

    Returns:
        Number of clusters detected by HDBSCAN (noise points excluded).
    """
    cfg   = load_config(config_path)
    dtype = _resolve_clustering_dtype(cfg)

    embeddings = _load_embeddings(posts_df, dtype=dtype)

    logger.info("n=%d  input_dim=%d  dtype=%s", len(posts_df), embeddings.shape[1], embeddings.dtype)

    embeddings = _maybe_apply_soft_zca(embeddings, cfg)
    reduced    = run_umap_for_clustering(embeddings, cfg)
    labels, _ = _run_hdbscan(reduced, cfg)

    k_optimal = len(set(labels)) - (1 if -1 in labels else 0)
    return k_optimal