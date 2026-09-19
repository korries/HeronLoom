"""Step 2b — K estimation via GMM-BIC; HDBSCAN pivot → penalized BIC sweep. Entry point: run()."""

import multiprocessing as mp
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits
from umap import UMAP

import k_estimator_hdbscan as _hdbscan_module
from k_estimator_hdbscan import _resolve_clustering_dtype
from preprocessing.whitening import soft_zca
from run_store import load_config
from utils.logger import get_logger

logger = get_logger(__name__)


_K_LEFT_MARGIN  = 48
_K_RIGHT_MARGIN = 48
_K_ABS_MIN      = 2
_DEFAULT_ALPHA  = 1.0


def _compute_k_bounds(posts_df: pd.DataFrame, config_path: str) -> tuple[int, int, int]:
    """Run HDBSCAN to derive (k_min, k_max, k_pivot) centred on the HDBSCAN pivot."""
    logger.info("pre-run HDBSCAN to determine K pivot...")
    k_pivot = _hdbscan_module.run(posts_df, config_path)
    k_min   = max(_K_ABS_MIN, k_pivot - _K_LEFT_MARGIN)
    k_max   = k_pivot + _K_RIGHT_MARGIN
    logger.info(
        "pivot=%d  bounds=[%d, %d]  margins=-%d/+%d",
        k_pivot, k_min, k_max, _K_LEFT_MARGIN, _K_RIGHT_MARGIN,
    )
    return k_min, k_max, k_pivot


def _find_final_k(
    valid_ks: list[int],
    bics: list[float],
    alpha: float = _DEFAULT_ALPHA,
) -> tuple[int, float, float]:
    """Select k via penalized BIC: score(k) = BIC(k) + α·range(BIC)·(k−k_min)/(k_max−k_min)."""
    K = np.array(valid_ks, dtype=float)
    B = np.array(bics,     dtype=float)

    bic_range = B.max() - B.min()
    k_range   = K.max() - K.min()

    if k_range == 0:
        idx = int(np.argmin(B))
        return int(K[idx]), float(B[idx]), float(B[idx])

    scores = B + alpha * bic_range * (K - K.min()) / k_range
    idx    = int(np.argmin(scores))
    return int(K[idx]), float(B[idx]), float(scores[idx])


def find_optimal_k(
    embeddings: np.ndarray,
    k_min: int,
    k_max: int,
    alpha: float = _DEFAULT_ALPHA,
    cfg_n_jobs: int | None = None,
    soft_zca_eps: float | None = None,
    gmm_dtype: np.dtype = np.float64,
) -> tuple[int, np.ndarray, np.ndarray]:
    """UMAP → 5d, then parallel GMM-BIC sweep over [k_min, k_max].

    Args:
        embeddings: input float32 array of shape (n, d).
        k_min: lower bound of the sweep range.
        k_max: upper bound of the sweep range.
        alpha: penalized BIC weight passed to _find_final_k.
        cfg_n_jobs: joblib parallelism for the GMM sweep.
        soft_zca_eps: if set, apply Soft-ZCA before UMAP to correct anisotropy.
            None = disabled. Typical values: 0.01 or 0.1.
        gmm_dtype: numpy dtype for the GMM-BIC sweep input (post-UMAP cast).
            float64 (default) is strongly recommended for numerical stability of
            the full-covariance Cholesky decomposition. float32 is allowed when
            clustering.clustering_float: float32 is set explicitly.

    Returns:
        Tuple of (k_optimal, umap_embeddings, seed_labels).

    Raises:
        RuntimeError: if no GMM converges across the sweep range.
    """
    n, d = embeddings.shape
    logger.info("n=%d  input_dim=%d", n, d)

    X = embeddings.copy()

    if soft_zca_eps is not None:
        logger.info("Soft-ZCA whitening  eps=%s", soft_zca_eps)
        X = soft_zca(X, eps=soft_zca_eps)
        # L2-renorm post-ZCA: vectors are no longer unit-norm after whitening
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.where(norms == 0, 1.0, norms)
        logger.info("Soft-ZCA + L2-renorm done  shape=%s", X.shape)

    logger.info("UMAP → 5d  (n_neighbors=15  min_dist=0.0  metric=cosine)")

    X = UMAP(
        n_components=5,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        n_jobs=-1,
        verbose=False,
    ).fit_transform(X)   # pass X (potentially whitened), not raw embeddings
    logger.info("UMAP done  shape=%s", X.shape)

    # UMAP always outputs float32; cast to the requested dtype only after
    # (rationale for float64 vs float32 is in this function's docstring).
    X = X.astype(gmm_dtype)
    logger.info("GMM input dtype cast → %s", X.dtype)

    n_jobs  = cfg_n_jobs if cfg_n_jobs is not None else -1
    total_k = k_max - k_min + 1

    logger.info("GMM-BIC sweep  k=[%d, %d]  alpha=%s  n_jobs=%s", k_min, k_max, alpha, n_jobs)

    def _fit_one(
        k: int, X: np.ndarray, lock, done, total: int,
    ) -> tuple[int, float, "GaussianMixture"] | None:
        """Fit a single GMM and return (k, BIC, gmm), or None on failure.

        Runs in its own process (loky backend, see Parallel() below), not a
        thread: covariance_type="full"'s per-component M-step loop holds the
        GIL, so threads would serialize across the sweep. `lock`/`done` are
        multiprocessing.Manager proxies rather than threading.Lock, which
        can't be pickled to a separate process.
        """
        try:
            with threadpool_limits(limits=1, user_api="blas"), warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type="full",
                    random_state=42,
                    n_init=3,
                    max_iter=50,
                )
                gmm.fit(X)
                bic_val = gmm.bic(X)
            with lock:
                done.value += 1
                print(f"\r[K-Estimator] {done.value}/{total}", end="", flush=True)
            return k, bic_val, gmm
        except Exception:
            with lock:
                done.value += 1
                print(f"\r[K-Estimator] {done.value}/{total} (skipped k={k})", end="", flush=True)
            return None

    # Default joblib backend (loky, process-based) instead of prefer="threads".
    # X is passed as an explicit delayed() argument rather than captured by
    # closure so joblib memory-maps it once and shares it across worker
    # processes, instead of re-pickling it for every task.
    with mp.Manager() as manager:
        _lock = manager.Lock()
        _done = manager.Value("i", 0)
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_one)(k, X, _lock, _done, total_k) for k in range(k_min, k_max + 1)
        )
    print()

    pairs: list[tuple[int, float, GaussianMixture]] = sorted(
        (item for item in results if item is not None), key=lambda x: x[0]
    )
    skipped = [k for k in range(k_min, k_max + 1) if not any(p[0] == k for p in pairs)]

    logger.debug("  %6s  %14s", "k", "BIC")
    logger.debug("  %s  %s", "─" * 6, "─" * 14)
    for k, bic_val, _ in pairs:
        logger.debug("  %6d  %14.1f", k, bic_val)
    if skipped:
        logger.warning("skipped (GMM fit failed): %s", skipped)
    logger.info("GMM-BIC sweep done  %d/%d k values converged  (full table in log file)",
                len(pairs), total_k)

    valid_ks: list[int]   = [k   for k, _, _ in pairs]
    bics:     list[float] = [bic for _, bic, _ in pairs]
    gmms:     list        = [g   for _, _, g   in pairs]

    if not bics:
        raise RuntimeError("no valid k found — check input data.")

    k_optimal, bic_opt, score_opt = _find_final_k(valid_ks, bics, alpha=alpha)
    k_raw = valid_ks[int(np.argmin(bics))]

    logger.info("BIC argmin (raw)       k=%d", k_raw)
    logger.info(
        "BIC argmin (penalized) k=%d  BIC=%.1f  score=%.1f",
        k_optimal, bic_opt, score_opt,
    )

    # GMM-optimal labels in the 5d UMAP space
    gmm_optimal = gmms[valid_ks.index(k_optimal)]
    seed_labels = gmm_optimal.predict(X).astype(np.int32)
    logger.info("seed_labels computed  shape=%s  k=%d", seed_labels.shape, k_optimal)

    return k_optimal, X, seed_labels


_CACHE_FILENAME = "k_cache.pkl"


def _cache_path(cfg: dict) -> Path:
    return Path(cfg["storage"]["processed_dir"]) / _CACHE_FILENAME


def _save_cache(k_optimal: int, cfg: dict, seed_labels: np.ndarray | None = None) -> None:
    path = _cache_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"k_optimal": k_optimal}
    if seed_labels is not None:
        payload["seed_labels"] = seed_labels
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    label_info = f"  seed_labels={seed_labels.shape}" if seed_labels is not None else ""
    logger.info("k_optimal=%d saved → %s%s", k_optimal, path, label_info)


def _load_cache(cfg: dict) -> tuple[int, np.ndarray | None]:
    """Load k_cache.pkl and return (k_optimal, seed_labels_or_None).

    Raises:
        FileNotFoundError: if k_cache.pkl does not exist.
    """
    path = _cache_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"cache not found: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    k_optimal   = data["k_optimal"]
    seed_labels = data.get("seed_labels", None)
    label_info  = f"  seed_labels={seed_labels.shape}" if seed_labels is not None else "  no seed_labels"
    logger.info("cache loaded  k_optimal=%d%s  (%s)", k_optimal, label_info, path)
    return k_optimal, seed_labels


def run(posts_df: pd.DataFrame, config_path: str = "config.yaml") -> tuple[int, np.ndarray]:
    """Run the full GMM-BIC pipeline and return (k_optimal, seed_labels).

    Args:
        posts_df: post table with an embedding_raw column (produced by step 1).
        config_path: path to config.yaml.

    Returns:
        Tuple of (k_optimal, seed_labels).
    """
    cfg = load_config(config_path)

    raw_embeddings = posts_df["embedding_raw"].values
    emb_list: list[np.ndarray] = []
    for emb in raw_embeddings:
        if isinstance(emb, bytes):
            emb = pickle.loads(emb)
        if isinstance(emb, (list, tuple)):
            emb = np.array(emb, dtype=np.float32)
        emb_list.append(emb)
    embeddings = np.stack(emb_list).astype(np.float32)

    clust_cfg = cfg.get("clustering", {})
    alpha     = clust_cfg.get("bic_penalty_alpha", _DEFAULT_ALPHA)
    n_jobs    = clust_cfg.get("gmm_n_jobs", -1)
    gmm_dtype = _resolve_clustering_dtype(cfg)

    logger.info("GMM-BIC dtype: %s  (clustering.clustering_float)", gmm_dtype)

    # soft-ZCA enabled only when gmm_input_space = "soft_zca"
    soft_zca_eps: float | None = None
    if clust_cfg.get("gmm_input_space") == "soft_zca":
        soft_zca_eps = float(clust_cfg.get("gmm_zca_eps", 0.01))

    k_min, k_max, _ = _compute_k_bounds(posts_df, config_path)
    k_optimal, _, seed_labels = find_optimal_k(
        embeddings, k_min, k_max,
        alpha=alpha, cfg_n_jobs=n_jobs,
        soft_zca_eps=soft_zca_eps,
        gmm_dtype=gmm_dtype,
    )

    return k_optimal, seed_labels