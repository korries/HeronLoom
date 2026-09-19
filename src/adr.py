"""Stage 3 — Adaptive Discriminant Refinement (ADR).

Thin pipeline wrapper around :class:`adr_refiner.ADRRefiner`: reads the
``adr:`` section of config.yaml, builds and fits a refiner, logs a
validation report, and persists the fitted refiner to disk.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from adr_refiner import ADRRefiner
from preprocessing.whitening import maybe_apply_soft_zca
from utils.logger import get_logger

logger = get_logger(__name__)


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    """L2-normalise each row of X (unit norm).

    Applied unconditionally before every LDA-GO/GMM step, even when
    ``maybe_apply_soft_zca`` has already normalised its output.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms == 0, 1.0, norms)


def _resolve_dtype(adr_cfg: dict) -> np.dtype:
    """Resolve adr_cfg['adr_float'] to a numpy dtype, falling back to float64 on any invalid value."""
    raw = str(adr_cfg.get("adr_float", "float64")).lower()
    if raw not in ("float32", "float64"):
        logger.warning("adr_float '%s' invalid — using float64", raw)
        raw = "float64"
    return np.float32 if raw == "float32" else np.float64


def _build_refiner(adr_cfg: dict, k_override: int) -> ADRRefiner:
    """Translate the adr: config section into an ADRRefiner instance."""
    n_dims_raw = adr_cfg.get("n_dims", "auto")
    n_dims = None if str(n_dims_raw).strip().lower() == "auto" else int(n_dims_raw)

    np_dtype = _resolve_dtype(adr_cfg)
    dtype_str = "float32" if np_dtype == np.float32 else "float64"

    lda_go_kwargs = {
        "lr":           float(adr_cfg.get("ldago_lr", 0.01)),
        "max_iter":     int(adr_cfg.get("ldago_max_iter", 500)),
        "tol":          float(adr_cfg.get("ldago_tol", 1e-4)),
        "force_ce":     bool(adr_cfg.get("ldago_force_ce", False)),
        "ce_optimizer": str(adr_cfg.get("ldago_ce_optimizer", "sgd")),
        "dtype":        dtype_str,
        "random_state": adr_cfg.get("ldago_random_state", adr_cfg.get("random_state", 0)),
        # Forwarded as-is to LDA-GO's "device" kwarg; None by default.
        "device":       adr_cfg.get("ldago_device", None),
    }

    return ADRRefiner(
        n_clusters=k_override,
        n_dims=n_dims,
        max_iter=int(adr_cfg.get("max_iter", 10)),
        covariance_type=str(adr_cfg.get("covariance_type", "diag")),
        gmm_repetitions=int(adr_cfg.get("gmm_repetitions", 10)),
        random_state=adr_cfg.get("random_state", 0),
        lda_go_kwargs=lda_go_kwargs,
        stop_on_loss_increase=bool(adr_cfg.get("ldago_stop_on_loss_increase", True)),
    )


def _validate_seed_labels(
    seed_labels: np.ndarray | None, n_samples: int, k_override: int
) -> np.ndarray | None:
    """Return seed_labels if usable, otherwise None (triggers a GMM bootstrap)."""
    if seed_labels is None:
        return None
    seed_labels = np.asarray(seed_labels)
    n_unique = len(set(seed_labels.tolist()))
    if len(seed_labels) != n_samples or n_unique != k_override:
        logger.warning(
            "Seed labels invalid (n=%d vs %d, clusters=%d vs K=%d) — "
            "falling back to a GMM bootstrap",
            len(seed_labels), n_samples, n_unique, k_override,
        )
        return None
    return seed_labels


def run_adr(
    X: np.ndarray,
    adr_cfg: dict,
    k_override: int,
    seed_labels: np.ndarray | None = None,
) -> tuple[ADRRefiner, np.ndarray, np.ndarray]:
    """Fit an ADRRefiner on X.

    Args:
        X: raw embeddings, shape (n, d).
        adr_cfg: the ``adr:`` config section.
        k_override: number of clusters from upstream HDBSCAN or GMM-BIC.
        seed_labels: optional initial partition (e.g. from GMM-BIC) used
            to start the refinement loop.

    Returns:
        (refiner, labels_final, probs)
    """
    np_dtype = _resolve_dtype(adr_cfg)
    X = np.asarray(maybe_apply_soft_zca(X, adr_cfg), dtype=np_dtype)
    X = _l2_normalize(X)
    logger.info("ADR input — shape=%s  dtype=%s  K=%d", X.shape, X.dtype, k_override)

    seed_labels = _validate_seed_labels(seed_labels, len(X), k_override)

    refiner = _build_refiner(adr_cfg, k_override)
    refiner.fit(X, seed_labels)
    probs = refiner.predict_proba(X)

    _log_validation_report(refiner, k_override, seed_labels)

    return refiner, refiner.labels_, probs


def _log_validation_report(
    refiner: ADRRefiner, K: int, seed_labels: np.ndarray | None
) -> None:
    """Log cluster count, ADR convergence, and — if seed_labels were used — seed-vs-final NMI."""
    labels = refiner.labels_
    n_clusters_found = len(set(labels.tolist()))
    converged = refiner.n_iter_ < refiner.max_iter

    logger.info(
        "ADR done — clusters=%d/%d  adr_iter=%d/%d  converged=%s  d=%d",
        n_clusters_found, K, refiner.n_iter_, refiner.max_iter, converged, refiner.n_dims_,
    )

    if seed_labels is not None:
        seed_vs_final_nmi = normalized_mutual_info_score(seed_labels, labels)
        logger.info(
            "NMI(seed -> final)=%.4f  (1.0=labels unchanged, 0.0=fully reshuffled)",
            seed_vs_final_nmi,
        )


def transform_lda(refiner: ADRRefiner, X: np.ndarray) -> np.ndarray:
    """Project embeddings (already soft-ZCA'd, same space as fit) into the LDA-GO subspace.

    Applies the same L2-normalisation as run_adr() before projecting, to
    match fit-time preprocessing.
    """
    X = _l2_normalize(np.asarray(X))
    return refiner.transform(X).astype(np.float32)


def predict_proba_lda(refiner: ADRRefiner, X: np.ndarray) -> np.ndarray:
    """Cluster membership probabilities for embeddings (already soft-ZCA'd, same space as fit)."""
    X = _l2_normalize(np.asarray(X))
    return refiner.predict_proba(X)


def _model_path(cfg: dict) -> Path:
    """Resolve the refiner's save path: <storage.processed_dir>/adr_refiner.joblib.

    Same pattern as every other stage (k_cache.pkl, nova_edges.parquet,
    labels.parquet, ...): always under cfg["storage"]["processed_dir"],
    which run_store.create_run() / resolve_run_for_resume() repoint at
    this run's own folder.
    """
    return Path(cfg["storage"]["processed_dir"]) / "adr_refiner.joblib"


def save_models(refiner: ADRRefiner, cfg: dict) -> None:
    path = _model_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(refiner, path)
    logger.info("Model saved: %s", path)


def load_models(cfg: dict) -> ADRRefiner | None:
    path = _model_path(cfg)
    if not path.exists():
        logger.warning("Model not found: %s", path)
        return None
    refiner = joblib.load(path)
    logger.info("Model loaded: %s", path)
    return refiner