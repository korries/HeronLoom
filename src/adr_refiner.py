"""Adaptive Discriminant Refinement (ADR): alternates an LDA-GO fit and a
GMM re-clustering for up to max_iter passes.

Adapted from TopiCLEAR's ADR loop (itself an application of the general
Adaptive Dimension Reduction concept — Ding & Li, ICML 2007), replacing
TopiCLEAR's closed-form LDA step with LDA-GO (Shen & Dong, arXiv:2506.06845v2)
and keeping its GMM re-clustering step. Full citations: docs/ALGORITHMS.md.
TopiCLEAR (Fujita et al., arXiv:2512.06694v2) attribution (MIT): THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import calinski_harabasz_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from lda_go import LDAGO
from utils.logger import get_logger

logger = get_logger(__name__)


class ADRRefiner(TransformerMixin, ClusterMixin, BaseEstimator):
    """Refines a clustering by alternating an LDA-GO fit and a GMM re-clustering.

    The loop stops after a pass once either of two conditions is met:
    the GMM's new labels exactly reproduce the labels LDA-GO started
    from this pass (NMI == 1.0 — a fixed point), or the following
    pass's LDA-GO starting loss is higher than this pass's, meaning the
    discriminant projection stopped improving (checked before that
    pass's GMM fit runs, so its cost is only paid when there's a reason
    to). A GMM collapse (fewer than n_clusters active components) is a
    third, safety-net exit. Not guaranteed to converge to a fixed point:
    runs up to max_iter passes and keeps the last one completed before
    stopping — there is no scored comparison across passes to pick a
    "best".

    Parameters
    ----------
    n_clusters : int
        Number of clusters K.
    n_dims : int or None
        Discriminant subspace dimension. None -> K - 1.
    max_iter : int
        Max LDA-GO <-> GMM passes.
    covariance_type : {"diag", "full"}
        GMM covariance type. Falls back to "diag" if "full" fails.
    gmm_repetitions : int
        n_init per GaussianMixture fit.
    random_state : int or None
        Seed for GMM fits and the bootstrap GMM (used if fit() gets no
        seed labels).
    lda_go_kwargs : dict or None
        Kwargs forwarded to LDAGO each pass (lr, max_iter, tol, force_ce,
        ce_optimizer, dtype, device, random_state).
    stop_on_loss_increase : bool
        Stop early if LDA-GO's starting loss rises between passes.

    Attributes
    ----------
    labels_ : ndarray, shape (n_samples,)
    rotation_ : ndarray, shape (n_features, n_dims)
        Alias for ldago_.L_.
    ldago_ : LDAGO
    gmm_ : GaussianMixture
    cluster_centers_ : ndarray
    error_ : float
        Negative of gmm_.score(X), i.e. negative per-sample average
        log-likelihood — not the total NLL over all samples.
    n_iter_ : int
        Index of the retained pass (1-based).
    nmi_history_ : list[float]
        NMI(new, old) labels at each completed pass.
    """

    def __init__(
        self,
        n_clusters: int,
        n_dims: int | None = None,
        max_iter: int = 10,
        covariance_type: str = "diag",
        gmm_repetitions: int = 10,
        random_state: int | None = None,
        lda_go_kwargs: dict | None = None,
        stop_on_loss_increase: bool = True,
    ) -> None:
        self.n_clusters = n_clusters
        self.n_dims = n_dims
        self.max_iter = max_iter
        self.covariance_type = covariance_type
        self.gmm_repetitions = gmm_repetitions
        self.random_state = random_state
        self.lda_go_kwargs = lda_go_kwargs or {}
        self.stop_on_loss_increase = stop_on_loss_increase

    @property
    def n_dims_(self) -> int:
        """Resolved subspace dimension (K - 1 when n_dims is None)."""
        return self.n_clusters - 1 if self.n_dims is None else self.n_dims

    def _fit_gmm(self, X: np.ndarray) -> GaussianMixture:
        """Fit a GMM with the configured covariance_type.

        Runs the ``gmm_repetitions`` restarts as parallel joblib processes
        (loky backend) instead of sklearn's built-in sequential n_init loop
        — same pattern as the GMM-BIC sweep in k_estimator_gmm.py.

        Retries once with covariance_type="diag" if "full" raises a
        ValueError on every restart (typically a singular covariance on a
        small/degenerate cluster).
        """
        n_components = self.n_clusters
        n_repeats = self.gmm_repetitions
        base_seed = self.random_state if self.random_state is not None else 0

        def _fit_one(seed: int, cov_type: str, X: np.ndarray, n_components: int):
            try:
                with threadpool_limits(limits=1, user_api="blas"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    gmm = GaussianMixture(
                        n_components=n_components,
                        covariance_type=cov_type,
                        n_init=1,
                        random_state=seed,
                    )
                    gmm.fit(X)
                    return gmm, gmm.score(X)
            except ValueError:
                return None

        def _run(cov_type: str):
            logger.info("Fitting GMM — K=%d n_init=%d covariance_type=%s", n_components, n_repeats, cov_type)
            # X passed as an explicit delayed() argument rather than closed
            # over, so joblib memory-maps it once and shares it across
            # worker processes instead of re-pickling it per restart.
            results = Parallel(n_jobs=-1)(
                delayed(_fit_one)(base_seed + i, cov_type, X, n_components)
                for i in range(n_repeats)
            )
            return [r for r in results if r is not None]

        valid = _run(self.covariance_type)
        if not valid:
            if self.covariance_type != "full":
                raise ValueError(
                    f"GMM fit failed on every restart ({n_repeats}/{n_repeats}) "
                    f"with covariance_type='{self.covariance_type}'"
                )
            logger.warning(
                "covariance_type='full' failed on every restart (%d/%d) — retrying with 'diag'",
                n_repeats, n_repeats,
            )
            valid = _run("diag")
            if not valid:
                raise ValueError("GMM fit failed on every restart, even with covariance_type='diag'")

        best_gmm, _ = max(valid, key=lambda pair: pair[1])
        return best_gmm

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> ADRRefiner:
        """Run the LDA-GO <-> GMM refinement loop.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Embeddings, already preprocessed upstream (e.g. whitened,
            L2-normalised). This class performs no preprocessing of its
            own — that keeps it a single-responsibility clustering step.
        y : np.ndarray or None
            Initial cluster assignment, e.g. from an external GMM-BIC
            sweep. If None, a direct GMM(K) fit on X provides the starting
            partition.
        """
        X = np.asarray(X)
        n_dims = self.n_dims_
        if n_dims >= X.shape[1]:
            raise ValueError(
                f"n_dims ({n_dims}) must be strictly less than the input "
                f"dimensionality ({X.shape[1]})."
            )

        if y is None:
            logger.info("No seed labels supplied — bootstrapping with GMM(K=%d) on the raw input", self.n_clusters)
            old_labels = self._fit_gmm(X).predict(X)
        else:
            old_labels = np.asarray(y).copy()

        nmi_history: list[float] = []
        prev_start_loss: float | None = None
        best: tuple | None = None
        n_iter_completed = 0

        for iteration in range(self.max_iter):
            ldago = LDAGO(d=n_dims, **self.lda_go_kwargs)
            X_subspace = ldago.fit_transform(X, old_labels)
            start_loss = ldago.loss_history_[0] if ldago.loss_history_ else None

            # Check the loss-increase stop condition right away, before
            # paying for the (expensive, multi-restart) GMM fit: if the
            # LDA-GO starting loss already rose vs. the last retained
            # iteration, this iteration will be discarded regardless of
            # what the GMM finds, so there's no point fitting it.
            if (
                self.stop_on_loss_increase
                and start_loss is not None
                and prev_start_loss is not None
                and start_loss > prev_start_loss
            ):
                logger.info(
                    "Iter %d — LDA-GO start loss rose (%.6f > %.6f at iter %d) — "
                    "stopping before the GMM fit, keeping iteration %d",
                    iteration + 1, start_loss, prev_start_loss, n_iter_completed, n_iter_completed,
                )
                break

            gmm = self._fit_gmm(X_subspace)
            new_labels = gmm.predict(X_subspace)
            n_active = len(set(new_labels.tolist()))

            if n_active < self.n_clusters:
                logger.warning(
                    "Iter %d — GMM collapse (%d/%d active clusters) — "
                    "reverting to the last stable iteration",
                    iteration + 1, n_active, self.n_clusters,
                )
                break

            nmi_score = float(normalized_mutual_info_score(new_labels, old_labels))
            nmi_history.append(nmi_score)
            try:
                ch_score = calinski_harabasz_score(X_subspace, new_labels)
                ch_s = f"{ch_score:.1f}"
            except Exception:
                ch_s = "n/a"
            logger.info(
                "ADR iter=%d — NMI=%.4f%s  CH=%s%s",
                iteration + 1, nmi_score, "  ✓" if nmi_score == 1.0 else "", ch_s,
                f"  start_loss={start_loss:.6f}" if start_loss is not None else "",
            )

            best = (new_labels, ldago, gmm, X_subspace)
            prev_start_loss = start_loss
            n_iter_completed = iteration + 1

            if nmi_score == 1.0:
                break
            old_labels = new_labels.copy()

        if best is None:
            raise RuntimeError(
                "ADR refinement collapsed on the very first iteration — no "
                "stable clustering was found. Check the seed labels and "
                "n_clusters, or relax covariance_type."
            )

        labels, ldago, gmm, X_subspace = best
        self.labels_ = labels
        self.ldago_ = ldago
        self.gmm_ = gmm
        self.rotation_ = ldago.L_
        self.cluster_centers_ = gmm.means_
        self.error_ = -gmm.score(X_subspace)
        self.n_iter_ = n_iter_completed
        self.nmi_history_ = nmi_history
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X into the fitted discriminant subspace."""
        check_is_fitted(self, ["ldago_"])
        return self.ldago_.transform(X)

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["gmm_"])
        return self.gmm_.predict(self.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["gmm_"])
        return self.gmm_.predict_proba(self.transform(X))