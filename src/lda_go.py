"""LDA-GO v2 — Gradient-Optimised LDA  (arXiv 2506.06845v2)

Faithful implementation of Algorithm 1:
  Cencheng Shen, Yuexiao Dong — "Linear Discriminant Analysis with
  Gradient Optimization", revised Apr 2026.

Algorithm overview
------------------
1. Within-class standardisation  (Section 3.1)
   Features are rescaled by within-class std before optimisation.
   The inverse transform is stored for inference on new data.

2. Precision parametrisation  Σ⁻¹ = L Lᵀ + σ²I  (Section 3.2)
   σ² is set via Oracle Approximating Shrinkage (OAS) for the NLL path
   and σ²=0 for the CE path.

3. Automatic loss selection via structural diagnostics  (Section 3.3)
   Signal sparsity r  = D_eff / p   (effective signal dimensionality)
   Excess kurtosis  κ = mean |e_j - 3| over features
   Rule: r < 0.10 OR κ > 10  → CE path,  otherwise → NLL path

4. Two optimisation paths  (Sections 3.4 / 3.5)
   NLL path: Gaussian negative log-likelihood optimised with Adam.
             gradient: ∂L_NLL/∂L = Σ̂_w,α L - L (σ²I_d + LᵀL)⁻¹
   CE path:  cross-entropy, gradient descent (SGD or Adam), σ²=0.
             O(npd + nKp) per iteration — no p×p intermediate matrix.

5. Initialisation  L⁰ = I_{p×d} + ε,  ε ~ N(0, 0.01²)  (Algorithm 1, lines 7/14)
   Paper defaults: max_iter = 500, lr = 0.01, tol = 1e-4.

6. Default rank  d = min(20, p)  (paper recommendation)

sklearn-compatible interface
----------------------------
  fit(X, y)            → trains L and σ²; stores L_, sigma2_
  transform(X)         → X_standardised @ L_                  (n × d)
  fit_transform(X, y)  → fit then transform
  predict(X)           → argmax discriminant scores            (n,)
  predict_proba(X)     → softmax probabilities                 (n × K)
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.special import softmax as _scipy_softmax

try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

# GPU (torch) — optional. If absent, everything runs on numpy/CPU.
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# GPU detection, ComfyUI-style (comfy/model_management.py): logs the torch
# version and detected GPU(s). Done from fit() rather than at module import
# time, because at import time the host application's logging is often not
# yet configured (no handlers attached => logger.info() is silently
# dropped). Logged once per process.
_GPU_INFO_LOGGED = False


def _log_torch_gpu_info() -> None:
    global _GPU_INFO_LOGGED
    if _GPU_INFO_LOGGED or not TORCH_AVAILABLE:
        return
    _GPU_INFO_LOGGED = True
    logger.info("LDA-GO — torch version: %s", torch.__version__)
    try:
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False
    logger.info("LDA-GO — CUDA available: %s", cuda_ok)
    if cuda_ok:
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            total_vram_mb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)
            logger.info("LDA-GO — GPU %d: %s — %.0f MB VRAM", i, name, total_vram_mb)


# Same "log once at INFO, then DEBUG" pattern as _log_torch_gpu_info: the
# selected backend never changes between ADR iterations within one process,
# so repeating it at INFO on every fit() call is just noise.
_DEVICE_INFO_LOGGED = False


def _log_device_once(device: str) -> None:
    global _DEVICE_INFO_LOGGED
    if not _DEVICE_INFO_LOGGED:
        _DEVICE_INFO_LOGGED = True
        logger.info("LDA-GO — running on torch:%s", device)
    else:
        logger.debug("LDA-GO — running on torch:%s", device)


def _oas_shrinkage(S: np.ndarray, n: int) -> tuple[float, float]:
    """Return (alpha, mu) where alpha is the OAS shrinkage coefficient
    and mu = tr(S)/p is the scaled trace.

    Parameters
    ----------
    S : (p, p) sample covariance matrix
    n : number of samples used to estimate S

    Returns
    -------
    alpha : float in [0, 1]
    mu    : float  (tr(S) / p)
    """
    p = S.shape[0]
    trace_S  = np.trace(S)
    trace_S2 = np.trace(S @ S)
    mu = trace_S / p

    # OAS formula (Chen et al. 2010, eq. 23)
    rho_num = ((1 - 2 / p) * trace_S2 + trace_S ** 2)
    rho_den = (n + 1 - 2 / p) * (trace_S2 - trace_S ** 2 / p)
    alpha = min(1.0, rho_num / rho_den) if rho_den > 0 else 1.0
    return float(alpha), float(mu)


class _Adam:
    def __init__(self, lr: float = 0.01, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr; self.b1 = beta1; self.b2 = beta2; self.eps = eps
        self.m = self.v = None; self.t = 0

    def step(self, grad: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class LDAGO:
    """LDA-GO v2 — sklearn-compatible classifier and subspace projector.

    Parameters
    ----------
    d : int or None
        Projection rank.  None, or any value outside (0, p], falls back
        to d = min(20, p) (paper default); an out-of-range value logs a
        warning rather than raising.
    max_iter : int
        Max gradient iterations (paper default: 500).
    tol : float
        Early-stop on ||∇L||_F  (paper default: 1e-4).
    lr : float
        Learning rate for both paths (paper default: 0.01).
    random_state : int or None
        Seed for the single random initialisation.
    force_ce : bool
        Force the CE path regardless of the diagnostics in Section 3.3.
    ce_optimizer : {"sgd", "adam"}
        Optimizer for the CE path only ("sgd" is the paper default). The
        NLL path always uses Adam.
    dtype : {"float64", "float32"}
        Numeric precision for all numpy arrays.
    device : str or None
        None (default) runs on numpy/CPU. "cuda" or "cpu" runs the CE
        path's gradient loop via PyTorch on that device instead (same
        formula, lr, optimizer, tol, max_iter — only the backend
        changes). Ignored on the NLL path, and falls back to numpy/CPU
        with a logged warning if torch isn't installed or CUDA isn't
        available.
    """

    def __init__(
        self,
        d: int | None = None,
        max_iter: int = 500,
        tol: float = 1e-4,
        lr: float = 0.01,
        random_state: int | None = None,
        force_ce: bool = False,
        ce_optimizer: str = "sgd",   # "sgd" (paper default) | "adam"
        dtype: str = "float64",      # "float64" | "float32"
        device: str | None = None,  # None | "cuda" | "cpu" — see class docstring
    ) -> None:
        self.d            = d
        self.max_iter     = max_iter
        self.tol          = tol
        self.lr           = lr
        self.random_state = random_state
        self.force_ce     = force_ce
        self.ce_optimizer = ce_optimizer
        self.dtype        = dtype
        self._np_dtype    = np.float32 if str(dtype).lower() == "float32" else np.float64
        self.device       = device

        # Set after fit()
        self.L_:            np.ndarray | None = None   # (p, d)
        self.sigma2_:       float                = 0.0
        self.mu_:           np.ndarray | None = None   # (K, p) standardised means
        self.pi_k_:         np.ndarray | None = None   # (K,)
        self.classes_:      np.ndarray | None = None   # (K,)
        self.x_mean_:       np.ndarray | None = None   # (p,) overall mean
        self.x_std_:        np.ndarray | None = None   # (p,) within-class std
        self.loss_path_:    str                  = ""     # "ce" | "nll"
        self.diagnostics_:  dict                 = {}
        self.n_iter_:       int                  = 0
        self.loss_history_: list                 = []     # loss at each internal grad step; [0] = start-of-fit loss (before any gradient step)


    def _standardise(self, X: np.ndarray, y: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
        """Compute and apply within-class standardisation.

        Stores self.x_mean_, self.x_std_ for use at prediction time.
        Returns (X_tilde, mu_tilde).
        """
        classes = self.classes_
        n, p    = X.shape

        x_bar = X.mean(axis=0)

        # Within-class residuals → per-feature std (exact: subtract per-class mean)
        mu_hat = np.zeros((len(classes), p))
        for k, cls in enumerate(classes):
            mask = y == cls
            mu_hat[k] = X[mask].mean(axis=0)
        R = X - mu_hat[np.searchsorted(classes, y)]   # (n, p)
        s = R.std(axis=0)
        s = np.where(s < 1e-10, 1.0, s)   # avoid division by zero

        self.x_mean_ = x_bar
        self.x_std_  = s

        X_tilde   = (X   - x_bar) / s
        mu_tilde  = (mu_hat - x_bar) / s   # (K, p)
        return X_tilde, mu_tilde

    def _apply_standardise(self, X: np.ndarray) -> np.ndarray:
        """Apply stored standardisation to new data."""
        return (X - self.x_mean_) / self.x_std_


    def _diagnostics(self, X_tilde: np.ndarray,
                     mu_tilde: np.ndarray) -> tuple[float, float]:
        """Compute signal sparsity r and excess kurtosis κ.

        Returns
        -------
        r : float   effective signal dimensionality / p  ∈ (0, 1]
        kappa : float   mean |excess kurtosis| across features
        """
        # Between-class variance per feature: b_j = Σ_k π_k (μ̃_kj - x̄_j)²
        # x̄_j of standardised data ≈ 0 (mean-centred), so b_j = Σ_k π_k μ̃_kj²
        pi  = self.pi_k_
        b   = (pi[:, np.newaxis] * mu_tilde ** 2).sum(axis=0)   # (p,)
        sum_b  = b.sum()
        sum_b2 = (b ** 2).sum()
        D_eff  = (sum_b ** 2 / sum_b2) if sum_b2 > 1e-12 else 1.0
        r      = D_eff / X_tilde.shape[1]

        # Excess kurtosis: e_j = kurtosis of within-class residuals per feature
        y_idx    = np.searchsorted(self.classes_, self._y_cache_)
        R_tilde  = X_tilde - mu_tilde[y_idx]                  # (n, p)
        m2       = (R_tilde ** 2).mean(axis=0)
        m4       = (R_tilde ** 4).mean(axis=0)
        e        = np.where(m2 > 1e-12, m4 / (m2 ** 2), 3.0)  # kurtosis
        kappa    = float(np.abs(e - 3).mean())

        self.diagnostics_ = {"r": float(r), "kappa": kappa,
                              "D_eff": float(D_eff)}
        return r, kappa


    @staticmethod
    def _ce_grad(X_tilde: np.ndarray, mu_tilde: np.ndarray,
                 Y_oh: np.ndarray, L: np.ndarray,
                 pi_k: np.ndarray, sigma2: float) -> tuple[float, np.ndarray]:
        """Return (loss, grad_L) for the cross-entropy path.

        Uses the direct O(npd + nKp) formulation from eq. (3) of the paper.
        No p×p intermediate matrix is formed.
        """
        n, p = X_tilde.shape
        K, _ = mu_tilde.shape

        Z = X_tilde @ L          # (n, d)   embedded data
        W = mu_tilde @ L         # (K, d)   embedded means

        # Discriminant scores δ_k(x_i) = z_i·w_k - ½‖w_k‖² + σ²(x̃_i·μ̃_k - ½‖μ̃_k‖²) + log π_k
        scores = Z @ W.T         # (n, K)   z_i · w_k
        scores -= 0.5 * (W ** 2).sum(axis=1)[np.newaxis, :]   # - ½‖w_k‖²
        if sigma2 > 0:
            iso   = X_tilde @ mu_tilde.T   # (n, K)
            iso  -= 0.5 * (mu_tilde ** 2).sum(axis=1)[np.newaxis, :]
            scores += sigma2 * iso
        scores += np.log(np.clip(pi_k, 1e-15, 1.0))[np.newaxis, :]

        probs  = _scipy_softmax(scores, axis=1)                # (n, K)
        loss   = -float((Y_oh * np.log(np.clip(probs, 1e-15, 1.0))).sum()) / n

        # Residual matrix R_ik = Y_ik - P_ik
        R      = Y_oh - probs                                  # (n, K)

        # grad_L = -1/n ( X̃ᵀ R W  +  M̃ᵀ (Rᵀ Z)  -  M̃ᵀ diag(1ᵀR) W )
        col_sum_R = R.sum(axis=0)                              # (K,)
        grad_L    = -(X_tilde.T @ R @ W
                      + mu_tilde.T @ (R.T @ Z)
                      - mu_tilde.T @ (col_sum_R[:, np.newaxis] * W)) / n
        return loss, grad_L

    # Same formula as _ce_grad, same lr, same optimizer (sgd/adam), same
    # tol, same max_iter, same best_L tracking as the numpy loop in fit().
    # Verified against the numpy version: identical loss trajectory,
    # final L differs only by ~1e-16 (float64 rounding). Only the device
    # running the matrix multiplications changes.

    def _fit_ce_torch(self, X_tilde: np.ndarray, mu_tilde: np.ndarray,
                       Y_oh: np.ndarray, pi_k: np.ndarray, L0: np.ndarray,
                       device: str) -> tuple[np.ndarray, list, int]:
        """Run the CE gradient loop on `device` via torch. Returns
        (best_L as numpy array, loss_history list, n_iter_done)."""
        torch_dtype = torch.float32 if self._np_dtype == np.float32 else torch.float64

        Xt     = torch.as_tensor(X_tilde, dtype=torch_dtype, device=device)
        Mt     = torch.as_tensor(mu_tilde, dtype=torch_dtype, device=device)
        Y      = torch.as_tensor(Y_oh, dtype=torch_dtype, device=device)
        log_pi = torch.log(torch.clamp(
            torch.as_tensor(pi_k, dtype=torch_dtype, device=device), min=1e-15))
        L      = torch.as_tensor(L0, dtype=torch_dtype, device=device)
        n      = Xt.shape[0]

        use_adam = self.ce_optimizer.lower() == "adam"
        if use_adam:
            # Exact replica of _Adam (same beta1/beta2/eps, same bias correction)
            m = torch.zeros_like(L); v = torch.zeros_like(L); t_step = 0
            b1, b2, eps_a = 0.9, 0.999, 1e-8

        loss_history: list = []
        best_loss  = float("inf")
        best_L     = L.clone()
        iters_done = 0

        for it in range(self.max_iter):
            Z = Xt @ L
            W = Mt @ L
            scores = Z @ W.T
            scores = scores - 0.5 * (W ** 2).sum(dim=1).unsqueeze(0)
            scores = scores + log_pi.unsqueeze(0)
            probs  = torch.softmax(scores, dim=1)
            loss_t = -(Y * torch.log(torch.clamp(probs, min=1e-15))).sum() / n

            R = Y - probs
            col_sum_R = R.sum(dim=0)
            grad_L = -(Xt.T @ R @ W + Mt.T @ (R.T @ Z)
                       - Mt.T @ (col_sum_R.unsqueeze(1) * W)) / n

            if use_adam:
                t_step += 1
                m = b1 * m + (1 - b1) * grad_L
                v = b2 * v + (1 - b2) * grad_L ** 2
                m_hat = m / (1 - b1 ** t_step)
                v_hat = v / (1 - b2 ** t_step)
                L = L - self.lr * m_hat / (torch.sqrt(v_hat) + eps_a)
            else:
                L = L - self.lr * grad_L

            loss_val = float(loss_t.item())
            loss_history.append(loss_val)
            iters_done = it + 1
            if loss_val < best_loss:
                best_loss = loss_val
                best_L    = L.clone()

            grad_norm = float(torch.linalg.norm(grad_L, ord="fro").item())
            if it == 0 or (it + 1) % 50 == 0:
                logger.debug(
                    "  LDA-GO[torch:%s] step %d/%d  loss=%.6f  best=%.6f  ‖∇L‖=%.2e",
                    device, it + 1, self.max_iter, loss_val, best_loss, grad_norm,
                )
            if grad_norm < self.tol:
                logger.debug(
                    "LDA-GO[torch:%s] converged at step %d  loss=%.6f  ‖∇L‖=%.2e < tol=%.2e",
                    device, it + 1, loss_val, grad_norm, self.tol,
                )
                break

        return best_L.cpu().numpy().astype(self._np_dtype), loss_history, iters_done


    @staticmethod
    def _nll_grad(Sw_alpha: np.ndarray, L: np.ndarray,
                  sigma2: float) -> tuple[float, np.ndarray]:
        """Return (loss, grad_L) for the NLL path.

        L_NLL = ½ tr(Σ̂_w,α Σ⁻¹) - ½ log det(Σ⁻¹)
        ∂L_NLL/∂L = Σ̂_w,α L - L (σ²I_d + LᵀL)⁻¹

        The log-det term: log det(LL' + σ²I) = log det(σ²I_d + L'L) + (p-d)log σ²
        We only need the gradient w.r.t. L, not the absolute loss value,
        but we track an approximate scalar loss for convergence monitoring.
        """
        inner  = sigma2 * np.eye(L.shape[1]) + L.T @ L        # (d, d)
        inner_inv = np.linalg.inv(inner)

        grad_L = Sw_alpha @ L - L @ inner_inv

        # inner is already computed above — this reuses it, so the log-det
        # term below costs only O(d³), not another O(p²) pass.
        p, d = L.shape
        trace_term  = 0.5 * float(np.trace(Sw_alpha @ (L @ L.T + sigma2 * np.eye(p))))
        sign, logdet_inner = np.linalg.slogdet(inner)
        logdet_term = 0.5 * (logdet_inner + (p - d) * np.log(max(sigma2, 1e-30)))
        loss = trace_term - logdet_term
        return loss, grad_L


    def fit(self, X: np.ndarray, y: np.ndarray) -> LDAGO:
        """Fit LDA-GO v2 to labelled embeddings.

        Parameters
        ----------
        X : (n, p) float array
        y : (n,)   integer-castable label array. Class ids need not be
            0 .. K-1 or contiguous — classes_ is np.unique(y), and all
            lookups go through it, so arbitrary integer ids work.

        Returns
        -------
        self
        """
        rng = np.random.default_rng(self.random_state)
        X   = np.asarray(X, dtype=self._np_dtype)
        y   = np.asarray(y, dtype=int)
        n, p = X.shape

        _log_torch_gpu_info()

        self.classes_ = np.unique(y)
        K = len(self.classes_)

        # Priors
        pi_k = np.array([(y == cls).sum() / n for cls in self.classes_])
        self.pi_k_ = pi_k

        # Cache y for diagnostics
        self._y_cache_ = y

        # Step 1: within-class standardisation
        X_tilde, mu_tilde = self._standardise(X, y)
        self.mu_ = mu_tilde                         # (K, p) standardised means

        # Step 2: rank d
        d_valid = self.d is not None and 0 < self.d <= p
        if self.d is not None and not d_valid:
            logger.warning(
                "LDA-GO: d=%s is out of range (0, p=%d] — falling back to "
                "d=min(20, p)=%d instead of raising.", self.d, p, min(20, p),
            )
        d = self.d if d_valid else min(20, p)

        # Step 3: structural diagnostics → loss selection
        r, kappa = self._diagnostics(X_tilde, mu_tilde)
        use_ce   = self.force_ce or (r < 0.10) or (kappa > 10.0)
        self.loss_path_ = "ce" if use_ce else "nll"
        logger.debug(
            "LDA-GO diagnostics — sparsity r=%.4f  kurtosis κ=%.2f  "
            "→ path=%s  d=%d",
            r, kappa, self.loss_path_.upper(), d,
        )

        # Step 4: parametrisation
        sigma2 = 0.0
        Sw_alpha = None

        if not use_ce:
            # Pooled within-class covariance
            Sw = np.zeros((p, p))
            for cls in self.classes_:
                mask = y == cls
                Xk   = X_tilde[mask]
                if mask.sum() > 1:
                    Sw += np.cov(Xk, rowvar=False) * (mask.sum() - 1)
            Sw /= max(n - K, 1)

            alpha, mu_oas = _oas_shrinkage(Sw, n)
            sigma2        = 1.0 / (alpha * mu_oas + 1e-10)
            Sw_alpha      = (1 - alpha) * Sw + alpha * mu_oas * np.eye(p)
            logger.debug("LDA-GO NLL — OAS α=%.4f  σ²=%.4f", alpha, sigma2)

        # Step 5: one-hot encode
        Y_oh = np.zeros((n, K))
        for k, cls in enumerate(self.classes_):
            Y_oh[y == cls, k] = 1.0

        # Step 6: single initialisation  L⁰ = I_{p×d} + ε
        L = np.eye(p, d) + rng.standard_normal((p, d)) * 0.01

        # Step 7: gradient optimisation
        self.loss_history_ = []   # reset for this fit() call

        # Checked BEFORE launching anything on cuda (same pattern as
        # ultralytics/torch_utils.py select_device — "if torch.cuda.is_available()"
        # checked upfront, not a try/except after a cuda failure): if
        # device="cuda" but torch.cuda.is_available() is False (no NVIDIA GPU,
        # or torch installed without CUDA support), fall back to numpy/CPU
        # directly, instead of crashing in _fit_ce_torch with "Torch not
        # compiled with CUDA enabled".
        device_available = True
        if self.device is not None and TORCH_AVAILABLE:
            if str(self.device).lower().startswith("cuda") and not torch.cuda.is_available():
                device_available = False
                logger.warning(
                    "LDA-GO: device=%s requested but torch.cuda.is_available() "
                    "is False (no NVIDIA GPU detected, or torch installed "
                    "without CUDA) — falling back to numpy/CPU.", self.device,
                )

        run_on_torch = (self.device is not None) and use_ce and TORCH_AVAILABLE and device_available
        if self.device is not None and use_ce and not TORCH_AVAILABLE:
            logger.warning(
                "LDA-GO: device=%s requested but torch is not installed — "
                "falling back to numpy/CPU (pip install torch).", self.device,
            )
        if self.device is not None and not use_ce:
            logger.warning(
                "LDA-GO: device=%s ignored — NLL path selected, only the CE "
                "path is ported to torch. Using numpy/CPU.",
                self.device,
            )

        if run_on_torch:
            _log_device_once(self.device)
            best_L, self.loss_history_, iters_done = self._fit_ce_torch(
                X_tilde, mu_tilde, Y_oh, pi_k, L, self.device)
            best_loss = self.loss_history_[-1] if self.loss_history_ else np.inf
            for lv in self.loss_history_:
                if lv < best_loss:
                    best_loss = lv
        else:
            adam       = _Adam(lr=self.lr)
            best_loss  = np.inf
            best_L     = L.copy()
            iters_done = 0
            use_adam_ce = self.ce_optimizer.lower() == "adam"

            logger.debug(
                "LDA-GO opt start — path=%s  d=%d  p=%d  K=%d  max_iter=%d  lr=%s  tol=%s",
                self.loss_path_.upper(), d, p, K, self.max_iter, self.lr, self.tol,
            )

            for it in range(self.max_iter):
                if use_ce:
                    loss, grad_L = self._ce_grad(
                        X_tilde, mu_tilde, Y_oh, L, pi_k, sigma2)
                    if use_adam_ce:
                        L = L - adam.step(grad_L)
                    else:
                        L = L - self.lr * grad_L
                else:
                    loss, grad_L = self._nll_grad(Sw_alpha, L, sigma2)
                    L = L - adam.step(grad_L)

                self.loss_history_.append(float(loss))

                iters_done = it + 1
                if loss < best_loss:
                    best_loss = loss
                    best_L    = L.copy()

                grad_norm = float(np.linalg.norm(grad_L, "fro"))

                if it == 0 or (it + 1) % 50 == 0:
                    logger.debug(
                        "  LDA-GO step %d/%d  loss=%.6f  best=%.6f  ‖∇L‖=%.2e",
                        it + 1, self.max_iter, loss, best_loss, grad_norm,
                    )

                if grad_norm < self.tol:
                    logger.debug(
                        "LDA-GO converged at step %d  loss=%.6f  ‖∇L‖=%.2e < tol=%.2e",
                        it + 1, loss, grad_norm, self.tol,
                    )
                    break

        self.L_       = best_L  # best L over all gradient steps, not the last iterate
        self.sigma2_  = sigma2
        self.n_iter_  = iters_done
        logger.info(
            "LDA-GO fit — n=%d p=%d K=%d d=%d  path=%s  loss=%.6f  steps=%d",
            n, p, K, d, self.loss_path_.upper(), best_loss, iters_done,
        )
        return self


    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X into the d-dimensional discriminant subspace.

        Applies the stored within-class standardisation before projecting.
        """
        self._check_fitted()
        X_tilde = self._apply_standardise(np.asarray(X, dtype=self._np_dtype))
        return X_tilde @ self.L_

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def _scores(self, X: np.ndarray) -> np.ndarray:
        """Discriminant scores (n × K) on standardised+projected data."""
        self._check_fitted()
        X_tilde = self._apply_standardise(np.asarray(X, dtype=self._np_dtype))
        Z  = X_tilde @ self.L_                     # (n, d)
        W  = self.mu_ @ self.L_                    # (K, d)

        scores = Z @ W.T                           # (n, K)
        scores -= 0.5 * (W ** 2).sum(axis=1)[np.newaxis, :]
        if self.sigma2_ > 0:
            iso   = X_tilde @ self.mu_.T
            iso  -= 0.5 * (self.mu_ ** 2).sum(axis=1)[np.newaxis, :]
            scores += self.sigma2_ * iso
        scores += np.log(np.clip(self.pi_k_, 1e-15, 1.0))[np.newaxis, :]
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        idx = np.argmax(self._scores(X), axis=1)
        return self.classes_[idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _scipy_softmax(self._scores(X), axis=1)


    def _check_fitted(self) -> None:
        if self.L_ is None:
            raise RuntimeError("LDA-GO: call fit() before transform() / predict().")