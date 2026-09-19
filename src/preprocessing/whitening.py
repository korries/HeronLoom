"""
Soft-ZCA whitening — implémentation unique partagée par tous les modules.

Corrige l'anisotropie des embeddings LLM (ex: Qwen3) avant UMAP/clustering
(Diera et al., 2024). Chaque module appelant (k_estimator_hdbscan.py,
k_estimator_gmm.py, adr.py) garde sa propre clé de config indépendante :

    hdbscan:
      soft_zca_input_space: true | false
      soft_zca_eps:         0.01

    clustering:
      gmm_input_space: "raw" | "soft_zca"
      gmm_zca_eps:     0.01

    adr:
      soft_zca_input_space: true | false
      soft_zca_eps:         0.01

Chaque section peut donc être activée/désactivée indépendamment des autres.
"""

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


def soft_zca(X: np.ndarray, eps: float = 0.01) -> np.ndarray:
    """Soft-ZCA whitening : corrige l'anisotropie tout en préservant l'orientation.

    W_ZCA = U (Λ + ε·I)^{-1/2} U^T   (Diera et al., 2024)

    Args:
        X:   matrice (n, d) — embeddings (L2-normalisés recommandé en entrée).
        eps: régulariseur eigenvalue (papier recommande 0.01 ou 0.1).
             0.0 = ZCA standard, valeurs plus grandes = whitening plus doux.

    Returns:
        X_w: matrice (n, d) whitened, NON re-normalisée.
             Appeler L2-normalisation après si nécessaire.
    """
    X = X - X.mean(axis=0)                        # centrage
    cov = np.cov(X, rowvar=False).astype(np.float64)
    U, lam, _ = np.linalg.svd(cov)                # SVD de la covariance
    W = U @ np.diag(1.0 / np.sqrt(lam + eps)) @ U.T
    return (X @ W.T).astype(np.float32)


def maybe_apply_soft_zca(
    X: np.ndarray,
    cfg_section: dict,
    enable_key: str = "soft_zca_input_space",
    eps_key: str = "soft_zca_eps",
    log_prefix: str = "",
) -> np.ndarray:
    """Applique Soft-ZCA + L2-renorm si cfg_section[enable_key] = true.

    Wrapper de config autour de soft_zca() : lit le flag d'activation et
    l'epsilon dans la sous-section de config passée par l'appelant, puis
    fait whitening + renormalisation L2 (les vecteurs ne sont plus sur la
    sphère unité après whitening — nécessaire si la suite utilise metric=cosine).

    Args:
        X:           embeddings (n, d).
        cfg_section: sous-section de config du module appelant
                      (ex: cfg["hdbscan"], cfg["adr"], cfg["clustering"]).
        enable_key:  nom de la clé bool d'activation dans cfg_section.
        eps_key:     nom de la clé eps dans cfg_section.
        log_prefix:  préfixe de log (ex: "HDBSCAN", "[ADR]", "[K-Estimator]").

    Returns:
        X inchangé si désactivé, sinon X whitened + L2-renormalisé.
    """
    if not cfg_section.get(enable_key, False):
        return X

    eps = float(cfg_section.get(eps_key, 0.01))
    logger.info("%sSoft-ZCA whitening  eps=%s", f"{log_prefix} " if log_prefix else "", eps)

    X_w = soft_zca(X, eps=eps)
    norms = np.linalg.norm(X_w, axis=1, keepdims=True)
    X_w = X_w / np.where(norms == 0, 1.0, norms)

    logger.info("%sSoft-ZCA + L2-renorm done  shape=%s", f"{log_prefix} " if log_prefix else "", X_w.shape)
    return X_w
