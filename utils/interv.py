"""
utils/interv.py — Feature ablation and LEACE concept erasure helpers.

Used by: N6, N7
"""

import numpy as np
from sklearn.utils.extmath import randomized_svd


# ── Feature ablation ──────────────────────────────────────────────────────────

def ablate_features(X: np.ndarray, feature_ids: list) -> np.ndarray:
    """
    Zero out specified SAE features in a protein feature matrix.

    Args:
        X:           (n, D_SAE) feature matrix
        feature_ids: List of feature indices to ablate

    Returns:
        Copy of X with specified columns set to 0.
    """
    X_abl = X.copy()
    valid = [i for i in feature_ids if i < X.shape[1]]
    if valid:
        X_abl[:, valid] = 0.0
    return X_abl


# ── LEACE concept direction ────────────────────────────────────────────────────

def compute_leace_direction(X_pos: np.ndarray, X_neg: np.ndarray,
                             reg: float = 1e-4,
                             n_components: int = 256) -> np.ndarray:
    """
    Compute the true LEACE concept direction via Fisher Linear Discriminant Analysis.

    Unlike the simple mean-difference direction (normalize(μ_pos − μ_neg)),
    this computes d = Σ_w^{-1}(μ_pos − μ_neg), where Σ_w is the pooled
    within-class covariance. This minimises information loss during erasure.

    For D_SAE=15360, uses randomized SVD for scalability (Woodbury identity).

    Args:
        X_pos:        (n_pos, D) feature matrix for positive class (toxins)
        X_neg:        (n_neg, D) feature matrix for negative class
        reg:          Regularization added to eigenvalues (prevents division by 0)
        n_components: Number of SVD components (larger = more accurate)

    Returns:
        Unit-normalized LEACE direction vector of shape (D,)
    """
    mu_pos = X_pos.mean(axis=0)
    mu_neg = X_neg.mean(axis=0)
    diff   = mu_pos - mu_neg

    # Pooled within-class scatter matrix (centred data)
    X_c     = np.concatenate([X_pos - mu_pos, X_neg - mu_neg], axis=0)
    n_total = len(X_c)
    n_comp  = min(n_components, n_total - 1, X_c.shape[1] - 1)

    try:
        U, S, Vt = randomized_svd(X_c, n_components=n_comp, random_state=42)
        # Eigenvalues of Σ_w in SVD subspace: λ_i = S_i² / (n − 2)
        lam     = S ** 2 / max(n_total - 2, 1)
        Vt_diff = Vt @ diff
        # (Σ_w + reg·I)^{-1} diff via Woodbury identity
        d = Vt.T @ (Vt_diff / (lam + reg))
        # Out-of-subspace residual (directions not captured by top SVD components)
        d += (diff - Vt.T @ Vt_diff) / reg
    except Exception as e:
        print(f'  WARNING: SVD failed ({e}) — falling back to mean-difference direction')
        d = diff.copy()

    norm = np.linalg.norm(d)
    return d / norm if norm > 1e-8 else diff / (np.linalg.norm(diff) + 1e-8)


# ── Concept erasure ────────────────────────────────────────────────────────────

def erase_direction(X: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Project out `direction` from all rows of X (LEACE concept erasure).

    x_erased = x − (x · d̂) d̂   where d̂ is the unit LEACE direction.

    The direction should be computed via compute_leace_direction() (Fisher LDA),
    which accounts for within-class covariance. This gives minimum information
    loss while removing the specific concept direction.

    Args:
        X:         (n, D) feature matrix
        direction: (D,) concept direction (need not be unit-norm)

    Returns:
        (n, D) array with concept direction projected out.
    """
    d = direction / (np.linalg.norm(direction) + 1e-8)  # ensure unit vector
    projections = X @ d                                   # (n,) dot products
    return X - np.outer(projections, d)                   # (n, D)
