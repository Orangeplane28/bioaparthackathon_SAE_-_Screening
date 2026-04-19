"""Test C — Centered Kernel Alignment (CKA) analysis.

Two uses:
  1. Consecutive-layer CKA: how much each layer transforms representations.
  2. Cross-class CKA per layer: how similar toxic vs benign representations are.
     Low value = model separates the two groups.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between representation matrices X and Y.

    Parameters
    ----------
    X, Y : (N, d) — same N, potentially different d

    Returns
    -------
    float in [0, 1] — 1 means identical representations (up to linear transform)
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    # L2-normalize each sample to unit norm (removes scale drift across layers)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)

    # Center columns
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # CKA = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    # This avoids forming N×N Gram matrices when d < N
    XtY = X.T @ Y
    XtX = X.T @ X
    YtY = Y.T @ Y

    numerator = float((XtY * XtY).sum())
    denom = float(np.sqrt((XtX * XtX).sum() * (YtY * YtY).sum()))
    return numerator / (denom + 1e-12)


def consecutive_layer_cka(X: np.ndarray) -> np.ndarray:
    """
    CKA between each consecutive pair of layers.

    Parameters
    ----------
    X : (N, n_layers, d_model)

    Returns
    -------
    (n_layers - 1,) array — value[i] = CKA(layer i, layer i+1)
    """
    n_layers = X.shape[1]
    cka_values = np.zeros(n_layers - 1)
    for i in tqdm(range(n_layers - 1), desc="Consecutive-layer CKA"):
        cka_values[i] = linear_cka(X[:, i, :], X[:, i + 1, :])
    return cka_values


def cross_class_cka(
    X: np.ndarray,
    labels: np.ndarray,
    positive_label: int = 1,
    max_samples: int = 200,
) -> np.ndarray:
    """
    CKA between toxic and benign subspaces at each layer.

    Subsamples to *max_samples* per class so the Gram matrices stay small.

    Parameters
    ----------
    X      : (N, n_layers, d_model)
    labels : (N,)

    Returns
    -------
    (n_layers,) array — lower = more diverged representations
    """
    rng = np.random.default_rng(42)
    pos_idx = np.where(labels == positive_label)[0]
    neg_idx = np.where(labels != positive_label)[0]

    n = min(max_samples, len(pos_idx), len(neg_idx))
    pos_idx = rng.choice(pos_idx, n, replace=False)
    neg_idx = rng.choice(neg_idx, n, replace=False)

    n_layers = X.shape[1]
    cka_values = np.zeros(n_layers)
    for layer in tqdm(range(n_layers), desc="Cross-class CKA"):
        cka_values[layer] = linear_cka(X[pos_idx, layer, :], X[neg_idx, layer, :])
    return cka_values
