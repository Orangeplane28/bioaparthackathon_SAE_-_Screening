"""Test B — Centroid distance and within-group variance per layer.

Computes a Fisher-like separation ratio: between-class centroid distance
divided by the sum of within-class mean distances.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm


def separation_ratio(
    X: np.ndarray,
    labels: np.ndarray,
    positive_label: int = 1,
) -> dict[str, np.ndarray]:
    """
    Compute per-layer geometric separation statistics.

    Parameters
    ----------
    X      : (N, n_layers, d_model)
    labels : (N,)

    Returns
    -------
    dict with keys:
        between_dist     : (n_layers,) L2 distance between class centroids
        within_positive  : (n_layers,) mean within-class distance for positive class
        within_negative  : (n_layers,) mean within-class distance for negative class
        ratio            : (n_layers,) between / (within_pos + within_neg)
    """
    n_layers = X.shape[1]
    pos_mask = labels == positive_label
    neg_mask = ~pos_mask
    X_pos = X[pos_mask]  # (N_pos, n_layers, d_model)
    X_neg = X[neg_mask]

    between = np.zeros(n_layers)
    within_pos = np.zeros(n_layers)
    within_neg = np.zeros(n_layers)

    for layer in tqdm(range(n_layers), desc="Centroid geometry"):
        c_pos = X_pos[:, layer, :].mean(0)
        c_neg = X_neg[:, layer, :].mean(0)

        between[layer] = float(np.linalg.norm(c_pos - c_neg))
        within_pos[layer] = float(
            np.mean(np.linalg.norm(X_pos[:, layer, :] - c_pos, axis=1))
        )
        within_neg[layer] = float(
            np.mean(np.linalg.norm(X_neg[:, layer, :] - c_neg, axis=1))
        )

    ratio = between / (within_pos + within_neg + 1e-12)

    return {
        "between_dist": between,
        "within_positive": within_pos,
        "within_negative": within_neg,
        "ratio": ratio,
    }
