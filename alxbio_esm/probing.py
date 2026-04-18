"""Test A — Linear probing AUROC per layer.

Trains a logistic regression on each layer's mean-pooled representation
and measures 5-fold cross-validated AUROC.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tqdm import tqdm


def linear_probe_auroc(
    X: np.ndarray,
    labels: np.ndarray,
    C: float = 0.1,
    cv: int = 5,
    max_iter: int = 1000,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Compute AUROC for a logistic regression probe at each layer.

    Parameters
    ----------
    X      : (N, n_layers, d_model)
    labels : (N,)

    Returns
    -------
    auroc : (n_layers,) mean cross-validated AUROC
    """
    n_layers = X.shape[1]
    aurocs = np.zeros(n_layers)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    for layer in tqdm(range(n_layers), desc="Linear probing"):
        X_layer = X[:, layer, :]  # (N, d_model)
        clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight="balanced",
            solver="lbfgs",
        )
        scores = cross_val_score(clf, X_layer, labels, cv=skf, scoring="roc_auc")
        aurocs[layer] = scores.mean()

    return aurocs
