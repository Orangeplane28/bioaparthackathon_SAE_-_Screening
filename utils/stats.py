"""
utils/stats.py — Statistical helpers shared across pipeline stages.

Used by: N4, N5, N6
"""

import numpy as np
from typing import List, Tuple, Callable, Optional
from sklearn.metrics import roc_auc_score


# ── Multiple testing correction ───────────────────────────────────────────────

def bh_correction(pvals: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg FDR correction.

    Args:
        pvals: Array of p-values
        alpha: FDR significance threshold

    Returns:
        q_values:       BH-adjusted p-values (monotone)
        is_significant: Boolean array (q <= alpha)
    """
    n = len(pvals)
    pvals = np.array(pvals, dtype=float)
    ranks = np.argsort(np.argsort(pvals)) + 1
    q_values = pvals * n / ranks
    # Make q-values monotone (cumulative min from largest p)
    sorted_idx = np.argsort(pvals)[::-1]
    q_sorted = q_values[sorted_idx]
    q_cummin = np.minimum.accumulate(q_sorted)
    q_values[sorted_idx] = q_cummin
    q_values = np.minimum(q_values, 1.0)
    return q_values, q_values <= alpha


# ── AUROC helpers ─────────────────────────────────────────────────────────────

def auroc_safe(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC, returning 0.5 on degenerate inputs."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return 0.5


def auroc_feature_vs_group(toxin_vecs: np.ndarray, control_vecs: np.ndarray,
                            feature_idx: int) -> float:
    """
    AUROC for a single SAE feature discriminating toxins (y=1) from controls (y=0).

    Args:
        toxin_vecs:   (n_tox, D_SAE) feature matrix
        control_vecs: (n_ctrl, D_SAE) feature matrix
        feature_idx:  Which feature to evaluate

    Returns:
        AUROC float in [0, 1]. Returns 0.5 if either group is empty.
    """
    if len(toxin_vecs) == 0 or len(control_vecs) == 0:
        return 0.5
    scores = np.concatenate([toxin_vecs[:, feature_idx],
                              control_vecs[:, feature_idx]])
    labels = np.concatenate([np.ones(len(toxin_vecs)),
                              np.zeros(len(control_vecs))])
    return auroc_safe(labels, scores)


def evaluate_classifier(clf, X: np.ndarray, y: np.ndarray,
                         label: str = '') -> dict:
    """
    Evaluate a sklearn classifier.

    Returns:
        dict with keys: auroc, auprc, f1
    """
    from sklearn.metrics import average_precision_score, f1_score
    if clf is None or len(X) == 0:
        return {'auroc': 0.5, 'auprc': 0.0, 'f1': 0.0}
    try:
        probs = clf.predict_proba(X)[:, 1]
        preds = clf.predict(X)
        auroc = auroc_safe(y, probs)
        auprc = average_precision_score(y, probs) if y.sum() > 0 else 0.0
        f1    = f1_score(y, preds, zero_division=0)
    except Exception:
        return {'auroc': 0.5, 'auprc': 0.0, 'f1': 0.0}
    if label:
        print(f'  {label:30s}: AUROC={auroc:.3f}, AUPRC={auprc:.3f}, F1={f1:.3f}')
    return {'auroc': float(auroc), 'auprc': float(auprc), 'f1': float(f1)}


# ── Bootstrap confidence intervals ────────────────────────────────────────────

def bootstrap_ci(values: np.ndarray,
                 stat_fn: Callable = np.mean,
                 n_boot: int = 2000,
                 ci: float = 0.95) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for stat_fn applied to values.

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    values = np.asarray(values)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    boot_stats = [
        stat_fn(np.random.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ]
    lo = np.percentile(boot_stats, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_stats, (1 + ci) / 2 * 100)
    return float(stat_fn(values)), float(lo), float(hi)


# ── Effect size ────────────────────────────────────────────────────────────────

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Pooled Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_var = ((n1 - 1) * group1.var() + (n2 - 1) * group2.var()) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var + 1e-8)
    return float((group1.mean() - group2.mean()) / pooled_std)
