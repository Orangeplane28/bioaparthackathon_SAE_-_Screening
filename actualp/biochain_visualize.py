"""
biochain_visualize.py
=====================
Standalone visualisation module for BioChain ML Engine v3.

Usage (after training completes):
    python biochain_visualize.py                          # loads biochain_model_v3.pt + re-scores test set
    python biochain_visualize.py --results biochain_results.json   # if you saved results dict
    python biochain_visualize.py --no-rerun               # skip re-scoring, only plot saved artefacts

Outputs (all saved to ./figures/):
    01_training_curves.png          – loss + AUC per epoch
    02_roc_curve.png                – ROC with AUC annotation
    03_pr_curve.png                 – Precision-Recall with AP annotation
    04_score_distributions.png      – P(threat) histogram, toxin vs benign
    05_calibration_diagram.png      – reliability diagram + ECE
    06_adversarial_eval.png         – hard-negative mutant bar chart
    07_fragment_importance.png      – top contributing fragments (MC dropout saliency)
    08_robustness_sweep.png         – AUC vs fragment count k=2..15
    09_cv_fold_summary.png          – per-fold AUC/AP (if CV results present)
    summary_panel.png               – 2×4 composite figure (all plots tiled)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── colour palette (dark BioChain theme) ──────────────────────────────────────
BG       = "#0a0a0a"
PANEL_BG = "#111111"
RED      = "#E61919"
GREEN    = "#4AF626"
GOLD     = "#f0c040"
BLUE     = "#4da6ff"
GREY     = "#666666"
TEXT     = "#eaeaea"
DIMTEXT  = "#999999"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    "#333333",
    "axes.labelcolor":   DIMTEXT,
    "xtick.color":       GREY,
    "ytick.color":       GREY,
    "text.color":        TEXT,
    "legend.facecolor":  PANEL_BG,
    "legend.edgecolor":  "#333333",
    "grid.color":        "#222222",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "font.family":       "monospace",
    "savefig.facecolor": BG,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
})

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ax_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#333333")
    ax.tick_params(colors=GREY)
    if title:
        ax.set_title(title, color=TEXT, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=DIMTEXT)
    if ylabel:
        ax.set_ylabel(ylabel, color=DIMTEXT)
    ax.grid(True)


def _save(fig, name: str):
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓  {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 01 – Training Curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(history: dict) -> Path:
    """
    history keys: train_loss, train_auc, val_auc  (lists, one value per epoch)
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("BioChain ML v3 — Training Curves", color=TEXT, fontsize=13, y=1.02)

    # loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color=RED, lw=2, label="Train Loss")
    _ax_style(ax, "Loss per Epoch", "Epoch", "BCE Loss")
    ax.legend()

    # AUC
    ax = axes[1]
    ax.plot(epochs, history["train_auc"], color=GREEN, lw=2, label="Train AUC")
    ax.plot(epochs, history["val_auc"],   color=GOLD,  lw=2, label="Val AUC",  linestyle="--")
    ax.axhline(0.90, color=RED, linestyle=":", alpha=0.6, label="Target 0.90")
    ax.set_ylim(0.4, 1.02)
    _ax_style(ax, "AUC per Epoch", "Epoch", "ROC-AUC")
    ax.legend()

    plt.tight_layout()
    return _save(fig, "01_training_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# 02 – ROC Curve
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc(y_true: np.ndarray, y_score: np.ndarray,
             fold_curves: list[tuple] | None = None) -> Path:
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5.5))

    # per-fold curves (faint)
    if fold_curves:
        for f_fpr, f_tpr in fold_curves:
            ax.plot(f_fpr, f_tpr, color=BLUE, alpha=0.2, lw=1)

    ax.plot(fpr, tpr, color=BLUE, lw=2.5,
            label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", lw=1, label="Random")

    # mark TPR @ FPR=5%
    idx = np.searchsorted(fpr, 0.05)
    if 0 < idx < len(tpr):
        ax.axvline(0.05, color=RED, linestyle=":", alpha=0.7)
        ax.scatter([fpr[idx]], [tpr[idx]], color=RED, zorder=5)
        ax.annotate(f"TPR={tpr[idx]:.1%}\n@ FPR=5%",
                    xy=(fpr[idx], tpr[idx]),
                    xytext=(0.15, tpr[idx] - 0.08),
                    color=RED, fontsize=8,
                    arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))

    _ax_style(ax, "ROC Curve", "False Positive Rate", "True Positive Rate")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(loc="lower right")
    plt.tight_layout()
    return _save(fig, "02_roc_curve.png")


# ══════════════════════════════════════════════════════════════════════════════
# 03 – Precision-Recall Curve
# ══════════════════════════════════════════════════════════════════════════════

def plot_pr(y_true: np.ndarray, y_score: np.ndarray) -> Path:
    from sklearn.metrics import precision_recall_curve, average_precision_score

    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(rec, prec, color=GREEN, lw=2.5, label=f"AP = {ap:.3f}")
    ax.axhline(baseline, color=GREY, linestyle="--", lw=1,
               label=f"Random (prevalence={baseline:.2f})")
    _ax_style(ax, "Precision-Recall Curve", "Recall", "Precision")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(loc="upper right")
    plt.tight_layout()
    return _save(fig, "03_pr_curve.png")


# ══════════════════════════════════════════════════════════════════════════════
# 04 – Score Distributions
# ══════════════════════════════════════════════════════════════════════════════

def plot_score_distributions(y_true: np.ndarray, y_score: np.ndarray,
                              threshold: float = 0.5) -> Path:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    bins = np.linspace(0, 1, 35)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(neg, bins=bins, color=GREEN, alpha=0.65, label=f"Benign  (n={len(neg)})",  density=True)
    ax.hist(pos, bins=bins, color=RED,   alpha=0.65, label=f"Toxin   (n={len(pos)})",  density=True)
    ax.axvline(threshold, color=GOLD, linestyle="--", lw=1.5, label=f"Threshold {threshold}")
    _ax_style(ax, "P(threat) Score Distributions", "P(threat)", "Density")
    ax.legend()
    plt.tight_layout()
    return _save(fig, "04_score_distributions.png")


# ══════════════════════════════════════════════════════════════════════════════
# 05 – Calibration Diagram
# ══════════════════════════════════════════════════════════════════════════════

def plot_calibration(y_true: np.ndarray, y_score: np.ndarray,
                     n_bins: int = 10) -> Path:
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centres, mean_pred, frac_pos, counts = [], [], [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_score >= lo) & (y_score < hi)
        if mask.sum() == 0:
            continue
        bin_centres.append((lo + hi) / 2)
        mean_pred.append(y_score[mask].mean())
        frac_pos.append(y_true[mask].mean())
        counts.append(mask.sum())

    mean_pred = np.array(mean_pred)
    frac_pos  = np.array(frac_pos)
    counts    = np.array(counts)
    ece = (np.abs(frac_pos - mean_pred) * counts / counts.sum()).sum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # reliability diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", lw=1, label="Perfect")
    ax.plot(mean_pred, frac_pos, "o-", color=BLUE, lw=2, ms=6, label=f"Model (ECE={ece:.4f})")
    _ax_style(ax, "Reliability Diagram", "Mean Predicted P(threat)", "Fraction Positive")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend()

    # count histogram
    ax = axes[1]
    ax.bar(mean_pred, counts, width=0.08, color=BLUE, alpha=0.7, edgecolor=PANEL_BG)
    _ax_style(ax, "Bin Sample Counts", "Mean Predicted P(threat)", "Count")

    fig.suptitle(f"Calibration  |  ECE = {ece:.4f}", color=TEXT, fontsize=12)
    plt.tight_layout()
    return _save(fig, "05_calibration_diagram.png")


# ══════════════════════════════════════════════════════════════════════════════
# 06 – Adversarial Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def plot_adversarial(adv_results: dict) -> Path:
    """
    adv_results expected keys:
      "per_mutant": list of dicts with keys "name", "score", "label"
      "fpr_on_mutants": float
    """
    per_mutant = adv_results.get("per_mutant", [])

    if not per_mutant:
        # Build a minimal demo if keys are missing
        per_mutant = [
            {"name": "Ricin E177Q",        "score": adv_results.get("ricin_e177q", 0.62),     "label": 0},
            {"name": "CRM197 (DT E148S)",  "score": adv_results.get("crm197",     0.71),      "label": 0},
            {"name": "Anthrax LF E687C",   "score": adv_results.get("anthrax_lf", 0.58),      "label": 0},
        ]

    names  = [m["name"]  for m in per_mutant]
    scores = [m["score"] for m in per_mutant]
    colours = [RED if s > 0.5 else GREEN for s in scores]

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.4), 4.5))
    bars = ax.bar(names, scores, color=colours, edgecolor=PANEL_BG, width=0.55)
    ax.axhline(0.5, color=GOLD, linestyle="--", lw=1.5, label="Threshold 0.5")

    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                s + 0.02, f"{s:.3f}",
                ha="center", va="bottom", color=TEXT, fontsize=8)

    fpr = adv_results.get("fpr_on_mutants", np.mean([s > 0.5 for s in scores]))
    _ax_style(ax,
              f"Adversarial Evaluation — Neutralised Toxin Mutants\n"
              f"FPR on mutants = {fpr:.1%}  (>50% = resists single-residue evasion)",
              "Mutant", "P(threat)")
    ax.set_ylim(0, 1.1)
    ax.legend()

    fp_patch = mpatches.Patch(color=RED,   label="Flagged as threat (FP)")
    tn_patch = mpatches.Patch(color=GREEN, label="Correctly classified benign (TN)")
    ax.legend(handles=[fp_patch, tn_patch], loc="upper right")

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    return _save(fig, "06_adversarial_eval.png")


# ══════════════════════════════════════════════════════════════════════════════
# 07 – Fragment Importance (MC-Dropout Saliency)
# ══════════════════════════════════════════════════════════════════════════════

def plot_fragment_importance(importances: list[float],
                              labels: list[str] | None = None) -> Path:
    importances = np.array(importances)
    n = len(importances)
    labels = labels or [f"Frag {i+1}" for i in range(n)]

    order = np.argsort(importances)[::-1]
    imp_sorted   = importances[order]
    label_sorted = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(7, n * 0.9), 4.5))
    colours = [RED if v > 0.5 else (GOLD if v > 0.3 else GREEN) for v in imp_sorted]
    ax.bar(label_sorted, imp_sorted, color=colours, edgecolor=PANEL_BG, width=0.6)
    ax.axhline(0.5, color=GOLD, linestyle="--", lw=1, alpha=0.7)
    _ax_style(ax, "Fragment Contribution to Threat Score",
              "Fragment", "Importance (mean MC-dropout ΔP)")
    ax.set_ylim(0, max(1.0, imp_sorted.max() * 1.15))
    plt.xticks(rotation=30, ha="right", fontsize=7)
    plt.tight_layout()
    return _save(fig, "07_fragment_importance.png")


# ══════════════════════════════════════════════════════════════════════════════
# 08 – Fragment-Count Robustness Sweep
# ══════════════════════════════════════════════════════════════════════════════

def plot_robustness(robustness: dict) -> Path:
    """
    Accepts either:
      {k (int or str): {"auc": float, "ap"?: float}, ...}
      {"by_fragment_count": [{"n_frags": k, "auc": ..., "ap"?: ...}, ...]}
    """
    if "by_fragment_count" in robustness and isinstance(
        robustness["by_fragment_count"], list
    ):
        robustness = {
            row["n_frags"]: {k: v for k, v in row.items() if k != "n_frags"}
            for row in robustness["by_fragment_count"]
        }
    ks   = sorted(int(k) for k in robustness)
    aucs = [robustness[str(k)]["auc"] if str(k) in robustness
            else robustness[k]["auc"] for k in ks]
    aps  = [robustness[str(k)].get("ap", np.nan) if str(k) in robustness
            else robustness[k].get("ap", np.nan) for k in ks]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ks, aucs, "o-", color=BLUE,  lw=2, ms=6, label="AUC")
    if not all(np.isnan(aps)):
        ax.plot(ks, aps, "s--", color=GOLD, lw=2, ms=5, label="AP")
    ax.axhline(0.9, color=RED, linestyle=":", alpha=0.6, label="Target 0.90")
    ax.set_xticks(ks)
    ax.set_ylim(0.4, 1.02)
    _ax_style(ax, "Fragment-Count Robustness Sweep", "Number of Fragments (k)", "Score")
    ax.legend()
    plt.tight_layout()
    return _save(fig, "08_robustness_sweep.png")


# ══════════════════════════════════════════════════════════════════════════════
# 09 – Cross-Validation Fold Summary
# ══════════════════════════════════════════════════════════════════════════════

def plot_cv_summary(fold_metrics: list[dict]) -> Path:
    """
    fold_metrics: list of dicts with keys auc, ap, tpr_at_fpr5
    """
    n = len(fold_metrics)
    aucs = [m.get("auc", 0) for m in fold_metrics]
    aps  = [m.get("ap",  0) for m in fold_metrics]
    tprs = [m.get("tpr_at_fpr5", m.get("tpr_fpr5", 0)) for m in fold_metrics]

    x = np.arange(n)
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(7, n * 1.4), 4.5))

    ax.bar(x - w, aucs, w, color=BLUE,  label="AUC",         edgecolor=PANEL_BG)
    ax.bar(x,     aps,  w, color=GREEN, label="AP",          edgecolor=PANEL_BG)
    ax.bar(x + w, tprs, w, color=GOLD,  label="TPR@FPR=5%",  edgecolor=PANEL_BG)

    # mean lines
    for vals, col in [(aucs, BLUE), (aps, GREEN), (tprs, GOLD)]:
        ax.axhline(np.mean(vals), color=col, linestyle="--", lw=1, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i+1}" for i in range(n)])
    ax.set_ylim(0, 1.12)
    _ax_style(ax, f"5-Fold CV Summary  |  AUC {np.mean(aucs):.3f}±{np.std(aucs):.3f}  "
                  f"AP {np.mean(aps):.3f}±{np.std(aps):.3f}",
              "Fold", "Score")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return _save(fig, "09_cv_fold_summary.png")


# ══════════════════════════════════════════════════════════════════════════════
# Summary Panel (2×4 tile)
# ══════════════════════════════════════════════════════════════════════════════

def build_summary_panel(saved_paths: list[Path]) -> Path:
    """Tile up to 8 individual plots into one composite figure."""
    paths = [p for p in saved_paths if p.exists()][:8]
    if not paths:
        return None

    from PIL import Image as PILImage

    imgs = [PILImage.open(p) for p in paths]
    # normalise heights
    target_h = 420
    resized = []
    for im in imgs:
        ratio = target_h / im.height
        resized.append(im.resize((int(im.width * ratio), target_h), PILImage.LANCZOS))

    cols = 4
    rows = (len(resized) + cols - 1) // cols
    # pad to full grid
    while len(resized) < rows * cols:
        resized.append(PILImage.new("RGB", (resized[0].width, target_h), color=(10, 10, 10)))

    row_imgs = []
    for r in range(rows):
        chunk = resized[r * cols: (r + 1) * cols]
        row_w = sum(im.width for im in chunk)
        row_img = PILImage.new("RGB", (row_w, target_h), color=(10, 10, 10))
        x = 0
        for im in chunk:
            row_img.paste(im, (x, 0))
            x += im.width
        row_imgs.append(row_img)

    total_h = target_h * rows
    total_w = row_imgs[0].width
    panel = PILImage.new("RGB", (total_w, total_h), color=(10, 10, 10))
    for r, row_img in enumerate(row_imgs):
        panel.paste(row_img, (0, r * target_h))

    out = FIGURES_DIR / "summary_panel.png"
    panel.save(out, dpi=(150, 150))
    print(f"  ✓  {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Re-score test set from checkpoint
# ══════════════════════════════════════════════════════════════════════════════

def rescore_from_checkpoint(ckpt_path: str = "biochain_model_v3.pt"):
    """
    Loads the checkpoint, rebuilds dataset, and returns (y_true, y_score, run_info).
    Falls back to None if biochain_ml_v3 is not importable.
    """
    try:
        import torch
        from biochain_ml_v3 import (
            build_dataset_from_uniprot,
            make_encoder,
            make_loader,
            SetTransformerClassifier,
            DEVICE,
        )
    except ImportError as e:
        print(f"  [warn] Cannot import biochain_ml_v3: {e}")
        return None, None, None

    if not Path(ckpt_path).exists():
        print(f"  [warn] Checkpoint not found: {ckpt_path}")
        return None, None, None

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    arch = ckpt.get("architecture", {})
    embed_dim = ckpt.get("embed_dim", arch.get("input_dim", 320))
    enc_name  = ckpt.get("encoder_name", "ESM-2 8M")

    use_esm3 = "ESM-3" in enc_name
    use_esm2 = "ESM-2" in enc_name and not use_esm3
    encoder, _ = make_encoder(use_esm3=use_esm3, use_esm2=use_esm2)

    model = SetTransformerClassifier(
        input_dim=embed_dim,
        d=arch.get("d", 128),
        h=arch.get("h", 4),
        m=arch.get("m", 8),
        n_isab=arch.get("n_isab", 2),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("  Building dataset from UniProt cache…")
    dataset = build_dataset_from_uniprot(random_seed=42, cache_path="uniprot_cache.json", n_augments=2)

    from sklearn.model_selection import GroupShuffleSplit
    groups = [fs.group_id for fs in dataset]
    labels = [fs.label  for fs in dataset]
    spl = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    _, te_idx = next(spl.split(dataset, labels, groups))
    test_data = [dataset[i] for i in te_idx]

    loader = make_loader(test_data, encoder, batch_size=16, shuffle=False)

    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            embs, lbls, mask = batch   # collate_fn returns (padded, labels, mask)
            embs = embs.to(DEVICE)
            _, probs = model(embs)
            all_scores.extend(probs.cpu().float().numpy().tolist())
            all_labels.extend(lbls.float().numpy().tolist())

    run_info = ckpt.get("run_info", {})
    return np.array(all_labels), np.array(all_scores), run_info


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="BioChain ML v3 — Visualization Suite")
    parser.add_argument("--checkpoint",  default="biochain_model_v3.pt",
                        help="Path to saved model checkpoint (default: biochain_model_v3.pt)")
    parser.add_argument("--results",     default=None,
                        help="Optional path to a JSON results file (biochain_results.json)")
    parser.add_argument("--no-rerun",    action="store_true",
                        help="Skip re-scoring the test set; only plot data in --results")
    args = parser.parse_args()

    print("\n━━━  BioChain Visualisation Suite  ━━━\n")

    # ── Load from JSON results if provided ───────────────────────────────────
    results: dict[str, Any] = {}
    if args.results and Path(args.results).exists():
        with open(args.results) as f:
            results = json.load(f)
        print(f"  Loaded results from {args.results}")

    # ── Re-score from checkpoint unless --no-rerun ────────────────────────────
    y_true = y_score = None
    run_info = results.get("run_info", {})

    if not args.no_rerun:
        print("  Re-scoring test set from checkpoint…")
        y_true, y_score, run_info_ckpt = rescore_from_checkpoint(args.checkpoint)
        if run_info_ckpt:
            run_info.update(run_info_ckpt)
    else:
        # Try to read scores stored in results JSON
        if "y_true" in results and "y_score" in results:
            y_true  = np.array(results["y_true"])
            y_score = np.array(results["y_score"])

    # Fall back to synthetic demo data so plots always render
    if y_true is None or y_score is None:
        print("  [demo] Generating synthetic scores for plot demonstration…")
        rng = np.random.default_rng(42)
        n_pos, n_neg = 60, 120
        y_true  = np.array([1]*n_pos + [0]*n_neg)
        y_score = np.concatenate([
            rng.beta(8, 2, n_pos),   # toxins → high scores
            rng.beta(2, 8, n_neg),   # benign → low scores
        ])

    # ── Plot collection ───────────────────────────────────────────────────────
    saved: list[Path] = []

    # 1. Training curves
    history = run_info.get("history", results.get("history", {}))
    if history and "train_loss" in history:
        print("  Plotting training curves…")
        saved.append(plot_training_curves(history))
    else:
        print("  [skip] No training history found — skipping training curves")

    # 2. ROC
    print("  Plotting ROC curve…")
    fold_curves = results.get("fold_roc_curves", None)
    saved.append(plot_roc(y_true, y_score, fold_curves=fold_curves))

    # 3. PR
    print("  Plotting PR curve…")
    saved.append(plot_pr(y_true, y_score))

    # 4. Score distributions
    print("  Plotting score distributions…")
    saved.append(plot_score_distributions(y_true, y_score))

    # 5. Calibration
    print("  Plotting calibration diagram…")
    saved.append(plot_calibration(y_true, y_score))

    # 6. Adversarial
    adv = run_info.get("adversarial", results.get("adversarial", {}))
    if adv:
        print("  Plotting adversarial evaluation…")
        saved.append(plot_adversarial(adv))
    else:
        print("  [demo] Generating adversarial demo data…")
        demo_adv = {
            "fpr_on_mutants": 0.60,
            "per_mutant": [
                {"name": "Ricin E177Q",       "score": 0.812, "label": 0},
                {"name": "CRM197 (DT E148S)", "score": 0.763, "label": 0},
                {"name": "Anthrax LF E687C",  "score": 0.698, "label": 0},
                {"name": "SEB Y89A",          "score": 0.341, "label": 0},
                {"name": "Aerolysin D434G",   "score": 0.290, "label": 0},
            ],
        }
        saved.append(plot_adversarial(demo_adv))

    # 7. Fragment importance
    fragment_imp = run_info.get("fragment_importances", results.get("fragment_importances", None))
    if fragment_imp:
        print("  Plotting fragment importance…")
        imps   = [fi.get("importance", fi) for fi in fragment_imp] if isinstance(fragment_imp[0], dict) else fragment_imp
        labels = [fi.get("sequence", f"Frag {i+1}")[:18]+"…" for i, fi in enumerate(fragment_imp)] \
                 if isinstance(fragment_imp[0], dict) else None
        saved.append(plot_fragment_importance(imps, labels))
    else:
        print("  [demo] Generating fragment importance demo…")
        rng = np.random.default_rng(7)
        demo_imp = sorted(rng.uniform(0.1, 0.95, 8).tolist(), reverse=True)
        saved.append(plot_fragment_importance(demo_imp))

    # 8. Robustness sweep
    robustness = run_info.get("robustness", results.get("robustness", {}))
    if robustness:
        print("  Plotting fragment-count robustness sweep…")
        saved.append(plot_robustness(robustness))
    else:
        print("  [demo] Generating robustness demo…")
        demo_rob = {k: {"auc": 0.72 + 0.03*i, "ap": 0.64 + 0.03*i}
                    for i, k in enumerate([2, 3, 5, 8, 10, 15])}
        saved.append(plot_robustness(demo_rob))

    # 9. CV summary
    fold_metrics = run_info.get("fold_metrics", results.get("fold_metrics", []))
    if fold_metrics:
        print("  Plotting CV fold summary…")
        saved.append(plot_cv_summary(fold_metrics))

    # ── Summary panel ─────────────────────────────────────────────────────────
    try:
        print("  Building summary panel…")
        build_summary_panel(saved)
    except ImportError:
        print("  [skip] Pillow not installed — skipping summary panel tile")
        print("         Install with:  pip install Pillow --break-system-packages")
    except Exception as e:
        print(f"  [warn] Summary panel failed: {e}")

    print(f"\n  All figures saved to ./{FIGURES_DIR}/\n")


if __name__ == "__main__":
    main()
