"""Visualisation utilities: UMAP plots and layer-analysis curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# UMAP scatter plots
# ---------------------------------------------------------------------------

def umap_grid(
    X: np.ndarray,
    labels: np.ndarray,
    layers: list[int],
    out_path: Path,
    model_tag: str = "",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> None:
    """
    2-D UMAP scatter for selected layers, saved as a single figure.

    Parameters
    ----------
    X      : (N, n_layers, d_model)
    labels : (N,) — 1=toxic, 0=benign
    layers : list of layer indices to plot
    """
    try:
        import umap
    except ImportError as exc:
        raise ImportError("umap-learn is required: uv add umap-learn") from exc

    n = len(layers)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    colors = np.where(labels == 1, "#e74c3c", "#2980b9")
    label_names = {1: "toxic", 0: "benign"}

    for ax, layer in zip(axes, layers):
        X_layer = X[:, layer, :].astype(np.float32)
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            verbose=False,
        )
        emb = reducer.fit_transform(X_layer)
        for lbl, color, name in [(1, "#e74c3c", "toxic"), (0, "#2980b9", "benign")]:
            mask = labels == lbl
            ax.scatter(emb[mask, 0], emb[mask, 1], c=color, s=10, alpha=0.7, label=name)
        ax.set_title(f"Layer {layer}", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].legend(loc="upper left", markerscale=2, fontsize=9)
    title = f"UMAP — {model_tag}" if model_tag else "UMAP"
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved UMAP → {out_path}")


_GROUP_COLORS: dict[str, str] = {
    "snakes": "#e67e22",
    "scorpions": "#27ae60",
    "spiders": "#8e44ad",
    "bacteria": "#2980b9",
}

# When group is unknown (no groups.tsv), color by toxicity label instead
_LABEL_COLORS: dict[int, str] = {1: "#e74c3c", 0: "#2980b9"}


def umap_grid_grouped(
    X: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    layers: list[int],
    out_path: Path,
    model_tag: str = "",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> None:
    """
    UMAP grid: rows = organism groups, columns = layers.

    Each cell shows only that group's sequences: red (toxic) / blue (benign).
    Row label on the left, layer index on top.
    Falls back to a single-row layout when all groups are "unknown".
    """
    try:
        import umap
    except ImportError as exc:
        raise ImportError("umap-learn is required: uv add umap-learn") from exc

    unique_groups = [g for g in sorted(set(groups)) if g != "unknown"]
    if not unique_groups:
        unique_groups = ["unknown"]

    n_rows = len(unique_groups)
    n_cols = len(layers)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)

    # Pre-compute one UMAP embedding per layer across ALL sequences so that
    # coordinates are comparable across rows within the same column.
    embeddings: dict[int, np.ndarray] = {}
    for layer in layers:
        X_layer = X[:, layer, :].astype(np.float32)
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            verbose=False,
        )
        embeddings[layer] = reducer.fit_transform(X_layer)

    for row_idx, grp in enumerate(unique_groups):
        grp_mask = groups == grp
        for col_idx, layer in enumerate(layers):
            ax = axes[row_idx][col_idx]
            emb = embeddings[layer]

            for lbl, lbl_name, color in [
                (1, "toxic", "#e74c3c"),
                (0, "benign", "#2980b9"),
            ]:
                mask = grp_mask & (labels == lbl)
                if mask.sum() == 0:
                    continue
                ax.scatter(
                    emb[mask, 0], emb[mask, 1],
                    c=color,
                    s=12,
                    alpha=0.7,
                    label=lbl_name,
                )

            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(f"Layer {layer}", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(grp, fontsize=11, fontweight="bold")

    # Figure-level legend: collect handles from all axes to ensure toxic+benign both appear
    seen: dict[str, object] = {}
    for row in axes:
        for ax in row:
            for h, l in zip(*ax.get_legend_handles_labels()):
                seen.setdefault(l, h)
    fig.legend(
        seen.values(), seen.keys(),
        loc="lower center", ncol=len(seen), markerscale=1.5, fontsize=9,
        frameon=False, bbox_to_anchor=(0.5, -0.02),
    )

    title = f"UMAP — {model_tag}" if model_tag else "UMAP"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved grouped UMAP → {out_path}")


# ---------------------------------------------------------------------------
# Layer-analysis summary plots
# ---------------------------------------------------------------------------

def plot_auroc(
    aurocs: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """
    Line plot of linear-probing AUROC vs layer for one or more models.

    Parameters
    ----------
    aurocs : mapping model_tag → (n_layers,) array
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for tag, vals in aurocs.items():
        ax.plot(vals, marker="o", markersize=3, label=tag)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="chance")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("5-fold AUROC")
    ax.set_title("Linear probe AUROC per layer")
    ax.legend()
    ax.set_ylim(0.4, 1.02)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] Saved AUROC plot → {out_path}")


def plot_separation_ratio(
    ratios: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Line plot of Fisher-like separation ratio vs layer."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for tag, vals in ratios.items():
        ax.plot(vals, marker="o", markersize=3, label=tag)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Separation ratio (between / within)")
    ax.set_title("Centroid separation ratio per layer")
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] Saved separation-ratio plot → {out_path}")


def plot_cka(
    consecutive: dict[str, np.ndarray],
    cross_class: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Two-panel CKA plot: consecutive-layer similarity and cross-class similarity."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for tag, vals in consecutive.items():
        ax1.plot(vals, marker="o", markersize=3, label=tag)
    ax1.set_xlabel("Layer i → i+1")
    ax1.set_ylabel("CKA similarity")
    ax1.set_title("Consecutive-layer CKA\n(low = large transformation)")
    ax1.legend()
    ax1.set_ylim(0, 1.05)

    for tag, vals in cross_class.items():
        ax2.plot(vals, marker="o", markersize=3, label=tag)
    ax2.set_xlabel("Layer index")
    ax2.set_ylabel("CKA similarity")
    ax2.set_title("Cross-class CKA (toxic vs benign)\n(low = groups diverged)")
    ax2.legend()
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] Saved CKA plot → {out_path}")


def plot_summary_panel(
    aurocs: np.ndarray,
    ratio: np.ndarray,
    cka_consec: np.ndarray,
    cka_cross: np.ndarray,
    out_path: Path,
    model_tag: str = "",
) -> None:
    """Single 4-panel summary figure for one model."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    title = f"Layer analysis — {model_tag}" if model_tag else "Layer analysis"
    fig.suptitle(title, fontsize=14)

    axes[0, 0].plot(aurocs, marker="o", markersize=3, color="#e74c3c")
    axes[0, 0].axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("Linear probe AUROC")
    axes[0, 0].set_xlabel("Layer")
    axes[0, 0].set_ylabel("AUROC")
    axes[0, 0].set_ylim(0.4, 1.02)

    axes[0, 1].plot(ratio, marker="o", markersize=3, color="#2ecc71")
    axes[0, 1].set_title("Centroid separation ratio")
    axes[0, 1].set_xlabel("Layer")
    axes[0, 1].set_ylabel("between / within")

    axes[1, 0].plot(cka_consec, marker="o", markersize=3, color="#3498db")
    axes[1, 0].set_title("Consecutive-layer CKA")
    axes[1, 0].set_xlabel("Layer i → i+1")
    axes[1, 0].set_ylabel("CKA")
    axes[1, 0].set_ylim(0, 1.05)

    axes[1, 1].plot(cka_cross, marker="o", markersize=3, color="#9b59b6")
    axes[1, 1].set_title("Cross-class CKA (toxic vs benign)")
    axes[1, 1].set_xlabel("Layer")
    axes[1, 1].set_ylabel("CKA")
    axes[1, 1].set_ylim(0, 1.05)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved summary panel → {out_path}")
