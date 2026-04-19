#!/usr/bin/env python
"""Step 3: Run all three representation-shift tests and generate figures.

Tests:
  A — Linear probing AUROC per layer
  B — Centroid separation ratio per layer
  C — CKA (consecutive layers + cross-class per layer)
  UMAP — 2-D scatter at layers 0, 8, 16, final
"""

import argparse
import json
from pathlib import Path

import numpy as np

from alxbio_esm.cka import consecutive_layer_cka, cross_class_cka
from alxbio_esm.embed import load_embeddings
from alxbio_esm.geometry import separation_ratio
from alxbio_esm.probing import linear_probe_auroc
from alxbio_esm.viz import (
    plot_summary_panel,
    umap_grid,
    umap_grid_grouped,
)

import warnings

warnings.filterwarnings("ignore")


def _load_groups(data_dir: Path | None, ids: list[str]) -> np.ndarray | None:
    """Return group array aligned to ids, or None if no groups.tsv found."""
    if data_dir is None:
        return None
    groups_path = Path(data_dir) / "groups.tsv"
    if not groups_path.exists():
        return None
    group_map: dict[str, str] = {}
    with open(groups_path) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                group_map[parts[0]] = parts[1]
    groups = np.array([group_map.get(uid, "unknown") for uid in ids])
    return groups


def run_for_model(
    emb_dir: Path,
    results_dir: Path,
    umap_layers: list[int],
    data_dir: Path | None = None,
) -> dict:
    model_tag = emb_dir.name
    print(f"\n{'=' * 60}\nAnalysing: {model_tag}\n{'=' * 60}")

    ids, X, labels = load_embeddings(emb_dir)
    n_layers = X.shape[1]

    # ---- Test A: linear probing ----
    print("\n[A] Linear probing …")
    aurocs = linear_probe_auroc(X, labels)

    # ---- Test B: centroid geometry ----
    print("\n[B] Centroid geometry …")
    geom = separation_ratio(X, labels)

    # ---- Test C: CKA ----
    print("\n[C] CKA …")
    cka_consec = consecutive_layer_cka(X)
    cka_cross = cross_class_cka(X, labels)

    # ---- UMAP ----
    print("\n[UMAP] Generating UMAP grid …")
    valid_umap_layers = sorted(set(min(l, n_layers - 1) for l in umap_layers))
    groups = _load_groups(data_dir, ids)
    if groups is not None:
        umap_grid_grouped(
            X,
            labels,
            groups,
            valid_umap_layers,
            out_path=results_dir / "figures" / f"umap_{model_tag}.png",
            model_tag=model_tag,
        )
    else:
        umap_grid(
            X,
            labels,
            valid_umap_layers,
            out_path=results_dir / "figures" / f"umap_{model_tag}.png",
            model_tag=model_tag,
        )

    # ---- Summary figure ----
    plot_summary_panel(
        aurocs=aurocs,
        ratio=geom["ratio"],
        cka_consec=cka_consec,
        cka_cross=cka_cross,
        out_path=results_dir / "figures" / f"summary_{model_tag}.png",
        model_tag=model_tag,
    )

    # ---- Save numeric results ----
    out_npz = results_dir / f"metrics_{model_tag}.npz"
    np.savez_compressed(
        out_npz,
        aurocs=aurocs,
        between_dist=geom["between_dist"],
        within_positive=geom["within_positive"],
        within_negative=geom["within_negative"],
        separation_ratio=geom["ratio"],
        cka_consecutive=cka_consec,
        cka_cross_class=cka_cross,
        labels=labels,
    )
    print(f"[analyze] Metrics saved → {out_npz}")

    summary = {
        "model": model_tag,
        "n_proteins": len(ids),
        "n_layers": int(n_layers),
        "peak_auroc_layer": int(np.argmax(aurocs)),
        "peak_auroc": float(np.max(aurocs)),
        "final_layer_auroc": float(aurocs[-1]),
        "peak_separation_ratio_layer": int(np.argmax(geom["ratio"])),
        "peak_separation_ratio": float(np.max(geom["ratio"])),
        "min_cross_class_cka_layer": int(np.argmin(cka_cross)),
        "min_cross_class_cka": float(np.min(cka_cross)),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emb-dirs",
        nargs="+",
        default=None,
        help="One or more embedding directories (model subdirs inside data/embeddings/). "
        "If omitted, auto-discovers all subdirectories of --emb-root.",
    )
    parser.add_argument("--emb-root", default="data/embeddings")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--umap-layers",
        nargs="+",
        type=int,
        default=[0, 8, 16, 33],
        help="Layer indices for UMAP visualisation",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Data directory containing groups.tsv (e.g. data/raw_eukaryotic). "
             "When present, UMAP is colored by organism group.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir) if args.data_dir else None

    if args.emb_dirs:
        emb_dirs = [Path(d) for d in args.emb_dirs]
    else:
        emb_root = Path(args.emb_root)
        emb_dirs = [d for d in sorted(emb_root.iterdir()) if d.is_dir()]

    if not emb_dirs:
        print("No embedding directories found. Run 02_embed.py first.")
        return

    all_summaries = []
    for emb_dir in emb_dirs:
        summary = run_for_model(emb_dir, results_dir, args.umap_layers, data_dir=data_dir)
        all_summaries.append(summary)

    summary_path = results_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n[analyze] Summary → {summary_path}")

    print("\n=== Results ===")
    for s in all_summaries:
        print(
            f"{s['model']:40s}  "
            f"peak AUROC={s['peak_auroc']:.3f} @ layer {s['peak_auroc_layer']:2d}  "
            f"final AUROC={s['final_layer_auroc']:.3f}  "
            f"sep_ratio={s['peak_separation_ratio']:.3f} @ layer {s['peak_separation_ratio_layer']:2d}"
        )


if __name__ == "__main__":
    main()
