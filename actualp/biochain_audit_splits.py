"""
BioChain — CV Split Auditor
============================
Prints a detailed breakdown of what goes into train vs. test for each
5-fold CV split, without loading any encoder or running any training.

Shows:
  - Total samples per split
  - Positive (toxin) families/organisms: count in train vs. test
  - Negative (benign) groups: count in train vs. test
  - Leakage check: any group appearing in both splits
  - Held-out marker on families never seen in training

Usage:
    python biochain_audit_splits.py
    python biochain_audit_splits.py --cache uniprot_cache.json --folds 5 --seed 42
"""

import argparse
import sys
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold

# Import dataset builder from the main pipeline so definitions stay in sync
from biochain_ml_v3 import build_dataset_from_uniprot

# ─────────────────────────────────────────────────────────────────────────────

def audit_splits(dataset, n_folds: int = 5, seed: int = 42):
    groups = np.array([fs.group_id for fs in dataset])
    labels = np.array([fs.label    for fs in dataset])
    skf    = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    W = 68   # table width

    print("\n" + "=" * W)
    print(f"  CV SPLIT AUDIT  —  {n_folds} folds  |  {len(dataset)} total samples")
    n_pos = int(labels.sum()); n_neg = len(labels) - n_pos
    print(f"  Full dataset: {n_pos} positives (toxin)  +  {n_neg} negatives (benign)")
    print("=" * W)

    all_fold_logs = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(dataset, labels, groups)):

        tr_data = [dataset[i] for i in tr_idx]
        te_data = [dataset[i] for i in te_idx]

        # ── Leakage check ────────────────────────────────────────────────────
        te_groups = set(dataset[i].group_id for i in te_idx)
        tr_groups = set(dataset[i].group_id for i in tr_idx)
        overlap   = te_groups & tr_groups

        # ── Per-group counts ─────────────────────────────────────────────────
        tr_pos = Counter(dataset[i].group_id for i in tr_idx if dataset[i].label == 1)
        tr_neg = Counter(dataset[i].group_id for i in tr_idx if dataset[i].label == 0)
        te_pos = Counter(dataset[i].group_id for i in te_idx if dataset[i].label == 1)
        te_neg = Counter(dataset[i].group_id for i in te_idx if dataset[i].label == 0)

        tr_pos_total = sum(tr_pos.values())
        tr_neg_total = sum(tr_neg.values())
        te_pos_total = sum(te_pos.values())
        te_neg_total = sum(te_neg.values())

        # ── Header ───────────────────────────────────────────────────────────
        print(f"\n{'━' * W}")
        print(f"  FOLD {fold+1} / {n_folds}")
        print(f"{'━' * W}")

        if overlap:
            print(f"  ⚠  LEAKAGE WARNING — {len(overlap)} group(s) in BOTH splits:")
            for g in sorted(overlap):
                print(f"       • {g}")
        else:
            print(f"  ✓  No group overlap between train and test")

        # ── Totals table ─────────────────────────────────────────────────────
        print(f"\n  {'':32s}  {'TRAIN':>8}  {'TEST':>8}  {'TOTAL':>8}")
        print(f"  {'─'*60}")
        print(f"  {'All samples':32s}  {len(tr_data):>8}  {len(te_data):>8}  {len(tr_data)+len(te_data):>8}")
        print(f"  {'Positives  (toxin)':32s}  {tr_pos_total:>8}  {te_pos_total:>8}  {tr_pos_total+te_pos_total:>8}")
        print(f"  {'Negatives  (benign)':32s}  {tr_neg_total:>8}  {te_neg_total:>8}  {tr_neg_total+te_neg_total:>8}")

        # ── Positive groups ───────────────────────────────────────────────────
        print(f"\n  POSITIVES — toxin families / organisms")
        print(f"  {'Group':34s}  {'Train':>6}  {'Test':>6}  {'Note'}")
        print(f"  {'·' * 60}")
        all_pos_groups = sorted(
            set(list(tr_pos) + list(te_pos)),
            key=lambda g: -(tr_pos.get(g, 0) + te_pos.get(g, 0))
        )
        for g in all_pos_groups:
            tr_n = tr_pos.get(g, 0)
            te_n = te_pos.get(g, 0)
            if tr_n == 0 and te_n > 0:
                note = "<-- HELD OUT (never in training)"
            elif tr_n > 0 and te_n == 0:
                note = "(train only this fold)"
            else:
                note = ""
            print(f"  {g[:34]:34s}  {tr_n:>6}  {te_n:>6}  {note}")

        # ── Negative groups ───────────────────────────────────────────────────
        print(f"\n  NEGATIVES — benign groups")
        print(f"  {'Group':34s}  {'Train':>6}  {'Test':>6}")
        print(f"  {'·' * 50}")
        all_neg_groups = sorted(
            set(list(tr_neg) + list(te_neg)),
            key=lambda g: -(tr_neg.get(g, 0) + te_neg.get(g, 0))
        )
        SHOW_NEG = 20
        for g in all_neg_groups[:SHOW_NEG]:
            print(f"  {g[:34]:34s}  {tr_neg.get(g,0):>6}  {te_neg.get(g,0):>6}")
        if len(all_neg_groups) > SHOW_NEG:
            rest = len(all_neg_groups) - SHOW_NEG
            rest_tr = sum(tr_neg.get(g, 0) for g in all_neg_groups[SHOW_NEG:])
            rest_te = sum(te_neg.get(g, 0) for g in all_neg_groups[SHOW_NEG:])
            print(f"  {'... (' + str(rest) + ' more benign groups)':34s}  {rest_tr:>6}  {rest_te:>6}")

        # ── Log for return ────────────────────────────────────────────────────
        all_fold_logs.append({
            "fold":                  fold + 1,
            "n_train":               len(tr_data),
            "n_test":                len(te_data),
            "train_pos":             tr_pos_total,
            "train_neg":             tr_neg_total,
            "test_pos":              te_pos_total,
            "test_neg":              te_neg_total,
            "group_overlap":         sorted(overlap),
            "test_positive_groups":  dict(te_pos),
            "train_positive_groups": dict(tr_pos),
            "test_negative_groups":  dict(te_neg),
            "train_negative_groups": dict(tr_neg),
        })

    # ── Global summary across all folds ──────────────────────────────────────
    print(f"\n{'=' * W}")
    print(f"  GLOBAL SUMMARY")
    print(f"{'=' * W}")
    any_leakage = any(len(fl["group_overlap"]) > 0 for fl in all_fold_logs)
    print(f"  Leakage across all folds: {'YES ⚠' if any_leakage else 'NONE ✓'}")

    # Which positive groups were held out across all folds
    print(f"\n  Held-out positive groups per fold:")
    print(f"  {'Fold':6s}  Held-out families/organisms")
    print(f"  {'─'*60}")
    for fl in all_fold_logs:
        held = sorted(
            g for g, n in fl["test_positive_groups"].items()
            if fl["train_positive_groups"].get(g, 0) == 0
        )
        print(f"  {fl['fold']:>4d}    {', '.join(held) if held else '(none fully held out)'}")

    print(f"\n  All positive groups seen across the full dataset:")
    all_groups_ever = set()
    for fl in all_fold_logs:
        all_groups_ever.update(fl["test_positive_groups"])
        all_groups_ever.update(fl["train_positive_groups"])
    for g in sorted(all_groups_ever):
        in_test_folds  = [fl["fold"] for fl in all_fold_logs if g in fl["test_positive_groups"]]
        in_train_folds = [fl["fold"] for fl in all_fold_logs if g in fl["train_positive_groups"]]
        print(f"  {'·'} {g[:40]:40s}  test_folds={in_test_folds}  train_folds={in_train_folds}")

    print(f"\n{'=' * W}")
    print("  [Done] No encoder loaded, no training run.")
    print("=" * W)

    return all_fold_logs


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioChain CV split auditor")
    parser.add_argument("--cache", type=str, default="uniprot_cache.json",
                        help="UniProt sequence cache path")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    print("[Audit] Building dataset (uses cache if available — no re-download)...")
    dataset = build_dataset_from_uniprot(random_seed=args.seed, cache_path=args.cache)

    audit_splits(dataset, n_folds=args.folds, seed=args.seed)
