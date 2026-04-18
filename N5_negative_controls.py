"""
N5 — Negative Control Analysis

Steps:
  1. Compute per-group mean activations (toxins, hard negs, general negs, all negs)
  2. Evaluate top discriminative features against hard and general negatives
  3. Categorise features: truly_specific / structural_only / non_discriminative
  4. 10,000-permutation test on triple overlap of convergent PFTs
  5. Cosine similarity matrix between toxin sub-groups
  6. Bootstrap confidence intervals on group mean AUROCs
  7. Save all results and figures

Run: python N5_negative_controls.py [--n-permutations 10000]
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    PROJECT_ROOT, DATA_DIR, SAE_DIR, SPLITS_DIR,
    FEAT_RANK_DIR, CONVERGENT_IDS, D_SAE, make_dirs
)
from utils.data import (
    load_master_dataset, load_splits, load_feature_ranking,
    build_feature_matrix
)
from utils.stats import (
    auroc_safe, auroc_feature_vs_group, bootstrap_ci,
    bh_correction, cohens_d
)


# ── Group activation summary ──────────────────────────────────────────────────

def group_activation_summary(feature_matrix: np.ndarray,
                               feature_ids: list) -> dict:
    """
    Compute mean / std / frequency for a set of features in a group.

    Args:
        feature_matrix: (n_proteins, D_SAE)
        feature_ids:    Indices of features to summarise

    Returns:
        dict with mean, std, freq arrays of length len(feature_ids)
    """
    sub = feature_matrix[:, feature_ids]
    return {
        'mean':  sub.mean(axis=0).tolist(),
        'std':   sub.std(axis=0).tolist(),
        'freq':  (sub > 0).mean(axis=0).tolist(),
    }


# ── Feature specificity categorisation ───────────────────────────────────────

def categorise_features(top_feature_ids: list,
                          X_tox: np.ndarray,
                          X_hard: np.ndarray,
                          X_gen: np.ndarray,
                          auroc_thresh_high: float = 0.70,
                          auroc_thresh_low:  float = 0.60) -> dict:
    """
    Categorise each top feature into one of three classes:

    - truly_specific:     AUROC vs hard ≥ 0.70 AND AUROC vs gen ≥ 0.70
                          (discriminates even structural analogues)
    - structural_only:    AUROC vs gen ≥ 0.70 but AUROC vs hard < 0.70
                          (captures general toxin signal but not topology-specific)
    - non_discriminative: Neither threshold met

    Returns:
        { 'truly_specific': [fids], 'structural_only': [fids],
          'non_discriminative': [fids], 'auroc_vs_hard': {fid: val}, ... }
    """
    truly_specific     = []
    structural_only    = []
    non_discriminative = []
    auroc_hard_map     = {}
    auroc_gen_map      = {}

    for fid in top_feature_ids:
        auc_h = auroc_feature_vs_group(X_tox, X_hard, fid)
        auc_g = auroc_feature_vs_group(X_tox, X_gen,  fid)
        auroc_hard_map[fid] = round(float(auc_h), 4)
        auroc_gen_map[fid]  = round(float(auc_g), 4)

        if auc_h >= auroc_thresh_high and auc_g >= auroc_thresh_high:
            truly_specific.append(fid)
        elif auc_g >= auroc_thresh_low:
            structural_only.append(fid)
        else:
            non_discriminative.append(fid)

    return {
        'truly_specific':     truly_specific,
        'structural_only':    structural_only,
        'non_discriminative': non_discriminative,
        'auroc_vs_hard':      auroc_hard_map,
        'auroc_vs_gen':       auroc_gen_map,
        'n_truly_specific':   len(truly_specific),
        'n_structural_only':  len(structural_only),
        'n_non_discriminative': len(non_discriminative),
    }


# ── Permutation test: triple overlap ─────────────────────────────────────────

def permutation_test_triple_overlap(X_tox: np.ndarray,
                                     toxin_pids: list,
                                     convergent_ids: list,
                                     top_feature_ids: list,
                                     n_permutations: int = 10_000,
                                     threshold: float = 0.5,
                                     seed: int = 42) -> dict:
    """
    10,000-permutation test asking:
    "Is the feature overlap among the three convergent PFTs
     greater than expected by chance?"

    Null distribution: repeatedly sample 3 random proteins from toxin_vecs
    and compute their pairwise triple-Jaccard overlap on the top feature set.

    Returns:
        { 'observed_overlap': float, 'p_value': float,
          'null_mean': float, 'null_std': float }
    """
    rng = np.random.default_rng(seed)

    # Locate convergent PFTs in feature matrix
    conv_row_indices = [i for i, pid in enumerate(toxin_pids)
                         if pid in convergent_ids]
    if len(conv_row_indices) < 3:
        return {
            'observed_overlap': None,
            'p_value': None,
            'note': f'Only {len(conv_row_indices)}/3 convergent PFTs in training set.',
        }

    top_ids = np.array(top_feature_ids)

    def triple_overlap(rows):
        """Jaccard of triple intersection / union on top_ids."""
        active = [set(np.where(X_tox[r, top_ids] > threshold)[0].tolist())
                  for r in rows]
        inter  = active[0] & active[1] & active[2]
        union  = active[0] | active[1] | active[2]
        return len(inter) / max(len(union), 1)

    observed = triple_overlap(conv_row_indices[:3])

    n_tox    = len(toxin_pids)
    null_dist = np.zeros(n_permutations)

    for i in range(n_permutations):
        rows = rng.choice(n_tox, size=3, replace=False).tolist()
        null_dist[i] = triple_overlap(rows)

    p_value = float((null_dist >= observed).mean())

    return {
        'observed_overlap': float(observed),
        'p_value':          p_value,
        'null_mean':        float(null_dist.mean()),
        'null_std':         float(null_dist.std()),
        'n_permutations':   n_permutations,
        'convergent_pids':  [toxin_pids[i] for i in conv_row_indices[:3]],
    }


# ── Cosine similarity matrix ──────────────────────────────────────────────────

def cosine_similarity_matrix(X: np.ndarray, pids: list,
                               feature_ids: list) -> dict:
    """
    Compute pairwise cosine similarity matrix across proteins in X
    restricted to feature_ids.

    Returns:
        { 'matrix': [[...]], 'pids': [...] }
    """
    sub = X[:, feature_ids].astype(np.float64)
    norms = np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8
    normed = sub / norms
    sim = normed @ normed.T
    return {
        'matrix': sim.tolist(),
        'pids':   list(pids),
    }


# ── Bootstrap CIs on group AUROCs ─────────────────────────────────────────────

def bootstrap_group_aurocs(X_tox: np.ndarray,
                             X_hard: np.ndarray,
                             X_gen: np.ndarray,
                             feature_ids: list,
                             n_boot: int = 2000) -> dict:
    """
    Bootstrap CIs for mean AUROC over top_feature_ids,
    for both hard and general negative comparisons.

    Returns:
        { 'hard': (mean, lo, hi), 'gen': (mean, lo, hi) }
    """
    auroc_hard_vals = np.array([
        auroc_feature_vs_group(X_tox, X_hard, fid) for fid in feature_ids
    ])
    auroc_gen_vals  = np.array([
        auroc_feature_vs_group(X_tox, X_gen,  fid) for fid in feature_ids
    ])

    hard_ci = bootstrap_ci(auroc_hard_vals, n_boot=n_boot)
    gen_ci  = bootstrap_ci(auroc_gen_vals,  n_boot=n_boot)

    return {
        'auroc_vs_hard_mean_ci': {
            'mean': round(hard_ci[0], 4),
            'lo':   round(hard_ci[1], 4),
            'hi':   round(hard_ci[2], 4),
        },
        'auroc_vs_gen_mean_ci': {
            'mean': round(gen_ci[0], 4),
            'lo':   round(gen_ci[1], 4),
            'hi':   round(gen_ci[2], 4),
        },
    }


# ── Figures ────────────────────────────────────────────────────────────────────

def save_figures(results: dict, output_dir: str):
    """Save matplotlib figures summarising the analysis."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  matplotlib not available — skipping figures')
        return

    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Specificity pie chart
    cats = results.get('feature_categories', {})
    n_ts  = cats.get('n_truly_specific', 0)
    n_so  = cats.get('n_structural_only', 0)
    n_nd  = cats.get('n_non_discriminative', 0)
    if n_ts + n_so + n_nd > 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie([n_ts, n_so, n_nd],
               labels=['Truly specific', 'Structural only', 'Non-discriminative'],
               autopct='%1.1f%%', startangle=140,
               colors=['#2196F3', '#FF9800', '#9E9E9E'])
        ax.set_title('Feature Specificity Categories')
        fig.savefig(os.path.join(output_dir, 'fig1_feature_specificity.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Figure 2: AUROC vs hard neg vs gen neg scatter
    auroc_h = list(cats.get('auroc_vs_hard', {}).values())
    auroc_g = list(cats.get('auroc_vs_gen',  {}).values())
    if auroc_h and auroc_g:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(auroc_h, auroc_g, alpha=0.5, s=20, color='steelblue')
        ax.axhline(0.70, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(0.70, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlabel('AUROC vs Hard Negatives')
        ax.set_ylabel('AUROC vs General Negatives')
        ax.set_title('Feature Discrimination Scatter')
        fig.savefig(os.path.join(output_dir, 'fig2_auroc_scatter.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Figure 3: Permutation test null distribution
    perm = results.get('permutation_test', {})
    if perm.get('observed_overlap') is not None and 'null_mean' in perm:
        n_perm = perm.get('n_permutations', 10000)
        rng = np.random.default_rng(0)
        null_approx = rng.normal(perm['null_mean'], perm['null_std'], n_perm)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(null_approx, bins=60, color='lightblue', edgecolor='white',
                label='Null distribution')
        ax.axvline(perm['observed_overlap'], color='red', linewidth=2,
                   label=f'Observed ({perm["observed_overlap"]:.3f})')
        ax.set_xlabel('Triple Feature Overlap (Jaccard)')
        ax.set_ylabel('Count')
        ax.set_title(f'Convergent PFT Triple Overlap\n(p = {perm["p_value"]:.4f})')
        ax.legend()
        fig.savefig(os.path.join(output_dir, 'fig3_permutation_test.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f'  Figures saved to {output_dir}/')


# ── Main ──────────────────────────────────────────────────────────────────────

def main(n_permutations: int = 10_000, n_boot: int = 2000):
    print('=' * 65)
    print('  N5 — NEGATIVE CONTROL ANALYSIS')
    print('=' * 65)
    make_dirs()

    master   = load_master_dataset(DATA_DIR)
    proteins = master['proteins']
    splits   = load_splits(SPLITS_DIR)
    ranking  = load_feature_ranking(FEAT_RANK_DIR)

    top_feature_ids = ranking.get('top_feature_ids', list(range(200)))
    print(f'  Loaded {len(top_feature_ids)} top feature IDs from ranking')

    # Group PID lists
    def group_pids(pids, label=None, categories=None):
        out = []
        for pid in pids:
            p = proteins.get(pid, {})
            if label is not None and p.get('label') != label:
                continue
            if categories is not None and p.get('category') not in categories:
                continue
            out.append(pid)
        return out

    all_pids = splits['train'] + splits['val'] + splits['test']

    toxin_pids    = group_pids(all_pids, label=1)
    hard_neg_pids = group_pids(all_pids, label=0,
                                categories=['hard_neg_membrane', 'hard_neg_ion_channel',
                                            'hard_neg_amphipathic', 'hard_neg_lipid_binding'])
    gen_neg_pids  = group_pids(all_pids, label=0,
                                categories=['gen_neg_metabolic', 'gen_neg_nuclear',
                                            'gen_neg_cytoskeletal', 'gen_neg_ribosomal'])

    if len(hard_neg_pids) < 5:
        hard_neg_pids = group_pids(all_pids, label=0)
    if len(gen_neg_pids) < 5:
        gen_neg_pids = group_pids(all_pids, label=0)

    print(f'  Groups — toxins:{len(toxin_pids)}, '
          f'hard_neg:{len(hard_neg_pids)}, gen_neg:{len(gen_neg_pids)}')

    # Build feature matrices
    print('\n[1/5] Building feature matrices...')
    X_tox,  _, tox_pids_used   = build_feature_matrix(toxin_pids,   proteins, SAE_DIR)
    X_hard, _, _                = build_feature_matrix(hard_neg_pids, proteins, SAE_DIR)
    X_gen,  _, _                = build_feature_matrix(gen_neg_pids,  proteins, SAE_DIR)

    if X_tox.shape[0] < 2:
        raise RuntimeError('Not enough toxin features. Run N3 first.')

    # 1. Group activation summaries
    print('\n[2/5] Computing group activation summaries...')
    tox_summary  = group_activation_summary(X_tox,  top_feature_ids)
    hard_summary = group_activation_summary(X_hard, top_feature_ids) if X_hard.shape[0] > 0 else {}
    gen_summary  = group_activation_summary(X_gen,  top_feature_ids) if X_gen.shape[0] > 0 else {}

    # 2. Feature specificity categorisation
    print('\n[3/5] Categorising feature specificity...')
    categories = categorise_features(
        top_feature_ids, X_tox,
        X_hard if X_hard.shape[0] > 0 else X_tox,
        X_gen  if X_gen.shape[0] > 0  else X_tox,
    )
    print(f'  Truly specific:      {categories["n_truly_specific"]}')
    print(f'  Structural only:     {categories["n_structural_only"]}')
    print(f'  Non-discriminative:  {categories["n_non_discriminative"]}')

    # 3. Bootstrap CIs
    print('\n[4/5] Bootstrap CIs on group mean AUROCs...')
    boot_cis = bootstrap_group_aurocs(
        X_tox,
        X_hard if X_hard.shape[0] > 0 else X_tox,
        X_gen  if X_gen.shape[0] > 0  else X_tox,
        top_feature_ids, n_boot=n_boot
    )
    print(f'  AUROC vs hard: {boot_cis["auroc_vs_hard_mean_ci"]}')
    print(f'  AUROC vs gen:  {boot_cis["auroc_vs_gen_mean_ci"]}')

    # 4. Permutation test
    print(f'\n[5/5] Permutation test (n={n_permutations:,})...')
    perm_result = permutation_test_triple_overlap(
        X_tox, list(tox_pids_used), CONVERGENT_IDS,
        top_feature_ids, n_permutations=n_permutations
    )
    p_val = perm_result.get('p_value')
    if p_val is not None:
        print(f'  Observed triple overlap: {perm_result["observed_overlap"]:.4f}')
        print(f'  Null mean:               {perm_result["null_mean"]:.4f} '
              f'± {perm_result["null_std"]:.4f}')
        print(f'  p-value:                 {p_val:.4f}')
    else:
        print(f'  {perm_result.get("note", "")}')

    # 5. Cosine similarity
    cos_sim = cosine_similarity_matrix(X_tox, list(tox_pids_used), top_feature_ids)

    # ── Save all results ──────────────────────────────────────────────────────
    results = {
        'top_feature_ids':       top_feature_ids,
        'group_summaries': {
            'toxins':    tox_summary,
            'hard_neg':  hard_summary,
            'gen_neg':   gen_summary,
        },
        'feature_categories': categories,
        'bootstrap_cis':      boot_cis,
        'permutation_test':   perm_result,
        'cosine_similarity':  cos_sim,
    }

    out_path = os.path.join(FEAT_RANK_DIR, 'negative_control_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results saved: {out_path}')

    # Figures
    fig_dir = os.path.join(PROJECT_ROOT, 'figures', 'N5')
    save_figures(results, fig_dir)

    print()
    print('=' * 65)
    print('  N5 RESULTS SUMMARY')
    print('=' * 65)
    print(f'  Truly specific features:    {categories["n_truly_specific"]}')
    print(f'  Structural-only features:   {categories["n_structural_only"]}')
    print(f'  AUROC vs hard (mean CI):    '
          f'{boot_cis["auroc_vs_hard_mean_ci"]["mean"]:.3f} '
          f'[{boot_cis["auroc_vs_hard_mean_ci"]["lo"]:.3f}–'
          f'{boot_cis["auroc_vs_hard_mean_ci"]["hi"]:.3f}]')
    if p_val is not None:
        sig = '✓ significant' if p_val < 0.05 else '✗ not significant'
        print(f'  Convergent PFT triple overlap: p={p_val:.4f} ({sig})')
    print()
    print('NEXT: python N6_causal_intervention.py')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='N5: Negative control analysis')
    parser.add_argument('--n-permutations', type=int, default=10_000,
                        help='Permutation test iterations (default: 10000)')
    parser.add_argument('--n-boot', type=int, default=2000,
                        help='Bootstrap iterations for CIs (default: 2000)')
    args = parser.parse_args()
    main(n_permutations=args.n_permutations, n_boot=args.n_boot)
