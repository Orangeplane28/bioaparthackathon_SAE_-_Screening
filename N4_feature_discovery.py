"""
N4 — SAE Feature Discovery

Steps:
  1. Load SAE feature activations for all proteins (train set)
  2. Compute per-feature statistics vs. negative controls (AUROC, Mann-Whitney U,
     Cohen's d, mean activation, activation frequency)
  3. BH FDR correction across all 15,360 features
  4. Rank by composite score; filter to top discriminative features
  5. Extract functional-site features for convergent PFTs
  6. Save feature ranking (both canonical filenames for downstream compatibility)

Run: python N4_feature_discovery.py [--top-k 200] [--batch 512]
"""

import os
import sys
import json
import argparse
import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    PROJECT_ROOT, DATA_DIR, SAE_DIR, SPLITS_DIR,
    FEAT_RANK_DIR, CONVERGENT_IDS, FUNCTIONAL_SITES,
    D_SAE, make_dirs
)
from utils.data import (
    load_master_dataset, load_splits, load_features, pool_features,
    build_feature_matrix
)
from utils.stats import bh_correction, auroc_safe, auroc_feature_vs_group, cohens_d


# ── Per-feature statistics ────────────────────────────────────────────────────

def compute_feature_stats(toxin_vecs: np.ndarray,
                           hard_neg_vecs: np.ndarray,
                           gen_neg_vecs: np.ndarray,
                           all_neg_vecs: np.ndarray,
                           batch_size: int = 512) -> dict:
    """
    Compute per-feature statistics across all D_SAE features.

    Args:
        toxin_vecs:    (n_tox, D_SAE)  – positive class
        hard_neg_vecs: (n_hard, D_SAE) – structurally similar negatives
        gen_neg_vecs:  (n_gen, D_SAE)  – general negatives
        all_neg_vecs:  (n_all, D_SAE)  – all negatives combined
        batch_size:    Features processed per batch (memory efficiency)

    Returns:
        dict with per-feature arrays: auroc_vs_all, auroc_vs_hard, auroc_vs_gen,
        pval_vs_all, cohen_d_vs_all, mean_tox, mean_neg, freq_tox, freq_neg
    """
    n_features = toxin_vecs.shape[1]
    assert hard_neg_vecs.shape[1] == n_features
    assert gen_neg_vecs.shape[1]  == n_features
    assert all_neg_vecs.shape[1]  == n_features

    auroc_vs_all  = np.zeros(n_features)
    auroc_vs_hard = np.zeros(n_features)
    auroc_vs_gen  = np.zeros(n_features)
    pval_vs_all   = np.ones(n_features)
    cohen_d_arr   = np.zeros(n_features)
    mean_tox      = np.zeros(n_features)
    mean_neg      = np.zeros(n_features)
    freq_tox      = np.zeros(n_features)
    freq_neg      = np.zeros(n_features)

    n_batches = (n_features + batch_size - 1) // batch_size

    for b in range(n_batches):
        start = b * batch_size
        end   = min(start + batch_size, n_features)

        t_b  = toxin_vecs[:, start:end]
        h_b  = hard_neg_vecs[:, start:end]
        g_b  = gen_neg_vecs[:, start:end]
        a_b  = all_neg_vecs[:, start:end]

        for fi in range(end - start):
            f = start + fi
            t_vals = t_b[:, fi]
            a_vals = a_b[:, fi]
            h_vals = h_b[:, fi]
            g_vals = g_b[:, fi]

            # AUROC
            auroc_vs_all[f]  = auroc_safe(
                np.concatenate([np.ones(len(t_vals)), np.zeros(len(a_vals))]),
                np.concatenate([t_vals, a_vals])
            )
            auroc_vs_hard[f] = auroc_safe(
                np.concatenate([np.ones(len(t_vals)), np.zeros(len(h_vals))]),
                np.concatenate([t_vals, h_vals])
            )
            auroc_vs_gen[f]  = auroc_safe(
                np.concatenate([np.ones(len(t_vals)), np.zeros(len(g_vals))]),
                np.concatenate([t_vals, g_vals])
            )

            # Mann-Whitney U (two-sided)
            if t_vals.std() > 1e-8 or a_vals.std() > 1e-8:
                try:
                    _, pval = scipy_stats.mannwhitneyu(
                        t_vals, a_vals, alternative='two-sided'
                    )
                    pval_vs_all[f] = float(pval)
                except Exception:
                    pval_vs_all[f] = 1.0
            else:
                pval_vs_all[f] = 1.0

            # Cohen's d
            cohen_d_arr[f] = cohens_d(t_vals, a_vals)

            # Mean activations
            mean_tox[f] = float(t_vals.mean())
            mean_neg[f] = float(a_vals.mean())

            # Activation frequency (fraction of proteins with non-zero activation)
            freq_tox[f] = float((t_vals > 0).mean())
            freq_neg[f] = float((a_vals > 0).mean())

        if (b + 1) % 10 == 0:
            print(f'  Feature stats: batch {b+1}/{n_batches} '
                  f'(features {start}–{end-1})')

    return {
        'auroc_vs_all':  auroc_vs_all,
        'auroc_vs_hard': auroc_vs_hard,
        'auroc_vs_gen':  auroc_vs_gen,
        'pval_vs_all':   pval_vs_all,
        'cohen_d':       cohen_d_arr,
        'mean_tox':      mean_tox,
        'mean_neg':      mean_neg,
        'freq_tox':      freq_tox,
        'freq_neg':      freq_neg,
    }


# ── Composite score & ranking ─────────────────────────────────────────────────

def rank_features(stats: dict, alpha: float = 0.05) -> dict:
    """
    Apply BH FDR correction and build composite ranking.

    Composite score = 0.5 * AUROC_vs_all + 0.25 * AUROC_vs_hard + 0.25 * AUROC_vs_gen
    (weights emphasise overall discrimination while penalising hard-neg leakage)

    A feature is 'significant' if:
      - q-value ≤ alpha, AND
      - AUROC_vs_all ≥ 0.60, AND
      - mean_tox > mean_neg

    Returns extended stats dict with q_values, is_significant, composite_score,
    sorted_indices.
    """
    q_values, is_sig_bh = bh_correction(stats['pval_vs_all'], alpha=alpha)

    auroc_sig  = stats['auroc_vs_all'] >= 0.60
    mean_check = stats['mean_tox'] > stats['mean_neg']
    is_significant = is_sig_bh & auroc_sig & mean_check

    composite = (
        0.50 * stats['auroc_vs_all'] +
        0.25 * stats['auroc_vs_hard'] +
        0.25 * stats['auroc_vs_gen']
    )

    sorted_indices = np.argsort(-composite)   # descending

    stats['q_values']       = q_values
    stats['is_significant'] = is_significant
    stats['composite_score'] = composite
    stats['sorted_indices'] = sorted_indices

    n_sig = int(is_significant.sum())
    print(f'  Significant features (BH q≤{alpha}, AUROC≥0.60, mean↑): {n_sig}')
    return stats


# ── Convergent PFT functional-site features ───────────────────────────────────

def extract_convergent_features(proteins: dict, sae_dir: str,
                                 convergent_ids: list,
                                 functional_sites: dict) -> dict:
    """
    For each convergent PFT, extract features most active at annotated
    functional sites (pore-forming loop, β-barrel stem, etc.).

    For each protein:
      - Pool mean activation for functional-site residues
      - Pool mean activation for non-functional residues
      - Return features where functional_mean > 2× non_functional_mean

    Returns:
        {pid: {'functional_features': [...], 'site_aurocs': [...], ...}}
    """
    results = {}

    for pid in convergent_ids:
        if pid not in proteins:
            continue

        feat_path = os.path.join(sae_dir, f'{pid}.npz')
        if not os.path.exists(feat_path):
            print(f'  WARNING: No SAE features for convergent PFT {pid}')
            continue

        try:
            data = np.load(feat_path)
            acts = data['features']  # (seq_len, D_SAE)
        except Exception as e:
            print(f'  WARNING: Could not load features for {pid}: {e}')
            continue

        seq_len = acts.shape[0]
        sites   = functional_sites.get(pid, [])

        if not sites:
            print(f'  No functional sites defined for {pid}')
            continue

        # Collect residue indices for each annotated site
        site_indices = [r for r in sites if 0 <= r < seq_len]

        if not site_indices:
            continue

        all_indices       = set(range(seq_len))
        non_site_indices  = list(all_indices - set(site_indices))

        site_acts     = acts[site_indices].mean(axis=0)     # (D_SAE,)
        non_site_acts = acts[non_site_indices].mean(axis=0) if non_site_indices else np.zeros(D_SAE)

        # Features enriched at functional sites
        ratio = site_acts / (non_site_acts + 1e-8)
        functional_feature_ids = np.where((ratio > 2.0) & (site_acts > 0.01))[0].tolist()

        results[pid] = {
            'functional_feature_ids': functional_feature_ids,
            'n_functional_features': len(functional_feature_ids),
            'site_mean_activations': site_acts[functional_feature_ids].tolist(),
            'n_site_residues': len(site_indices),
        }
        print(f'  {pid}: {len(functional_feature_ids)} site-enriched features')

    return results


# ── Cross-toxin overlap ───────────────────────────────────────────────────────

def compute_cross_toxin_overlap(toxin_vecs: np.ndarray,
                                 toxin_pids: list,
                                 top_feature_ids: list,
                                 threshold: float = 0.5) -> dict:
    """
    For each pair of toxins, compute the fraction of top features
    that are both active (> threshold) — a proxy for mechanistic similarity.

    Returns:
        { 'pairwise_overlap': {(pid_i, pid_j): fraction}, 'triple_overlap': fraction }
    """
    top_ids = np.array(top_feature_ids)
    active  = {}

    for i, pid in enumerate(toxin_pids):
        vec = toxin_vecs[i, top_ids]
        active[pid] = set(np.where(vec > threshold)[0].tolist())

    pairwise = {}
    pids_list = list(active.keys())
    for i in range(len(pids_list)):
        for j in range(i + 1, len(pids_list)):
            pi, pj = pids_list[i], pids_list[j]
            union = active[pi] | active[pj]
            if union:
                pairwise[(pi, pj)] = len(active[pi] & active[pj]) / len(union)
            else:
                pairwise[(pi, pj)] = 0.0

    # Triple overlap (convergent PFTs specifically)
    conv_active = [active.get(cid, set()) for cid in pids_list if cid in active]
    if len(conv_active) >= 3:
        triple_inter = conv_active[0].copy()
        for s in conv_active[1:]:
            triple_inter &= s
        triple_union = conv_active[0].copy()
        for s in conv_active[1:]:
            triple_union |= s
        triple_overlap = len(triple_inter) / max(len(triple_union), 1)
    else:
        triple_overlap = 0.0

    return {
        'pairwise_overlap': {f'{k[0]}__{k[1]}': v for k, v in pairwise.items()},
        'triple_overlap': triple_overlap,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(top_k: int = 200, batch_size: int = 512, alpha: float = 0.05):
    print('=' * 65)
    print('  N4 — SAE FEATURE DISCOVERY')
    print('=' * 65)
    make_dirs()
    os.makedirs(FEAT_RANK_DIR, exist_ok=True)

    master   = load_master_dataset(DATA_DIR)
    proteins = master['proteins']
    splits   = load_splits(SPLITS_DIR)
    train_pids = splits['train']

    print(f'  Train proteins: {len(train_pids)}')

    # Build feature matrices for each group (train set only)
    print('\n[1/5] Building group feature matrices (train set)...')

    def pids_by_category(pids, label=None, categories=None):
        """Filter PIDs by label and/or category list."""
        out = []
        for pid in pids:
            p = proteins.get(pid, {})
            if label is not None and p.get('label') != label:
                continue
            if categories is not None and p.get('category') not in categories:
                continue
            out.append(pid)
        return out

    toxin_pids   = pids_by_category(train_pids, label=1)
    hard_neg_pids = pids_by_category(train_pids, label=0,
                                      categories=[
                                          'hard_neg_membrane', 'hard_neg_ion_channel',
                                          'hard_neg_amphipathic', 'hard_neg_lipid_binding'
                                      ])
    gen_neg_pids  = pids_by_category(train_pids, label=0,
                                      categories=[
                                          'gen_neg_metabolic', 'gen_neg_nuclear',
                                          'gen_neg_cytoskeletal', 'gen_neg_ribosomal'
                                      ])

    # If category-based filtering yields too few, fall back to label-based
    if len(hard_neg_pids) < 5:
        hard_neg_pids = pids_by_category(train_pids, label=0)
        print('  NOTE: Falling back to all negatives for hard_neg group')
    if len(gen_neg_pids) < 5:
        gen_neg_pids = pids_by_category(train_pids, label=0)

    all_neg_pids = list({p for p in train_pids
                          if proteins.get(p, {}).get('label') == 0})

    print(f'  Toxins:      {len(toxin_pids)}')
    print(f'  Hard negs:   {len(hard_neg_pids)}')
    print(f'  Gen negs:    {len(gen_neg_pids)}')
    print(f'  All negs:    {len(all_neg_pids)}')

    X_tox,  y_tox,  _ = build_feature_matrix(toxin_pids,   proteins, SAE_DIR)
    X_hard, y_hard, _ = build_feature_matrix(hard_neg_pids, proteins, SAE_DIR)
    X_gen,  y_gen,  _ = build_feature_matrix(gen_neg_pids,  proteins, SAE_DIR)
    X_neg,  y_neg,  _ = build_feature_matrix(all_neg_pids,  proteins, SAE_DIR)

    if X_tox.shape[0] < 2:
        raise RuntimeError('Not enough toxin SAE features found. Run N3 first.')

    print(f'  Feature matrix shapes: tox={X_tox.shape}, neg={X_neg.shape}')
    print(f'  D_SAE = {X_tox.shape[1]}')

    # 2. Compute per-feature statistics
    print('\n[2/5] Computing per-feature statistics (batched)...')
    feat_stats = compute_feature_stats(
        X_tox, X_hard, X_gen, X_neg, batch_size=batch_size
    )

    # 3. BH correction + ranking
    print('\n[3/5] Applying BH FDR correction and ranking...')
    feat_stats = rank_features(feat_stats, alpha=alpha)
    sorted_idx = feat_stats['sorted_indices']
    top_feature_ids = sorted_idx[:top_k].tolist()

    # Select only significant features for output (capped at top_k)
    sig_idx = sorted_idx[feat_stats['is_significant'][sorted_idx]][:top_k]
    sig_feature_ids = sig_idx.tolist()
    if len(sig_feature_ids) == 0:
        print('  WARNING: No features passed significance threshold.'
              f' Using top-{top_k} by composite score.')
        sig_feature_ids = top_feature_ids

    print(f'  Top-{top_k} features (by composite score): first 5 = {top_feature_ids[:5]}')

    # 4. Convergent PFT functional-site features
    print('\n[4/5] Extracting convergent PFT functional-site features...')
    convergent_results = extract_convergent_features(
        proteins, SAE_DIR, CONVERGENT_IDS, FUNCTIONAL_SITES
    )

    # Add convergent site features to the significant set
    conv_site_features = set()
    for pid, res in convergent_results.items():
        conv_site_features.update(res.get('functional_feature_ids', []))
    print(f'  Convergent PFT site-enriched features: {len(conv_site_features)}')

    # 5. Cross-toxin overlap (preview)
    print('\n[5/5] Computing cross-toxin feature overlap...')
    tox_pids_for_overlap = [p for p in toxin_pids
                              if os.path.exists(os.path.join(SAE_DIR, f'{p}.npz'))]
    overlap_results = {}
    if len(tox_pids_for_overlap) >= 2:
        X_tox_filt, _, pids_filt = build_feature_matrix(
            tox_pids_for_overlap, proteins, SAE_DIR
        )
        overlap_results = compute_cross_toxin_overlap(
            X_tox_filt, pids_filt, sig_feature_ids
        )
        print(f'  Triple overlap (convergent PFTs): '
              f'{overlap_results.get("triple_overlap", 0):.4f}')

    # ── Save results ──────────────────────────────────────────────────────────
    feature_ranking = {
        'top_feature_ids':         sig_feature_ids,
        'top_k':                   top_k,
        'alpha_fdr':               alpha,
        'n_significant':           int(feat_stats['is_significant'].sum()),
        'per_feature': {
            'auroc_vs_all':    feat_stats['auroc_vs_all'].tolist(),
            'auroc_vs_hard':   feat_stats['auroc_vs_hard'].tolist(),
            'auroc_vs_gen':    feat_stats['auroc_vs_gen'].tolist(),
            'q_values':        feat_stats['q_values'].tolist(),
            'cohen_d':         feat_stats['cohen_d'].tolist(),
            'mean_tox':        feat_stats['mean_tox'].tolist(),
            'mean_neg':        feat_stats['mean_neg'].tolist(),
            'freq_tox':        feat_stats['freq_tox'].tolist(),
            'freq_neg':        feat_stats['freq_neg'].tolist(),
            'composite_score': feat_stats['composite_score'].tolist(),
            'is_significant':  feat_stats['is_significant'].tolist(),
        },
        'convergent_site_features': {
            pid: res for pid, res in convergent_results.items()
        },
        'cross_toxin_overlap': overlap_results,
    }

    # Save as both expected filenames for downstream compatibility
    for fname in ('feature_ranking.json', 'top_toxin_features.json'):
        path = os.path.join(FEAT_RANK_DIR, fname)
        with open(path, 'w') as f:
            json.dump(feature_ranking, f, indent=2)
    print(f'  Feature ranking saved to {FEAT_RANK_DIR}/')
    print(f'    → feature_ranking.json')
    print(f'    → top_toxin_features.json  (canonical alias for N5/N6/N7)')

    # Print top-10 summary
    print()
    print('=' * 65)
    print('  N4 RESULTS SUMMARY')
    print('=' * 65)
    print(f'  Total features tested:   {X_tox.shape[1]}')
    print(f'  Significant (BH q≤{alpha}):  {int(feat_stats["is_significant"].sum())}')
    print(f'  Top feature IDs (first 10): {sig_feature_ids[:10]}')
    top5 = sig_feature_ids[:5]
    for fid in top5:
        print(f'    Feature {fid:5d}: AUROC={feat_stats["auroc_vs_all"][fid]:.3f}, '
              f'q={feat_stats["q_values"][fid]:.2e}, '
              f'd={feat_stats["cohen_d"][fid]:.2f}')
    print()
    print('NEXT: python N5_negative_controls.py')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='N4: SAE feature discovery')
    parser.add_argument('--top-k',  type=int,   default=200,
                        help='Max features to keep in ranking (default: 200)')
    parser.add_argument('--batch',  type=int,   default=512,
                        help='Feature batch size for stats computation (default: 512)')
    parser.add_argument('--alpha',  type=float, default=0.05,
                        help='BH FDR threshold (default: 0.05)')
    args = parser.parse_args()
    main(top_k=args.top_k, batch_size=args.batch, alpha=args.alpha)
