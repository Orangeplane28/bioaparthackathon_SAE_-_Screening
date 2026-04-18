"""
N6 — Causal Intervention Analysis

Steps:
  1. Ablation dose-response: zero out top-k features, measure AUROC drop
  2. True LEACE concept erasure (Fisher LDA direction via utils/interv)
  3. Six erasure configurations (tox vs hard, gen, all × ablation / LEACE)
  4. Targeted functional-site intervention (site residues only)
  5. Selectivity ratio: intervention effect on toxins vs. negatives
  6. Held-out family evaluation (test-set families not seen during training)
  7. Save all results and figures

Run: python N6_causal_intervention.py [--top-k 50]
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    PROJECT_ROOT, DATA_DIR, SAE_DIR, SPLITS_DIR,
    FEAT_RANK_DIR, CONVERGENT_IDS, FUNCTIONAL_SITES,
    D_SAE, make_dirs
)
from utils.data import (
    load_master_dataset, load_splits, load_feature_ranking,
    build_feature_matrix, load_classifier
)
from utils.stats import auroc_safe, evaluate_classifier
from utils.interv import ablate_features, compute_leace_direction, erase_direction


# ── Ablation dose-response ────────────────────────────────────────────────────

def ablation_dose_response(clf,
                            X_tox: np.ndarray,
                            X_neg: np.ndarray,
                            y_tox: np.ndarray,
                            y_neg: np.ndarray,
                            sorted_feature_ids: list,
                            ablation_steps: list = None) -> dict:
    """
    Successively zero out the top-k discriminative features and measure
    the classifier AUROC on the combined (toxin + negative) set.

    Args:
        clf:                Fitted sklearn classifier (Pipeline)
        X_tox / X_neg:     Feature matrices
        y_tox / y_neg:     Labels (all 1s and 0s respectively)
        sorted_feature_ids: Ranked feature IDs (most discriminative first)
        ablation_steps:     List of k-values to test; default = [0,1,2,5,10,20,50,100,200]

    Returns:
        { 'k_values': [...], 'auroc': [...], 'auroc_drop': [...] }
    """
    if ablation_steps is None:
        ablation_steps = [0, 1, 2, 5, 10, 20, 50, 100, 200]

    X_all = np.concatenate([X_tox, X_neg], axis=0)
    y_all = np.concatenate([y_tox, y_neg], axis=0)

    baseline_auroc = evaluate_classifier(clf, X_all, y_all)['auroc']
    auroc_list = [baseline_auroc]
    k_list     = [0]

    for k in ablation_steps:
        if k == 0:
            continue
        fids = sorted_feature_ids[:k]
        X_abl = ablate_features(X_all, fids)
        auc   = evaluate_classifier(clf, X_abl, y_all, f'ablate-{k}')['auroc']
        auroc_list.append(auc)
        k_list.append(k)

    drops = [baseline_auroc - a for a in auroc_list]
    return {
        'k_values':   k_list,
        'auroc':      auroc_list,
        'auroc_drop': drops,
        'baseline_auroc': baseline_auroc,
    }


# ── LEACE erasure configurations ──────────────────────────────────────────────

def run_erasure_configs(clf,
                         X_tox_tr: np.ndarray,
                         X_hard_tr: np.ndarray,
                         X_gen_tr: np.ndarray,
                         X_tox_te: np.ndarray,
                         X_hard_te: np.ndarray,
                         X_gen_te: np.ndarray,
                         y_tox_te: np.ndarray,
                         y_hard_te: np.ndarray,
                         y_gen_te: np.ndarray,
                         top_feature_ids: list) -> dict:
    """
    Six erasure configurations:
      1. Ablation only (tox vs hard)
      2. Ablation only (tox vs gen)
      3. Ablation only (tox vs all)
      4. LEACE erasure (tox vs hard)
      5. LEACE erasure (tox vs gen)
      6. LEACE erasure (tox vs all)

    For each, evaluate the classifier on toxin+hard and toxin+gen test sets.

    Returns:
        { config_name: { 'auroc_tox_vs_hard': float, 'auroc_tox_vs_gen': float } }
    """
    X_allneg_tr = np.concatenate([X_hard_tr, X_gen_tr], axis=0)
    X_allneg_te = np.concatenate([X_hard_te, X_gen_te], axis=0)
    y_allneg_te = np.concatenate([y_hard_te, y_gen_te], axis=0)

    def eval_config(X_tox_eval, X_hard_eval, X_gen_eval):
        """Evaluate a classifier after some transformation has been applied."""
        res = {}
        # Tox vs hard
        if len(X_hard_eval) > 0:
            Xc = np.concatenate([X_tox_eval, X_hard_eval])
            yc = np.concatenate([y_tox_te[:len(X_tox_eval)],
                                  y_hard_te[:len(X_hard_eval)]])
            res['auroc_tox_vs_hard'] = evaluate_classifier(clf, Xc, yc)['auroc']
        # Tox vs gen
        if len(X_gen_eval) > 0:
            Xc = np.concatenate([X_tox_eval, X_gen_eval])
            yc = np.concatenate([y_tox_te[:len(X_tox_eval)],
                                  y_gen_te[:len(X_gen_eval)]])
            res['auroc_tox_vs_gen'] = evaluate_classifier(clf, Xc, yc)['auroc']
        return res

    results = {}

    # Config 1–3: ablation
    for name, fids in [
        ('ablation_vs_hard', top_feature_ids[:50]),
        ('ablation_vs_gen',  top_feature_ids[:50]),
        ('ablation_vs_all',  top_feature_ids[:50]),
    ]:
        X_tox_e  = ablate_features(X_tox_te,  fids)
        X_hard_e = ablate_features(X_hard_te, fids)
        X_gen_e  = ablate_features(X_gen_te,  fids)
        results[name] = eval_config(X_tox_e, X_hard_e, X_gen_e)

    # Config 4–6: LEACE erasure
    for name, X_pos_tr, X_neg_tr, X_neg_te, y_neg_te_arr in [
        ('leace_vs_hard', X_tox_tr, X_hard_tr, X_hard_te, y_hard_te),
        ('leace_vs_gen',  X_tox_tr, X_gen_tr,  X_gen_te,  y_gen_te),
        ('leace_vs_all',  X_tox_tr, X_allneg_tr, X_allneg_te, y_allneg_te),
    ]:
        if len(X_pos_tr) < 4 or len(X_neg_tr) < 4:
            results[name] = {'note': 'Insufficient samples for LEACE'}
            continue
        try:
            direction = compute_leace_direction(X_pos_tr, X_neg_tr)
            X_tox_e  = erase_direction(X_tox_te,  direction)
            X_hard_e = erase_direction(X_hard_te, direction)
            X_gen_e  = erase_direction(X_gen_te,  direction)
            results[name] = eval_config(X_tox_e, X_hard_e, X_gen_e)
        except Exception as e:
            results[name] = {'error': str(e)}

    return results


# ── Targeted functional-site intervention ─────────────────────────────────────

def targeted_site_intervention(proteins: dict,
                                 sae_dir: str,
                                 convergent_ids: list,
                                 functional_sites: dict,
                                 top_feature_ids: list) -> dict:
    """
    For each convergent PFT, zero out the top discriminative features
    in the functional-site residues only, and measure how much the
    per-protein mean activation drops.

    Returns:
        { pid: { 'pre_mean': float, 'post_mean': float, 'drop_frac': float } }
    """
    results = {}

    for pid in convergent_ids:
        if pid not in proteins:
            continue

        feat_path = os.path.join(sae_dir, f'{pid}.npz')
        if not os.path.exists(feat_path):
            continue

        try:
            data = np.load(feat_path)
            acts = data['features']   # (seq_len, D_SAE)
        except Exception as e:
            print(f'  WARNING: Cannot load {pid}: {e}')
            continue

        seq_len  = acts.shape[0]
        sites    = functional_sites.get(pid, [])
        site_idx = [r for r in sites if 0 <= r < seq_len]

        if not site_idx:
            continue

        site_idx   = list(set(site_idx))
        top_fids   = [f for f in top_feature_ids if f < acts.shape[1]]

        pre_mean   = float(acts[site_idx][:, top_fids].mean())

        acts_abl   = acts.copy()
        acts_abl[np.ix_(site_idx, top_fids)] = 0.0
        post_mean  = float(acts_abl[site_idx][:, top_fids].mean())

        drop_frac  = (pre_mean - post_mean) / (pre_mean + 1e-8)
        results[pid] = {
            'pre_mean':  round(pre_mean, 6),
            'post_mean': round(post_mean, 6),
            'drop_frac': round(drop_frac, 4),
            'n_site_residues': len(site_idx),
            'n_ablated_features': len(top_fids),
        }
        print(f'  {pid}: site activation {pre_mean:.4f} → {post_mean:.4f} '
              f'(drop {drop_frac:.1%})')

    return results


# ── Selectivity ratio ─────────────────────────────────────────────────────────

def compute_selectivity(clf,
                         X_tox: np.ndarray, y_tox: np.ndarray,
                         X_neg: np.ndarray, y_neg: np.ndarray,
                         top_feature_ids: list) -> dict:
    """
    Selectivity = (AUROC drop on toxins) / (AUROC drop on negatives)

    A high ratio means ablation specifically hurts toxin classification
    rather than disrupting all predictions indiscriminately.
    """
    # Baseline
    auc_tox_base = evaluate_classifier(clf, X_tox, y_tox)['auroc']
    auc_neg_base = evaluate_classifier(clf, X_neg, y_neg)['auroc']

    # Ablated
    X_tox_abl = ablate_features(X_tox, top_feature_ids)
    X_neg_abl = ablate_features(X_neg, top_feature_ids)

    auc_tox_abl = evaluate_classifier(clf, X_tox_abl, y_tox)['auroc']
    auc_neg_abl = evaluate_classifier(clf, X_neg_abl, y_neg)['auroc']

    drop_tox = auc_tox_base - auc_tox_abl
    drop_neg = auc_neg_base - auc_neg_abl
    selectivity = drop_tox / (drop_neg + 1e-8)

    return {
        'auroc_tox_baseline':  round(float(auc_tox_base), 4),
        'auroc_tox_ablated':   round(float(auc_tox_abl), 4),
        'auroc_neg_baseline':  round(float(auc_neg_base), 4),
        'auroc_neg_ablated':   round(float(auc_neg_abl), 4),
        'drop_tox':            round(float(drop_tox), 4),
        'drop_neg':            round(float(drop_neg), 4),
        'selectivity_ratio':   round(float(selectivity), 4),
    }


# ── Held-out family evaluation ────────────────────────────────────────────────

def held_out_family_eval(clf,
                          proteins: dict,
                          splits: dict,
                          sae_dir: str,
                          top_feature_ids: list) -> dict:
    """
    Evaluate the classifier on test-set families, separately for each
    protein family present in the test split.

    Returns:
        { family: { 'n_proteins': int, 'auroc': float, 'auroc_ablated': float } }
    """
    test_pids = splits['test']
    family_pids = {}
    for pid in test_pids:
        fam = proteins.get(pid, {}).get('family', 'unknown')
        family_pids.setdefault(fam, []).append(pid)

    results = {}
    for fam, pids in family_pids.items():
        if len(pids) < 2:
            continue
        X, y, _ = build_feature_matrix(pids, proteins, sae_dir)
        if X.shape[0] < 2 or len(np.unique(y)) < 2:
            continue

        res_full = evaluate_classifier(clf, X, y, f'family {fam}')
        X_abl    = ablate_features(X, top_feature_ids)
        res_abl  = evaluate_classifier(clf, X_abl, y)

        results[fam] = {
            'n_proteins':    len(pids),
            'auroc':         round(res_full['auroc'], 4),
            'auprc':         round(res_full['auprc'], 4),
            'auroc_ablated': round(res_abl['auroc'],  4),
            'auroc_drop':    round(res_full['auroc'] - res_abl['auroc'], 4),
        }

    return results


# ── Figures ────────────────────────────────────────────────────────────────────

def save_figures(dose_resp: dict, erasure_configs: dict,
                  site_interv: dict, fig_dir: str):
    """Save key intervention figures."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  matplotlib not available — skipping figures')
        return

    os.makedirs(fig_dir, exist_ok=True)

    # Figure 1: Ablation dose-response
    if dose_resp.get('k_values'):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(dose_resp['k_values'], dose_resp['auroc'],
                marker='o', color='steelblue', linewidth=2)
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Chance')
        ax.set_xlabel('Features ablated (top-k)')
        ax.set_ylabel('Classifier AUROC')
        ax.set_title('Feature Ablation Dose-Response')
        ax.legend()
        fig.savefig(os.path.join(fig_dir, 'fig1_ablation_dose_response.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Figure 2: Erasure config comparison
    if erasure_configs:
        names  = list(erasure_configs.keys())
        auc_h  = [erasure_configs[n].get('auroc_tox_vs_hard', 0) for n in names]
        auc_g  = [erasure_configs[n].get('auroc_tox_vs_gen',  0) for n in names]
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(x - 0.2, auc_h, 0.4, label='vs Hard neg', color='#2196F3')
        ax.bar(x + 0.2, auc_g, 0.4, label='vs Gen neg',  color='#FF9800')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel('AUROC')
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title('AUROC after Erasure Interventions')
        ax.legend()
        fig.savefig(os.path.join(fig_dir, 'fig2_erasure_configs.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Figure 3: Functional site intervention
    if site_interv:
        pids_plot   = list(site_interv.keys())
        pre_vals    = [site_interv[p]['pre_mean']  for p in pids_plot]
        post_vals   = [site_interv[p]['post_mean'] for p in pids_plot]
        x = np.arange(len(pids_plot))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x - 0.2, pre_vals,  0.4, label='Pre-ablation',  color='#4CAF50')
        ax.bar(x + 0.2, post_vals, 0.4, label='Post-ablation', color='#F44336')
        ax.set_xticks(x)
        ax.set_xticklabels(pids_plot, fontsize=9)
        ax.set_ylabel('Mean site activation')
        ax.set_title('Targeted Functional-Site Intervention')
        ax.legend()
        fig.savefig(os.path.join(fig_dir, 'fig3_site_intervention.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f'  Figures saved to {fig_dir}/')


# ── Main ──────────────────────────────────────────────────────────────────────

def main(top_k: int = 50, ablation_steps: list = None):
    print('=' * 65)
    print('  N6 — CAUSAL INTERVENTION ANALYSIS')
    print('=' * 65)
    make_dirs()

    if ablation_steps is None:
        ablation_steps = [0, 1, 2, 5, 10, 20, 50, 100, 200]

    master   = load_master_dataset(DATA_DIR)
    proteins = master['proteins']
    splits   = load_splits(SPLITS_DIR)
    ranking  = load_feature_ranking(FEAT_RANK_DIR)

    top_feature_ids = ranking.get('top_feature_ids', list(range(200)))
    top_k_ids = top_feature_ids[:top_k]
    print(f'  Using top-{top_k} features for intervention')

    # Group PIDs
    def group_pids(split_key, label=None, categories=None):
        out = []
        for pid in splits[split_key]:
            p = proteins.get(pid, {})
            if label is not None and p.get('label') != label:
                continue
            if categories is not None and p.get('category') not in categories:
                continue
            out.append(pid)
        return out

    # Load classifiers
    from config import BASELINES_DIR
    clf_sae = load_classifier(BASELINES_DIR)
    if clf_sae is None:
        raise RuntimeError('No trained classifier found. Run N3 first.')
    print(f'  Loaded classifier: {type(clf_sae).__name__}')

    # Build matrices
    print('\n[1/6] Building feature matrices...')
    train_tox_pids  = group_pids('train', label=1)
    train_hard_pids = group_pids('train', label=0,
                                  categories=['hard_neg_membrane', 'hard_neg_ion_channel',
                                              'hard_neg_amphipathic', 'hard_neg_lipid_binding'])
    train_gen_pids  = group_pids('train', label=0,
                                  categories=['gen_neg_metabolic', 'gen_neg_nuclear',
                                              'gen_neg_cytoskeletal', 'gen_neg_ribosomal'])
    if len(train_hard_pids) < 5:
        train_hard_pids = group_pids('train', label=0)
    if len(train_gen_pids) < 5:
        train_gen_pids = group_pids('train', label=0)

    test_tox_pids  = group_pids('test', label=1)
    test_hard_pids = group_pids('test', label=0,
                                 categories=['hard_neg_membrane', 'hard_neg_ion_channel',
                                             'hard_neg_amphipathic', 'hard_neg_lipid_binding'])
    test_gen_pids  = group_pids('test', label=0,
                                 categories=['gen_neg_metabolic', 'gen_neg_nuclear',
                                             'gen_neg_cytoskeletal', 'gen_neg_ribosomal'])
    if len(test_hard_pids) < 5:
        test_hard_pids = group_pids('test', label=0)
    if len(test_gen_pids) < 5:
        test_gen_pids = group_pids('test', label=0)

    all_neg_pids = [p for p in splits['test']
                    if proteins.get(p, {}).get('label') == 0]

    X_tr_tox,  y_tr_tox,  _ = build_feature_matrix(train_tox_pids,  proteins, SAE_DIR)
    X_tr_hard, y_tr_hard, _ = build_feature_matrix(train_hard_pids, proteins, SAE_DIR)
    X_tr_gen,  y_tr_gen,  _ = build_feature_matrix(train_gen_pids,  proteins, SAE_DIR)
    X_te_tox,  y_te_tox,  _ = build_feature_matrix(test_tox_pids,   proteins, SAE_DIR)
    X_te_hard, y_te_hard, _ = build_feature_matrix(test_hard_pids,  proteins, SAE_DIR)
    X_te_gen,  y_te_gen,  _ = build_feature_matrix(test_gen_pids,   proteins, SAE_DIR)
    X_neg,     y_neg,     _ = build_feature_matrix(all_neg_pids,    proteins, SAE_DIR)

    if X_te_tox.shape[0] < 2:
        raise RuntimeError('Not enough test toxins. Run N3 first.')

    # 1. Ablation dose-response
    print('\n[2/6] Ablation dose-response...')
    X_all_te = np.concatenate([X_te_tox, X_te_hard], axis=0) if X_te_hard.shape[0] > 0 else X_te_tox
    y_all_te = np.concatenate([y_te_tox, y_te_hard], axis=0) if X_te_hard.shape[0] > 0 else y_te_tox
    y_ones   = np.ones(len(X_te_tox), dtype=int)
    y_zeros  = np.zeros(len(X_neg), dtype=int) if X_neg.shape[0] > 0 else np.zeros(0, dtype=int)

    dose_resp = ablation_dose_response(
        clf_sae, X_te_tox, X_neg if X_neg.shape[0] > 0 else X_te_hard,
        y_ones, y_zeros if len(y_zeros) > 0 else np.zeros(len(X_te_hard), dtype=int),
        top_feature_ids, ablation_steps=ablation_steps
    )
    print(f'  Baseline AUROC: {dose_resp["baseline_auroc"]:.3f}')
    print(f'  After ablating {top_k} features: '
          f'AUROC={dose_resp["auroc"][-1] if dose_resp["auroc"] else "N/A"}')

    # 2–3. LEACE erasure configs
    print('\n[3/6] Running 6 erasure configurations...')
    erasure_configs = run_erasure_configs(
        clf_sae,
        X_tr_tox, X_tr_hard, X_tr_gen,
        X_te_tox, X_te_hard, X_te_gen,
        y_te_tox, y_te_hard, y_te_gen,
        top_k_ids
    )

    # 4. Targeted functional-site intervention
    print('\n[4/6] Targeted functional-site intervention...')
    site_interv = targeted_site_intervention(
        proteins, SAE_DIR, CONVERGENT_IDS, FUNCTIONAL_SITES, top_k_ids
    )

    # 5. Selectivity ratio
    print('\n[5/6] Selectivity ratio...')
    selectivity = {}
    if X_neg.shape[0] > 0 and len(np.unique(y_neg)) > 1:
        selectivity = compute_selectivity(
            clf_sae,
            X_te_tox, y_te_tox,
            X_neg,    y_neg,
            top_k_ids
        )
        print(f'  Selectivity ratio: {selectivity.get("selectivity_ratio", "N/A")}')

    # 6. Held-out family evaluation
    print('\n[6/6] Held-out family evaluation...')
    family_results = held_out_family_eval(
        clf_sae, proteins, splits, SAE_DIR, top_k_ids
    )
    print(f'  Evaluated {len(family_results)} test-set families')

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        'top_k':               top_k,
        'top_k_feature_ids':   top_k_ids,
        'ablation_dose_response': dose_resp,
        'erasure_configs':        erasure_configs,
        'site_intervention':      site_interv,
        'selectivity':            selectivity,
        'held_out_families':      family_results,
    }

    out_dir  = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'causal_intervention_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results saved: {out_path}')

    # Figures
    fig_dir = os.path.join(PROJECT_ROOT, 'figures', 'N6')
    save_figures(dose_resp, erasure_configs, site_interv, fig_dir)

    print()
    print('=' * 65)
    print('  N6 RESULTS SUMMARY')
    print('=' * 65)
    print(f'  Baseline AUROC:         {dose_resp["baseline_auroc"]:.3f}')
    if dose_resp['auroc']:
        min_idx = int(np.argmin(dose_resp['auroc']))
        print(f'  Min AUROC after ablation: {min(dose_resp["auroc"]):.3f} '
              f'(k={dose_resp["k_values"][min_idx]})')
    for cfg, res in erasure_configs.items():
        h = res.get('auroc_tox_vs_hard', None)
        g = res.get('auroc_tox_vs_gen',  None)
        if h is not None or g is not None:
            h_str = f'{h:.3f}' if h is not None else 'N/A'
            g_str = f'{g:.3f}' if g is not None else 'N/A'
            print(f'  {cfg:<22s}: vs_hard={h_str}, vs_gen={g_str}')
    if selectivity:
        print(f'  Selectivity ratio:      {selectivity.get("selectivity_ratio", "N/A")}')
    print()
    print('NEXT: python N7_external_validation.py')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='N6: Causal intervention analysis')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Number of top features for intervention (default: 50)')
    args = parser.parse_args()
    main(top_k=args.top_k)
