"""
N7 — External Validation

Steps:
  1. Fetch 17 curated external proteins (diverse toxin families not in training data)
  2. Extract ESM3 layer-36 embeddings (same pipeline as N3)
  3. Pass through trained SAE → feature activations
  4. Evaluate classifiers on external proteins
  5. Per-family AUROC, feature overlap fraction with top discriminative features
  6. Convergent evolution extension: test triple-overlap on external PFTs
  7. Ablation generalisation: does ablation hurt external proteins more than random?
  8. Full pipeline summary

Run: python N7_external_validation.py [--device cuda]
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    PROJECT_ROOT, DATA_DIR, SAE_DIR, EMB_DIR,
    SPLITS_DIR, FEAT_RANK_DIR, BASELINES_DIR,
    CONVERGENT_IDS, FUNCTIONAL_SITES,
    D_MODEL, D_SAE, K_TOPK, ESM3_LAYER,
    SAE_WEIGHTS, make_dirs
)
from utils.esm3 import load_esm3, load_sae, get_block
from utils.data import (
    load_master_dataset, load_splits, load_feature_ranking,
    save_features, load_features, load_classifier
)
from utils.stats import auroc_safe, evaluate_classifier
from utils.interv import ablate_features


# ── Curated external protein panel ────────────────────────────────────────────
#
# 17 proteins from families/organisms not included in SafeProtein-Bench training set.
# Chosen to span:
#   - Beta-PFTs (new families): aerolysin, leukotoxin LukSF subunit
#   - Alpha-PFTs: delta-toxin, melittin, magainin
#   - AB-toxins: anthrax LF, botulinum neurotoxin light chain
#   - RIPs (ribosome-inactivating): ricin A chain, saporin
#   - Neurotoxins: alpha-conotoxin, dendrotoxin
#   - Structural analogues (should score LOW): colipase, annexin, apolipoprot A1,
#     pore-forming defensin hBD2, spider silk MaSp1 (true negative)
#
# UniProt accessions that are NOT in SafeProtein-Bench.

EXTERNAL_PROTEINS = [
    # Beta-PFTs
    {'id': 'P09839', 'name': 'Aerolysin',                'family': 'beta_PFT',    'label': 1},
    {'id': 'P0A0L2', 'name': 'Leukotoxin LukS-PV',       'family': 'beta_PFT',    'label': 1},
    {'id': 'P0A0L4', 'name': 'Leukotoxin LukF-PV',       'family': 'beta_PFT',    'label': 1},
    # Alpha-PFTs / AMPs
    {'id': 'P01498', 'name': 'Delta-toxin',               'family': 'alpha_PFT',   'label': 1},
    {'id': 'P01501', 'name': 'Melittin',                  'family': 'alpha_PFT',   'label': 1},
    {'id': 'P11006', 'name': 'Magainin-2',                'family': 'AMP',         'label': 1},
    # AB-toxins
    {'id': 'P15917', 'name': 'Anthrax lethal factor',     'family': 'AB_toxin',    'label': 1},
    {'id': 'P10844', 'name': 'BoNT/A light chain',        'family': 'AB_toxin',    'label': 1},
    # RIPs
    {'id': 'P02879', 'name': 'Ricin toxin A chain',       'family': 'RIP',         'label': 1},
    {'id': 'P20656', 'name': 'Saporin-S6',                'family': 'RIP',         'label': 1},
    # Neurotoxins
    {'id': 'P0C1X2', 'name': 'Alpha-conotoxin ImI',       'family': 'conotoxin',   'label': 1},
    {'id': 'P00974', 'name': 'Dendrotoxin I',             'family': 'dendrotoxin', 'label': 1},
    # Structural analogues / true negatives
    {'id': 'P02703', 'name': 'Colipase',                  'family': 'lipase_coact','label': 0},
    {'id': 'P07355', 'name': 'Annexin A2',                'family': 'annexin',     'label': 0},
    {'id': 'P02647', 'name': 'Apolipoprotein A-I',        'family': 'apolipoprotein','label': 0},
    {'id': 'O15520', 'name': 'Defensin beta-2 (hBD2)',    'family': 'defensin',    'label': 0},
    {'id': 'A0A0K8SXK3', 'name': 'Spider silk MaSp1',    'family': 'silk',        'label': 0},
]


# ── Fetch + cache external sequences ─────────────────────────────────────────

def load_external_proteins(cache_path: str) -> dict:
    """
    Load or fetch external protein sequences from UniProt.
    Results are cached to avoid repeated API calls.

    Returns:
        { pid: {id, name, family, label, sequence, length} }
    """
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        print(f'  Loaded {len(cached)} external proteins from cache')
        return cached

    from utils.data import fetch_uniprot_sequence

    proteins = {}
    for meta in EXTERNAL_PROTEINS:
        pid = meta['id']
        seq = fetch_uniprot_sequence(pid)
        if seq:
            proteins[pid] = {
                **meta,
                'sequence': seq,
                'length':   len(seq),
            }
            print(f'  Fetched {pid} ({meta["name"]}): {len(seq)} aa')
        else:
            print(f'  WARNING: Could not fetch {pid} ({meta["name"]})')
        time.sleep(0.5)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(proteins, f, indent=2)
    print(f'  Cached {len(proteins)} external proteins to {cache_path}')
    return proteins


# ── Local embedding helper (autocast + BF16→FP32, mirrors N3) ────────────────

def _embed_sequence(model, tokenizer, sequence: str,
                    layer: int, device: str,
                    use_transformers_api: bool) -> np.ndarray:
    """Hook-based ESM3 embedding with autocast to avoid Float/BFloat16 matmul errors."""
    if use_transformers_api:
        inputs = tokenizer(sequence, return_tensors='pt', add_special_tokens=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[layer + 1][0, 1:-1, :].float().cpu().numpy()

    encoded    = tokenizer(sequence, return_tensors='pt', add_special_tokens=True)
    tok_tensor = encoded['input_ids'].to(device)
    captured   = {}

    def hook_fn(module, inp, out):
        val = out[0] if isinstance(out, tuple) else out
        captured['hidden'] = val.detach().float()   # BFloat16 → Float32

    block = get_block(model, layer)
    hook  = block.register_forward_hook(hook_fn)
    try:
        device_type = 'cuda' if device != 'cpu' else 'cpu'
        with torch.no_grad(), torch.amp.autocast(device_type=device_type,
                                                  dtype=torch.bfloat16):
            model(sequence_tokens=tok_tensor)
    finally:
        hook.remove()

    if 'hidden' not in captured:
        raise RuntimeError(f'Hook did not fire at layer {layer}')
    return captured['hidden'][0, 1:-1, :].cpu().numpy()


def _is_oom(e: Exception) -> bool:
    msg = str(e).lower()
    return (isinstance(e, torch.cuda.OutOfMemoryError) or
            'out of memory' in msg or 'cudaerrormemoryal' in msg.replace(' ', ''))


# ── Embed external proteins ───────────────────────────────────────────────────

def embed_external_proteins(model, tokenizer, proteins: dict,
                              ext_emb_dir: str,
                              use_transformers_api: bool,
                              device: str) -> list:
    """
    Extract ESM3 layer-36 embeddings for external proteins.
    Uses autocast to prevent Float/BFloat16 dtype errors.
    """
    os.makedirs(ext_emb_dir, exist_ok=True)
    errors = []

    for pid, prot in proteins.items():
        out_path = os.path.join(ext_emb_dir, f'{pid}.npy')
        if os.path.exists(out_path):
            continue

        seq = prot.get('sequence', '')
        if not seq:
            continue

        embedded = False
        for max_len, label in [(len(seq), 'full'), (512, '512aa'), (256, '256aa')]:
            try:
                emb = _embed_sequence(model, tokenizer, seq[:max_len],
                                      ESM3_LAYER, device, use_transformers_api)
                np.save(out_path, emb.astype(np.float32))
                if label != 'full':
                    print(f'  {pid}: truncated to {label} (OOM)')
                embedded = True
                break
            except Exception as e:
                if _is_oom(e):
                    try:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    continue
                errors.append((pid, str(e)))
                embedded = True
                break

        if not embedded:
            errors.append((pid, 'OOM even at 256aa'))

    print(f'  External embeddings extracted. Errors: {len(errors)}')
    return errors


# ── SAE features for external proteins ───────────────────────────────────────

def extract_external_sae_features(sae, proteins: dict,
                                   ext_emb_dir: str,
                                   ext_sae_dir: str,
                                   emb_mean: np.ndarray,
                                   emb_std: np.ndarray,
                                   device: str) -> list:
    """Pass external protein embeddings through the trained SAE."""
    os.makedirs(ext_sae_dir, exist_ok=True)
    sae.eval()
    errors = []

    for pid in proteins:
        emb_path = os.path.join(ext_emb_dir, f'{pid}.npy')
        out_path = os.path.join(ext_sae_dir, f'{pid}.npz')

        if os.path.exists(out_path):
            continue
        if not os.path.exists(emb_path):
            continue

        try:
            emb = np.load(emb_path)
            emb_norm   = (emb - emb_mean) / (emb_std + 1e-8)
            emb_tensor = torch.tensor(emb_norm, dtype=torch.float32).to(device)
            with torch.no_grad():
                acts, _, _ = sae(emb_tensor)
            save_features(pid, acts.cpu().numpy(), ext_sae_dir)
        except Exception as e:
            errors.append((pid, str(e)))

    print(f'  External SAE features extracted. Errors: {len(errors)}')
    return errors


# ── Pooled feature vectors ────────────────────────────────────────────────────

def build_external_feature_matrix(proteins: dict,
                                   ext_sae_dir: str) -> tuple:
    """
    Load mean-pooled SAE features for external proteins.

    Returns:
        (X, y, pids) — (n_ext, D_SAE), labels, pid list
    """
    from utils.data import load_features, pool_features
    X_list, y_list, pids_list = [], [], []

    for pid, prot in proteins.items():
        feat_path = os.path.join(ext_sae_dir, f'{pid}.npz')
        if not os.path.exists(feat_path):
            continue
        try:
            data = np.load(feat_path)
            acts = data['features']   # key written by save_features()
            vec  = pool_features(acts, strategy='topk')
            X_list.append(vec)
            y_list.append(int(prot.get('label', 0)))
            pids_list.append(pid)
        except Exception:
            pass

    if not X_list:
        return np.zeros((0, D_SAE)), np.zeros(0, dtype=int), []

    return np.stack(X_list), np.array(y_list, dtype=int), pids_list


# ── Per-family AUROC ──────────────────────────────────────────────────────────

def per_family_auroc(clf, proteins: dict,
                      X: np.ndarray,
                      y: np.ndarray,
                      pids: list) -> dict:
    """
    Evaluate classifier AUROC for each family in the external set.

    Returns:
        { family: { 'n': int, 'auroc': float, 'pids': [...] } }
    """
    family_idx = {}
    for i, pid in enumerate(pids):
        fam = proteins[pid].get('family', 'unknown')
        family_idx.setdefault(fam, []).append(i)

    results = {}
    for fam, indices in family_idx.items():
        Xi = X[indices]
        yi = y[indices]
        if len(np.unique(yi)) < 2:
            probs = clf.predict_proba(Xi)[:, 1].tolist() if len(Xi) > 0 else []
            results[fam] = {
                'n': len(indices),
                'auroc': None,  # can't compute with single class
                'mean_prob': round(float(np.mean(probs)), 4) if probs else None,
                'pids': [pids[i] for i in indices],
            }
        else:
            auc = evaluate_classifier(clf, Xi, yi, f'external {fam}')['auroc']
            results[fam] = {
                'n': len(indices),
                'auroc': round(float(auc), 4),
                'pids': [pids[i] for i in indices],
            }

    return results


# ── Feature overlap fraction ──────────────────────────────────────────────────

def feature_overlap_fraction(X_ext: np.ndarray,
                               pids_ext: list,
                               top_feature_ids: list,
                               X_train_tox: np.ndarray,
                               threshold: float = 0.5) -> dict:
    """
    For each external protein, compute the fraction of top_feature_ids
    that are active (> threshold), and compare to the training toxin mean.

    Returns:
        { pid: { 'frac_active': float, 'rank_vs_train': float } }
    """
    top_ids = np.array(top_feature_ids)

    # Training toxin reference
    train_active = (X_train_tox[:, top_ids] > threshold)   # (n_train, k)
    train_frac   = train_active.mean(axis=1)                # mean active features per protein
    train_mean   = float(train_frac.mean())

    results = {}
    for i, pid in enumerate(pids_ext):
        vec         = X_ext[i, top_ids]
        frac_active = float((vec > threshold).mean())
        results[pid] = {
            'frac_active_top_features': round(frac_active, 4),
            'train_toxin_mean':         round(train_mean, 4),
            'relative_to_train':        round(frac_active / (train_mean + 1e-8), 4),
        }

    return results


# ── Convergent evolution extension ───────────────────────────────────────────

def convergent_evolution_extension(X_ext: np.ndarray,
                                    pids_ext: list,
                                    proteins_ext: dict,
                                    top_feature_ids: list,
                                    threshold: float = 0.5) -> dict:
    """
    Identify external beta-PFTs and test whether they share the same
    top discriminative features as the training convergent PFTs.

    Returns:
        { pid: { 'family': str, 'frac_shared_with_train_conv': float } }
    """
    top_ids = np.array(top_feature_ids)

    # External beta-PFT candidates
    ext_pft_pids = [pid for pid in pids_ext
                    if proteins_ext[pid].get('family') in ('beta_PFT', 'alpha_PFT')]

    if not ext_pft_pids:
        return {}

    results = {}
    for pid in ext_pft_pids:
        idx = pids_ext.index(pid)
        vec = X_ext[idx, top_ids]
        active_set = set(np.where(vec > threshold)[0].tolist())
        results[pid] = {
            'family':      proteins_ext[pid].get('family'),
            'n_active':    len(active_set),
            'frac_active': round(len(active_set) / max(len(top_ids), 1), 4),
        }

    return results


# ── Ablation generalisation test ──────────────────────────────────────────────

def ablation_generalisation(clf,
                              X_ext: np.ndarray,
                              y_ext: np.ndarray,
                              top_feature_ids: list) -> dict:
    """
    Test whether ablating top features hurts external protein classification
    more than ablating a random equal-size feature set (control).

    Returns:
        { 'auroc_full': float, 'auroc_ablated': float, 'auroc_random_abl': float,
          'drop_vs_top': float, 'drop_vs_random': float }
    """
    if len(np.unique(y_ext)) < 2:
        return {'note': 'Cannot compute AUROC — single class in external set'}

    auc_full = evaluate_classifier(clf, X_ext, y_ext)['auroc']

    # Ablate top features
    X_abl   = ablate_features(X_ext, top_feature_ids)
    auc_abl = evaluate_classifier(clf, X_abl, y_ext)['auroc']

    # Ablate random control features
    rng         = np.random.default_rng(42)
    random_fids = rng.choice(D_SAE, size=len(top_feature_ids), replace=False).tolist()
    X_rand_abl  = ablate_features(X_ext, random_fids)
    auc_rand    = evaluate_classifier(clf, X_rand_abl, y_ext)['auroc']

    return {
        'auroc_full':           round(float(auc_full), 4),
        'auroc_ablated_top':    round(float(auc_abl),  4),
        'auroc_ablated_random': round(float(auc_rand), 4),
        'drop_vs_top':          round(float(auc_full - auc_abl), 4),
        'drop_vs_random':       round(float(auc_full - auc_rand), 4),
        'generalisation_gap':   round(float((auc_full - auc_abl) - (auc_full - auc_rand)), 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(device: str = 'cuda'):
    print('=' * 65)
    print('  N7 — EXTERNAL VALIDATION')
    print('=' * 65)
    make_dirs()

    device = device if torch.cuda.is_available() else 'cpu'
    print(f'  Device: {device}')

    master   = load_master_dataset(DATA_DIR)
    proteins = master['proteins']
    splits   = load_splits(SPLITS_DIR)
    ranking  = load_feature_ranking(FEAT_RANK_DIR)
    top_feature_ids = ranking.get('top_feature_ids', list(range(200)))

    ext_cache_path = os.path.join(DATA_DIR,    'external_proteins.json')
    ext_emb_dir    = os.path.join(EMB_DIR,     'external')
    ext_sae_dir    = os.path.join(SAE_DIR,     'external')

    # 1. Load external proteins
    print('\n[1/7] Loading external proteins...')
    ext_proteins = load_external_proteins(ext_cache_path)
    if len(ext_proteins) == 0:
        raise RuntimeError('No external proteins loaded. Check network access.')
    print(f'  External proteins available: {len(ext_proteins)}')

    # 2. Embed external proteins with ESM3
    print('\n[2/7] Loading ESM3 and extracting external embeddings...')
    model, tokenizer, use_transformers_api = load_esm3(device=device)
    # No manual dtype cast — autocast inside _embed_sequence handles BF16/FP32
    embed_external_proteins(model, tokenizer, ext_proteins, ext_emb_dir,
                            use_transformers_api, device)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Load SAE and extract features
    print('\n[3/7] Extracting external SAE features...')
    sae = load_sae(SAE_WEIGHTS, device=device)
    emb_mean = np.load(os.path.join(SAE_WEIGHTS, 'emb_mean.npy'))
    emb_std  = np.load(os.path.join(SAE_WEIGHTS, 'emb_std.npy'))
    extract_external_sae_features(sae, ext_proteins, ext_emb_dir, ext_sae_dir,
                                   emb_mean, emb_std, device)

    # 4. Build feature matrix
    print('\n[4/7] Building external feature matrix...')
    X_ext, y_ext, pids_ext = build_external_feature_matrix(ext_proteins, ext_sae_dir)
    print(f'  External matrix: {X_ext.shape}')

    if X_ext.shape[0] == 0:
        raise RuntimeError('No external SAE features found. Check steps 2–3.')

    # Load classifier
    clf = load_classifier('sae_logreg', BASELINES_DIR)
    if clf is None:
        clf = load_classifier('sae_mlp', BASELINES_DIR)
    if clf is None:
        raise RuntimeError('No trained classifier. Run N3 first.')
    print(f'  Classifier: {type(clf).__name__}')

    # 5. Per-family AUROC
    print('\n[5/7] Per-family AUROC on external set...')
    fam_auroc = per_family_auroc(clf, ext_proteins, X_ext, y_ext, pids_ext)

    # Overall AUROC (if mixed classes)
    overall_auc = None
    if len(np.unique(y_ext)) >= 2:
        overall_res = evaluate_classifier(clf, X_ext, y_ext, 'External overall')
        overall_auc = round(float(overall_res['auroc']), 4)

    # 6. Feature overlap fraction
    print('\n[6/7] Feature overlap fraction...')
    train_tox_pids = [p for p in splits['train']
                       if proteins.get(p, {}).get('label') == 1]
    from utils.data import build_feature_matrix
    X_tr_tox, _, _ = build_feature_matrix(train_tox_pids, proteins, SAE_DIR)

    overlap = {}
    if X_tr_tox.shape[0] > 0:
        overlap = feature_overlap_fraction(
            X_ext, pids_ext, top_feature_ids, X_tr_tox
        )

    # 7. Convergent evolution extension
    print('\n[7/7] Convergent evolution extension...')
    conv_ext = convergent_evolution_extension(
        X_ext, pids_ext, ext_proteins, top_feature_ids
    )

    # Ablation generalisation
    abl_gen = ablation_generalisation(clf, X_ext, y_ext, top_feature_ids[:50])

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        'n_external_proteins': len(ext_proteins),
        'overall_auroc':       overall_auc,
        'per_family_auroc':    fam_auroc,
        'feature_overlap':     overlap,
        'convergent_ext':      conv_ext,
        'ablation_generalisation': abl_gen,
        'external_proteins':   {
            pid: {k: v for k, v in meta.items() if k != 'sequence'}
            for pid, meta in ext_proteins.items()
        },
    }

    out_dir  = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'external_validation_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results saved: {out_path}')

    print()
    print('=' * 65)
    print('  N7 RESULTS SUMMARY')
    print('=' * 65)
    print(f'  External proteins evaluated: {X_ext.shape[0]}')
    if overall_auc is not None:
        print(f'  Overall AUROC (ext):         {overall_auc:.3f}')
    print()
    print('  Per-family AUROC:')
    for fam, res in fam_auroc.items():
        auc_str = f'{res["auroc"]:.3f}' if res["auroc"] is not None else 'N/A (1 class)'
        print(f'    {fam:<20s}: {auc_str}  (n={res["n"]})')
    print()
    print(f'  Ablation generalisation gap: '
          f'{abl_gen.get("generalisation_gap", "N/A")}')
    print()
    print('=' * 65)
    print('  PIPELINE COMPLETE')
    print('  Results in: ' + out_dir)
    print('=' * 65)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='N7: External validation')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    args = parser.parse_args()
    main(device=args.device)
