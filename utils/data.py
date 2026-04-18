"""
utils/data.py — Data I/O helpers shared across pipeline stages.

Used by: N2, N3, N4, N5, N6, N7
"""

import os
import json
import time
import numpy as np
import requests
from typing import Optional, List, Dict, Tuple

from config import (
    SAE_DIR, DATA_DIR, SPLITS_DIR, BASELINES_DIR,
    UNIPROT_BASE, RATE_LIMIT_PAUSE, D_SAE
)


# ── Feature file I/O ──────────────────────────────────────────────────────────

def load_features(pid: str, sae_dir: str = SAE_DIR) -> Optional[np.ndarray]:
    """
    Load per-residue SAE feature matrix for a protein.

    Returns:
        np.ndarray of shape (seq_len, D_SAE), or None if file not found.
    """
    path = os.path.join(sae_dir, f'{pid}.npz')
    if not os.path.exists(path):
        return None
    return np.load(path)['features']


def save_features(pid: str, features: np.ndarray, sae_dir: str = SAE_DIR):
    """Save per-residue SAE feature matrix as compressed npz."""
    os.makedirs(sae_dir, exist_ok=True)
    np.savez_compressed(os.path.join(sae_dir, f'{pid}.npz'), features=features)


def pool_features(features: np.ndarray, strategy: str = 'mean',
                  topk_frac: float = 0.1) -> np.ndarray:
    """
    Pool per-residue features (seq_len, D_SAE) → protein vector (D_SAE,).

    Strategies:
        'mean'  — mean over residues
        'max'   — max over residues
        'topk'  — mean of top-k% residues by activation magnitude
    """
    if strategy == 'mean':
        return features.mean(axis=0)
    elif strategy == 'max':
        return features.max(axis=0)
    elif strategy == 'topk':
        k = max(1, int(features.shape[0] * topk_frac))
        magnitudes = np.abs(features).sum(axis=1)
        top_idx = np.argsort(magnitudes)[::-1][:k]
        return features[top_idx].mean(axis=0)
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy!r}")


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_master_dataset(data_dir: str = DATA_DIR) -> Dict:
    """Load master dataset JSON. Tries ESM3-specific name first."""
    for fname in ['master_dataset_esm3.json', 'master_dataset.json']:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError(
        f'No master dataset found in {data_dir}. Run N2 first.'
    )


def load_splits(splits_dir: str = SPLITS_DIR) -> Dict[str, List[str]]:
    """Load train/val/test protein ID lists."""
    splits = {}
    for split_name in ['train', 'val', 'test']:
        path = os.path.join(splits_dir, f'{split_name}_pids.json')
        if os.path.exists(path):
            with open(path) as f:
                splits[split_name] = json.load(f)
        else:
            splits[split_name] = []
    return splits


def load_feature_ranking(feat_rank_dir: str) -> Dict:
    """
    Load N4 feature ranking. Tries canonical name first, then fallback.
    Returns the full ranking dict with 'top_feature_ids' key.
    """
    for fname in ['top_toxin_features.json', 'feature_ranking.json']:
        path = os.path.join(feat_rank_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            print(f'Feature ranking loaded from: {fname}')
            return data
    print('WARNING: No feature ranking found — using placeholder IDs 0..199')
    return {'top_feature_ids': list(range(200))}


def load_classifier(baselines_dir: str = BASELINES_DIR):
    """Load trained SAE classifier (joblib pickle). Returns None if not found."""
    import joblib
    for fname in ['sae_logreg.pkl', 'sae_mlp.pkl', 'raw_logreg.pkl']:
        path = os.path.join(baselines_dir, fname)
        if os.path.exists(path):
            clf = joblib.load(path)
            print(f'Classifier loaded: {fname}')
            return clf
    print('WARNING: No saved classifier found. Run N3 first.')
    return None


def build_feature_matrix(pid_list: List[str], proteins: Dict,
                         sae_dir: str = SAE_DIR,
                         pooling: str = 'mean') -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build (X, y, valid_pids) feature matrix for a list of protein IDs.

    Returns:
        X:          (n, D_SAE) float32 array
        y:          (n,) int array of labels (1=toxin, 0=benign)
        valid_pids: list of PIDs that had feature files
    """
    X, y, valid_pids = [], [], []
    for pid in pid_list:
        feats = load_features(pid, sae_dir)
        if feats is None:
            continue
        X.append(pool_features(feats, pooling))
        y.append(int(proteins.get(pid, {}).get('label', 0)))
        valid_pids.append(pid)
    if X:
        return np.stack(X).astype(np.float32), np.array(y, dtype=int), valid_pids
    return np.zeros((0, D_SAE), dtype=np.float32), np.array([], dtype=int), []


# ── UniProt API ───────────────────────────────────────────────────────────────

def fetch_uniprot_sequence(uniprot_id: str, retries: int = 3,
                            pause: float = RATE_LIMIT_PAUSE) -> Optional[str]:
    """
    Fetch protein sequence from UniProt REST API.

    Returns:
        Amino acid sequence string, or None if not found / fetch failed.
    """
    url = f'{UNIPROT_BASE}/{uniprot_id}.fasta'
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                lines = resp.text.strip().split('\n')
                return ''.join(l for l in lines if not l.startswith('>'))
            elif resp.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(pause * (attempt + 1))
    return None


def uniprot_search(query: str, fields: str = 'accession,sequence,protein_name,organism_name',
                   size: int = 50) -> List[Dict]:
    """
    Search UniProt REST API and return parsed results.

    Args:
        query:  UniProt query string (e.g. 'keyword:toxin reviewed:yes')
        fields: Comma-separated response fields
        size:   Max results to return

    Returns:
        List of dicts with keys from `fields`.
    """
    url = f'{UNIPROT_BASE}/search'
    params = {'query': query, 'fields': fields, 'format': 'json', 'size': size}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        results = resp.json().get('results', [])
        return results
    except Exception as e:
        print(f'UniProt search failed for query={query!r}: {e}')
        return []


def fetch_protein_family(uniprot_id: str, pause: float = RATE_LIMIT_PAUSE) -> str:
    """Fetch protein family annotation from UniProt JSON endpoint."""
    url = f'{UNIPROT_BASE}/{uniprot_id}.json'
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            for comment in data.get('comments', []):
                if comment.get('commentType') == 'SIMILARITY':
                    texts = comment.get('texts', [])
                    if texts:
                        return texts[0].get('value', f'cluster_{uniprot_id[:3]}')
            names = data.get('proteinDescription', {})
            rec = names.get('recommendedName', {})
            full = rec.get('fullName', {}).get('value', '')
            if full:
                return full.split(',')[0].strip()
    except Exception:
        pass
    time.sleep(pause)
    return f'cluster_{uniprot_id[:3]}'
