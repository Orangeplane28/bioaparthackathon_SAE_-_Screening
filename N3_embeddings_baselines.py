"""
N3 — ESM3 Embeddings, SAE Training, and Baseline Classifiers

Steps:
  1. Extract ESM3 layer-36 embeddings for all proteins (checkpointed)
  2. Train TopK SAE on training set embeddings (50 epochs)
  3. Extract SAE feature activations for all proteins
  4. Train and evaluate 4 baseline classifiers (LogReg, MLP on raw + SAE features)

GPU strongly recommended for steps 1–3.

Run: python N3_embeddings_baselines.py [--device cuda] [--epochs 50]
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    PROJECT_ROOT, DATA_DIR, SAE_DIR, EMB_DIR, SPLITS_DIR,
    BASELINES_DIR, SAE_WEIGHTS, D_MODEL, D_SAE, K_TOPK, ESM3_LAYER,
    SAE_N_EPOCHS, SAE_BATCH, SAE_LR, SAE_AUX_LAMBDA, make_dirs
)
from utils.esm3 import TopKSAE, load_esm3, embed_esm3
from utils.data import (
    load_master_dataset, load_splits, load_features, save_features, pool_features,
    build_feature_matrix, load_classifier
)
from utils.stats import evaluate_classifier


def embed_all_proteins(model, tokenizer, proteins: dict, emb_dir: str,
                        use_transformers_api: bool, device: str,
                        layer: int = ESM3_LAYER):
    """Extract ESM3 embeddings for all proteins with checkpointing."""
    os.makedirs(emb_dir, exist_ok=True)
    errors = []
    n_total = len(proteins)

    for i, (pid, prot) in enumerate(proteins.items()):
        out_path = os.path.join(emb_dir, f'{pid}.npy')
        if os.path.exists(out_path):
            continue  # already done

        seq = prot.get('sequence', '')
        if not seq:
            continue

        try:
            emb = embed_esm3(model, tokenizer, seq, layer=layer,
                             device=device, use_transformers_api=use_transformers_api)
            np.save(out_path, emb.astype(np.float32))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            try:
                emb = embed_esm3(model, tokenizer, seq[:512], layer=layer,
                                 device=device, use_transformers_api=use_transformers_api)
                np.save(out_path, emb.astype(np.float32))
                print(f'  {pid}: truncated to 512 aa (OOM)')
            except Exception as e2:
                errors.append((pid, str(e2)))
        except Exception as e:
            errors.append((pid, str(e)))

        if (i + 1) % 50 == 0:
            print(f'  Embeddings: {i+1}/{n_total} processed')

    print(f'  Embedding extraction complete. Errors: {len(errors)}')
    return errors


def train_sae(emb_dir: str, train_pids: list, sae_weights_dir: str,
              n_epochs: int = SAE_N_EPOCHS, batch_size: int = SAE_BATCH,
              lr: float = SAE_LR, aux_lambda: float = SAE_AUX_LAMBDA,
              device: str = 'cpu') -> TopKSAE:
    """Train TopK SAE on training set embeddings."""
    os.makedirs(sae_weights_dir, exist_ok=True)

    # Load all training embeddings into memory
    print('  Loading training embeddings...')
    all_embs = []
    for pid in train_pids:
        path = os.path.join(emb_dir, f'{pid}.npy')
        if os.path.exists(path):
            emb = np.load(path)        # (seq_len, D_MODEL)
            all_embs.append(emb)

    if not all_embs:
        raise RuntimeError('No training embeddings found. Run embedding extraction first.')

    # Compute normalization stats
    concat = np.concatenate(all_embs, axis=0)   # (N_tokens, D_MODEL)
    emb_mean = concat.mean(axis=0).astype(np.float32)
    emb_std  = concat.std(axis=0).astype(np.float32) + 1e-8
    np.save(os.path.join(sae_weights_dir, 'emb_mean.npy'), emb_mean)
    np.save(os.path.join(sae_weights_dir, 'emb_std.npy'),  emb_std)
    print(f'  Training tokens: {len(concat):,}')

    # Normalize
    concat_norm = (concat - emb_mean) / emb_std

    # Build tensor dataset
    X_tensor = torch.tensor(concat_norm, dtype=torch.float32)
    dataset   = torch.utils.data.TensorDataset(X_tensor)
    loader    = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, drop_last=True)

    sae = TopKSAE(d_model=D_MODEL, n_features=D_SAE, k=K_TOPK).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    print(f'  Training SAE: {n_epochs} epochs, batch={batch_size}, lr={lr}')
    best_loss = float('inf')

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches  = 0

        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            acts, pre_acts, x_hat = sae(x_batch)

            l_recon = ((x_batch - x_hat) ** 2).mean()
            l_aux   = ((pre_acts - acts.detach()) ** 2).mean()
            loss    = l_recon + aux_lambda * l_aux

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()
            sae.normalize_decoder()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch+1:3d}/{n_epochs}: loss={avg_loss:.6f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(sae.state_dict(), os.path.join(sae_weights_dir, 'sae_best.pt'))

    torch.save(sae.state_dict(), os.path.join(sae_weights_dir, 'sae_layer36_final.pt'))
    print(f'  SAE training complete. Best loss: {best_loss:.6f}')
    return sae


def extract_sae_features_all(sae, proteins: dict, emb_dir: str, sae_dir: str,
                               emb_mean: np.ndarray, emb_std: np.ndarray,
                               device: str):
    """Pass all protein embeddings through the trained SAE."""
    os.makedirs(sae_dir, exist_ok=True)
    sae.eval()
    errors = []

    for i, pid in enumerate(proteins):
        emb_path = os.path.join(emb_dir, f'{pid}.npy')
        if not os.path.exists(emb_path):
            continue

        out_path = os.path.join(sae_dir, f'{pid}.npz')
        if os.path.exists(out_path):
            continue

        try:
            emb = np.load(emb_path)
            emb_norm = (emb - emb_mean) / (emb_std + 1e-8)
            emb_tensor = torch.tensor(emb_norm, dtype=torch.float32).to(device)
            with torch.no_grad():
                acts, _, _ = sae(emb_tensor)
            save_features(pid, acts.cpu().numpy(), sae_dir)
        except Exception as e:
            errors.append((pid, str(e)))

        if (i + 1) % 100 == 0:
            print(f'  SAE features: {i+1}/{len(proteins)} processed')

    print(f'  SAE feature extraction complete. Errors: {len(errors)}')
    return errors


def train_baselines(proteins: dict, splits: dict, sae_dir: str,
                    baselines_dir: str, emb_dir: str = None):
    """Train and evaluate LogReg and MLP on raw ESM3 and SAE features."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    os.makedirs(baselines_dir, exist_ok=True)
    results = {}

    for feat_type, feat_dir in [('SAE', sae_dir), ('ESM3_raw', emb_dir)]:
        if feat_dir is None or not os.path.exists(feat_dir):
            continue

        X_tr, y_tr, _ = build_feature_matrix(splits['train'], proteins, feat_dir)
        X_va, y_va, _ = build_feature_matrix(splits['val'],   proteins, feat_dir)
        X_te, y_te, _ = build_feature_matrix(splits['test'],  proteins, feat_dir)

        if X_tr.shape[0] < 10:
            continue

        for clf_name, clf_obj in [
            ('LogReg',  LogisticRegression(max_iter=500, C=0.1,
                                           class_weight='balanced', random_state=42, n_jobs=-1)),
            ('MLP',     MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=200,
                                      random_state=42, early_stopping=True)),
        ]:
            name = f'{feat_type}_{clf_name}'
            print(f'  Training {name}...')
            pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf_obj)])
            pipe.fit(X_tr, y_tr)
            joblib.dump(pipe, os.path.join(baselines_dir, f'{name.lower()}.pkl'))

            results[name] = {
                'train': evaluate_classifier(pipe, X_tr, y_tr, f'{name} train'),
                'val':   evaluate_classifier(pipe, X_va, y_va, f'{name} val'),
                'test':  evaluate_classifier(pipe, X_te, y_te, f'{name} test'),
            }

    # Save results
    with open(os.path.join(baselines_dir, 'baseline_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('  Baseline results saved.')
    return results


def main(device: str = 'cuda', n_epochs: int = SAE_N_EPOCHS):
    print('=' * 65)
    print('  N3 — ESM3 EMBEDDINGS, SAE TRAINING, BASELINES')
    print('=' * 65)
    make_dirs()

    master   = load_master_dataset(DATA_DIR)
    proteins = master['proteins']
    splits   = load_splits(SPLITS_DIR)

    device = device if torch.cuda.is_available() else 'cpu'
    print(f'  Device: {device}')
    print(f'  Proteins: {len(proteins)}')
    print()

    # 1. Load ESM3
    print('[1/4] Loading ESM3...')
    model, tokenizer, use_transformers_api = load_esm3(device=device)

    # 2. Embed all proteins
    print('\n[2/4] Extracting ESM3 embeddings (checkpointed)...')
    embed_all_proteins(model, tokenizer, proteins, EMB_DIR,
                       use_transformers_api, device)

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Train SAE
    print('\n[3/4] Training TopK SAE...')
    sae = train_sae(EMB_DIR, splits['train'], SAE_WEIGHTS,
                    n_epochs=n_epochs, device=device)

    # Load normalization stats
    emb_mean = np.load(os.path.join(SAE_WEIGHTS, 'emb_mean.npy'))
    emb_std  = np.load(os.path.join(SAE_WEIGHTS, 'emb_std.npy'))

    # Extract SAE features for all proteins
    extract_sae_features_all(sae, proteins, EMB_DIR, SAE_DIR, emb_mean, emb_std, device)

    # 4. Train baselines
    print('\n[4/4] Training baseline classifiers...')
    results = train_baselines(proteins, splits, SAE_DIR, BASELINES_DIR, EMB_DIR)

    print()
    print('=' * 65)
    print('  N3 RESULTS SUMMARY')
    print('=' * 65)
    for model_name, res in results.items():
        test_r = res.get('test', {})
        print(f'  {model_name:<25s}: AUROC={test_r.get("auroc", 0):.3f}, '
              f'F1={test_r.get("f1", 0):.3f}')
    print()
    print('NEXT: python N4_feature_discovery.py')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='N3: Embeddings, SAE training, baselines')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--epochs', type=int, default=SAE_N_EPOCHS)
    args = parser.parse_args()
    main(device=args.device, n_epochs=args.epochs)
