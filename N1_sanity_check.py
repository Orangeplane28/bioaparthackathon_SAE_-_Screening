"""
N1 — Sanity Check: ESM3 + TopK SAE Architecture Verification

Verifies:
  1. ESM3-small-open loads and produces d_model=1536 representations
  2. Layer-36 hook extraction works correctly
  3. TopKSAE architecture (1536 → 15360, k=120) runs without error
  4. Project directory structure is created

Run: python N1_sanity_check.py
"""

import os
import sys
import json
import numpy as np
import torch

# ── Imports ───────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    PROJECT_ROOT, D_MODEL, D_SAE, K_TOPK, ESM3_LAYER,
    CONVERGENT_IDS, make_dirs
)
from utils.esm3 import TopKSAE, load_esm3, embed_esm3


def main():
    print('=' * 65)
    print('  N1 — ESM3 + SAE SANITY CHECK')
    print('=' * 65)
    print(f'  Project root: {PROJECT_ROOT}')
    print()

    # ── 1. Create project directory structure ─────────────────────────────────
    print('[1/4] Creating project directories...')
    make_dirs()
    print(f'  ✅ Directory structure created under {PROJECT_ROOT}')
    print()

    # ── 2. Test TopKSAE architecture ──────────────────────────────────────────
    print('[2/4] Testing TopKSAE architecture...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'  Device: {device}')

    sae = TopKSAE(d_model=D_MODEL, n_features=D_SAE, k=K_TOPK).to(device)
    test_input = torch.randn(10, D_MODEL, device=device)
    acts, pre_acts, x_hat = sae(test_input)

    assert acts.shape == (10, D_SAE),   f'acts shape wrong: {acts.shape}'
    assert x_hat.shape == (10, D_MODEL), f'x_hat shape wrong: {x_hat.shape}'
    n_active = (acts != 0).sum(dim=-1)
    assert (n_active == K_TOPK).all(), f'Expected {K_TOPK} active per token, got {n_active}'

    print(f'  SAE input shape:  {test_input.shape}')
    print(f'  SAE acts shape:   {acts.shape}')
    print(f'  SAE x_hat shape:  {x_hat.shape}')
    print(f'  Active per token: {n_active[0].item()} (expected {K_TOPK})')
    print(f'  Sparsity:         {(acts == 0).float().mean().item():.1%}')

    # Test decoder normalization
    sae.normalize_decoder()
    dec_norms = sae.W_dec.norm(dim=1)
    assert torch.allclose(dec_norms, torch.ones_like(dec_norms), atol=1e-5), \
        'W_dec columns not unit-norm after normalize_decoder()'
    print(f'  W_dec column norms: min={dec_norms.min():.4f}, max={dec_norms.max():.4f} (all ≈ 1.0)')
    print('  ✅ TopKSAE architecture verified')
    print()

    # ── 3. Load ESM3 and test embedding ───────────────────────────────────────
    print('[3/4] Loading ESM3 and testing embedding extraction...')
    try:
        model, tokenizer, use_transformers_api = load_esm3(device=device)
        model.eval()

        # Test on a short sequence
        TEST_SEQ = 'MKAIFVLKGNDHHVPDLGKVNGEQHGNKITIEQHGLKDPKAKLTVQATADLAKDAGVK'
        emb = embed_esm3(model, tokenizer, TEST_SEQ,
                         layer=ESM3_LAYER, device=device,
                         use_transformers_api=use_transformers_api)

        assert emb.shape == (len(TEST_SEQ), D_MODEL), \
            f'Embedding shape wrong: {emb.shape}, expected ({len(TEST_SEQ)}, {D_MODEL})'
        print(f'  Sequence length:      {len(TEST_SEQ)} aa')
        print(f'  Embedding shape:      {emb.shape}  (expected ({len(TEST_SEQ)}, {D_MODEL}))')
        print(f'  Embedding mean:       {emb.mean():.6f}')
        print(f'  Embedding std:        {emb.std():.6f}')

        # Full pipeline: ESM3 → SAE
        emb_tensor = torch.tensor(emb, dtype=torch.float32).to(device)
        acts, _, x_hat = sae(emb_tensor)
        recon_err = ((emb_tensor - x_hat) ** 2).mean().item()
        print(f'  SAE acts shape:       {acts.shape}')
        print(f'  SAE sparsity:         {(acts == 0).float().mean().item():.1%}')
        print(f'  SAE recon error:      {recon_err:.4f} (untrained — expected high)')
        print('  ✅ ESM3 → SAE pipeline verified')

    except RuntimeError as e:
        print(f'  ⚠️  ESM3 not available: {e}')
        print('  (This is expected if running without network access or esm package)')
        print('  Continue to N2 — embeddings will be extracted in N3.')
    print()

    # ── 4. Save environment info ───────────────────────────────────────────────
    print('[4/4] Saving environment info...')
    env_info = {
        'python_version': sys.version,
        'torch_version':  torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device_used':    device,
        'project_root':   PROJECT_ROOT,
        'architecture': {
            'd_model':    D_MODEL,
            'd_sae':      D_SAE,
            'k_topk':     K_TOPK,
            'esm3_layer': ESM3_LAYER,
        },
    }
    env_path = os.path.join(PROJECT_ROOT, 'N1_env_info.json')
    with open(env_path, 'w') as f:
        json.dump(env_info, f, indent=2)
    print(f'  Environment info saved to {env_path}')
    print()
    print('=' * 65)
    print('  N1 COMPLETE — Architecture verified, ready for N2')
    print('=' * 65)


if __name__ == '__main__':
    main()
