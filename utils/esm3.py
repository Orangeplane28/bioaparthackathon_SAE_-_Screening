"""
utils/esm3.py — ESM3 model loading, embedding extraction, and TopK SAE.

Used by: N1, N3, N7
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── TopK Sparse Autoencoder ───────────────────────────────────────────────────

class TopKSAE(nn.Module):
    """
    Sparse Autoencoder with TopK activation.

    Architecture:
        Encoder: x → (x - b_pre) @ W_enc + b_enc → TopK(pre_acts) → acts
        Decoder: acts @ W_dec + b_dec → x_hat

    Args:
        d_model:    Input dimension (ESM3 d_model = 1536)
        n_features: SAE width (10× expansion = 15360)
        k:          Number of active features per token (120)
    """

    def __init__(self, d_model: int = 1536, n_features: int = 15360, k: int = 120):
        super().__init__()
        self.d_model    = d_model
        self.n_features = n_features
        self.k          = k

        self.b_pre = nn.Parameter(torch.zeros(d_model))
        self.W_enc = nn.Parameter(torch.empty(d_model, n_features))
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.W_dec = nn.Parameter(torch.empty(n_features, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def encode(self, x: torch.Tensor):
        """Returns (acts, pre_acts). Acts is sparse (TopK non-zero)."""
        pre = (x - self.b_pre) @ self.W_enc + self.b_enc
        topk_v, topk_i = torch.topk(pre, self.k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, topk_i, F.relu(topk_v))
        return acts, pre

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        acts, pre_acts = self.encode(x)
        x_hat = self.decode(acts)
        return acts, pre_acts, x_hat

    @torch.no_grad()
    def normalize_decoder(self):
        """Keep W_dec columns unit-norm (call after each optimizer step)."""
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)


# ── ESM3 model loading ────────────────────────────────────────────────────────

def load_esm3(device: str = 'cuda'):
    """
    Load ESM3-small-open model and tokenizer.

    Tries the official EvolutionaryScale `esm` package first,
    then falls back to the HuggingFace checkpoint.

    Returns:
        model, tokenizer, use_transformers_api (bool)

    Raises:
        RuntimeError if neither path succeeds.
    """
    # Path 1: official esm package (v3.x API — underscores, EsmSequenceTokenizer)
    try:
        from esm.models.esm3 import ESM3
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
        tokenizer = EsmSequenceTokenizer()
        model = ESM3.from_pretrained('esm3_sm_open_v1').to(device).eval()
        print('  ESM3 loaded via official esm package')
        return model, tokenizer, False
    except Exception as e1:
        print(f'  Official esm package unavailable: {e1}')

    # Path 2: HuggingFace EvolutionaryScale checkpoint
    try:
        from transformers import AutoTokenizer, AutoModel
        hf_id = 'EvolutionaryScale/esm3-sm-open-v1'
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModel.from_pretrained(hf_id).to(device).eval()
        print(f'ESM3 loaded via HuggingFace ({hf_id})')
        return model, tokenizer, True
    except Exception as e2:
        print(f'  HuggingFace ESM3 unavailable: {e2}')

    # Hard stop — do NOT fall back to ESM2 (d_model mismatch: 1280 ≠ 1536)
    raise RuntimeError(
        'Cannot load ESM3. Install via:\n'
        '  pip install esm   (official EvolutionaryScale package)\n'
        'or ensure HuggingFace hub access for EvolutionaryScale/esm3-sm-open-v1.\n'
        'Do NOT substitute ESM2 — d_model mismatch (ESM2=1280, ESM3=1536) '
        'will produce garbage SAE activations.'
    )


def load_sae(sae_weights_dir: str, device: str = 'cpu',
             d_model: int = 1536, n_features: int = 15360, k: int = 120) -> TopKSAE:
    """Load trained SAE weights from disk."""
    import os
    sae = TopKSAE(d_model=d_model, n_features=n_features, k=k)
    for fname in ['sae_layer36_final.pt', 'sae_final.pt', 'sae_best.pt']:
        path = os.path.join(sae_weights_dir, fname)
        if os.path.exists(path):
            sae.load_state_dict(torch.load(path, map_location='cpu'))
            sae = sae.to(device).eval()
            print(f'SAE loaded from {fname}')
            return sae
    print('WARNING: No SAE weights found — using untrained SAE (run N3 first)')
    return sae.to(device).eval()


# ── ESM3 embedding extraction ─────────────────────────────────────────────────

def get_block(model, layer_idx: int):
    """
    Return the transformer block at layer_idx from an ESM3 model,
    trying multiple known attribute name schemes.
    """
    for attr in ['transformer', 'encoder', 'esm', 'model']:
        base = getattr(model, attr, None)
        if base is None:
            continue
        for sub in ['blocks', 'layers', 'layer', 'transformer_stack']:
            blocks = getattr(base, sub, None)
            if blocks is not None:
                try:
                    return blocks[layer_idx]
                except (IndexError, TypeError):
                    continue
    raise AttributeError(
        f'Block {layer_idx} not found in model. '
        'Inspect model.named_modules() to find the correct attribute path.'
    )


def embed_esm3(model, tokenizer, sequence: str, layer: int = 36,
               device: str = 'cuda', use_transformers_api: bool = False) -> np.ndarray:
    """
    Extract layer-{layer} residue representations from ESM3.

    Args:
        model:                ESM3 model (official or HuggingFace)
        tokenizer:            Corresponding tokenizer
        sequence:             Amino acid sequence string
        layer:                Layer index to extract (default 36 = 75% depth)
        device:               'cuda' or 'cpu'
        use_transformers_api: True if model was loaded via HuggingFace transformers

    Returns:
        np.ndarray of shape (seq_len, d_model) — no BOS/EOS tokens
    """
    if use_transformers_api:
        inputs = tokenizer(sequence, return_tensors='pt', add_special_tokens=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # hidden_states[0] = embedding layer, [layer+1] = transformer layer
        return outputs.hidden_states[layer + 1][0, 1:-1, :].float().cpu().numpy()

    # Hook-based extraction for official ESM package (v3.x)
    # Use the transformers-style tokenizer call — works for EsmSequenceTokenizer
    encoded = tokenizer(sequence, return_tensors='pt', add_special_tokens=True)
    tok_tensor = encoded['input_ids'].to(device)

    captured = {}

    def hook_fn(module, inp, out):
        val = out[0] if isinstance(out, tuple) else out
        captured['hidden'] = val.detach().float()

    block = get_block(model, layer)
    hook = block.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(sequence_tokens=tok_tensor)
    finally:
        hook.remove()

    if 'hidden' not in captured:
        raise RuntimeError(
            f'Hook did not capture hidden state at layer {layer}. '
            'Inspect model.named_modules() to verify block structure.'
        )

    # Remove BOS and EOS special tokens → shape (seq_len, d_model)
    return captured['hidden'][0, 1:-1, :].cpu().numpy()


def extract_sae_features(model, tokenizer, sae, sequence: str,
                          emb_mean: np.ndarray = None, emb_std: np.ndarray = None,
                          layer: int = 36, device: str = 'cuda',
                          use_transformers_api: bool = False,
                          max_seq_len: int = 1022) -> np.ndarray:
    """
    Full ESM3 → normalize → SAE pipeline for a single sequence.

    Returns:
        np.ndarray of shape (seq_len, D_SAE) — per-residue SAE activations
    """
    seq = sequence[:max_seq_len]
    emb = embed_esm3(model, tokenizer, seq, layer=layer,
                     device=device, use_transformers_api=use_transformers_api)

    if emb_mean is not None and emb_std is not None:
        emb = (emb - emb_mean) / (emb_std + 1e-8)

    emb_tensor = torch.tensor(emb, dtype=torch.float32).to(device)
    with torch.no_grad():
        acts, _, _ = sae(emb_tensor)
    return acts.cpu().numpy()
