"""ESM3 per-layer mean-pooled embeddings via forward hooks.

ESM3 does not expose hidden states through its forward() API, so we register
hooks on the encoder output and each transformer block to collect intermediate
representations.

Layer indexing (same convention as embed.py):
  layer 0       : EncodeInputs output (token embeddings combined across tracks)
  layer 1..N    : output of transformer blocks 1..N
  (final norm is applied internally but not captured as a separate layer)

Prerequisites
-------------
1. Accept the license at huggingface.co/EvolutionaryScale/esm3-sm-open-v1
2. huggingface-cli login  (or set HF_TOKEN env var)

The first call to load_esm3() will download ~2.9 GB of weights into the HF
cache (~/.cache/huggingface/).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from alxbio_esm.device_utils import clear_cache, coerce_precision, get_device

# Load .env from the project root (or any parent) so HF_TOKEN is available
# before huggingface_hub tries to authenticate. Safe no-op if .env is absent.
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)

Precision = Literal["bf16", "fp16", "fp32"]

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

_DEFAULT_MODEL = "esm3_sm_open_v1"  # 1.4B params, d_model=1536, 48 layers


# ─────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────

def load_esm3(
    model_name: str = _DEFAULT_MODEL,
    device: str = "auto",
    precision: Precision = "bf16",
) -> object:
    """
    Load an ESM3 model with a chosen numeric precision.

    from_pretrained() downloads weights from HuggingFace on first call and
    always casts to bfloat16 on GPU. We override that cast here so precision
    is an explicit parameter.

    Parameters
    ----------
    model_name : currently only "esm3_sm_open_v1" (the open-access small model)
    device     : "cuda" or "cpu"
    precision  : "bf16" (default, recommended), "fp16", or "fp32"

    Returns
    -------
    ESM3 model in eval mode at the requested precision.
    """
    try:
        from esm.models.esm3 import ESM3
    except ImportError as exc:
        raise ImportError("Run: uv add esm") from exc

    device = get_device(device)
    precision = coerce_precision(precision, device)
    dtype = _DTYPE_MAP[precision]
    print(f"[esm3] Loading {model_name} in {precision} on {device} …")

    model = ESM3.from_pretrained(model_name, device=torch.device(device))
    model = model.to(dtype).eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    n_layers = len(model.transformer.blocks)
    d_model = model.transformer.blocks[0].ffn[0].weight.shape[1]
    print(f"[esm3] {n_params:.1f}B params | {n_layers} layers | d_model={d_model} | {precision}")
    return model


# ─────────────────────────────────────────────────────────────
# Per-layer embedding via forward hooks
# ─────────────────────────────────────────────────────────────

class _LayerCollector:
    """Registers forward hooks to collect per-layer hidden states."""

    def __init__(self, model) -> None:
        self._outputs: dict[int, torch.Tensor] = {}
        self._hooks: list = []

        # Layer 0: encoder output (pre-transformer input)
        def encoder_hook(module, inp, out):
            self._outputs[0] = out.detach()

        self._hooks.append(model.encoder.register_forward_hook(encoder_hook))

        # Layers 1..N: each transformer block output
        for i, block in enumerate(model.transformer.blocks):
            def block_hook(module, inp, out, idx=i):
                self._outputs[idx + 1] = out.detach()

            self._hooks.append(block.register_forward_hook(block_hook))

    def collect(self) -> dict[int, torch.Tensor]:
        return dict(self._outputs)

    def reset(self) -> None:
        self._outputs.clear()

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


@torch.no_grad()
def embed_sequence(
    seq: str,
    model,
    collector: _LayerCollector,
    device: str,
    max_len: int = 256,
) -> np.ndarray:
    """
    Embed a single amino-acid sequence using all ESM3 transformer layers.

    Returns
    -------
    np.ndarray of shape (n_layers + 1, d_model)
        Layer 0 = encoder output; layers 1..N = transformer block outputs.
        BOS/EOS tokens are stripped before mean-pooling over residues.
    """
    from esm.sdk.api import ESMProtein

    seq = seq[:max_len]
    collector.reset()

    protein = ESMProtein(sequence=seq)
    tokens = model.encode(protein)

    model.forward(sequence_tokens=tokens.sequence.unsqueeze(0))

    raw = collector.collect()
    n_layers = len(raw)

    layer_embeds = np.stack(
        [
            raw[i][0, 1:-1].float().mean(0).cpu().numpy()
            for i in range(n_layers)
        ]
    )  # (n_layers, d_model)
    return layer_embeds


def embed_dataset(
    ids: list[str],
    seqs: list[str],
    labels: list[int],
    out_dir: Path,
    model_name: str = _DEFAULT_MODEL,
    device: str = "auto",
    precision: Precision = "bf16",
    max_len: int = 256,
    skip_existing: bool = True,
) -> Path:
    """
    Embed all sequences with ESM3, saving per-protein .npz files.

    Each file contains:
        layers : (n_layers+1, d_model) float32
        label  : int scalar (0 or 1)

    Returns
    -------
    Path to the model-specific output directory.
    """
    out_dir = Path(out_dir)
    model_dir = out_dir / model_name.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)

    model = load_esm3(model_name, device, precision)
    collector = _LayerCollector(model)

    try:
        for pid, seq, label in tqdm(zip(ids, seqs, labels), total=len(ids), desc="Embedding ESM3"):
            out_path = model_dir / f"{pid}.npz"
            if skip_existing and out_path.exists():
                continue
            try:
                layers = embed_sequence(seq, model, collector, device, max_len)
                np.savez_compressed(out_path, layers=layers, label=np.int8(label))
            except RuntimeError as exc:
                print(f"[esm3] WARN: skipping {pid}: {exc}")
                clear_cache(device)
    finally:
        collector.remove()

    print(f"[esm3] Embeddings saved to {model_dir}")
    return model_dir
