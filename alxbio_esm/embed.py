"""ESM2 per-layer mean-pooled embeddings.

Saves a compressed .npz per protein with shape (n_layers, d_model).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

ModelKey = Literal[
    "facebook/esm2_t6_8M_UR50D",
    "facebook/esm2_t12_35M_UR50D",
    "facebook/esm2_t30_150M_UR50D",
    "facebook/esm2_t33_650M_UR50D",
    "facebook/esm2_t36_3B_UR50D",
]

Precision = Literal["fp32", "fp16", "bf16"]

_DEFAULT_MODEL: ModelKey = "facebook/esm2_t33_650M_UR50D"

_DTYPE_MAP: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def load_model(
    model_name: str = _DEFAULT_MODEL,
    device: str = "cuda",
    precision: Precision = "fp16",
) -> tuple[EsmModel, object, str]:
    """Load ESM2 model + tokenizer. Returns (model, tokenizer, device)."""
    print(f"[embed] Loading {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name, output_hidden_states=True)
    model = model.to(_DTYPE_MAP[precision]).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[embed] Loaded {n_params:.0f}M params on {device}.")
    return model, tokenizer, device


@torch.no_grad()
def embed_sequence(
    seq: str,
    model: EsmModel,
    tokenizer,
    device: str,
    max_len: int = 500,
) -> np.ndarray:
    """
    Embed a single sequence using all transformer layers.

    Returns
    -------
    np.ndarray of shape (n_layers, d_model)
        n_layers = 1 (embedding) + n_transformer_layers
        Residue dimension is mean-pooled (CLS/EOS tokens excluded).
    """
    seq = seq[:max_len]
    inputs = tokenizer(seq, return_tensors="pt").to(device)
    out = model(**inputs, output_hidden_states=True)
    # hidden_states: tuple of (1, L+2, d_model) — index 0 is embed layer
    layer_embeds = np.stack(
        [h[0, 1:-1].float().mean(0).cpu().numpy() for h in out.hidden_states]
    )  # (n_layers, d_model)
    return layer_embeds


def embed_dataset(
    ids: list[str],
    seqs: list[str],
    labels: list[int],
    out_dir: Path,
    model_name: str = _DEFAULT_MODEL,
    device: str = "cuda",
    precision: Precision = "fp16",
    max_len: int = 500,
    skip_existing: bool = True,
) -> Path:
    """
    Embed all sequences and save per-protein .npz files.

    Each file contains:
        layers : (n_layers, d_model) float32
        label  : int scalar (0 or 1)

    Returns
    -------
    Path to the output directory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device = load_model(model_name, device, precision)
    model_tag = model_name.split("/")[-1]
    model_dir = out_dir / model_tag
    model_dir.mkdir(exist_ok=True)

    for pid, seq, label in tqdm(zip(ids, seqs, labels), total=len(ids), desc="Embedding"):
        out_path = model_dir / f"{pid}.npz"
        if skip_existing and out_path.exists():
            continue
        try:
            layers = embed_sequence(seq, model, tokenizer, device, max_len)
            np.savez_compressed(out_path, layers=layers, label=np.int8(label))
        except RuntimeError as exc:
            # OOM or tokenisation error — skip and warn
            print(f"[embed] WARN: skipping {pid}: {exc}")
            torch.cuda.empty_cache()

    print(f"[embed] Embeddings saved to {model_dir}")
    return model_dir


def load_embeddings(
    emb_dir: Path,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Load all .npz files from a model embedding directory.

    Returns
    -------
    ids    : list of protein IDs
    X      : (N, n_layers, d_model) float32
    labels : (N,) int
    """
    emb_dir = Path(emb_dir)
    files = sorted(emb_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {emb_dir}")

    ids, arrays, label_list = [], [], []
    for f in tqdm(files, desc="Loading embeddings"):
        d = np.load(f)
        ids.append(f.stem)
        arrays.append(d["layers"])
        label_list.append(int(d["label"]))

    X = np.stack(arrays)  # (N, n_layers, d_model)
    labels = np.array(label_list, dtype=np.int32)
    print(f"[embed] Loaded {len(ids)} proteins, shape {X.shape}")
    return ids, X, labels
