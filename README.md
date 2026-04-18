# alxbio-esm

Layer-by-layer analysis of whether ESM2 encodes bacterial toxicity in its representations — without fine-tuning. Three tests per model: linear probe AUROC, centroid separation ratio, and CKA. Outputs per-layer metrics and UMAP figures.

## Setup

```bash
uv sync
```

Requires CUDA. Tested on RTX 4050 6 GB with ESM2-150M and ESM2-650M in FP16.

---

## Pipeline

### Step 1 — Download data

**Standard** (bacterial Swiss-Prot, length-stratified):
```bash
uv run python scripts/01_download.py --max-per-class 500
```

**Organism-matched** (eliminates species-level bias — use this for cleaner results):
```bash
uv run python scripts/01b_organism_matched.py
# writes data/raw_matched/toxic.fasta + benign.fasta
```

Both scripts deduplicate at 40% sequence identity and report organism distribution.

### Step 2 — Embed

```bash
# 150M first (fast, ~10s), then 650M
uv run python scripts/02_embed.py \
  --data-dir data/raw_matched \
  --out-dir data/embeddings_matched \
  --model facebook/esm2_t30_150M_UR50D

uv run python scripts/02_embed.py \
  --data-dir data/raw_matched \
  --out-dir data/embeddings_matched \
  --model facebook/esm2_t33_650M_UR50D
```

Each sequence is saved as a `.npz` with shape `(n_layers, d_model)` — mean-pooled over residues, all layers retained.

### Step 3 — Analyse

```bash
uv run python scripts/03_analyze.py \
  --emb-root data/embeddings_matched \
  --results-dir results/matched \
  --umap-layers 0 8 16 33
```

Outputs per model:
- `results/matched/figures/summary_<model>.png` — 4-panel: AUROC, separation ratio, consecutive CKA, cross-class CKA
- `results/matched/figures/umap_<model>.png` — UMAP at selected layers
- `results/matched/metrics_<model>.npz` — raw arrays for all metrics
- `results/matched/summary.json` — peak values table

---

## Available models

Pass any of these to `--model` in step 2:

| Model | Layers | d_model | VRAM (FP16) |
|---|---|---|---|
| `facebook/esm2_t6_8M_UR50D` | 6 | 320 | <1 GB |
| `facebook/esm2_t12_35M_UR50D` | 12 | 480 | <1 GB |
| `facebook/esm2_t30_150M_UR50D` | 30 | 640 | ~0.3 GB |
| `facebook/esm2_t33_650M_UR50D` | 33 | 1280 | ~2.5 GB |
| `facebook/esm2_t36_3B_UR50D` | 36 | 2560 | ~12 GB |

---

## Using ESM3

ESM3 uses EvolutionaryScale's own `esm` package, not Transformers. The rest of the pipeline (analysis, viz) is model-agnostic — only `alxbio_esm/embed.py` needs a new backend.

**Install:**
```bash
uv add esm  # EvolutionaryScale's package
```

**Get model access:** ESM3 weights are gated on HuggingFace (`esm3_sm_open_v1`). Request access at `huggingface.co/EvolutionaryScale/esm3-sm-open-v1`, then `huggingface-cli login`.

**Drop-in replacement for `embed_sequence` in `alxbio_esm/embed.py`:**

```python
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig

def load_esm3(device="cuda"):
    model = ESM3.from_pretrained("esm3_sm_open_v1").to(device).eval()
    return model

@torch.no_grad()
def embed_sequence_esm3(seq, model, device, max_len=500):
    seq = seq[:max_len]
    protein = ESMProtein(sequence=seq)
    # encode_inputs returns a tensor of shape (1, L+2, d_model) per track
    tensor = model.encode(protein)
    # forward_and_sample with output_hidden_states returns all layer embeddings
    out = model.forward(
        sequence_tokens=tensor.sequence.unsqueeze(0).to(device),
        output_hidden_states=True,
    )
    # out.hidden_states: tuple of (1, L+2, d) for each layer
    return np.stack([h[0, 1:-1].float().mean(0).cpu().numpy()
                     for h in out.hidden_states])  # (n_layers, d)
```

Then call `embed_dataset(..., model_name="esm3_sm_open_v1")` — the rest of the pipeline is unchanged. The `model_tag` derived from the name will automatically separate ESM3 embeddings from ESM2 ones in the output directories.

> ESM3-small has 1.4B parameters across sequence, structure, and function tracks. Run with FP16 and `max_len=256` on a 6 GB card to stay within VRAM.
