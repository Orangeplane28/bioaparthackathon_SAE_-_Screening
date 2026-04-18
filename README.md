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

ESM3 is fully supported via `alxbio_esm/embed_esm3.py`. The analysis pipeline (`03_analyze.py`) is model-agnostic — no changes needed there.

**One-time setup:**
```bash
# 1. Accept the license at huggingface.co/EvolutionaryScale/esm3-sm-open-v1
# 2. Log in (downloads ~2.9 GB weights on first run)
hf auth login
# or: export HF_TOKEN=hf_...
```

The environment variable name is **`HF_TOKEN`**.

**Run:**
```bash
uv run python scripts/02_embed.py \
  --data-dir data/raw_matched \
  --out-dir data/embeddings_matched \
  --model esm3_sm_open_v1 \
  --precision bf16 \
  --max-len 256

uv run python scripts/03_analyze.py \
  --emb-root data/embeddings_matched \
  --results-dir results/matched \
  --umap-layers 0 12 24 48
```

**Precision options** (same flag for both ESM2 and ESM3):

| `--precision` | dtype | VRAM (ESM3-small) | Notes |
|---|---|---|---|
| `bf16` | bfloat16 | ~3 GB | default for ESM3, recommended |
| `fp16` | float16 | ~3 GB | default for ESM2 |
| `fp32` | float32 | ~6 GB | reference only, slow |

> ESM3-small: 1.4B params, d_model=1536, 48 layers. Uses forward hooks internally (no `output_hidden_states` in its API) — this is handled transparently in `embed_esm3.py`.
