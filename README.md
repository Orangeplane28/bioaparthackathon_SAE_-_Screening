# alxbio-esm

Layer-by-layer analysis of whether ESM2 encodes toxicity in its representations — without fine-tuning. Three tests per model: linear probe AUROC, centroid separation ratio, and CKA. Outputs per-layer metrics and UMAP figures.

## Setup

```bash
uv sync
```

Device agnostic

---

## Datasets

All sequences from [UniProt Swiss-Prot](https://www.uniprot.org/) (reviewed only). Toxic class = keyword [KW-0800](https://www.uniprot.org/keywords/KW-0800). Deduplicated at 40% sequence identity.

### Bacterial (taxonomy_id:2)

| Dataset | Toxic | Benign | Notes |
|---|---|---|---|
| Standard (`data/raw/`) | 197 | 315 | Length-stratified; E. coli K12 dominates benign (184/315) — organism bias present |
| Organism-matched (`data/raw_matched/`) | 138 | 159 | Benign drawn from same taxon as each toxic protein — bias eliminated |

**Source:** `taxonomy_id:2 AND keyword:KW-0800 AND reviewed:true` (480 total pre-dedup)

### Eukaryotic — venomous clades (`data/raw_eukaryotic/`)

Benign proteins drawn from the same venomous clade to eliminate cross-kingdom bias.

| Clade | Taxonomy ID | Toxic | Benign |
|---|---|---|---|
| Snakes (Serpentes) | [8570](https://www.uniprot.org/taxonomy/8570) | 872 | 262 |
| Scorpions (Scorpiones) | [6843](https://www.uniprot.org/taxonomy/6843) | 494 | 327 |
| Spiders (Araneae) | [6893](https://www.uniprot.org/taxonomy/6893) | 373 | 63 |
| **Total** | | **1,739** | **652** |

Group membership tracked in `data/raw_eukaryotic/groups.tsv` — used for group-colored UMAP.

---

## Pipeline

### Step 1 — Download data

**Standard** (bacterial Swiss-Prot, length-stratified):
```bash
uv run python scripts/01_download.py 
```

**Organism-matched** (eliminates species-level bias — use this for cleaner results):
```bash
uv run python scripts/01b_organism_matched.py
# writes data/raw_matched/toxic.fasta + benign.fasta
```

**Eukaryotic clade-matched** (snakes / scorpions / spiders):
```bash
uv run python scripts/01c_eukaryotic.py
# writes data/raw_eukaryotic/toxic.fasta + benign.fasta + groups.tsv
```

All scripts deduplicate at 40% sequence identity and report organism distribution.

### Step 2 — Embed

```bash
# Bacterial organism-matched
uv run python scripts/02_embed.py \
  --data-dir data/raw_matched \
  --out-dir data/embeddings_matched \
  --model facebook/esm2_t33_650M_UR50D

# Eukaryotic (groups.tsv detected automatically — loads all sequences)
uv run python scripts/02_embed.py \
  --data-dir data/raw_eukaryotic \
  --out-dir data/embeddings_eukaryotic \
  --model facebook/esm2_t33_650M_UR50D
```

Each sequence is saved as a `.npz` with shape `(n_layers, d_model)` — mean-pooled over residues, all layers retained.

### Step 3 — Analyse

```bash
# Bacterial
uv run python scripts/03_analyze.py \
  --emb-root data/embeddings_matched \
  --results-dir results/matched \
  --umap-layers 0 8 16 33

# Eukaryotic (--data-dir enables group-colored UMAP: rows=clade, cols=layer)
uv run python scripts/03_analyze.py \
  --emb-root data/embeddings_eukaryotic \
  --data-dir data/raw_eukaryotic \
  --results-dir results/eukaryotic \
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
