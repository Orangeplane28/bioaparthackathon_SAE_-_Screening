# AIxBio Hackathon — ESM3 SAE Toxin Classifier

Interpretable protein toxicity detection using ESM3 sparse autoencoders (SAEs).

The pipeline trains a TopK SAE on ESM3-small-open layer-36 representations, discovers discriminative SAE features for dangerous proteins, validates them against hard structural analogues, and probes causality with targeted feature ablation and LEACE concept erasure.

---

## Repository structure

```
├── config.py                    # All paths, constants, hyperparameters
├── utils/
│   ├── esm3.py                  # TopKSAE, load_esm3(), embed_esm3(), load_sae()
│   ├── data.py                  # Dataset loading, feature I/O, UniProt helpers
│   ├── stats.py                 # AUROC, BH FDR, bootstrap CIs, Cohen's d
│   └── interv.py                # Feature ablation, LEACE concept erasure (Fisher LDA)
├── N1_sanity_check.py           # Verify ESM3 + SAE architecture
├── N2_data_preparation.py       # Fetch proteins, annotate families, split
├── N3_embeddings_baselines.py   # ESM3 embeddings + SAE training + baselines
├── N4_feature_discovery.py      # Per-feature stats, BH FDR, ranking
├── N5_negative_controls.py      # Specificity, permutation test, bootstrap CIs
├── N6_causal_intervention.py    # Ablation dose-response + LEACE erasure
├── N7_external_validation.py    # 17 held-out proteins from unseen families
└── requirements.txt
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# Install ESM3 (official package, recommended):
pip install esm
```

### 2. Run the pipeline in order

```bash
# Verify architecture (CPU, ~30 s)
python N1_sanity_check.py

# Fetch proteins and build splits (~5–10 min, needs internet)
python N2_data_preparation.py --max-proteins 60

# Extract ESM3 embeddings, train SAE, train baselines
# GPU strongly recommended (8–24 h on CPU, ~1 h on A100)
python N3_embeddings_baselines.py --device cuda --epochs 50

# Discover discriminative SAE features
python N4_feature_discovery.py --top-k 200 --batch 512

# Negative control analysis + permutation test
python N5_negative_controls.py --n-permutations 10000

# Causal intervention (ablation + LEACE)
python N6_causal_intervention.py --top-k 50

# External validation on 17 held-out proteins
python N7_external_validation.py --device cuda
```

### 3. Outputs

| Script | Key outputs |
|--------|-------------|
| N2 | `data/master_dataset_esm3.json`, `data/splits/` |
| N3 | `embeddings/`, `sae_features/`, `baselines/`, `sae_weights/` |
| N4 | `analysis/feature_ranking/feature_ranking.json` (also `top_toxin_features.json`) |
| N5 | `analysis/feature_ranking/negative_control_results.json`, `figures/N5/` |
| N6 | `results/causal_intervention_results.json`, `figures/N6/` |
| N7 | `results/external_validation_results.json` |

---

## Architecture

| Component | Spec |
|-----------|------|
| Protein LM | ESM3-small-open (`esm3-sm-open-v1`), 1.4 B params |
| Layer tapped | 36 / 48 (≈ 75% depth) |
| d_model | 1 536 |
| SAE width | 15 360 (10× expansion) |
| SAE sparsity | TopK, k = 120 active per residue |
| Classifier | LogReg + MLP on mean-pooled SAE features |

> **Important — d_model mismatch**: ESM3 uses d_model = 1 536. ESM2-3B uses 1 280.
> Never substitute one for the other. `utils/esm3.load_esm3()` enforces this with
> an explicit `RuntimeError` if neither the official ESM package nor the HuggingFace
> `EvolutionaryScale/esm3-sm-open-v1` checkpoint can be loaded.

---

## Key design choices

**Hard negatives** — membrane proteins, ion channels, amphipathic-helix proteins, and
lipid-binding proteins are included alongside general metabolic/nuclear/cytoskeletal
negatives so the classifier must learn toxin-specific signal rather than generic
membrane-associating features.

**Family-based splits** — train/val/test are split at the protein-family level
(disjoint families) to prevent homology leakage. The three convergent PFTs
(alpha-hemolysin P09616, FraC P61914, ClyA P77335) are forced into the test set.

**True LEACE** — `utils/interv.compute_leace_direction()` computes the Fisher LDA
direction `d = Σ_w⁻¹(μ_pos − μ_neg)` via randomized SVD (Woodbury identity), not the
naive mean-difference normalisation. This gives minimum information loss during erasure.

**Dual JSON save in N4** — `feature_ranking.json` and `top_toxin_features.json` are
written simultaneously so N5/N6/N7 can find the ranking under either filename.

---

## Hardware requirements

| Task | Minimum | Recommended |
|------|---------|-------------|
| N1 sanity check | CPU, 4 GB RAM | — |
| N2 data prep | CPU, 4 GB RAM | — |
| N3 embeddings | 16 GB RAM + GPU 8 GB VRAM | A100 40 GB |
| N3 SAE training | GPU 8 GB VRAM | A100 40 GB |
| N4–N7 analysis | CPU, 16 GB RAM | — |

---

## Citation

If you use this code, please cite:

- ESM3: Hayes et al., *Science* 2024 — `EvolutionaryScale/esm3-sm-open-v1`
- SafeProtein-Bench: apart-research/SafeProtein-Bench
- LEACE: Belrose & Mallen 2023 (<https://arxiv.org/abs/2306.03819>)
