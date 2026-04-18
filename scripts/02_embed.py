#!/usr/bin/env python
"""Step 2: Embed sequences with ESM2 or ESM3, saving per-layer representations.

ESM2 models are loaded via HuggingFace Transformers (no auth required).
ESM3 models require:
  1. Accept license at huggingface.co/EvolutionaryScale/esm3-sm-open-v1
  2. huggingface-cli login  (or HF_TOKEN env var)
"""

import argparse
from pathlib import Path

from alxbio_esm.data import load_dataset

ESM2_MODELS = [
    "facebook/esm2_t6_8M_UR50D",
    "facebook/esm2_t12_35M_UR50D",
    "facebook/esm2_t30_150M_UR50D",
    "facebook/esm2_t33_650M_UR50D",
    "facebook/esm2_t36_3B_UR50D",
]

ESM3_MODELS = [
    "esm3_sm_open_v1",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw_matched")
    parser.add_argument("--out-dir", default="data/embeddings_matched")
    parser.add_argument(
        "--model",
        default="facebook/esm2_t33_650M_UR50D",
        choices=ESM2_MODELS + ESM3_MODELS,
    )
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="Truncate sequences to this length. Defaults: 500 for ESM2, 256 for ESM3.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        default=None,
        choices=["fp32", "fp16", "bf16"],
        help="Numeric precision. Defaults: fp16 for ESM2, bf16 for ESM3.",
    )
    args = parser.parse_args()

    ids, seqs, labels = load_dataset(
        Path(args.data_dir), max_per_class=args.max_per_class
    )

    is_esm3 = args.model in ESM3_MODELS
    precision = args.precision or ("bf16" if is_esm3 else "fp16")
    max_len = args.max_len or (256 if is_esm3 else 500)

    if is_esm3:
        from alxbio_esm.embed_esm3 import embed_dataset
    else:
        from alxbio_esm.embed import embed_dataset

    embed_dataset(
        ids=ids,
        seqs=seqs,
        labels=labels,
        out_dir=Path(args.out_dir),
        model_name=args.model,
        device=args.device,
        precision=precision,
        max_len=max_len,
    )


if __name__ == "__main__":
    main()
