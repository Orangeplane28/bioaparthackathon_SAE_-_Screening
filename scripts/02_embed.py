#!/usr/bin/env python
"""Step 2: Embed all sequences with ESM2, saving per-layer representations."""

import argparse
from pathlib import Path

from alxbio_esm.data import load_dataset
from alxbio_esm.embed import embed_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw", help="Directory with toxic/benign FASTA")
    parser.add_argument("--out-dir", default="data/embeddings", help="Embedding output directory")
    parser.add_argument(
        "--model",
        default="facebook/esm2_t33_650M_UR50D",
        choices=[
            "facebook/esm2_t6_8M_UR50D",
            "facebook/esm2_t12_35M_UR50D",
            "facebook/esm2_t30_150M_UR50D",
            "facebook/esm2_t33_650M_UR50D",
            "facebook/esm2_t36_3B_UR50D",
        ],
        help="ESM2 model to use",
    )
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument("--max-len", type=int, default=500, help="Truncate sequences to this length")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 (use FP32)")
    args = parser.parse_args()

    ids, seqs, labels = load_dataset(
        Path(args.data_dir), max_per_class=args.max_per_class
    )
    embed_dataset(
        ids=ids,
        seqs=seqs,
        labels=labels,
        out_dir=Path(args.out_dir),
        model_name=args.model,
        device=args.device,
        fp16=not args.no_fp16,
        max_len=args.max_len,
    )


if __name__ == "__main__":
    main()
