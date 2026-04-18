#!/usr/bin/env python
"""Step 1: Download and prepare balanced bacterial toxic/benign sequences.

Bias mitigations:
  - Both classes: bacterial-only (taxonomy_id:2) + reviewed Swiss-Prot
  - KW-0929 (antimicrobial) excluded symmetrically (neither class)
  - Length-stratified sampling: benign matched to toxic length histogram
  - Sequence redundancy removal at ~40% identity (CD-HIT or Python fallback)
  - Organism distribution reported for manual audit
"""

import argparse
from pathlib import Path

from alxbio_esm.data import load_dataset, prepare_dataset
from alxbio_esm.dedup import dedup_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument(
        "--benign-pool-multiplier",
        type=int,
        default=6,
        help="Download this many × max_per_class benign sequences before stratified sampling",
    )
    parser.add_argument("--min-len", type=int, default=50)
    parser.add_argument("--max-len", type=int, default=1022)
    parser.add_argument("--bin-width", type=int, default=50, help="Length bin width for stratification")
    parser.add_argument("--identity", type=float, default=0.4, help="Sequence identity threshold for dedup")
    parser.add_argument("--overwrite", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # Step 1–4: download, filter, length-stratify, organism audit
    prepare_dataset(
        raw_dir=out_dir,
        max_per_class=args.max_per_class,
        benign_pool_multiplier=args.benign_pool_multiplier,
        min_len=args.min_len,
        max_len=args.max_len,
        bin_width=args.bin_width,
        overwrite=args.overwrite,
    )

    # Step 5: sequence deduplication (writes toxic_nr.fasta / benign_nr.fasta)
    toxic_nr, benign_nr = dedup_dataset(out_dir, identity=args.identity)

    # Rename _nr files to the canonical names expected by downstream scripts
    final_toxic = out_dir / "toxic.fasta"
    final_benign = out_dir / "benign.fasta"
    toxic_nr.rename(final_toxic)
    benign_nr.rename(final_benign)
    print(f"\n[download] Final files: {final_toxic}, {final_benign}")

    # Report final counts
    ids, _, labels = load_dataset(out_dir, max_per_class=args.max_per_class)
    import numpy as np
    labels_arr = np.array(labels)
    print(
        f"[download] Dataset ready: "
        f"{(labels_arr == 1).sum()} toxic, {(labels_arr == 0).sum()} benign."
    )


if __name__ == "__main__":
    main()
