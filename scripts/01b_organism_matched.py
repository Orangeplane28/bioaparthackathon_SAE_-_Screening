#!/usr/bin/env python
"""Step 1b: Build an organism-matched dataset to eliminate organism-level bias.

For every taxon that contributes toxic proteins, benign proteins are downloaded
from the **same taxon**. This forces the model to discriminate toxic from
non-toxic within the same bacterium — no cross-species shortcuts.

Writes data/raw_matched/toxic.fasta and data/raw_matched/benign.fasta.
Downstream scripts (02_embed, 03_analyze) work on any data-dir, so just pass
--data-dir data/raw_matched to them.
"""

import argparse
from pathlib import Path

from alxbio_esm.data import prepare_organism_matched_dataset
from alxbio_esm.dedup import dedup_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw",
                        help="Directory containing toxic_raw.fasta from step 01")
    parser.add_argument("--out-dir", default="data/raw_matched")
    parser.add_argument("--min-per-taxon", type=int, default=2,
                        help="Skip taxa with fewer than this many toxic proteins")
    parser.add_argument("--min-len", type=int, default=50)
    parser.add_argument("--max-len", type=int, default=1022)
    parser.add_argument("--bin-width", type=int, default=50)
    parser.add_argument("--identity", type=float, default=0.4,
                        help="Sequence identity threshold for deduplication")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # Build organism-matched dataset
    prepare_organism_matched_dataset(
        raw_dir=Path(args.raw_dir),
        out_dir=out_dir,
        min_len=args.min_len,
        max_len=args.max_len,
        min_per_taxon=args.min_per_taxon,
        bin_width=args.bin_width,
        overwrite=args.overwrite,
    )

    # Deduplicate
    toxic_nr, benign_nr = dedup_dataset(out_dir, identity=args.identity)
    toxic_nr.rename(out_dir / "toxic.fasta")
    benign_nr.rename(out_dir / "benign.fasta")

    print(f"\n[matched] Dataset ready in {out_dir}/")


if __name__ == "__main__":
    main()
