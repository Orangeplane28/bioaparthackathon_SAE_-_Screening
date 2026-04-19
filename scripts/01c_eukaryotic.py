#!/usr/bin/env python
"""Step 1c: Download clade-matched eukaryotic toxic/benign dataset.

Downloads Swiss-Prot reviewed toxins (KW-0800) and benign proteins from
three venomous clades — benign drawn from the same clade to eliminate
cross-kingdom organism bias:

  Snakes    (taxonomy_id:8570)  ~2,061 toxic /  676 benign
  Scorpions (taxonomy_id:6843)  ~2,435 toxic /  733 benign
  Spiders   (taxonomy_id:6893)  ~1,440 toxic /  169 benign

Writes:
  data/raw_eukaryotic/toxic.fasta
  data/raw_eukaryotic/benign.fasta
  data/raw_eukaryotic/groups.tsv   (uniprot_id → snakes/scorpions/spiders)

Downstream scripts (02_embed, 03_analyze) accept --data-dir data/raw_eukaryotic.
Pass --grouped-umap to 03_analyze to get group-colored UMAP output.
"""

import argparse
from pathlib import Path

from alxbio_esm.data import load_dataset_with_groups, prepare_eukaryotic_dataset
from alxbio_esm.dedup import dedup_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/raw_eukaryotic")
    parser.add_argument("--min-len", type=int, default=50)
    parser.add_argument("--max-len", type=int, default=1022)
    parser.add_argument("--bin-width", type=int, default=50)
    parser.add_argument("--identity", type=float, default=0.4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    prepare_eukaryotic_dataset(
        out_dir=out_dir,
        min_len=args.min_len,
        max_len=args.max_len,
        bin_width=args.bin_width,
        overwrite=args.overwrite,
    )

    # Deduplicate within each class
    toxic_nr, benign_nr = dedup_dataset(out_dir, identity=args.identity)
    toxic_nr.rename(out_dir / "toxic.fasta")
    benign_nr.rename(out_dir / "benign.fasta")

    # Final counts with group breakdown
    import numpy as np
    ids, _, labels, groups = load_dataset_with_groups(out_dir, max_per_class=10_000)
    labels_arr = np.array(labels)
    groups_arr = np.array(groups)
    print(f"\n[eukaryotic] Final dataset: {(labels_arr==1).sum()} toxic, {(labels_arr==0).sum()} benign")
    for grp in sorted(set(groups)):
        mask = groups_arr == grp
        t = ((labels_arr == 1) & mask).sum()
        b = ((labels_arr == 0) & mask).sum()
        print(f"  {grp:12s}  {t:4d} toxic  {b:4d} benign")
    print(f"\n[eukaryotic] Dataset ready in {out_dir}/")
    print(f"[eukaryotic] Groups metadata: {out_dir / 'groups.tsv'}")


if __name__ == "__main__":
    main()
