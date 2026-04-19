"""UniProt data download, FASTA parsing, and dataset preparation utilities.

Bias mitigations applied:
  - taxonomy_id:2 on both classes (bacterial-only)
  - reviewed:true on both classes
  - KW-0929 excluded from neither class (symmetric treatment)
  - Length-stratified sampling: benign sampled to match toxic length histogram
  - Organism distribution reported post-download for manual audit
  - Organism-matched pairing: benign queried per toxic taxon ID (see prepare_organism_matched_dataset)
  - Eukaryotic clade-matched: benign from same venomous clade (see prepare_eukaryotic_dataset)
"""

from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import requests
from tqdm import tqdm

_UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb/search"

_KW_TOXIN = "KW-0800"
_KW_VIRULENCE = "KW-0843"
# KW-0929 (antimicrobial) intentionally NOT excluded — many bacterial toxins
# carry both KW-0800 and KW-0929 (bacteriocins, colicins, pore-forming toxins).
# Excluding it only from benign would create an asymmetric hole in the negative class.

QUERIES = {
    "toxic": (
        f"(taxonomy_id:2) AND (keyword:{_KW_TOXIN}) AND (reviewed:true)"
    ),
    "benign": (
        f"(taxonomy_id:2) AND (reviewed:true)"
        f" NOT (keyword:{_KW_TOXIN})"
        f" NOT (keyword:{_KW_VIRULENCE})"
    ),
}

# Venomous eukaryotic clades: taxon_id → group label
# Snakes (Serpentes), Scorpions (Scorpiones), Spiders (Araneae)
EUKARYOTIC_GROUPS: dict[str, str] = {
    "8570": "snakes",
    "6843": "scorpions",
    "6893": "spiders",
}

# ─────────────────────────────────────────────────────────────
# FASTA utilities
# ─────────────────────────────────────────────────────────────

def parse_fasta(text: str) -> list[tuple[str, str]]:
    """Return list of (header, sequence) from a FASTA string."""
    records: list[tuple[str, str]] = []
    header = seq_parts = None
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq_parts)))  # type: ignore[arg-type]
            header = line[1:].strip()
            seq_parts = []
        elif header is not None:
            seq_parts.append(line.strip())  # type: ignore[union-attr]
    if header is not None:
        records.append((header, "".join(seq_parts)))  # type: ignore[arg-type]
    return records


def parse_fasta_file(path: Path) -> list[tuple[str, str]]:
    return parse_fasta(Path(path).read_text())


def write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for header, seq in records:
            fh.write(f">{header}\n{seq}\n")


def extract_uniprot_id(header: str) -> str:
    m = re.search(r"\|([A-Z0-9]+)\|", header)
    return m.group(1) if m else header.split()[0]


def extract_organism(header: str) -> str:
    """Extract OS= organism name from a UniProt FASTA header."""
    m = re.search(r"OS=(.+?)\s+OX=", header)
    return m.group(1).strip() if m else "Unknown"


def extract_taxon_id(header: str) -> str | None:
    """Extract OX= NCBI taxon ID from a UniProt FASTA header."""
    m = re.search(r"OX=(\d+)", header)
    return m.group(1) if m else None


# ─────────────────────────────────────────────────────────────
# UniProt download
# ─────────────────────────────────────────────────────────────

def _stream_fasta_pages(
    query: str,
    page_size: int = 500,
    max_records: int | None = None,
    sleep_between: float = 0.5,
) -> Iterator[str]:
    """Yield raw FASTA text pages from UniProt REST API (cursor-based pagination)."""
    params: dict = {"query": query, "format": "fasta", "size": page_size}
    fetched = 0
    url: str | None = _UNIPROT_BASE

    while url:
        resp = requests.get(url, params=params if url == _UNIPROT_BASE else None, timeout=60)
        resp.raise_for_status()
        text = resp.text.strip()
        if not text:
            break
        yield text

        fetched += text.count(">")
        if max_records and fetched >= max_records:
            break

        link_header = resp.headers.get("Link", "")
        next_url = None
        for part in link_header.split(","):
            if 'rel="next"' in part:
                m = re.search(r"<([^>]+)>", part)
                if m:
                    next_url = m.group(1)
        url = next_url
        params = {}
        if url:
            time.sleep(sleep_between)


def download_class_raw(
    label: str,
    out_path: Path,
    max_records: int = 2000,
    page_size: int = 500,
    overwrite: bool = False,
) -> list[tuple[str, str]]:
    """
    Download UniProt FASTA sequences for *label* ("toxic" or "benign").

    Returns the parsed (header, sequence) list and writes *out_path*.
    Uses the file on disk if it already exists and has content (unless overwrite=True).
    """
    out_path = Path(out_path)

    if not overwrite and out_path.exists() and out_path.stat().st_size > 0:
        records = parse_fasta_file(out_path)
        print(f"[data] {out_path} cached with {len(records)} records — skipping download.")
        return records

    query = QUERIES[label]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records_collected: list[tuple[str, str]] = []
    tmp_path = out_path.with_suffix(".tmp")

    with tmp_path.open("w") as fh, tqdm(desc=f"Downloading {label}", unit="seq") as pbar:
        for page_text in _stream_fasta_pages(query, page_size=page_size, max_records=max_records):
            page_records = parse_fasta(page_text)
            remaining = max_records - len(records_collected)
            for header, seq in page_records[:remaining]:
                fh.write(f">{header}\n{seq}\n")
                records_collected.append((header, seq))
                pbar.update(1)
            if len(records_collected) >= max_records:
                break

    tmp_path.rename(out_path)
    print(f"[data] Wrote {len(records_collected)} {label} sequences → {out_path}")
    return records_collected


# ─────────────────────────────────────────────────────────────
# Bias 1 fix: length-stratified sampling
# ─────────────────────────────────────────────────────────────

def length_bins(seqs: list[tuple[str, str]], bin_width: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Return (lengths, bin_edges) for a sequence list."""
    lengths = np.array([len(s) for _, s in seqs])
    lo = int(lengths.min() // bin_width * bin_width)
    hi = int(lengths.max() // bin_width * bin_width) + bin_width
    edges = np.arange(lo, hi + bin_width, bin_width)
    return lengths, edges


def length_stratify(
    target_records: list[tuple[str, str]],
    pool_records: list[tuple[str, str]],
    bin_width: int = 50,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """
    Sample *pool_records* so its length distribution matches *target_records*.

    Parameters
    ----------
    target_records : the class whose length histogram we want to replicate (toxic)
    pool_records   : the larger pool to sample from (benign)

    Returns
    -------
    Subset of pool_records length-matched to target_records.
    """
    rng = np.random.default_rng(seed)
    target_lens, edges = length_bins(target_records, bin_width)
    hist, _ = np.histogram(target_lens, bins=edges)

    # Group pool by bin
    pool_by_bin: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for rec in pool_records:
        l = len(rec[1])
        if edges[0] <= l < edges[-1]:
            bin_idx = int((l - edges[0]) // bin_width)
            pool_by_bin[bin_idx].append(rec)

    result: list[tuple[str, str]] = []
    for i, count in enumerate(hist):
        available = pool_by_bin[i]
        n = min(int(count), len(available))
        if n > 0:
            chosen = rng.choice(len(available), n, replace=False)
            result.extend(available[j] for j in chosen)

    print(
        f"[data] Length-stratified benign: {len(result)} sequences "
        f"matched to toxic length distribution (bins of {bin_width} aa)."
    )
    return result


# ─────────────────────────────────────────────────────────────
# Bias 4 mitigation: organism distribution audit
# ─────────────────────────────────────────────────────────────

def report_organism_distribution(
    toxic_records: list[tuple[str, str]],
    benign_records: list[tuple[str, str]],
    top_n: int = 10,
) -> dict[str, Counter]:
    """
    Print and return organism counts for each class.

    If the top-N organisms differ heavily between classes, organism bias is present.
    """
    orgs: dict[str, Counter] = {}
    for label, records in [("toxic", toxic_records), ("benign", benign_records)]:
        counts: Counter = Counter(extract_organism(h) for h, _ in records)
        orgs[label] = counts
        print(f"\n[data] Top-{top_n} organisms in {label} class:")
        for org, n in counts.most_common(top_n):
            print(f"  {n:4d}  {org}")

    toxic_top = {o for o, _ in orgs["toxic"].most_common(top_n)}
    benign_top = {o for o, _ in orgs["benign"].most_common(top_n)}
    overlap = toxic_top & benign_top
    print(
        f"\n[data] Organism overlap (top-{top_n}): "
        f"{len(overlap)}/{top_n} species shared between classes."
    )
    if len(overlap) < top_n // 2:
        print("[data] WARNING: Low organism overlap — organism bias may inflate results.")
    return orgs


# ─────────────────────────────────────────────────────────────
# High-level pipeline
# ─────────────────────────────────────────────────────────────

def prepare_dataset(
    raw_dir: Path,
    max_per_class: int = 500,
    benign_pool_multiplier: int = 6,
    min_len: int = 50,
    max_len: int = 1022,
    bin_width: int = 50,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """
    Full dataset preparation pipeline:

    1. Download toxic sequences (up to *max_per_class*)
    2. Download benign pool (up to *max_per_class* × *benign_pool_multiplier*)
    3. Filter both by [min_len, max_len]
    4. Length-stratify benign to match toxic histogram
    5. Write final balanced toxic.fasta / benign_balanced.fasta

    Returns paths to the two final FASTA files.
    """
    raw_dir = Path(raw_dir)

    # 1. Download toxic
    toxic_all = download_class_raw(
        "toxic",
        raw_dir / "toxic_raw.fasta",
        max_records=max_per_class * 2,  # extra buffer for length filtering
        overwrite=overwrite,
    )

    # 2. Download benign pool
    pool_size = max_per_class * benign_pool_multiplier
    benign_pool = download_class_raw(
        "benign",
        raw_dir / "benign_raw.fasta",
        max_records=pool_size,
        overwrite=overwrite,
    )

    # 3. Filter by length
    toxic_filtered = [(h, s) for h, s in toxic_all if min_len <= len(s) <= max_len]
    benign_filtered = [(h, s) for h, s in benign_pool if min_len <= len(s) <= max_len]
    toxic_filtered = toxic_filtered[:max_per_class]

    print(
        f"[data] After length filter [{min_len}, {max_len}]: "
        f"{len(toxic_filtered)} toxic, {len(benign_filtered)} benign (pool)."
    )

    # 4. Length-stratify benign
    benign_matched = length_stratify(toxic_filtered, benign_filtered, bin_width=bin_width)

    # 5. Write final files
    toxic_out = raw_dir / "toxic.fasta"
    benign_out = raw_dir / "benign.fasta"
    write_fasta(toxic_filtered, toxic_out)
    write_fasta(benign_matched, benign_out)
    print(f"[data] Final dataset: {len(toxic_filtered)} toxic, {len(benign_matched)} benign.")

    # 6. Organism audit
    report_organism_distribution(toxic_filtered, benign_matched)

    return toxic_out, benign_out


# ─────────────────────────────────────────────────────────────
# Load for downstream use
# ─────────────────────────────────────────────────────────────

def load_dataset(
    data_dir: Path,
    max_per_class: int = 500,
    min_len: int = 50,
    max_len: int = 1022,
) -> tuple[list[str], list[str], list[int]]:
    """
    Return (ids, sequences, labels) from toxic.fasta / benign.fasta in *data_dir*.

    Labels: 1 = toxic, 0 = benign.
    """
    ids: list[str] = []
    seqs: list[str] = []
    labels: list[int] = []

    for label_int, label_str in [(1, "toxic"), (0, "benign")]:
        fasta_path = Path(data_dir) / f"{label_str}.fasta"
        if not fasta_path.exists():
            raise FileNotFoundError(
                f"{fasta_path} not found. Run scripts/01_download.py first."
            )
        records = parse_fasta_file(fasta_path)
        count = 0
        for header, seq in records:
            if not (min_len <= len(seq) <= max_len):
                continue
            ids.append(extract_uniprot_id(header))
            seqs.append(seq)
            labels.append(label_int)
            count += 1
            if count >= max_per_class:
                break
        print(f"[data] Loaded {count} {label_str} sequences.")

    return ids, seqs, labels


# ─────────────────────────────────────────────────────────────
# Organism-matched dataset (eliminates organism bias)
# ─────────────────────────────────────────────────────────────

def _fetch_benign_for_taxon(
    taxon_id: str,
    n_wanted: int,
    pool_multiplier: int = 5,
    sleep_between: float = 0.3,
) -> list[tuple[str, str]]:
    """Fetch up to n_wanted benign proteins from a specific NCBI taxon."""
    query = (
        f"(taxonomy_id:{taxon_id}) AND (reviewed:true)"
        f" NOT (keyword:{_KW_TOXIN})"
        f" NOT (keyword:{_KW_VIRULENCE})"
    )
    records: list[tuple[str, str]] = []
    for page_text in _stream_fasta_pages(
        query,
        page_size=min(200, n_wanted * pool_multiplier),
        max_records=n_wanted * pool_multiplier,
        sleep_between=sleep_between,
    ):
        records.extend(parse_fasta(page_text))
        if len(records) >= n_wanted * pool_multiplier:
            break
    return records


def prepare_organism_matched_dataset(
    raw_dir: Path,
    out_dir: Path,
    min_len: int = 50,
    max_len: int = 1022,
    min_per_taxon: int = 2,
    bin_width: int = 50,
    overwrite: bool = False,
    seed: int = 42,
) -> tuple[Path, Path]:
    """
    Build an organism-matched toxic/benign dataset.

    For every taxon ID that contributes toxic proteins, benign proteins are
    downloaded from **that same taxon**. This eliminates organism-level signal
    as a confound: the classifier must discriminate toxic from non-toxic within
    the same bacterium, not E. coli from Staph.

    Algorithm
    ---------
    1. Load existing toxic_raw.fasta (from raw_dir).
    2. Group toxic proteins by OX= taxon ID.
    3. For each taxon with >= min_per_taxon toxic proteins, query UniProt for
       benign proteins from the same taxon.
    4. Length-stratify the per-taxon benign pool to match the per-taxon toxic
       length distribution.
    5. Write toxic.fasta and benign.fasta to out_dir.

    Returns
    -------
    (toxic_out, benign_out) paths.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    toxic_out = out_dir / "toxic.fasta"
    benign_out = out_dir / "benign.fasta"

    if (
        not overwrite
        and toxic_out.exists() and toxic_out.stat().st_size > 0
        and benign_out.exists() and benign_out.stat().st_size > 0
    ):
        print(f"[data] Organism-matched dataset cached in {out_dir} — skipping.")
        return toxic_out, benign_out

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # 1. Load all toxic sequences
    toxic_raw_path = raw_dir / "toxic_raw.fasta"
    if not toxic_raw_path.exists():
        raise FileNotFoundError(f"{toxic_raw_path} not found. Run 01_download.py first.")
    all_toxic = parse_fasta_file(toxic_raw_path)
    all_toxic = [(h, s) for h, s in all_toxic if min_len <= len(s) <= max_len]

    # 2. Group toxic by taxon ID
    toxic_by_taxon: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for header, seq in all_toxic:
        ox = extract_taxon_id(header)
        if ox:
            toxic_by_taxon[ox].append((header, seq))

    # Sort by count descending so largest taxa are processed first
    taxa_sorted = sorted(toxic_by_taxon.items(), key=lambda x: -len(x[1]))
    taxa_to_use = [(ox, recs) for ox, recs in taxa_sorted if len(recs) >= min_per_taxon]

    print(
        f"[data] Organism-matched: {len(taxa_to_use)} taxa with >= {min_per_taxon} toxic proteins "
        f"(out of {len(toxic_by_taxon)} total taxa)."
    )

    # 3 & 4. For each taxon, fetch benign and length-stratify
    matched_toxic: list[tuple[str, str]] = []
    matched_benign: list[tuple[str, str]] = []
    per_taxon_stats: list[dict] = []

    for ox, t_recs in tqdm(taxa_to_use, desc="Organism-matched download"):
        name = extract_organism(t_recs[0][0])
        benign_pool = _fetch_benign_for_taxon(ox, n_wanted=len(t_recs))

        # Filter by length
        benign_pool = [(h, s) for h, s in benign_pool if min_len <= len(s) <= max_len]

        if not benign_pool:
            print(f"  [skip] OX={ox} ({name}): no benign proteins found.")
            continue

        # Length-stratify: match benign length dist to toxic length dist for this taxon
        if len(benign_pool) >= len(t_recs):
            benign_selected = length_stratify(t_recs, benign_pool, bin_width=bin_width, seed=seed)
            # Fallback if stratification yields too few (rare bins): take what we can
            if len(benign_selected) < max(1, len(t_recs) // 2):
                n = min(len(t_recs), len(benign_pool))
                idx = rng.choice(len(benign_pool), n, replace=False)
                benign_selected = [benign_pool[i] for i in idx]
        else:
            # Fewer benign than toxic — take all benign, subsample toxic to match
            benign_selected = benign_pool
            idx = rng.choice(len(t_recs), len(benign_pool), replace=False)
            t_recs = [t_recs[i] for i in idx]

        matched_toxic.extend(t_recs)
        matched_benign.extend(benign_selected)
        per_taxon_stats.append({
            "taxon_id": ox,
            "name": name,
            "n_toxic": len(t_recs),
            "n_benign": len(benign_selected),
        })

    if not matched_toxic:
        raise RuntimeError("Organism-matched download produced 0 paired sequences.")

    write_fasta(matched_toxic, toxic_out)
    write_fasta(matched_benign, benign_out)

    print(f"\n[data] Organism-matched dataset: {len(matched_toxic)} toxic, {len(matched_benign)} benign.")
    print(f"[data] Per-taxon breakdown:")
    for s in per_taxon_stats:
        print(f"  OX={s['taxon_id']:8s}  {s['n_toxic']:3d} toxic / {s['n_benign']:3d} benign  {s['name']}")

    report_organism_distribution(matched_toxic, matched_benign, top_n=10)
    return toxic_out, benign_out


# ─────────────────────────────────────────────────────────────
# Eukaryotic clade-matched dataset (snakes / scorpions / spiders)
# ─────────────────────────────────────────────────────────────

def prepare_eukaryotic_dataset(
    out_dir: Path,
    min_len: int = 50,
    max_len: int = 1022,
    bin_width: int = 50,
    overwrite: bool = False,
    sleep_between: float = 0.5,
) -> tuple[Path, Path, Path]:
    """
    Download and prepare a clade-matched eukaryotic toxic/benign dataset.

    For each venomous clade (snakes=8570, scorpions=6843, spiders=6893):
      - All Swiss-Prot reviewed toxins (KW-0800) from that clade
      - Benign proteins from the same clade, length-stratified to match toxic

    Writes to out_dir:
      toxic.fasta, benign.fasta — merged across groups
      groups.tsv               — uniprot_id → group name for every sequence

    Returns (toxic_path, benign_path, groups_path).
    """
    out_dir = Path(out_dir)
    toxic_out = out_dir / "toxic.fasta"
    benign_out = out_dir / "benign.fasta"
    groups_out = out_dir / "groups.tsv"

    if (
        not overwrite
        and toxic_out.exists() and toxic_out.stat().st_size > 0
        and benign_out.exists() and benign_out.stat().st_size > 0
        and groups_out.exists()
    ):
        print(f"[data] Eukaryotic dataset cached in {out_dir} — skipping.")
        return toxic_out, benign_out, groups_out

    out_dir.mkdir(parents=True, exist_ok=True)

    all_toxic: list[tuple[str, str]] = []
    all_benign: list[tuple[str, str]] = []
    group_map: dict[str, str] = {}  # uniprot_id → group name

    for taxon_id, group_name in EUKARYOTIC_GROUPS.items():
        print(f"\n[data] Downloading {group_name} (taxonomy_id:{taxon_id})...")

        toxic_query = (
            f"(taxonomy_id:{taxon_id}) AND (keyword:{_KW_TOXIN}) AND (reviewed:true)"
        )
        benign_query = (
            f"(taxonomy_id:{taxon_id}) AND (reviewed:true)"
            f" NOT (keyword:{_KW_TOXIN})"
            f" NOT (keyword:{_KW_VIRULENCE})"
        )

        # Download all toxins for this clade
        toxic_raw: list[tuple[str, str]] = []
        for page_text in _stream_fasta_pages(toxic_query, page_size=500, sleep_between=sleep_between):
            toxic_raw.extend(parse_fasta(page_text))

        # Download benign pool (3× toxic count as target)
        benign_raw: list[tuple[str, str]] = []
        benign_target = max(len(toxic_raw) * 3, 500)
        for page_text in _stream_fasta_pages(
            benign_query, page_size=500, max_records=benign_target, sleep_between=sleep_between
        ):
            benign_raw.extend(parse_fasta(page_text))
            if len(benign_raw) >= benign_target:
                break

        # Length filter
        toxic_filt = [(h, s) for h, s in toxic_raw if min_len <= len(s) <= max_len]
        benign_filt = [(h, s) for h, s in benign_raw if min_len <= len(s) <= max_len]
        print(
            f"[data] {group_name}: {len(toxic_filt)} toxic, {len(benign_filt)} benign "
            f"after length filter [{min_len}, {max_len}]."
        )

        if not toxic_filt:
            print(f"[data] WARNING: No toxic sequences for {group_name} — skipping.")
            continue

        # Length-stratify benign to match toxic length distribution
        if benign_filt:
            benign_matched = length_stratify(toxic_filt, benign_filt, bin_width=bin_width)
        else:
            benign_matched = []
            print(f"[data] WARNING: No benign sequences for {group_name}.")

        # Record group membership
        for h, _ in toxic_filt:
            group_map[extract_uniprot_id(h)] = group_name
        for h, _ in benign_matched:
            group_map[extract_uniprot_id(h)] = group_name

        all_toxic.extend(toxic_filt)
        all_benign.extend(benign_matched)
        print(f"[data] {group_name}: kept {len(toxic_filt)} toxic, {len(benign_matched)} benign.")

    if not all_toxic:
        raise RuntimeError("Eukaryotic download produced 0 sequences.")

    write_fasta(all_toxic, toxic_out)
    write_fasta(all_benign, benign_out)

    with open(groups_out, "w") as f:
        f.write("uniprot_id\tgroup\n")
        for uid, grp in group_map.items():
            f.write(f"{uid}\t{grp}\n")

    print(f"\n[data] Eukaryotic dataset: {len(all_toxic)} toxic, {len(all_benign)} benign.")
    report_organism_distribution(all_toxic, all_benign)
    return toxic_out, benign_out, groups_out


def load_dataset_with_groups(
    data_dir: Path,
    max_per_class: int = 5000,
    min_len: int = 50,
    max_len: int = 1022,
) -> tuple[list[str], list[str], list[int], list[str]]:
    """
    Like load_dataset but also returns per-sequence group labels from groups.tsv.

    Returns (ids, sequences, labels, groups).
    Sequences without a groups.tsv entry get group "unknown".
    """
    ids, seqs, labels = load_dataset(
        data_dir, max_per_class=max_per_class, min_len=min_len, max_len=max_len
    )

    group_map: dict[str, str] = {}
    groups_path = Path(data_dir) / "groups.tsv"
    if groups_path.exists():
        with open(groups_path) as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    group_map[parts[0]] = parts[1]

    groups = [group_map.get(uid, "unknown") for uid in ids]
    return ids, seqs, labels, groups
