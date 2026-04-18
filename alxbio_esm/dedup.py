"""Bias 3 fix: sequence redundancy removal.

Implements a greedy identity-based clustering (CD-HIT-like) using k-mer Jaccard
similarity as a fast proxy for sequence identity. For ~1000 sequences this runs
in seconds without external binaries.

Optionally wraps the real CD-HIT binary if available on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

from alxbio_esm.data import parse_fasta, write_fasta


# ─────────────────────────────────────────────────────────────
# K-mer Jaccard (Python fallback)
# ─────────────────────────────────────────────────────────────

def _kmer_set(seq: str, k: int = 5) -> frozenset[str]:
    return frozenset(seq[i : i + k] for i in range(len(seq) - k + 1))


def _jaccard(a: frozenset, b: frozenset) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def greedy_dedup(
    records: list[tuple[str, str]],
    threshold: float = 0.4,
    k: int = 5,
) -> list[tuple[str, str]]:
    """
    Greedy sequence deduplication using k-mer Jaccard similarity.

    Two sequences are considered redundant if their k-mer Jaccard similarity
    exceeds *threshold*. The representative (first encountered) is kept.

    A Jaccard threshold of ~0.4 on 5-mers correlates approximately with 40%
    sequence identity, consistent with the CD-HIT-0.4 cutoff used in ToxDL 2.0.

    Parameters
    ----------
    records   : list of (header, sequence)
    threshold : Jaccard similarity cutoff (0.4 ≈ 40% seq identity)
    k         : k-mer length

    Returns
    -------
    Deduplicated list of (header, sequence).
    """
    reps: list[tuple[str, str]] = []
    rep_kmers: list[frozenset] = []

    for header, seq in tqdm(records, desc="Deduplicating"):
        kmers = _kmer_set(seq, k)
        redundant = any(_jaccard(kmers, r) > threshold for r in rep_kmers)
        if not redundant:
            reps.append((header, seq))
            rep_kmers.append(kmers)

    print(f"[dedup] {len(records)} → {len(reps)} sequences after dedup (threshold={threshold}).")
    return reps


# ─────────────────────────────────────────────────────────────
# CD-HIT wrapper (if available)
# ─────────────────────────────────────────────────────────────

def _cdhit_available() -> bool:
    return shutil.which("cd-hit") is not None


def cdhit_dedup(
    records: list[tuple[str, str]],
    identity: float = 0.4,
    word_size: int = 2,
) -> list[tuple[str, str]]:
    """
    Deduplicate using the real CD-HIT binary.

    Falls back to greedy_dedup if CD-HIT is not on PATH.
    """
    if not _cdhit_available():
        print("[dedup] cd-hit not found on PATH — using Python k-mer fallback.")
        return greedy_dedup(records, threshold=identity)

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / "input.fasta"
        out_path = Path(tmpdir) / "output.fasta"
        write_fasta(records, in_path)

        cmd = [
            "cd-hit",
            "-i", str(in_path),
            "-o", str(out_path),
            "-c", str(identity),
            "-n", str(word_size),
            "-d", "0",        # full header in cluster file
            "-T", "0",        # use all CPU threads
            "-M", "4000",     # 4GB memory limit
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[dedup] CD-HIT failed: {result.stderr[:500]}\nFalling back to Python.")
            return greedy_dedup(records, threshold=identity)

        deduplicated = parse_fasta(out_path.read_text())

    print(f"[dedup] CD-HIT: {len(records)} → {len(deduplicated)} sequences.")
    return deduplicated


def dedup_dataset(
    raw_dir: Path,
    out_dir: Path | None = None,
    identity: float = 0.4,
    prefer_cdhit: bool = True,
) -> tuple[Path, Path]:
    """
    Deduplicate toxic.fasta and benign.fasta in *raw_dir*.

    Writes *toxic_nr.fasta* and *benign_nr.fasta* to *out_dir* (defaults to raw_dir).
    Returns paths to the two deduplicated files.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir) if out_dir else raw_dir

    fn = cdhit_dedup if prefer_cdhit else greedy_dedup

    results = {}
    for label in ("toxic", "benign"):
        in_path = raw_dir / f"{label}.fasta"
        out_path = out_dir / f"{label}_nr.fasta"

        if out_path.exists() and out_path.stat().st_size > 0:
            from alxbio_esm.data import parse_fasta_file
            nr = parse_fasta_file(out_path)
            print(f"[dedup] {out_path} cached with {len(nr)} records — skipping.")
            results[label] = out_path
            continue

        from alxbio_esm.data import parse_fasta_file
        records = parse_fasta_file(in_path)
        nr = fn(records, identity=identity) if prefer_cdhit else fn(records, threshold=identity)
        write_fasta(nr, out_path)
        results[label] = out_path

    return results["toxic"], results["benign"]
