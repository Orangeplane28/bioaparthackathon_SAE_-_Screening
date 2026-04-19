"""
N2 — Data Preparation: SafeProtein-Bench + Hard Negatives + Family-Based Splits

Steps:
  1. Fetch 429 dangerous proteins from SafeProtein-Bench (GitHub)
  2. Fetch 4 categories of hard negatives from UniProt:
       - Membrane proteins (not toxin/antimicrobial)
       - Ion channels
       - Amphipathic helix / signal peptide proteins
       - Lipid-binding proteins
  3. Fetch general negatives (metabolic, nuclear, cytoskeletal)
  4. Annotate protein families via UniProt
  5. Split into train/val/test by family (70/15/15)
     - Convergent PFTs (P09616, P61914, P77335) forced into test

Run: python N2_data_preparation.py [--max-proteins N]
"""

import os
import sys
import json
import time
import random
import argparse
import requests
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    PROJECT_ROOT, DATA_DIR, SPLITS_DIR, CONVERGENT_IDS,
    UNIPROT_BASE, RATE_LIMIT_PAUSE, make_dirs
)
from utils.data import (
    fetch_uniprot_sequence, fetch_protein_family, uniprot_search
)


# ── SafeProtein-Bench ─────────────────────────────────────────────────────────
SAFE_PROTEIN_BENCH_URL = (
    'https://raw.githubusercontent.com/jigang-fan/'
    'SafeProtein/main/SafeProtein_Bench.json'
)

# Fallback: curated UniProt accessions for well-characterised dangerous proteins.
# Covers beta-PFTs, alpha-PFTs, AB-toxins, neurotoxins, and plant/bacterial RIPs.
# These are fetched from UniProt when SafeProtein-Bench is unavailable.
FALLBACK_TOXIN_IDS = [
    # ── Beta pore-forming toxins ──────────────────────────────────────────────
    'P09616',  # Alpha-hemolysin (S. aureus) — heptameric beta-PFT
    'P61914',  # Fragaceatoxin C / FraC (A. fragacea) — actinoporin beta-PFT
    'P77335',  # Cytolysin A / ClyA (E. coli) — alpha-helical PFT
    'P09839',  # Aerolysin (A. hydrophila)
    'P0A0L2',  # Leukotoxin LukS-PV (S. aureus)
    'P0A0L4',  # Leukotoxin LukF-PV (S. aureus)
    'P07659',  # Epsilon-toxin (C. perfringens)
    'P15108',  # Perfringolysin O / theta-toxin (C. perfringens) — CDC
    'P0C0S2',  # Listeriolysin O / LLO (L. monocytogenes) — CDC
    'P0C0T6',  # Pneumolysin (S. pneumoniae) — CDC
    'Q8GGC1',  # Streptolysin O (S. pyogenes) — CDC
    # ── Alpha pore-forming toxins / AMPs ─────────────────────────────────────
    'P01498',  # Delta-toxin (S. aureus)
    'P01501',  # Melittin (A. mellifera honeybee)
    'P11006',  # Magainin-2 (X. laevis frog)
    'P08043',  # Alamethicin (T. viride)
    # ── AB-type toxins ────────────────────────────────────────────────────────
    'P15917',  # Anthrax lethal factor / LF (B. anthracis)
    'P13423',  # Anthrax edema factor / EF (B. anthracis)
    'P10844',  # Botulinum neurotoxin A light chain (C. botulinum)
    'P0DPI0',  # Tetanus toxin light chain (C. tetani)
    'P0AAL5',  # Diphtheria toxin (C. diphtheriae)
    'P00588',  # Cholera toxin A subunit (V. cholerae)
    'P04406',  # Shiga toxin A subunit (S. dysenteriae)
    # ── Ribosome-inactivating proteins (RIPs) ────────────────────────────────
    'P02879',  # Ricin A chain (R. communis)
    'P20656',  # Saporin-S6 (S. officinalis)
    'P06280',  # Abrin A chain (A. precatorius)
    'P09762',  # Mistletoe lectin I A chain (V. album)
    # ── Snake venom toxins ────────────────────────────────────────────────────
    'P01388',  # Alpha-bungarotoxin (B. multicinctus)
    'P60775',  # Cobratoxin (N. naja)
    'P00601',  # Phospholipase A2 (N. naja) — venom PLA2
    'P00278',  # Phospholipase A2 (C. atrox)
    'P0C8G5',  # Taipoxin alpha chain (O. scutellatus)
    # ── Spider / scorpion toxins ──────────────────────────────────────────────
    'P00974',  # Dendrotoxin I (D. polylepis)
    'P58809',  # Charybdotoxin (L. quinquestriatus)
    'P01497',  # Omega-agatoxin IVA (A. aperta)
    # ── Cone snail (conotoxins) ───────────────────────────────────────────────
    'P0C1X2',  # Alpha-conotoxin ImI (C. imperialis)
    'P50983',  # Mu-conotoxin GIIIA (C. geographus)
    # ── Plant toxins ──────────────────────────────────────────────────────────
    'P01579',  # Viscotoxin A1 (V. album) — thionin-type
    'P07695',  # Pokeweed antiviral protein / PAP (P. americana)
    # ── Fungal / microbial toxins ─────────────────────────────────────────────
    'P15284',  # Aflatoxin B1 biosynthesis protein (A. parasiticus) — polyketide
    'P16404',  # Fumonisin B1 biosynthesis (F. proliferatum)
]


def fetch_safe_protein_bench() -> dict:
    """
    Download SafeProtein-Bench 429-protein dangerous protein list.

    Falls back to fetching FALLBACK_TOXIN_IDS from UniProt if the
    SafeProtein-Bench URL is unavailable (404, network error, etc.).
    """
    # Try SafeProtein-Bench first
    try:
        resp = requests.get(SAFE_PROTEIN_BENCH_URL, timeout=30)
        resp.raise_for_status()
        raw = resp.json()
        # Normalise to {pid: {name, sequence}} regardless of source format.
        # SafeProtein_Bench.json uses capitalised keys: "Sequence", "Length"
        data = {}
        for pid, info in raw.items():
            seq = info.get('Sequence') or info.get('sequence') or ''
            if seq:
                data[pid] = {'name': info.get('name', pid), 'sequence': seq}
        print(f'  SafeProtein-Bench: {len(data)} proteins')
        return data
    except Exception as e:
        print(f'  WARNING: Could not fetch SafeProtein-Bench ({e})')
        print('  Falling back to curated UniProt fallback list '
              f'({len(FALLBACK_TOXIN_IDS)} accessions)...')

    # Fallback: fetch each accession individually from UniProt
    data = {}
    ok, fail = 0, 0
    for uid in FALLBACK_TOXIN_IDS:
        try:
            seq = fetch_uniprot_sequence(uid)
            if seq:
                data[uid] = {'name': uid, 'sequence': seq}
                ok += 1
            else:
                fail += 1
        except Exception as e2:
            print(f'    WARNING: Could not fetch {uid}: {e2}')
            fail += 1
        time.sleep(RATE_LIMIT_PAUSE)

    print(f'  Fallback toxins fetched: {ok} OK, {fail} failed')
    if ok == 0:
        print('  ERROR: No toxins could be fetched. '
              'Check internet access or add proteins manually to data/toxins.json')
    return data


# ── Hard negative queries ─────────────────────────────────────────────────────
# Design rationale: queries are intentionally spread across multiple organisms
# (human, E. coli, yeast, Arabidopsis) so the classifier cannot separate toxins
# from negatives on the basis of organism-level embedding signal.
# All queries exclude keyword:toxin and keyword:antimicrobial.
HARD_NEG_QUERIES = {
    # Human membrane proteins — similar membrane topology to beta-PFTs
    'hard_neg_membrane': (
        'organism_id:9606 AND keyword:membrane AND reviewed:true '
        'NOT keyword:toxin NOT keyword:antimicrobial'
    ),
    # Ion channels from multiple organisms — similar pore architecture
    'hard_neg_ion_channel': (
        'keyword:"ion channel" AND reviewed:true '
        'NOT keyword:toxin NOT keyword:antimicrobial'
    ),
    # Coiled-coil / amphipathic helices from bacteria (non-human counterweight)
    'hard_neg_amphipathic': (
        'keyword:"coiled coil" AND organism_id:83333 AND reviewed:true '
        'NOT keyword:toxin NOT keyword:antimicrobial'
    ),
    # Lipid-binding from plants — avoids human-only clustering
    'hard_neg_lipid_binding': (
        'keyword:"lipid-binding" AND organism_id:3702 AND reviewed:true '
        'NOT keyword:toxin NOT keyword:antimicrobial'
    ),
    # Bacterial secreted proteins — same secretion context as many toxins, non-toxic
    'hard_neg_bacterial_secreted': (
        'keyword:secreted AND organism_id:83333 AND reviewed:true '
        'NOT keyword:toxin NOT keyword:antimicrobial NOT keyword:virulence'
    ),
    # Fungal extracellular proteins — same extracellular context, non-toxic
    'hard_neg_fungal_extracellular': (
        'keyword:secreted AND organism_id:559292 AND reviewed:true '
        'NOT keyword:toxin NOT keyword:antimicrobial'
    ),
}

GENERAL_NEG_QUERIES = {
    # Human metabolic — classic easy negative
    'gen_neg_metabolic':    'keyword:glycolysis AND reviewed:true AND organism_id:9606',
    # E. coli metabolic — non-human counterweight
    'gen_neg_ecoli_metabolic': (
        'keyword:glycolysis AND reviewed:true AND organism_id:83333'
    ),
    # Human nuclear
    'gen_neg_nuclear':      'keyword:nucleus AND reviewed:true AND organism_id:9606',
    # Human cytoskeletal
    'gen_neg_cytoskeletal': 'keyword:cytoskeleton AND reviewed:true AND organism_id:9606',
    # Human ribosomal
    'gen_neg_ribosomal':    'keyword:ribosome AND reviewed:true AND organism_id:9606',
    # Plant ribosomal — non-animal, non-fungal, non-bacterial diversity
    'gen_neg_plant_ribosomal': (
        'keyword:ribosome AND reviewed:true AND organism_id:3702'
    ),
}

# ── Length range of toxins (used to filter negatives for length matching) ──────
# SafeProtein-Bench toxins span ~10–1500 aa. We keep negatives within 3× of the
# toxin median (~300 aa) to prevent the classifier from learning length as a proxy.
NEG_MIN_LEN = 10
NEG_MAX_LEN = 1500


def fetch_proteins_from_query(category: str, query: str,
                               max_per_category: int = 60) -> list:
    """
    Fetch proteins for a UniProt query and return as list of dicts.

    Applies a length filter (NEG_MIN_LEN..NEG_MAX_LEN) to negatives so the
    classifier cannot trivially learn sequence-length as a toxicity proxy.
    """
    results = uniprot_search(query, size=max_per_category * 2)  # over-fetch to absorb filtering
    proteins = []
    for r in results:
        if len(proteins) >= max_per_category:
            break
        uid = r.get('primaryAccession', '')
        if not uid:
            continue
        seq_data = r.get('sequence', {})
        seq = seq_data.get('value', '')
        if not seq:
            seq = fetch_uniprot_sequence(uid)
        if not seq or len(seq) < 10:
            continue
        # Length filter: discard proteins far outside the toxin length range
        if not (NEG_MIN_LEN <= len(seq) <= NEG_MAX_LEN):
            continue
        name = (r.get('proteinDescription', {})
                  .get('recommendedName', {})
                  .get('fullName', {})
                  .get('value', uid))
        proteins.append({
            'id': uid, 'name': name, 'sequence': seq,
            'length': len(seq), 'category': category,
        })
        time.sleep(RATE_LIMIT_PAUSE)
    print(f'  {category}: {len(proteins)} proteins')
    return proteins


# ── Family annotation & splitting ─────────────────────────────────────────────

def annotate_families(proteins: dict) -> dict:
    """Add 'family' field to each protein via UniProt JSON endpoint."""
    print('  Annotating protein families...')
    for i, (pid, prot) in enumerate(proteins.items()):
        if 'family' not in prot:
            prot['family'] = fetch_protein_family(pid)
        if (i + 1) % 50 == 0:
            print(f'    Annotated {i+1}/{len(proteins)} proteins')
    return proteins


def family_based_split(proteins: dict, convergent_ids: list,
                        train_frac: float = 0.70,
                        val_frac:   float = 0.15,
                        seed: int = 42) -> dict:
    """
    Split proteins into train/val/test by protein family (disjoint families).

    Convergent PFTs are always forced into test set.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Group PIDs by family
    family_to_pids = defaultdict(list)
    for pid, prot in proteins.items():
        if pid not in convergent_ids:
            family_to_pids[prot.get('family', f'cluster_{pid[:3]}')].append(pid)

    families = list(family_to_pids.keys())
    random.shuffle(families)

    n = len(families)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_families = set(families[:n_train])
    val_families   = set(families[n_train:n_train + n_val])

    train_pids, val_pids, test_pids = [], [], []

    for fam, pids in family_to_pids.items():
        if fam in train_families:
            train_pids.extend(pids)
        elif fam in val_families:
            val_pids.extend(pids)
        else:
            test_pids.extend(pids)

    # Force convergent PFTs into test
    for cid in convergent_ids:
        if cid in proteins:
            for lst in [train_pids, val_pids]:
                if cid in lst:
                    lst.remove(cid)
            if cid not in test_pids:
                test_pids.append(cid)

    return {
        'train': train_pids,
        'val':   val_pids,
        'test':  test_pids,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(max_proteins: int = 60):
    print('=' * 65)
    print('  N2 — DATA PREPARATION')
    print('=' * 65)
    make_dirs()

    all_proteins = {}  # pid → protein dict

    # 1. Toxins from SafeProtein-Bench
    print('\n[1/5] Fetching SafeProtein-Bench dangerous proteins...')
    spb = fetch_safe_protein_bench()
    for pid, info in spb.items():
        seq = info.get('sequence', '') or fetch_uniprot_sequence(pid)
        if not seq:
            continue
        all_proteins[pid] = {
            'id': pid,
            'name': info.get('name', pid),
            'sequence': seq,
            'length': len(seq),
            'label': 1,
            'is_hard_negative': False,
            'category': 'toxin',
        }
        time.sleep(RATE_LIMIT_PAUSE)
    print(f'  Toxins loaded: {len(all_proteins)}')

    # 2. Hard negatives
    print('\n[2/5] Fetching hard negatives...')
    for category, query in HARD_NEG_QUERIES.items():
        prots = fetch_proteins_from_query(category, query, max_proteins)
        for p in prots:
            pid = p['id']
            if pid not in all_proteins:
                all_proteins[pid] = {
                    **p,
                    'label': 0,
                    'is_hard_negative': True,
                }

    # 3. General negatives
    print('\n[3/5] Fetching general negatives...')
    for category, query in GENERAL_NEG_QUERIES.items():
        prots = fetch_proteins_from_query(category, query, max_proteins)
        for p in prots:
            pid = p['id']
            if pid not in all_proteins:
                all_proteins[pid] = {
                    **p,
                    'label': 0,
                    'is_hard_negative': False,
                }

    # 4. Ensure convergent PFTs are present
    print('\n[4/5] Ensuring convergent PFTs are present...')
    CONVERGENT_META = {
        'P09616': {'name': 'Alpha-hemolysin', 'organism': 'S. aureus'},
        'P61914': {'name': 'Fragaceatoxin C (FraC)', 'organism': 'A. fragacea'},
        'P77335': {'name': 'Cytolysin A (ClyA)', 'organism': 'E. coli'},
    }
    for cid, meta in CONVERGENT_META.items():
        if cid not in all_proteins:
            seq = fetch_uniprot_sequence(cid)
            if seq:
                all_proteins[cid] = {
                    'id': cid, 'label': 1, 'is_hard_negative': False,
                    'sequence': seq, 'length': len(seq), 'category': 'toxin',
                    **meta,
                }
                print(f'  Added convergent PFT: {cid} ({meta["name"]})')
        else:
            print(f'  Convergent PFT already present: {cid}')

    # 5. Annotate families & split
    print('\n[5/5] Annotating families and splitting...')
    all_proteins = annotate_families(all_proteins)
    splits = family_based_split(all_proteins, CONVERGENT_IDS)

    # Save
    # Organism breakdown — sanity-check that negatives span multiple organisms
    from collections import Counter
    neg_orgs = Counter()
    for p in all_proteins.values():
        if p.get('label') == 0:
            org = p.get('organism', p.get('category', 'unknown'))
            neg_orgs[org] += 1

    master = {
        'proteins': all_proteins,
        'metadata': {
            'convergent_ids':   CONVERGENT_IDS,
            'n_toxins':         sum(1 for p in all_proteins.values() if p.get('label') == 1),
            'n_hard_neg':       sum(1 for p in all_proteins.values() if p.get('is_hard_negative')),
            'n_gen_neg':        sum(1 for p in all_proteins.values()
                                     if p.get('label') == 0 and not p.get('is_hard_negative')),
            'neg_length_range': [NEG_MIN_LEN, NEG_MAX_LEN],
            'neg_categories':   list(set(
                p.get('category', '') for p in all_proteins.values() if p.get('label') == 0
            )),
        }
    }
    master_path = os.path.join(DATA_DIR, 'master_dataset_esm3.json')
    with open(master_path, 'w') as f:
        json.dump(master, f, indent=2)
    print(f'  Master dataset saved: {master_path}')

    os.makedirs(SPLITS_DIR, exist_ok=True)
    for split_name, pids in splits.items():
        path = os.path.join(SPLITS_DIR, f'{split_name}_pids.json')
        with open(path, 'w') as f:
            json.dump(pids, f, indent=2)

    # Summary
    print()
    print('=' * 65)
    print('  N2 RESULTS SUMMARY')
    print('=' * 65)
    print(f'  Total proteins:    {len(all_proteins)}')
    print(f'  Toxins (label=1):  {master["metadata"]["n_toxins"]}')
    print(f'  Hard negatives:    {master["metadata"]["n_hard_neg"]}')
    print(f'  General negatives: {master["metadata"]["n_gen_neg"]}')
    print(f'  Train split:       {len(splits["train"])} proteins')
    print(f'  Val split:         {len(splits["val"])} proteins')
    print(f'  Test split:        {len(splits["test"])} proteins')
    print()
    print('NEXT: python N3_embeddings_baselines.py  (requires GPU)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='N2: Data preparation')
    parser.add_argument('--max-proteins', type=int, default=60,
                        help='Max proteins per UniProt query category')
    args = parser.parse_args()
    main(max_proteins=args.max_proteins)
