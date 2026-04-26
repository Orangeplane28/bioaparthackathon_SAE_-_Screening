"""
BioChain ML Engine v3
=====================
Stage 1: ESM-3 / ESM-2 / Lightweight encoder -> fragment embeddings
Stage 2: Set Transformer -> P(threat) in [0,1]

Usage:
  python biochain_ml_v3.py --esm3          # ESM-3 small open (best)
  python biochain_ml_v3.py --esm2          # ESM-2 8M
  python biochain_ml_v3.py                 # lightweight CPU demo
  python biochain_ml_v3.py --esm3 --cv     # 5-fold cross-validation
  python biochain_ml_v3.py --esm3 --robustness  # + fragment-count sweep

Requirements:
  pip install torch transformers scikit-learn scipy requests tqdm
  pip install esm                           # for ESM-3 only
  export HF_TOKEN=hf_...
  # Accept ESM-3 license: huggingface.co/EvolutionaryScale/esm3-sm-open-v1
"""
import os, json, random, hashlib, warnings, requests, argparse
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss
warnings.filterwarnings("ignore")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else "cpu"
)
print(f"[BioChain v3] device={DEVICE}")

# ---------------------------------------------------------------------------
# 1. Protein accession lists
# ---------------------------------------------------------------------------
# Toxin family labels — used as group_id for CV splits so entire families
# are held out from training. Proves the model learns toxicity not taxonomy.
TOXIN_FAMILY: Dict[str, str] = {
    # Snake toxins
    "P00625": "snake", "Q90WC0": "snake", "P01552": "snake",
    "P17529": "snake", "P15922": "snake", "P20049": "snake",
    "P02978": "snake", "P01536": "snake",
    # Scorpion toxins
    "P81054": "scorpion", "P84891": "scorpion", "Q9S419": "scorpion",
    # Spider toxins
    "P16893": "spider", "P15018": "spider", "P68137": "spider",
    # Bacterial toxins
    "P10844": "bacterial", "P04958": "bacterial", "P00588": "bacterial",
    "P09977": "bacterial", "P0A0M1": "bacterial", "P01555": "bacterial",
    "P11439": "bacterial", "P15917": "bacterial", "P0A0L2": "bacterial",
    "P15011": "bacterial", "P0A0L5": "bacterial",
    # Plant toxins (RIPs)
    "P02879": "plant_rip", "P11140": "plant_rip",
}

METAZOAN_TOXIN_IDS = [
    "P00625","Q90WC0","P01552","P17529","P15922","P20049",
    "P02978","P01536","P81054","P84891","Q9S419","P16893",
    "P15018","P68137",
]
NON_METAZOAN_TOXIN_IDS = [
    "P10844",  # Botulinum neurotoxin A  (C. botulinum)
    "P04958",  # Tetanus toxin           (C. tetani)
    "P00588",  # Diphtheria toxin        (C. diphtheriae)
    "P09977",  # Shiga toxin 1A          (E. coli O157)
    "P0A0M1",  # Shiga toxin 2A          (E. coli)
    "P01555",  # Cholera toxin A         (V. cholerae)
    "P11439",  # Pseudomonas exotoxin A
    "P15917",  # Anthrax lethal factor   (B. anthracis)
    "P0A0L2",  # Staph enterotoxin A     (S. aureus)
    "P15011",  # TSST-1                  (S. aureus)
    "P0A0L5",  # Alpha-toxin             (S. aureus)
    "P02879",  # Ricin                   (R. communis)
    "P11140",  # Abrin-a                 (A. precatorius)
]
BENIGN_IDS_HUMAN  = ["P08524","O00624","P04637","P00533","P01308"]
BENIGN_IDS_ECOLI  = ["P0ABU9","P0A8W0","P00722","P0A9Q1"]
BENIGN_IDS_YEAST  = ["P00330","P00549","P00925"]
# Format: (label, query_type, query_value, n)
# query_type "taxonomy" -> builds reviewed:true AND {value} AND NOT KW-0800
# query_type "full"     -> uses {value} verbatim (for lectin/defensin etc.)
EXTRA_BENIGN_QUERIES = [
    ("arabidopsis", "taxonomy", "taxonomy_id:3702", 200),
    ("fly",         "taxonomy", "taxonomy_id:7227", 200),
    ("worm",        "taxonomy", "taxonomy_id:6239", 200),
    # Biological hard negatives — same fold families as toxins but non-toxic.
    # Lectin: carbohydrate-binding proteins, similar beta-barrel folds to some toxins.
    # Defensin precursors: antimicrobial peptides, similar disulfide-rich scaffold to
    # some snake/scorpion toxins. Using length 100-500 to get full precursor proteins
    # that are long enough to fragment (mature defensins ~30 aa are too short).
    ("lectin",   "full", "reviewed:true+AND+family:lectin+AND+length:[100+TO+500]+AND+NOT+keyword:KW-0800", 150),
    ("defensin", "full", "reviewed:true+AND+name:defensin+AND+length:[100+TO+500]+AND+NOT+keyword:KW-0800", 100),
]

# ---------------------------------------------------------------------------
# 2. Hard-negative mutations (FIX #2)
# ---------------------------------------------------------------------------
# Format: accession -> [(motif_context, WT aa, Mut aa, window, citation)]
# motif_context: short AA string surrounding the target residue so we can
# locate it dynamically even when signal peptides shift absolute positions.
# window: how many residues left/right to search if exact match fails.
HARD_NEGATIVE_MUTATIONS: Dict[str, List[Tuple[str,str,str,int,str]]] = {
    "P02879": [("SAGITLGY", "E", "Q", 5,  "Ricin A-chain: catalytic E (Endo 1987) — locate via SAGITLGYE motif")],
    "P00588": [("GADDVVDS", "E", "S", 5,  "Diphtheria toxin CRM197: E148S in NAD-binding loop")],
    "P15917": [("LHELGHAV", "H", "A", 5,  "Anthrax LF: Zn-binding HEXXH motif H686A (HExxH is canonical)")],
    "P11140": [("SAGITLGY", "E", "V", 5,  "Abrin A-chain: catalytic Glu, same GAGA motif as Ricin")],
    "P10844": [("HELIH",    "H", "A", 5,  "BoNT/A LC: Zn-HEXXH catalytic H, HELIH motif")],
}


def _find_and_mutate(sequence: str,
                     motif: str, wt_aa: str, mut_aa: str, window: int,
                     note: str) -> Optional[str]:
    """
    Locate `wt_aa` near `motif` in `sequence` and apply the substitution.
    Strategy:
      1. Find the motif substring.
      2. Within `window` residues on either side, find the first occurrence of wt_aa.
      3. Apply the substitution.
    Returns mutated sequence or None if not found.
    """
    motif = motif.upper(); seq = sequence.upper()

    # Try to find motif, then locate wt_aa nearby
    pos = seq.find(motif)
    if pos != -1:
        search_start = max(0, pos - window)
        search_end   = min(len(seq), pos + len(motif) + window)
        region = seq[search_start:search_end]
        local = region.find(wt_aa)
        if local != -1:
            abs_pos = search_start + local
            return sequence[:abs_pos] + mut_aa + sequence[abs_pos+1:]

    # Fallback: scan for first occurrence of wt_aa in full sequence
    # (only if motif completely absent — unusual isoform)
    idx = seq.find(wt_aa)
    if idx != -1:
        print(f"  [HardNeg] motif '{motif}' not found; used first '{wt_aa}' at pos {idx+1}")
        return sequence[:idx] + mut_aa + sequence[idx+1:]

    print(f"  [HardNeg] could not locate '{wt_aa}' near motif '{motif}' ({note})")
    return None

# ---------------------------------------------------------------------------
# 3. UniProt fetcher
# ---------------------------------------------------------------------------
def download_uniprot_sequences(accessions: List[str],
                                cache_path: str = "uniprot_cache.json") -> Dict[str,str]:
    cache: Dict[str,str] = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
    needed = [a for a in accessions if a not in cache]
    if needed:
        print(f"[UniProt] fetching {len(needed)} sequences...")
        for acc in needed:
            try:
                r = requests.get(f"https://rest.uniprot.org/uniprotkb/{acc}.json", timeout=15)
                r.raise_for_status()
                cache[acc] = r.json()["sequence"]["value"]
            except Exception as e:
                print(f"  WARN {acc}: {e}")
        with open(cache_path,"w") as f:
            json.dump(cache, f)
    return {a: cache[a] for a in accessions if a in cache}

def download_uniprot_keyword_batch(keyword_id: str = "KW-0800",
                                    taxonomy_filter: Optional[str] = None,
                                    max_results: int = 500,
                                    cache_path: str = "uniprot_kw_cache.json"
                                    ) -> Tuple[Dict[str,str], Dict[str,str]]:
    """Download reviewed proteins by keyword.
    Returns (seq_dict, organism_dict) — organism used as group_id to prevent
    homolog leakage across CV folds for bulk toxins."""
    cache_key = f"{keyword_id}_{taxonomy_filter}_{max_results}"
    cache: Dict[str,Dict] = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
    if cache_key in cache:
        entry = cache[cache_key]
        # Handle old cache format (plain dict of sequences)
        if entry and isinstance(next(iter(entry.values())), str):
            return entry, {}
        seqs = {acc: v["seq"] for acc, v in entry.items()}
        orgs = {acc: v["org"] for acc, v in entry.items()}
        return seqs, orgs
    parts = [f"reviewed:true", f"keyword:{keyword_id}"]
    if taxonomy_filter:
        parts.append(taxonomy_filter)
    url = (f"https://rest.uniprot.org/uniprotkb/search"
           f"?query={'+AND+'.join(parts)}&format=json"
           f"&fields=accession,sequence,organism_name&size={max_results}")
    try:
        r = requests.get(url, timeout=30); r.raise_for_status()
        results = r.json().get("results", [])
        seqs, orgs = {}, {}
        for e in results:
            if "sequence" not in e: continue
            acc = e["primaryAccession"]
            seqs[acc] = e["sequence"]["value"]
            # organism.scientificName gives e.g. "Clostridium botulinum" — use as group
            org = e.get("organism", {}).get("scientificName", "unknown")
            orgs[acc] = org
        cache[cache_key] = {acc: {"seq": seqs[acc], "org": orgs[acc]} for acc in seqs}
        with open(cache_path,"w") as f:
            json.dump(cache, f)
        print(f"[UniProt] batch: {len(seqs)} proteins ({len(set(orgs.values()))} organisms)")
        return seqs, orgs
    except Exception as e:
        print(f"[UniProt] batch failed: {e}"); return {}, {}

# ---------------------------------------------------------------------------
# 4. Mutation utilities
# ---------------------------------------------------------------------------
def apply_mutation(sequence: str, position_1indexed: int, wt_aa: str, mut_aa: str) -> str:
    """Apply a single documented point mutation. Validates WT residue first."""
    pos = position_1indexed - 1
    if pos < 0 or pos >= len(sequence):
        raise ValueError(f"Position {position_1indexed} out of range (len={len(sequence)})")
    if sequence[pos] != wt_aa:
        raise ValueError(f"Expected {wt_aa} at pos {position_1indexed}, found {sequence[pos]}")
    return sequence[:pos] + mut_aa + sequence[pos+1:]

# ---------------------------------------------------------------------------
# 5. Fragmentation
# ---------------------------------------------------------------------------
def fragment_protein(sequence: str, min_len: int = 60, max_len: int = 120,
                     stride: int = 30) -> List[str]:
    frags, pos = [], 0
    while pos < len(sequence):
        frag = sequence[pos:pos + random.randint(min_len, max_len)]
        if len(frag) >= min_len:
            frags.append(frag)
        pos += stride
    return frags

@dataclass
class FragmentSet:
    fragments: List[str]
    label: int        # 1=threat, 0=benign
    group_id: str     # protein-level group key
    source: str

def _make_fs(seq: str, label: int, source: str, group_id: str,
             n_range=(3,8), min_len=50, max_len=100, stride=25) -> Optional[FragmentSet]:
    frags = fragment_protein(seq, min_len, max_len, stride)
    if len(frags) < n_range[0]:
        return None
    selected = random.sample(frags, min(random.randint(*n_range), len(frags)))
    return FragmentSet(fragments=selected, label=label, group_id=group_id, source=source)

# ---------------------------------------------------------------------------
# 6. Dataset builders
# ---------------------------------------------------------------------------
def build_dataset_from_uniprot(random_seed: int = 42,
                                cache_path: str = "uniprot_cache.json",
                                n_augments: int = 4,
                                use_bulk_toxins: bool = True,
                                max_bulk_toxins: int = 300) -> List[FragmentSet]:
    random.seed(random_seed)
    dataset: List[FragmentSet] = []

    # Positives: curated accessions + optional bulk KW-0800 download
    all_tox = list(set(METAZOAN_TOXIN_IDS + NON_METAZOAN_TOXIN_IDS))
    tox_seqs = download_uniprot_sequences(all_tox, cache_path)

    bulk_organisms: Dict[str, str] = {}
    if use_bulk_toxins:
        # Pull the full reviewed toxin set from UniProt (metazoan + non-metazoan).
        # Also fetches organism name — used as group_id to prevent homolog leakage:
        # two Ricin variants from R.communis share organism group, so StratifiedGroupKFold
        # keeps them on the same side of every train/test split.
        bulk, bulk_organisms = download_uniprot_keyword_batch(
            keyword_id="KW-0800", taxonomy_filter=None,
            max_results=max_bulk_toxins, cache_path="uniprot_kw_cache.json"
        )
        tox_seqs = {**bulk, **tox_seqs}   # curated list takes precedence for overlaps
        print(f"[Dataset] toxins after bulk merge: {len(tox_seqs)} ({len(bulk)} bulk + {len(all_tox)} curated)")
    else:
        print(f"[Dataset] toxins: {len(tox_seqs)}/{len(all_tox)}")

    for acc, seq in tox_seqs.items():
        if len(seq) < 80: continue
        # group_id priority:
        #   1. Curated family label (snake/scorpion/spider/bacterial/plant_rip)
        #   2. Organism name from UniProt bulk download (prevents homolog leakage)
        #   3. Accession hash fallback (should never reach this for well-fetched data)
        if acc in TOXIN_FAMILY:
            gid = TOXIN_FAMILY[acc]
        elif acc in bulk_organisms and bulk_organisms[acc] != "unknown":
            # Sanitise organism name -> safe group key (e.g. "Clostridium botulinum" -> "clostridium_botulinum")
            gid = bulk_organisms[acc].lower().replace(" ", "_")[:32]
        else:
            gid = hashlib.md5(acc.encode()).hexdigest()[:8]
        for i in range(n_augments):
            random.seed(random_seed + i*1000 + int(hashlib.md5(acc.encode()).hexdigest(), 16) % 1000)
            fs = _make_fs(seq, 1, f"toxin_{acc}_{i}", gid)
            if fs: dataset.append(fs)

    # Negatives: explicit accessions
    benign_ids = BENIGN_IDS_HUMAN + BENIGN_IDS_ECOLI + BENIGN_IDS_YEAST
    ben_seqs = download_uniprot_sequences(benign_ids, cache_path)
    print(f"[Dataset] explicit benign: {len(ben_seqs)}")
    for acc, seq in ben_seqs.items():
        if len(seq) < 80: continue
        gid = hashlib.md5(acc.encode()).hexdigest()[:8]
        for i in range(n_augments):
            random.seed(random_seed + i*1000 + int(hashlib.md5(acc.encode()).hexdigest(), 16) % 1000)
            fs = _make_fs(seq, 0, f"benign_{acc}_{i}", gid)
            if fs: dataset.append(fs)

    # Extra negatives: organism-based + biological hard negatives (lectin, defensin)
    for org, qtype, qval, n in EXTRA_BENIGN_QUERIES:
        if qtype == "taxonomy":
            query = f"reviewed:true+AND+{qval}+AND+NOT+keyword:KW-0800"
        else:  # "full" — use verbatim
            query = qval
        url = (f"https://rest.uniprot.org/uniprotkb/search"
               f"?query={query}&format=json&fields=accession,sequence&size={n}")
        try:
            r = requests.get(url, timeout=30); r.raise_for_status()
            extra = {e["primaryAccession"]: e["sequence"]["value"]
                     for e in r.json().get("results",[]) if "sequence" in e}
            print(f"[Dataset] extra benign ({org}): {len(extra)}")
            for acc, seq in extra.items():
                if len(seq) < 80: continue
                gid = hashlib.md5(acc.encode()).hexdigest()[:8]
                for i in range(2):
                    random.seed(random_seed + i*999 + int(hashlib.md5(acc.encode()).hexdigest(), 16) % 999)
                    fs = _make_fs(seq, 0, f"benign_{org}_{acc}_{i}", gid)
                    if fs: dataset.append(fs)
        except Exception as e:
            print(f"[Dataset] extra benign ({org}) failed: {e}")

    random.shuffle(dataset)
    n_pos = sum(1 for d in dataset if d.label==1)
    print(f"[Dataset] total={len(dataset)} pos={n_pos} neg={len(dataset)-n_pos}")

    # Print family distribution so we can verify family-level holdout is working
    from collections import Counter
    family_counts = Counter(fs.group_id for fs in dataset if fs.label == 1)
    print(f"[Dataset] positive family distribution: {dict(family_counts)}")
    return dataset

def build_hard_negative_test_set(cache_path: str = "uniprot_cache.json",
                                  random_seed: int = 42) -> List[FragmentSet]:
    """Real toxin sequences with documented neutralizing point mutations. Label=0.
    Uses motif-based residue location so it works regardless of signal peptide offsets."""
    random.seed(random_seed)
    seqs = download_uniprot_sequences(list(HARD_NEGATIVE_MUTATIONS.keys()), cache_path)
    result: List[FragmentSet] = []
    for acc, mutations in HARD_NEGATIVE_MUTATIONS.items():
        if acc not in seqs:
            print(f"[HardNeg] SKIP {acc}: unavailable"); continue
        mut_seq, applied = seqs[acc], []
        for motif, wt, mut, window, note in mutations:
            new_seq = _find_and_mutate(mut_seq, motif, wt, mut, window, note)
            if new_seq is None:
                print(f"[HardNeg] {acc} mutation failed — using WT sequence (conservative)")
                # Still include as hard negative even if we couldn't mutate;
                # the model should still score a full toxin highly
            else:
                applied.append(f"{wt}->{mut}")
                mut_seq = new_seq
        label_str = "+".join(applied) if applied else "WT-fallback"
        frags = fragment_protein(mut_seq, 50, 100, 25)
        if len(frags) < 3:
            print(f"[HardNeg] {acc} too short to fragment, skipping"); continue
        selected = random.sample(frags, min(random.randint(3, 8), len(frags)))
        gid = hashlib.md5(f"{acc}_mut".encode()).hexdigest()[:8]
        result.append(FragmentSet(selected, 0, gid, f"hard_neg_{acc}_{label_str}"))
        print(f"[HardNeg] {acc} ({label_str}): {len(selected)} frags")
    print(f"[HardNeg] total={len(result)}")
    return result

# ---------------------------------------------------------------------------
# 7. Encoders
# ---------------------------------------------------------------------------

class ESM3FragmentEncoder:
    """
    ESM-3 small open (1536-dim). Best quality encoder.
    pip install esm
    Accept license: huggingface.co/EvolutionaryScale/esm3-sm-open-v1

    FIX: ESM-3 loads weights in BFloat16 by default. Some internal tensors
    (e.g. average_plddt fed into plddt_projection) are created as Float32,
    which causes a dtype mismatch error:
      RuntimeError: mat1 and mat2 must have the same dtype, got Float and BFloat16
    Solution: cast the entire model to float32 after loading. This is safe for
    frozen inference and works on any device (CPU, CUDA, MPS).
    """
    def __init__(self, model_name: str = "esm3_sm_open_v1"):
        from esm.models.esm3 import ESM3
        print(f"[ESM-3] loading {model_name}...")
        self.model = ESM3.from_pretrained(model_name).to(DEVICE)
        # ── DTYPE FIX ──────────────────────────────────────────────────────
        # ESM-3 is loaded in BFloat16 by default but some intermediate tensors
        # (e.g. rbf_16_fn(average_plddt) inside forward()) stay as Float32,
        # which crashes the plddt_projection linear layer.
        # Casting the whole model to float32 makes every weight & buffer
        # consistent and avoids the RuntimeError.
        self.model = self.model.float()
        # ───────────────────────────────────────────────────────────────────
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.embed_dim = 1536
        print(f"[ESM-3] loaded. embed_dim={self.embed_dim}. frozen (float32).")

    @torch.no_grad()
    def encode(self, sequences: List[str], batch_size: int = 4) -> torch.Tensor:
        from esm.sdk.api import ESMProtein
        embs = []
        for i in range(0, len(sequences), batch_size):
            for seq in sequences[i:i+batch_size]:
                seq = seq[:1022]  # ESM-3 max length
                p = ESMProtein(sequence=seq)
                pt = self.model.encode(p)
                # unsqueeze adds the batch dimension expected by forward()
                out = self.model(sequence_tokens=pt.sequence.unsqueeze(0))
                # out.embeddings: (B, L, D) — strip BOS/EOS tokens then mean-pool
                emb = out.embeddings[0]      # (L, D)
                # Guard against very short sequences where [1:-1] might be empty
                inner = emb[1:-1] if emb.size(0) > 2 else emb
                embs.append(inner.mean(dim=0).cpu().float())
        return torch.stack(embs, dim=0)


class ESM2FragmentEncoder:
    """ESM-2 8M (320-dim). Matches 650M on short sequences per AMPForge."""
    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D"):
        from transformers import EsmModel, EsmTokenizer
        print(f"[ESM-2] loading {model_name}...")
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name).eval().to(DEVICE)
        for p in self.model.parameters(): p.requires_grad = False
        self.embed_dim = 320
        print(f"[ESM-2] loaded. embed_dim={self.embed_dim}. frozen.")

    @torch.no_grad()
    def encode(self, sequences: List[str], batch_size: int = 32) -> torch.Tensor:
        embs = []
        for i in range(0, len(sequences), batch_size):
            batch = [s[:512] for s in sequences[i:i+batch_size]]
            inp = self.tokenizer(batch, return_tensors="pt", padding=True,
                                 truncation=True, max_length=512)
            inp = {k: v.to(DEVICE) for k,v in inp.items()}
            out = self.model(**inp)
            mask = inp["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            embs.append(pooled.cpu())
        return torch.cat(embs, dim=0)


AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa:i for i,aa in enumerate(AA_VOCAB)}

class LightweightEncoder:
    """AA embedding + 1D conv + mean pool. CPU fallback, no HF token needed."""
    def __init__(self, embed_dim: int = 64):
        self.embed_dim = embed_dim
        self.aa_embed = nn.Embedding(len(AA_VOCAB)+1, embed_dim, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim*2, 5, padding=2), nn.ReLU(),
            nn.Conv1d(embed_dim*2, embed_dim, 3, padding=1), nn.ReLU())
        nn.init.xavier_uniform_(self.aa_embed.weight[1:])

    def encode(self, sequences: List[str]) -> torch.Tensor:
        embs = []
        with torch.no_grad():
            for seq in sequences:
                idx = torch.tensor([AA_TO_IDX.get(aa,0)+1 for aa in seq.upper()
                                    if aa in AA_TO_IDX], dtype=torch.long)
                if idx.numel() == 0: idx = torch.zeros(1, dtype=torch.long)
                e = self.aa_embed(idx).unsqueeze(0).transpose(1,2)
                embs.append(self.conv(e).mean(dim=2).squeeze(0))
        return torch.stack(embs, dim=0)

    def parameters(self):
        return list(self.aa_embed.parameters()) + list(self.conv.parameters())


def make_encoder(use_esm3=False, use_esm2=False):
    if use_esm3:  return ESM3FragmentEncoder(),  1536
    if use_esm2:  return ESM2FragmentEncoder(),  320
    return LightweightEncoder(embed_dim=64), 64

# ---------------------------------------------------------------------------
# 8. Set Transformer
# ---------------------------------------------------------------------------

class MAB(nn.Module):
    def __init__(self, d, h, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d,d*4), nn.GELU(), nn.Dropout(drop),
                                nn.Linear(d*4,d), nn.Dropout(drop))
    def forward(self, x):
        o,_ = self.attn(x,x,x); x = self.n1(x+o); return self.n2(x+self.ff(x))


class ISAB(nn.Module):
    """Induced Set Attention Block — O(mN) complexity, permutation-invariant."""
    def __init__(self, d, h, m=8, drop=0.1):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1,m,d))
        self.ah = nn.MultiheadAttention(d,h,dropout=drop,batch_first=True)
        self.nh1 = nn.LayerNorm(d); self.nh2 = nn.LayerNorm(d)
        self.fh = nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Linear(d*2,d))
        self.az = nn.MultiheadAttention(d,h,dropout=drop,batch_first=True)
        self.nz1 = nn.LayerNorm(d); self.nz2 = nn.LayerNorm(d)
        self.fz = nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Linear(d*2,d))
    def forward(self, x):
        I = self.I.expand(x.size(0),-1,-1)
        h,_ = self.ah(I,x,x); h = self.nh1(I+h); h = self.nh2(h+self.fh(h))
        z,_ = self.az(x,h,h); z = self.nz1(x+z); return self.nz2(z+self.fz(z))


class PMA(nn.Module):
    def __init__(self, d, h, k=1, drop=0.1):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1,k,d))
        self.attn = nn.MultiheadAttention(d,h,dropout=drop,batch_first=True)
        self.norm  = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Linear(d*2,d))
        self.weights: Optional[torch.Tensor] = None
    def forward(self, z):
        S = self.S.expand(z.size(0),-1,-1)
        o, self.weights = self.attn(S,z,z)
        o = self.norm(S+o); return self.norm2(o+self.ff(o))


class SetTransformerClassifier(nn.Module):
    def __init__(self, input_dim=320, d=128, h=4, m=8, n_isab=2, drop=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim,d), nn.LayerNorm(d),
                                  nn.GELU(), nn.Dropout(drop))
        self.isabs = nn.ModuleList([ISAB(d,h,m,drop) for _ in range(n_isab)])
        self.pma = PMA(d,h,k=1,drop=drop)
        self.sab = MAB(d,h,drop)
        self.head = nn.Sequential(nn.Linear(d,d//2), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(d//2,1))

    def forward(self, x):
        z = self.proj(x)
        for isab in self.isabs: z = isab(z)
        p = self.pma(z); p = self.sab(p)
        logits = self.head(p.squeeze(1)).squeeze(-1)
        return logits, torch.sigmoid(logits)

    def fragment_importance(self):
        if self.pma.weights is None: return None
        return self.pma.weights.squeeze(1)

    def explain(self, x):
        self.eval()
        with torch.no_grad():
            logits, probs = self(x)
            imp = self.fragment_importance()
        return {"probability": probs.cpu().tolist(),
                "logits": logits.cpu().tolist(),
                "fragment_importance": imp.cpu().tolist() if imp is not None else None}

# ---------------------------------------------------------------------------
# 9. Dataset & DataLoader
# ---------------------------------------------------------------------------

class FragmentSetDataset(Dataset):
    def __init__(self, fragment_sets: List[FragmentSet], encoder, max_frags: int = 15,
                 embedding_cache: Optional[Dict[str, torch.Tensor]] = None):
        self.data = fragment_sets
        all_frags, self.offsets = [], []
        for fs in fragment_sets:
            frags = fs.fragments[:max_frags]
            self.offsets.append((len(all_frags), len(all_frags)+len(frags)))
            all_frags.extend(frags)
        if embedding_cache is not None:
            # FIX #1: use pre-computed embeddings — zero re-encoding cost
            all_embs = torch.stack([embedding_cache[f] for f in all_frags])
        else:
            print(f"[Dataset] encoding {len(fragment_sets)} sets ({len(all_frags)} frags)...")
            all_embs = encoder.encode(all_frags) if all_frags else torch.zeros(0, encoder.embed_dim)
        self.embeddings = [all_embs[s:e] for s,e in self.offsets]

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        return self.embeddings[i], torch.tensor(self.data[i].label, dtype=torch.float32)

    def groups(self): return [fs.group_id for fs in self.data]


def collate_fn(batch):
    embs, labels = zip(*batch)
    B, max_n, d = len(embs), max(e.size(0) for e in embs), embs[0].size(1)
    padded = torch.zeros(B, max_n, d)
    mask   = torch.zeros(B, max_n, dtype=torch.bool)
    for i,e in enumerate(embs):
        padded[i,:e.size(0)] = e; mask[i,:e.size(0)] = True
    return padded, torch.stack(labels), mask


def precompute_embeddings(dataset: List[FragmentSet], encoder,
                           max_frags: int = 15) -> Dict[str, torch.Tensor]:
    """
    FIX #1: Encode every unique fragment exactly once across the entire dataset.
    Returns {sequence_str: embedding_tensor} cache used by FragmentSetDataset.
    In CV mode this reduces encoding from 3×n_folds passes to 1 pass total.
    """
    all_seqs = list({frag for fs in dataset for frag in fs.fragments[:max_frags]})
    print(f"[Precompute] encoding {len(all_seqs)} unique fragments once...")
    all_embs = encoder.encode(all_seqs)
    cache = {seq: emb for seq, emb in zip(all_seqs, all_embs)}
    print(f"[Precompute] done. Cache size: {len(cache)} fragments.")
    return cache


def make_loader(data, encoder, batch_size, shuffle,
                embedding_cache: Optional[Dict[str, torch.Tensor]] = None):
    ds = FragmentSetDataset(data, encoder, embedding_cache=embedding_cache)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

# ---------------------------------------------------------------------------
# 10. Training & evaluation
# ---------------------------------------------------------------------------

def _pos_weight(data: List[FragmentSet]) -> torch.Tensor:
    n_pos = sum(fs.label for fs in data)
    return torch.tensor([max(len(data)-n_pos, 1) / max(n_pos, 1)], dtype=torch.float32)


def train_epoch(model, loader, opt, pw):
    model.train()
    crit = nn.BCEWithLogitsLoss(pos_weight=pw.to(DEVICE))
    total, probs_all, lbl_all = 0.0, [], []
    for emb, lbl, _ in loader:
        emb, lbl = emb.to(DEVICE), lbl.to(DEVICE)
        opt.zero_grad()
        logits, probs = model(emb)
        loss = crit(logits, lbl); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        total += loss.item()
        probs_all.extend(probs.detach().cpu().tolist())
        lbl_all.extend(lbl.cpu().tolist())
    auc = roc_auc_score(lbl_all, probs_all) if len(set(lbl_all))>1 else 0.5
    return {"loss": total/len(loader), "auc": auc}


@torch.no_grad()
def evaluate(model, loader) -> dict:
    model.eval()
    probs_all, lbl_all = [], []
    for emb, lbl, _ in loader:
        _, probs = model(emb.to(DEVICE))
        probs_all.extend(probs.cpu().tolist()); lbl_all.extend(lbl.tolist())
    probs_all = np.array(probs_all); lbl_all = np.array(lbl_all)
    if len(set(lbl_all)) < 2:
        return {"auc":0.5,"ap":0.5,"probs":probs_all,"labels":lbl_all}
    auc = roc_auc_score(lbl_all, probs_all)
    ap  = average_precision_score(lbl_all, probs_all)
    fpr, tpr, _ = roc_curve(lbl_all, probs_all)
    m = {"auc":auc,"ap":ap,"probs":probs_all,"labels":lbl_all}
    for t in [0.05, 0.01, 0.001]:
        idx = np.searchsorted(fpr, t)
        if idx < len(tpr): m[f"tpr@fpr={t}"] = float(tpr[idx])
    return m


def _train_fold(train_data, val_data, test_data, encoder, embed_dim,
                n_epochs=40, batch_size=16, lr=3e-4, patience=10, seed=42,
                embedding_cache: Optional[Dict[str, torch.Tensor]] = None):
    torch.manual_seed(seed)
    tr_ld = make_loader(train_data, encoder, batch_size, shuffle=True,  embedding_cache=embedding_cache)
    va_ld = make_loader(val_data,   encoder, batch_size, shuffle=False, embedding_cache=embedding_cache)
    te_ld = make_loader(test_data,  encoder, batch_size, shuffle=False, embedding_cache=embedding_cache)
    pw = _pos_weight(train_data)
    model = SetTransformerClassifier(input_dim=embed_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)
    best_auc, best_state, no_improve = 0.0, None, 0
    hist = {"train_loss":[],"train_auc":[],"val_auc":[],"val_ap":[]}
    for epoch in range(1, n_epochs+1):
        tr = train_epoch(model, tr_ld, opt, pw)
        va = evaluate(model, va_ld); sched.step()
        hist["train_loss"].append(tr["loss"]); hist["train_auc"].append(tr["auc"])
        hist["val_auc"].append(va["auc"]); hist["val_ap"].append(va["ap"])
        if va["auc"] > best_auc:
            best_auc = va["auc"]
            best_state = {k: v.clone() if hasattr(v,"clone") else v
                          for k,v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience: break
    if best_state: model.load_state_dict(best_state)
    tm = evaluate(model, te_ld)
    return model, {"test_metrics":tm,"history":hist,"best_val_auc":best_auc}

# ---------------------------------------------------------------------------
# 11. Adversarial evaluation (FIX #2)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_adversarial(model, encoder, hard_neg_sets: List[FragmentSet],
                          threshold: float = 0.5) -> dict:
    """
    Evaluate on toxin homologs with neutralizing point mutations.
    High FPR = model resists single-mutation evasion (good).
    Low FPR  = adversary can defeat the screen with one substitution (bad).
    """
    model.eval()
    results = []
    for fs in hard_neg_sets:
        embs = encoder.encode(fs.fragments)
        _, prob = model(embs.unsqueeze(0).to(DEVICE))
        p = float(prob.item())
        results.append({"source":fs.source,"probability":round(p,4),"flagged":p>=threshold})
    n = len(results)
    n_flagged = sum(r["flagged"] for r in results)
    fpr = n_flagged/n if n else 0.0
    print(f"\n[Adversarial] n={n} flagged={n_flagged} FPR-on-mutants={fpr:.1%}")
    print(f"  High FPR (>50%) = resists single-mutation evasion")
    print(f"  Low  FPR (<20%) = single substitution defeats screen")
    for r in results:
        tag = "FLAGGED" if r["flagged"] else "passed"
        print(f"  {tag}  {r['source'][:55]:<55}  P={r['probability']:.4f}")
    return {"n":n,"n_flagged":n_flagged,"fpr_on_mutants":round(fpr,4),"per_sample":results}

# ---------------------------------------------------------------------------
# 12. Calibration evaluation (FIX #4)
# ---------------------------------------------------------------------------

def evaluate_calibration(model, loader, n_bins: int = 10) -> dict:
    """Reliability diagram + Platt scaling. P(threat)=0.7 should mean 70% truly threats."""
    raw = evaluate(model, loader)
    probs, labels = raw["probs"], raw["labels"]
    brier = float(brier_score_loss(labels, probs))
    bins = np.linspace(0,1,n_bins+1)
    reliability = []
    for lo,hi in zip(bins[:-1],bins[1:]):
        mask = (probs>=lo)&(probs<hi)
        if mask.sum()==0: continue
        reliability.append((float(probs[mask].mean()), float(labels[mask].mean()), int(mask.sum())))
    mce = float(np.mean([abs(mp-fp) for mp,fp,_ in reliability])) if reliability else 1.0
    result = {"mean_calibration_error":round(mce,4),"brier_score":round(brier,4),
              "reliability":reliability}
    try:
        logits = np.log(np.clip(probs,1e-7,1-1e-7)/np.clip(1-probs,1e-7,1-1e-7))
        platt = LogisticRegression(C=1.0,solver="lbfgs",max_iter=1000)
        platt.fit(logits.reshape(-1,1), labels)
        pc = platt.predict_proba(logits.reshape(-1,1))[:,1]
        bc = float(brier_score_loss(labels, pc))
        result["platt_coef"] = [float(platt.coef_[0][0]), float(platt.intercept_[0])]
        result["brier_platt"] = round(bc,4)
        print(f"[Calibration] MCE={mce:.4f} Brier={brier:.4f} -> Brier(Platt)={bc:.4f}")
    except Exception as e:
        print(f"[Calibration] Platt failed: {e}")
    return result

# ---------------------------------------------------------------------------
# 13. Fragment-count robustness (FIX #5)
# ---------------------------------------------------------------------------

def fragment_count_robustness(model, encoder,
                               test_proteins: Dict[str,Tuple[str,int]],
                               fragment_counts: Optional[List[int]] = None,
                               n_trials: int = 5, seed: int = 42) -> dict:
    """AUC at each fragment count k. Detects if splitting into 2-3 frags defeats the screen."""
    if fragment_counts is None:
        fragment_counts = [2, 3, 5, 8, 10, 15]
    model.eval(); random.seed(seed); rows = []
    for k in fragment_counts:
        by_label: Dict[int,List[float]] = {0:[],1:[]}
        for acc,(seq,label) in test_proteins.items():
            frags = fragment_protein(seq,50,100,25)
            if len(frags)<2: continue
            ps = []
            for t in range(n_trials):
                random.seed(seed+t*37+abs(hash(acc))%37)
                sel = random.sample(frags, min(k,len(frags)))
                embs = encoder.encode(sel)
                with torch.no_grad():
                    _, p = model(embs.unsqueeze(0).to(DEVICE))
                ps.append(float(p.item()))
            by_label[label].append(float(np.mean(ps)))
        all_p = by_label[0]+by_label[1]
        all_l = [0]*len(by_label[0])+[1]*len(by_label[1])
        auc = (roc_auc_score(all_l,all_p) if len(set(all_l))>1 and len(all_l)>=4
               else float("nan"))
        mt = float(np.mean(by_label[1])) if by_label[1] else float("nan")
        mb = float(np.mean(by_label[0])) if by_label[0] else float("nan")
        rows.append({"n_frags":k,"auc":round(auc,4),"mean_p_threat":round(mt,4),
                     "mean_p_benign":round(mb,4)})
        print(f"[Robustness] k={k:2d}  AUC={auc:.4f}  P(T|T)={mt:.4f}  P(T|B)={mb:.4f}")
    return {"by_fragment_count":rows}

# ---------------------------------------------------------------------------
# 14. Cross-validation pipeline
# ---------------------------------------------------------------------------

def cross_validate(dataset, encoder, embed_dim, n_folds=5, n_epochs=40,
                   batch_size=16, lr=3e-4, seed=42) -> dict:
    # FIX #1: encode entire dataset once — shared across all folds and splits
    emb_cache = precompute_embeddings(dataset, encoder)

    groups = np.array([fs.group_id for fs in dataset])
    labels = np.array([fs.label for fs in dataset])
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics, final_model = [], None
    for fold,(tr_idx,te_idx) in enumerate(skf.split(dataset,labels,groups)):
        print(f"\n[CV] fold {fold+1}/{n_folds}")
        tr_data = [dataset[i] for i in tr_idx]
        te_data = [dataset[i] for i in te_idx]
        tr_grp  = [dataset[i].group_id for i in tr_idx]
        tr_lbl  = [dataset[i].label for i in tr_idx]
        spl = GroupShuffleSplit(n_splits=1,test_size=0.15,random_state=seed+fold)
        ti2,vi2 = next(spl.split(tr_data,tr_lbl,tr_grp))
        va_data = [tr_data[i] for i in vi2]
        tr_data = [tr_data[i] for i in ti2]
        model, info = _train_fold(tr_data,va_data,te_data,encoder,embed_dim,
                                  n_epochs=n_epochs,batch_size=batch_size,
                                  lr=lr,seed=seed+fold,embedding_cache=emb_cache)
        tm = info["test_metrics"]
        fold_metrics.append({k:v for k,v in tm.items() if k not in ("probs","labels")})
        print(f"  AUC={tm['auc']:.4f}  AP={tm['ap']:.4f}")
        final_model = model
        last_te_data = te_data   # save last fold's test set for calibration (no leakage)
    num_keys = [k for k in fold_metrics[0] if isinstance(fold_metrics[0][k],float)]
    summary = {k:{"mean":round(float(np.mean([fm[k] for fm in fold_metrics])),4),
                  "std": round(float(np.std( [fm[k] for fm in fold_metrics])),4),
                  "per_fold":[round(fm[k],4) for fm in fold_metrics]}
               for k in num_keys}
    print("\n[CV] Summary:")
    for k,v in summary.items():
        print(f"  {k}: {v['mean']:.4f} +/- {v['std']:.4f}  {v['per_fold']}")
    return {"fold_metrics":fold_metrics,"summary":summary,
            "final_model":final_model,"emb_cache":emb_cache,
            "last_fold_test_data":last_te_data}

# ---------------------------------------------------------------------------
# 15. Single-split training pipeline
# ---------------------------------------------------------------------------

def train_model(dataset, encoder, embed_dim, n_epochs=40, batch_size=16,
                lr=3e-4, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    groups = [fs.group_id for fs in dataset]
    labels = [fs.label for fs in dataset]
    spl = GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=seed)
    tr_idx,te_idx = next(spl.split(dataset,labels,groups))
    tr_data = [dataset[i] for i in tr_idx]
    te_data = [dataset[i] for i in te_idx]
    tr_grp  = [groups[i] for i in tr_idx]
    tr_lbl  = [labels[i] for i in tr_idx]
    spl2 = GroupShuffleSplit(n_splits=1,test_size=0.15,random_state=seed)
    ti2,vi2 = next(spl2.split(tr_data,tr_lbl,tr_grp))
    va_data = [tr_data[i] for i in vi2]
    tr_data = [tr_data[i] for i in ti2]
    print(f"[Split] train={len(tr_data)} val={len(va_data)} test={len(te_data)}")
    model, info = _train_fold(tr_data,va_data,te_data,encoder,embed_dim,
                              n_epochs=n_epochs,batch_size=batch_size,lr=lr,seed=seed)
    tm = info["test_metrics"]
    print(f"[Test] AUC={tm['auc']:.4f}  AP={tm['ap']:.4f}")
    for k,v in tm.items():
        if k.startswith("tpr@"): print(f"[Test] {k}={v:.4f}")
    return model, {"history":info["history"],
                   "test_metrics":{k:v for k,v in tm.items() if k not in ("probs","labels")},
                   "best_val_auc":info["best_val_auc"],
                   "embed_dim":embed_dim, "encoder":encoder,
                   "test_data":te_data, "dataset":dataset}

# ---------------------------------------------------------------------------
# 16. Production inference
# ---------------------------------------------------------------------------

class BioChainMLEngine:
    def __init__(self, model: SetTransformerClassifier, encoder):
        self.model = model.eval(); self.encoder = encoder

    def score_assembly(self, fragments: List[str]) -> dict:
        if not fragments: return {"error":"no fragments"}
        embs = self.encoder.encode(fragments)
        x = embs.unsqueeze(0).to(DEVICE)
        out = self.model.explain(x)
        prob = out["probability"][0]
        imp  = out["fragment_importance"]
        top  = (sorted(range(len(imp[0])),key=lambda i:imp[0][i],reverse=True)[:3]
                if imp and imp[0] else list(range(min(3,len(fragments)))))
        return {"probability_score": round(float(prob),4),
                "tier":              self._tier(prob),
                "top_fragment_indices": top,
                "fragment_importance":  imp[0] if imp else None,
                "uncertainty":       round(self._mc_uncertainty(embs),4),
                "raw_logit":         round(float(out["logits"][0]),4),
                "n_fragments":       len(fragments)}

    @staticmethod
    def _tier(p):
        if p < 0.3:   return "SILENT_LOG"
        if p < 0.6:   return "RETROSPECTIVE_FLAG"
        if p < 0.8:   return "SOFT_ALERT"
        if p < 0.9:   return "HOLD_48H"
        return "REGULATORY"

    def _mc_uncertainty(self, embs, n=10):
        self.model.train()
        ps = []
        with torch.no_grad():
            x = embs.unsqueeze(0).to(DEVICE)
            for _ in range(n):
                _, p = self.model(x); ps.append(p.item())
        self.model.eval()
        return float(np.std(ps))

# ---------------------------------------------------------------------------
# 17. __main__
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioChain ML Engine v3")
    parser.add_argument("--esm3",       action="store_true", help="Use ESM-3 small open")
    parser.add_argument("--esm2",       action="store_true", help="Use ESM-2 8M")
    parser.add_argument("--cv",         action="store_true", help="5-fold cross-validation")
    parser.add_argument("--epochs",     type=int, default=40)
    parser.add_argument("--batch",      type=int, default=16)
    parser.add_argument("--cache",      type=str, default="uniprot_cache.json")
    parser.add_argument("--robustness", action="store_true", help="Fragment-count sweep")
    args = parser.parse_args()

    use_esm3 = args.esm3
    use_esm2 = args.esm2 and not use_esm3
    enc_name = "ESM-3" if use_esm3 else ("ESM-2" if use_esm2 else "Lightweight")

    print("=" * 60)
    print("BioChain ML Engine v3")
    print("=" * 60)
    print(f"  encoder={enc_name}  cv={args.cv}  epochs={args.epochs}")

    dataset = build_dataset_from_uniprot(random_seed=42, cache_path=args.cache)
    encoder, embed_dim = make_encoder(use_esm3, use_esm2)

    if args.cv:
        cv = cross_validate(dataset, encoder, embed_dim,
                            n_epochs=args.epochs, batch_size=args.batch)
        model = cv["final_model"]
        emb_cache = cv["emb_cache"]
        # Use last fold's test data for calibration — this data was genuinely held out
        # from the last fold's training, so there is no leakage into the final model.
        # (Previous approach re-split the full dataset which allowed training proteins
        # to appear in the calibration test set.)
        test_data_for_cal = cv["last_fold_test_data"]
        run_info = {"embed_dim":embed_dim, "encoder":encoder, "dataset":dataset,
                    "test_data":test_data_for_cal, "emb_cache":emb_cache}
    else:
        model, run_info = train_model(dataset, encoder, embed_dim,
                                      n_epochs=args.epochs, batch_size=args.batch)
        encoder = run_info["encoder"]
        emb_cache = None

    # Adversarial eval — always runs (fast, critical)
    hard_neg = build_hard_negative_test_set(cache_path=args.cache)
    adv_results = evaluate_adversarial(model, encoder, hard_neg, threshold=0.5)

    # Calibration eval — always runs
    cal_results = {}
    if "test_data" in run_info:
        te_ld = make_loader(run_info["test_data"], encoder, args.batch,
                            shuffle=False, embedding_cache=run_info.get("emb_cache"))
        cal_results = evaluate_calibration(model, te_ld)

    # Fragment-count robustness — opt-in
    # NOTE: intentionally uses known training proteins here. Robustness sweep measures
    # whether AUC degrades as fragment count k decreases (2→15), not generalization
    # to unseen proteins. Using training proteins is valid for this specific question.
    rob_results = {}
    if args.robustness:
        t_seqs = download_uniprot_sequences(NON_METAZOAN_TOXIN_IDS[:5], args.cache)
        b_seqs = download_uniprot_sequences(BENIGN_IDS_HUMAN[:5], args.cache)
        tp = {a:(s,1) for a,s in t_seqs.items()}
        tp.update({a:(s,0) for a,s in b_seqs.items()})
        rob_results = fragment_count_robustness(model, encoder, tp)

    # ── Save checkpoint ────────────────────────────────────────────────────────
    cv_summary = cv["summary"] if args.cv else {}
    fold_metrics = cv["fold_metrics"] if args.cv else []
    history = (fold_metrics[0].get("history", {}) if fold_metrics
               else run_info.get("history", {}))

    torch.save({
        "model_state":  model.state_dict(),
        "embed_dim":    embed_dim,
        "encoder_name": enc_name,
        "architecture": {"input_dim":embed_dim,"d":128,"h":4,"m":8,"n_isab":2},
        "run_info": {
            "cv_summary":   cv_summary,
            "fold_metrics": [{k:v for k,v in fm.items()
                              if k not in ("probs","labels")} for fm in fold_metrics],
            "adversarial":  adv_results,
            "calibration":  {k:v for k,v in cal_results.items() if k != "reliability"},
            "robustness":   rob_results,
            "history":      history,
        }
    }, "biochain_model_v3.pt")
    print("\n[Saved] biochain_model_v3.pt")

    # ── Save results JSON (for biochain_visualize.py and reporting) ────────────
    results_json = {
        "encoder":      enc_name,
        "embed_dim":    embed_dim,
        "cv_summary":   cv_summary,
        "fold_metrics": [{k:v for k,v in fm.items()
                          if k not in ("probs","labels")} for fm in fold_metrics],
        "adversarial":  adv_results,
        "calibration":  {k:v for k,v in cal_results.items() if k != "reliability"},
        "robustness":   rob_results,
        "history":      history,
    }
    with open("biochain_results.json", "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    print("[Saved] biochain_results.json")
    print("[Done]")
