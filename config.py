"""
config.py — Central configuration for the AIxBio ESM3 SAE project.

Set the AIXBIO_ROOT environment variable to override the default project root:
    export AIXBIO_ROOT=/path/to/your/project
"""

import os

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.environ.get(
    'AIXBIO_ROOT',
    os.path.join(os.path.expanduser('~'), 'AIxBio_Hackathon_ESM3')
)

# ── Directory layout ──────────────────────────────────────────────────────────
DATA_DIR      = os.path.join(PROJECT_ROOT, 'data')
SAE_DIR       = os.path.join(PROJECT_ROOT, 'sae_features')
EMB_DIR       = os.path.join(PROJECT_ROOT, 'embeddings')
SPLITS_DIR    = os.path.join(DATA_DIR, 'splits')
FEAT_RANK_DIR = os.path.join(PROJECT_ROOT, 'analysis', 'feature_ranking')
ANALYSIS_DIR  = os.path.join(PROJECT_ROOT, 'analysis')
INTERV_DIR    = os.path.join(PROJECT_ROOT, 'analysis', 'intervention')
NEG_CTRL_DIR  = os.path.join(ANALYSIS_DIR, 'negative_controls')
BASELINES_DIR = os.path.join(PROJECT_ROOT, 'baselines')
SAE_WEIGHTS   = os.path.join(PROJECT_ROOT, 'sae_weights')
FIG_DIR       = os.path.join(ANALYSIS_DIR, 'figures')
EXT_DIR       = os.path.join(PROJECT_ROOT, 'external_validation')

# ── ESM3-small-open architecture ──────────────────────────────────────────────
D_MODEL    = 1536    # ESM3 hidden dimension
D_SAE      = 15360   # SAE width (10× expansion)
K_TOPK     = 120     # TopK sparse activation
ESM3_LAYER = 36      # Layer index (75% depth: 36/48)

# ── Convergent pore-forming toxins ────────────────────────────────────────────
CONVERGENT_IDS = ['P09616', 'P61914', 'P77335']

FUNCTIONAL_SITES = {
    'P09616': list(range(14, 34)) + list(range(106, 128)),  # Alpha-hemolysin
    'P61914': list(range(18, 36)) + [111, 114, 134],        # FraC
    'P77335': list(range(0, 15))  + list(range(55, 80)),    # ClyA
}

PROTEIN_NAMES = {
    'P09616': 'Alpha-hemolysin',
    'P61914': 'Fragaceatoxin C (FraC)',
    'P77335': 'Cytolysin A (ClyA)',
}

# ── Training hyperparameters (N3) ─────────────────────────────────────────────
SAE_N_EPOCHS  = 50
SAE_BATCH     = 2048
SAE_LR        = 3e-4
SAE_AUX_LAMBDA = 1e-3

# ── Feature discovery thresholds (N4) ────────────────────────────────────────
GLOBAL_FREQ_THRESHOLD = 0.80
FDR_ALPHA             = 0.05
AUROC_MIN             = 0.60
COHENS_D_MIN          = 0.30

# ── External UniProt API ──────────────────────────────────────────────────────
UNIPROT_BASE      = 'https://rest.uniprot.org/uniprotkb'
RATE_LIMIT_PAUSE  = 0.5   # seconds between requests


def make_dirs():
    """Create all project directories."""
    dirs = [
        DATA_DIR, SAE_DIR, EMB_DIR, SPLITS_DIR,
        FEAT_RANK_DIR, ANALYSIS_DIR, INTERV_DIR, NEG_CTRL_DIR,
        BASELINES_DIR, SAE_WEIGHTS, FIG_DIR, EXT_DIR,
        os.path.join(DATA_DIR, 'splits'),
        os.path.join(DATA_DIR, 'convergent_toxins'),
        os.path.join(DATA_DIR, 'hard_negatives'),
        os.path.join(DATA_DIR, 'general_negatives'),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
