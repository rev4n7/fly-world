"""Which neurons we pull from male-cns:v1.0, and why.

Core types are named explicitly (justified in scoping/SCOPING.md). Middle-layer
interneurons are NOT hand-picked: they are selected by synapse-count thresholds
applied to the real connectome in fetch.py.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SERVER = "neuprint.janelia.org"
DATASET = "male-cns:v1.0"

# --- sensory populations (stimulus is injected here) ---
LOOMING_TYPES = ["LC4", "LPLC2", "LC6"]   # looming / approaching-object detectors
PURSUIT_TYPES = ["LC10a"]                  # small-object pursuit (courtship chase) detectors
HEARING_PREFIX = "JO"                      # Johnston's organ: antenna sound / air-vibration sensors

# --- descending / motor readouts ---
ESCAPE_TYPES = ["DNp01", "DNp02", "DNp04", "DNp06", "DNp11", "TTMn", "PSI"]  # DNp01 = Giant Fiber
STEER_TYPES = ["DNa02", "DNa03", "DNa01", "DNp09"]

# --- middle-layer selection thresholds (synapse counts, per middle cell) ---
GF_MID_MIN_IN = 100    # >= this many synapses from looming/hearing sensors ...
GF_MID_MIN_OUT = 50    # ... and >= this many onto a Giant Fiber
STEER_MID_MIN_IN = 100   # >= from LC10a ...
STEER_MID_MIN_OUT = 100  # ... and >= onto DNa02/DNa03
# feedback layer: ANY neuron (any transmitter) receiving >= IN synapses from the circuit and
# sending >= OUT synapses back onto its middle/output neurons. Restores the local balance
# (mostly inhibition) that a bare feed-forward selection throws away.
FEEDBACK_MIN_IN = 100
FEEDBACK_MIN_OUT = 100
JO_MIN_OUT = 5         # JO cell kept if it sends >= this many synapses into the circuit

# --- Fly World additions (flypac/fetch_world.py; evidence in scoping/flyworld/FINDINGS.md) ---
# Courtship. Names from the dataset's `synonyms` annotation.
VAB3_PREFIX = "AN09B017"   # vAB3 ascending neurons (Yu 2010, von Philipsborn 2011): leg pheromone taste -> P1
P1_PREFIX = "pC1"          # pC1/P1 cluster: male-specific fru/dsx courtship-arousal neurons
SONG_TYPES = ["pIP10"]     # courtship-song command descending neuron (P2b, Kimura 2008)
TOUCH_MIN_OUT = 20         # leg taste neuron kept if it sends >= this many synapses to vAB3
COURT_MID_MIN_IN = 100     # middle cell: >= this from vAB3/LC10a (or from pC1, for the song layer) ...
COURT_MID_MIN_OUT = 100    # ... and >= this onto pC1 (or onto pIP10)
# Smell. Food / vinegar-attractive glomeruli: DM1 (Or42b) + VA2 (Or92a) (Semmelhack & Wang 2009),
# DP1m (Ir64a, acid attraction; Ai et al. 2010), DM4 (Or59b) + DM2 (Or22a) (fruit esters).
FOOD_GLOMERULI = ["DM1", "VA2", "DP1m", "DM4", "DM2"]
SMELL_PN_MIN_IN = 100      # projection neuron kept if it gets >= this from food-odour ORNs
SMELL_HOP_MIN = 10         # every hop of a PN -> A -> (B ->) DNa02/DNa03 route must have >= this many synapses
# Eating. Mouthpart taste neurons (labellum, taste pegs, pharynx) -> MN9 proboscis motor neuron.
FEED_TYPES = ["MN9"]
MOUTH_GRN_PREFIXES = ["LB", "claw_tpGRN", "dorsal_tpGRN", "PhG"]
TASTE_HOP_MIN = 10         # every hop of a taste neuron -> A -> (B ->) MN9 route must have >= this many synapses

# --- synapse sign from predicted neurotransmitter ---
# Same convention as the Shiu et al. 2024 whole-brain LIF model: ACh excitatory,
# GABA and glutamate inhibitory (fly CNS glutamate mostly acts via GluCl).
# Modulatory / unclear transmitters get sign 0 (not modelled as fast synapses).
NT_SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}

# --- eye geometry (approximate literature values, NOT connectome data) ---
# Per-eye azimuth coverage in degrees: 0 = straight ahead, positive = toward that
# eye's side, 180 = straight behind. Frontal edge crosses the midline (binocular
# overlap); the rear edge leaves a blind spot behind the fly.
AZ_FRONT, AZ_BACK = -15.0, 160.0
EL_BOTTOM, EL_TOP = -60.0, 60.0
