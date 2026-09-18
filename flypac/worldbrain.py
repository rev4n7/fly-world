"""Fly World brain: the Fly-Pacman LIF network on data/world/, with more senses and live ablation.

Same neuron model and weights as flypac.brain (Shiu et al. 2024 constants, weight = synapse
count x 0.275 mV x NT sign). Added:
  * senses:   smell (food-odour ORNs), mouth taste (taste -> MN9 route), leg touch of a female
              (leg taste cells -> vAB3), female vision (LC10a), predators (looming + hearing as before)
  * readouts: pIP10 (courtship song), MN9 (proboscis extension = eating), pC1/P1 (arousal)
  * ablation: a removed neuron loses every incoming and outgoing synapse and gets no sensory drive,
              i.e. it is taken out of the network, not just hidden
Anything that is not connectome data is marked DESIGN.
"""
import numpy as np
from scipy import sparse

from . import brain as B
from . import config as C

WORLD_DIR = C.DATA_DIR / "world"

# Tested and left out (scoping/flyworld/FINDINGS.md): the 460 cells on food-odour -> steering routes.
# With them, odour on EITHER antenna alone turns the fly left, and 160+ cells fire > 50 Hz even at a
# 2 Hz receptor rate. The wiring carries no usable odour direction to DNa02, so smell is shown
# (receptors + projection neurons) but does not steer; the fly finds food by wandering.
# The PNs' few remaining outputs (484 synapses) land on LEFT-side cells only (DNb05_L, CB0356_L),
# which made the fly circle left whenever any food existed; those onward edges are cut too, so
# smell is display-only: ORN -> PN, nothing further.
DROP_ROLES = ["middle_smell"]
SMELL_ROLES = ["sense_smell", "smell_pn"]

# Taste cells that food activates. The dataset doesn't say which mouth taste cells are sugar cells.
# Stimulating each of the 31 taste types alone (scoping/flyworld/FINDINGS.md) shows that only five
# drive MN9 through their own wiring: labellar LB3b/c/d and pharyngeal PhG1b/c; the other 26 don't
# (bitter, water, or unknown). The labellum touches food first (the pharynx tastes only after
# swallowing), so food = the three labellar types. Stimulating all 31 at once leaves MN9 silent,
# and so does adding PhG1b/c to the LB3 types.
FOOD_TASTE_TYPES = ["LB3b", "LB3c", "LB3d"]

# ---- DESIGN: sensory encoding for the new senses (stimulus -> Poisson rate) ----
WSENSE = dict(
    smell_rate_max=2.0,    # Hz, food-odour ORNs at saturating odour. Low because each ORN has ~180 synapses
                           # onto its PN, so at 0.275 mV/synapse one ORN spike alone fires the PN; at 2 Hz
                           # PNs go ~37 Hz next to food, ~8 Hz at 12 units, ~1 Hz at 20 units
    smell_half=0.35,       # odour units giving tanh(1) = 76% of max
    smell_len=6.0,         # units; odour falls off as exp(-distance / smell_len) around each food
    antenna_sep=0.15,      # units from the midline to each antenna (fly body ~1 unit)
    taste_rate=30.0,       # Hz, FOOD_TASTE_TYPES while the mouth is on food
    touch_rate=60.0,       # Hz, leg taste cells (vAB3 inputs) while touching the female
    contact_dist=0.7,      # units; closer than this = touching
)


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


class WorldBrain(B.Brain):
    def __init__(self, seed=0, data_dir=None):
        super().__init__(seed=seed, data_dir=data_dir or self._prepared())
        n, roles = self.neurons, self.neurons.role.to_numpy()
        # DESIGN: receptor neurons (odour, taste, leg-taste) fire from the stimulus at their dendrites
        # in the periphery. Synapses onto their axon terminals in the brain (e.g. the dense ORN->ORN
        # contacts) modulate release, which this model can't express; as spike-driving inputs they make
        # the odour channel run away (ORNs at ~180 Hz for a 60 Hz stimulus). So their inputs are dropped.
        receptor = np.isin(roles, ["sense_smell", "sense_taste", "sense_touch"])
        self.W = sparse.csc_matrix(sparse.diags((~receptor).astype(float)) @ self.W)
        self.W_full = self.W.copy()
        self.alive = np.ones(self.N, bool)
        self.smell = np.flatnonzero(roles == "sense_smell")
        side = n.side.to_numpy()[self.smell]
        self.smell_side = np.where(side == "L", 0, np.where(side == "R", 1, 2))  # 2 = antenna unknown: average
        self.taste = np.flatnonzero((roles == "sense_taste") & n.type.isin(FOOD_TASTE_TYPES).to_numpy())
        self.touch = np.flatnonzero(roles == "sense_touch")
        types = n.type.to_numpy()
        self.song = np.flatnonzero(types == "pIP10")
        self.feed = np.flatnonzero(types == "MN9")
        self.p1 = np.flatnonzero(roles == "court_p1")
        self.gf = np.array([self.out["DNp01_L"], self.out["DNp01_R"]])

    @staticmethod
    def _prepared(src=WORLD_DIR):
        """src/ minus DROP_ROLES and the smell relays' onward edges, cached in src/sim/."""
        import pandas as pd
        out = src / "sim"
        src_time = max((src / f).stat().st_mtime for f in ["neurons.csv", "edges.csv"])
        if not (out / "edges.csv").exists() or (out / "edges.csv").stat().st_mtime < src_time:
            n = pd.read_csv(src / "neurons.csv")
            e = pd.read_csv(src / "edges.csv")
            n = n[~n.role.isin(DROP_ROLES)]
            e = e[e.pre.isin(n.bodyId) & e.post.isin(n.bodyId)]
            smell = set(n.bodyId[n.role.isin(SMELL_ROLES)])
            e = e[~(e.pre.isin(smell) & ~e.post.isin(smell))]
            out.mkdir(exist_ok=True)
            n.to_csv(out / "neurons.csv", index=False)
            e.to_csv(out / "edges.csv", index=False)
        return out

    # ------------------------------------------------------------------ ablation
    def set_network(self, removed=(), syn_gain=1.0, inh_gain=1.0):
        """Rebuild the live network: remove neurons (row indices) and scale synapses.

        removed:  cells taken out, with every incoming and outgoing synapse
        syn_gain: every synapse x this (diffuse damage, e.g. 0.6 = all connections 40% weaker)
        inh_gain: inhibitory (GABA / glutamate) synapses x this on top (weakened brakes)
        """
        self.alive[:] = True
        self.alive[np.asarray(list(removed), int)] = False
        W = self.W_full.copy()
        W.data = W.data * syn_gain * np.where(W.data < 0, inh_gain, 1.0)
        keep = sparse.diags(self.alive.astype(float))
        self.W = (keep @ W @ keep).tocsc()
        self.W.eliminate_zeros()
        dead = ~self.alive
        self.v[dead] = B.V_REST
        self.g[dead] = 0.0
        self.rates[dead] = 0.0
        self.buf = [s[self.alive[s]] for s in self.buf]   # spikes already in flight from removed cells vanish

    def set_ablated(self, idx):
        self.set_network(idx)

    # ------------------------------------------------------------------ senses
    def sense_world(self, predators=(), females=(), foods=(), heading=0.0):
        """predators: ghost-style dicts (see Brain.sense). females: dicts with bearing, dist, size, visible.
        foods: dicts with dist and bearing. Bearings are body-frame degrees, +right."""
        S, W = B.SENSE, WSENSE
        # predators reuse Fly-Pacman's looming + hearing encoding; females drive LC10a the way pellets did
        self.sense(ghosts=predators, pellets=[f for f in females if f["visible"]])
        r = self.input_rate
        # smell: odour at each antenna (left, right), both antennae on the fly's midline +- antenna_sep
        conc = np.zeros(2)
        for f in foods:
            for k, sgn in enumerate((-1, 1)):
                a = np.radians(f["bearing"])
                fx, fy = f["dist"] * np.cos(a), f["dist"] * np.sin(a) - sgn * W["antenna_sep"]
                conc[k] += np.exp(-np.hypot(fx, fy) / W["smell_len"])
        per_cell = np.append(conc, conc.mean())[self.smell_side]
        r[self.smell] = W["smell_rate_max"] * np.tanh(per_cell / W["smell_half"])
        on_food = any(f["dist"] < W["contact_dist"] for f in foods)
        r[self.taste] = W["taste_rate"] if on_food else 0.0
        touching = any(f["dist"] < W["contact_dist"] for f in females)
        r[self.touch] = W["touch_rate"] if touching else 0.0
        r[~self.alive] = 0.0
        self.odour = conc
        self.on_food, self.touching = on_food, touching

    # ------------------------------------------------------------------ readout
    def readout(self, counts):
        m = self.motor(counts)
        r = self.rates
        m.update(song=float(r[self.song].sum()), feed=float(r[self.feed].sum()),
                 arousal=float(r[self.p1].mean()), gf_rate=float(r[self.gf].sum()))
        return m
