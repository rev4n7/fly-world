"""Neuron sim: leaky integrate-and-fire network built from the real male-cns:v1.0 weights.

Neuron model and constants follow the published whole-brain Drosophila LIF model
(Shiu et al. 2024, Nature 634:210): conductance-style synapses, weight per spike =
W_SYN * synapse_count * NT sign, Poisson-driven sensory neurons.

Pieces that are NOT connectome data are marked "DESIGN" and collected in SENSE/MOTOR.
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from . import config as C

# ---- LIF constants (Shiu et al. 2024) ----
V_REST = -52.0      # mV (also reset)
V_TH = -45.0        # mV
T_MBR = 20e-3       # s   membrane time constant
TAU_SYN = 5e-3      # s   synaptic decay
T_REF = 2.2e-3      # s   refractory
T_DLY = 1.8e-3      # s   synaptic delay
W_SYN = 0.275       # mV per synapse per spike
F_POI = 250         # Poisson input strength, in units of W_SYN
DT = 1e-4           # s

# ---- DESIGN: spike-frequency adaptation (not in Shiu et al.) ----
# Off by default: once the feedback layer (config.FEEDBACK_*) is included, the real
# inhibition ends reverberation by itself. Kept as a switch for experiments.
ADAPT_B = 0.0       # mV per spike
TAU_ADAPT = 0.3     # s

# ---- DESIGN: sensory encoding (stimulus -> Poisson rate of sensory neurons) ----
SENSE = dict(
    loom_rate_max=200.0,   # Hz, looming cells (LC4/LPLC2/LC6)
    loom_half=1500.0,      # deg/s of angular expansion giving ~76% of max rate
    pursuit_rate_max=120.0,  # Hz, LC10a
    pursuit_half=6.0,      # deg of small-object "salience"
    pursuit_pref_size=15.0,  # deg; LC10a prefer small objects, bigger ones fade out
    hear_rate_max=80.0,    # Hz, JO cells
    hear_half=1.0,         # air-vibration units above threshold
    hear_threshold=0.5,    # air-vibration units; below this the antennae report nothing
    hear_r0=1.0,           # tiles; near-field vibration ~ speed * (r0/d)^3 (steep, like real near-field sound)
    ear_directionality=0.3,  # 0 = no L/R difference, 0.5 = fully one-sided
)

# ---- DESIGN: motor decoding ----
MOTOR = dict(
    rate_tau=0.15,         # s, smoothing of output firing rates
    explore_rate=4.0,      # Hz background Poisson drive to DNa02 L/R (spontaneous wandering)
)


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def mirror_hearing(neurons, edges):
    """Right ear := mirror of the better-traced left ear (decision 2026-09-16, option B).

    Right-side JO cells and their edges are dropped. Every left JO cell gets a mirrored copy
    (bodyId -> -bodyId, side R). A connection to/from neuron X is copied to the same cell type
    on the opposite side, split evenly if that type has several cells there; midline or
    unpaired partners map to themselves. All weights are real left-ear synapse counts.
    """
    jo = neurons.role == "sense_hearing"
    drop = set(neurons.index[jo & (neurons.side != "L")])
    left_jo = neurons.index[jo & (neurons.side == "L")]
    neurons = neurons.drop(index=list(drop))
    edges = edges[~edges.pre.isin(drop) & ~edges.post.isin(drop)]

    flip = {"L": "R", "R": "L"}
    by_type_side = neurons.groupby(["type", "side"]).groups

    def partners(bid):
        if bid in left_jo:
            return [(-bid, 1.0)]
        t, s = neurons.at[bid, "type"], neurons.at[bid, "side"]
        other = list(by_type_side.get((t, flip.get(s)), [])) if s in flip else []
        if not other:
            return [(bid, 1.0)]
        return [(o, 1.0 / len(other)) for o in other]

    touched = edges[edges.pre.isin(left_jo) | edges.post.isin(left_jo)]
    new = []
    for pre, post, w in touched[["pre", "post", "weight"]].itertuples(index=False):
        for p2, fp in partners(pre):
            for q2, fq in partners(post):
                new.append((p2, q2, w * fp * fq))
    mirrored = neurons.loc[left_jo].copy()
    mirrored.index = -mirrored.index
    mirrored["side"] = "R"
    mirrored["instance"] = mirrored["instance"].astype(str) + " [mirrored L->R]"
    neurons = pd.concat([neurons, mirrored])
    edges = pd.concat([edges, pd.DataFrame(new, columns=["pre", "post", "weight"])])
    edges = edges.groupby(["pre", "post"], as_index=False).weight.sum()
    return neurons, edges


@dataclass
class Brain:
    seed: int = 0
    exclude_lc_recurrence: bool = False
    data_dir: object = None            # folder with neurons.csv / edges.csv (default: data/)
    neurons: pd.DataFrame = field(init=False)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        d = self.data_dir or C.DATA_DIR
        n = pd.read_csv(d / "neurons.csv").set_index("bodyId")
        e = pd.read_csv(d / "edges.csv")
        n, e = mirror_hearing(n, e)
        if self.exclude_lc_recurrence:
            vis = set(n.index[n.role.isin(["sense_looming", "sense_pursuit"])])
            e = e[~(e.pre.isin(vis) & e.post.isin(vis))]
        self.neurons = n
        self.edges = e
        self.N = len(n)
        idx = pd.Series(np.arange(self.N), index=n.index)
        sign = n.sign.to_numpy()
        pre, post = idx[e.pre].to_numpy(), idx[e.post].to_numpy()
        vals = W_SYN * e.weight.to_numpy() * sign[pre]
        self.W = sparse.csc_matrix((vals, (post, pre)), shape=(self.N, self.N))
        self.idx = idx

        roles = n.role.to_numpy()
        self.loom = np.flatnonzero(roles == "sense_looming")
        self.purs = np.flatnonzero(roles == "sense_pursuit")
        self.hear = np.flatnonzero(roles == "sense_hearing")
        side = n.side.to_numpy()
        # body-frame azimuth: 0 ahead, +right, -left
        eye_az = n.azimuth.to_numpy()
        self.body_az = np.where(side == "R", eye_az, -eye_az)
        self.el = n.elevation.to_numpy()
        self.rf_az = n.rf_az_sd.to_numpy()
        self.rf_el = n.rf_el_sd.to_numpy()
        self.hear_right = side[self.hear] == "R"

        def one(t, s):
            return int(idx[n.index[(n.type == t) & (n.side == s)][0]])
        self.out = {f"{t}_{s}": one(t, s) for t in ["DNp01", "DNp02", "DNp04", "DNp06", "DNp11", "DNa02", "TTMn"]
                    for s in "LR"}
        self.dna02 = np.array([self.out["DNa02_L"], self.out["DNa02_R"]])
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.v = np.full(self.N, V_REST)
        self.g = np.zeros(self.N)
        self.ref_until = np.zeros(self.N)
        self.adapt = np.zeros(self.N)
        self.t = 0.0
        self.delay_steps = int(round(T_DLY / DT))
        self.buf = [np.zeros(0, dtype=int) for _ in range(self.delay_steps)]
        # spikes of the last few 0.1 ms steps: the only cells that can still be refractory
        self.recent = deque(maxlen=int(np.ceil(T_REF / DT)) + 2)
        self.rates = np.zeros(self.N)      # smoothed Hz, every neuron (for the side panel)
        self.input_rate = np.zeros(self.N)

    # ------------------------------------------------------------------ senses
    def _rf_overlap(self, cells, bearing, theta, elev=0.0):
        daz = wrap180(self.body_az[cells] - bearing)
        s_obj = theta / 2.0
        return (np.exp(-daz ** 2 / (2 * (self.rf_az[cells] ** 2 + s_obj ** 2)))
                * np.exp(-(self.el[cells] - elev) ** 2 / (2 * (self.rf_el[cells] ** 2 + s_obj ** 2))))

    def sense(self, ghosts=(), pellets=()):
        """Set sensory Poisson rates from the scene.

        ghosts:  dicts with bearing (deg, body frame, +right), dist (tiles), size (tiles),
                 closing (tiles/s, >0 approaching), speed (tiles/s), visible (line of sight)
        pellets: dicts with bearing, dist, size, visible
        """
        S = SENSE
        r = np.zeros(self.N)
        loom_drive = np.zeros(len(self.loom))
        purs_drive = np.zeros(len(self.purs))
        ear = np.zeros(2)  # left, right
        for gh in ghosts:
            d = max(gh["dist"], 0.2)
            theta = np.degrees(2 * np.arctan(gh["size"] / (2 * d)))
            if gh["visible"] and gh["closing"] > 0:
                dtheta = np.degrees(gh["size"] * gh["closing"] / (d ** 2 + gh["size"] ** 2 / 4))
                loom_drive += dtheta * self._rf_overlap(self.loom, gh["bearing"], theta)
            # air vibration reaches the antennae around walls: game may pass path distance
            dh = max(gh.get("hear_dist", d), 0.3)
            a = gh["speed"] * (S["hear_r0"] / dh) ** 3
            lat = np.sin(np.radians(gh["bearing"])) * S["ear_directionality"]
            ear += a * np.array([0.5 - lat, 0.5 + lat]) * 2
        for p in pellets:
            if not p["visible"]:
                continue
            theta = np.degrees(2 * np.arctan(p["size"] / (2 * max(p["dist"], 0.2))))
            sal = theta * np.exp(-theta / S["pursuit_pref_size"])
            purs_drive += sal * self._rf_overlap(self.purs, p["bearing"], theta)
        r[self.loom] = S["loom_rate_max"] * np.tanh(loom_drive / S["loom_half"])
        r[self.purs] = S["pursuit_rate_max"] * np.tanh(purs_drive / S["pursuit_half"])
        ear = np.maximum(ear - S["hear_threshold"], 0.0)
        r[self.hear] = S["hear_rate_max"] * np.tanh(np.where(self.hear_right, ear[1], ear[0]) / S["hear_half"])
        r[self.dna02] += MOTOR["explore_rate"]
        self.input_rate = r

    # ------------------------------------------------------------------ dynamics
    def step(self, duration):
        """Advance the network; returns spike counts per neuron over `duration` seconds.

        Same model as step_reference() (identical spikes for the same seed), organised for speed:
        spike propagation is vectorised, and only cells that fired within the last refractory
        period are checked for refractoriness instead of all N cells every 0.1 ms.
        """
        if ADAPT_B:
            return self.step_reference(duration)
        n_steps = max(1, int(round(duration / DT)))
        N = self.N
        counts = np.zeros(N, dtype=np.int32)
        driven = np.flatnonzero(self.input_rate > 0)
        hits = self.rng.random((n_steps, len(driven))) < (self.input_rate[driven] * DT) if len(driven) else None
        w_in = F_POI * W_SYN
        decay = np.exp(-DT / TAU_SYN)
        leak = DT / T_MBR
        indptr, indices, data = self.W.indptr, self.W.indices, self.W.data
        v, g, ref_until = self.v, self.g, self.ref_until
        recent = self.recent
        dv = np.empty(N)
        for step in range(n_steps):
            sp = self.buf.pop(0)
            if len(sp):
                if len(sp) == 1:
                    j = sp[0]
                    lo, hi = indptr[j], indptr[j + 1]
                    g[indices[lo:hi]] += data[lo:hi]      # a column has no repeated rows
                else:
                    starts = indptr[sp]
                    lens = indptr[sp + 1] - starts
                    sl = np.repeat(starts - np.cumsum(lens) + lens, lens) + np.arange(lens.sum())
                    g += np.bincount(indices[sl], weights=data[sl], minlength=N)
            if hits is not None:
                g[driven[hits[step]]] += w_in
            np.subtract(v, V_REST, out=dv)
            np.subtract(g, dv, out=dv)                    # dv = g - (v - V_REST)
            dv *= leak
            cand = [r for r in recent if len(r)]
            if cand:
                cand = np.concatenate(cand)
                dv[cand[self.t < ref_until[cand]]] = 0.0  # refractory cells don't integrate
            v += dv
            g *= decay
            spk = np.flatnonzero(v >= V_TH)
            if len(spk):
                v[spk] = V_REST
                ref_until[spk] = self.t + T_REF
                counts[spk] += 1
            recent.append(spk)
            self.buf.append(spk)
            self.t += DT
        # Flush decayed synaptic input to exactly 0. Left alone, silent cells' g decays into subnormal
        # floats (< 1e-308), which x86 CPUs process ~100x slower (a quiet brain ran 8x slower).
        g[np.abs(g) < 1e-9] = 0.0
        a = 1 - np.exp(-duration / MOTOR["rate_tau"])
        self.rates += a * (counts / duration - self.rates)
        return counts

    def step_reference(self, duration):
        """Straightforward version of step() (kept for checking and for ADAPT_B experiments)."""
        n_steps = max(1, int(round(duration / DT)))
        N = self.N
        counts = np.zeros(N, dtype=np.int32)
        driven = np.flatnonzero(self.input_rate > 0)
        # Poisson input for the whole window at once: hits[step, k] -> driven[k] gets a kick
        hits = self.rng.random((n_steps, len(driven))) < (self.input_rate[driven] * DT) if len(driven) else None
        w_in = F_POI * W_SYN
        decay = np.exp(-DT / TAU_SYN)
        a_decay = np.exp(-DT / TAU_ADAPT)
        leak = DT / T_MBR
        indptr, indices, data = self.W.indptr, self.W.indices, self.W.data
        v, g, ref_until = self.v, self.g, self.ref_until
        for step in range(n_steps):
            sp = self.buf.pop(0)
            if len(sp):
                if len(sp) == 1:
                    j = sp[0]
                    lo, hi = indptr[j], indptr[j + 1]
                    g[indices[lo:hi]] += data[lo:hi]      # a column has no repeated rows
                else:
                    sl = np.concatenate([np.arange(indptr[j], indptr[j + 1]) for j in sp])
                    g += np.bincount(indices[sl], weights=data[sl], minlength=N)
            if hits is not None:
                g[driven[hits[step]]] += w_in
            active = self.t >= ref_until
            dv = g - (v - V_REST)
            if ADAPT_B:
                dv -= self.adapt
            v += active * leak * dv
            g *= decay
            if ADAPT_B:
                self.adapt *= a_decay
            spk = np.flatnonzero(v >= V_TH)
            if len(spk):
                v[spk] = V_REST
                ref_until[spk] = self.t + T_REF
                counts[spk] += 1
                if ADAPT_B:
                    self.adapt[spk] += ADAPT_B
            self.buf.append(spk)
            self.t += DT
        # Flush decayed synaptic input to exactly 0. Left alone, silent cells' g decays into subnormal
        # floats (< 1e-308), which x86 CPUs process ~100x slower (a quiet brain ran 8x slower).
        g[np.abs(g) < 1e-9] = 0.0
        a = 1 - np.exp(-duration / MOTOR["rate_tau"])
        self.rates += a * (counts / duration - self.rates)
        return counts

    # ------------------------------------------------------------------ motor readout
    def motor(self, counts):
        """Decode movement from descending-neuron activity (DESIGN: fixed motor map).

        jump:       any Giant Fiber (DNp01) spike in this window
        away_right: + means push off toward the right (left-side escape DNs more active)
        away_fwd:   + means jump forward (rear-threat DNp11 > front-threat DNp02)
        turn_right: DNa02_R - DNa02_L (DNa02 activity predicts ipsilateral turning)
        """
        o, r = self.out, self.rates
        side = lambda s: sum(r[o[f"{t}_{s}"]] for t in ["DNp01", "DNp02", "DNp04", "DNp11"])
        return dict(
            jump=int(counts[o["DNp01_L"]] + counts[o["DNp01_R"]]) > 0,
            gf_spikes=(int(counts[o["DNp01_L"]]), int(counts[o["DNp01_R"]])),
            away_right=side("L") - side("R"),
            away_fwd=(r[o["DNp11_L"]] + r[o["DNp11_R"]]) - (r[o["DNp02_L"]] + r[o["DNp02_R"]]),
            turn_right=r[o["DNa02_R"]] - r[o["DNa02_L"]],
        )
