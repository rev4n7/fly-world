"""Fly World logic (no graphics): an open arena, user-spawned entities, a fly driven only by its brain.

No "if predator near then flee" or "if female then approach" rule. Every tick:
  entities -> brain.sense_world() -> brain.step() -> neuron readout -> fly movement
Movement comes from descending/motor neurons only: DNa02 L/R (turning), Giant Fiber (takeoff),
MN9 (proboscis extension: stop and eat). Ablating a neuron changes behaviour only through the network.
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .worldbrain import WorldBrain

FRAME = 0.02            # s of simulated time per tick
WORLD_W, WORLD_H = 30.0, 19.0   # units (1 unit ~ 1 fly body length)

# ---- DESIGN: body / world constants (not connectome data) ----
FLY_WALK = 2.5          # units/s at WALK_REF Hz of DNa02 (L+R)
WALK_REF = 10.0         # Hz; resting DNa02 L+R (exploration drive) ~ 10 Hz = normal walking
WALK_MAX = 1.3          # x FLY_WALK at most
FLY_JUMP = 8.0          # units/s during an escape burst
JUMP_TIME = 0.35        # s
TURN_GAIN = 2.5         # deg/s of turning per Hz of DNa02 R-L difference
MAX_TURN = 400.0        # deg/s
DIR_WINDOW = 0.3        # s of neuron activity used to set takeoff direction
FEED_ON = 10.0          # Hz of MN9 (both sides) = proboscis extended: the fly stops and eats
EAT_RATE = 0.12         # fraction of a food item eaten per second of feeding
SONG_ON = 10.0          # Hz of pIP10 = wing extended (song)
PREDATOR_SIZE, PREDATOR_SPEED = 1.3, 2.0
PREDATOR_REST = 2.5     # s a predator pauses after a catch or a missed lunge
FEMALE_SIZE, FEMALE_SPEED = 0.6, 0.7
FOOD_SIZE = 0.7
CATCH_DIST = 0.7

# ---- injury lab (DESIGN: amounts and speeds; what gets damaged is real) ----
DECLINE_RATE = 0.004    # fraction of the living neurons lost per second in "slow decline"
SEIZURE_HZ, SEIZURE_FRAC = 80.0, 0.08   # seizure = >= 8% of non-sensory cells firing >= 80 Hz

# ---- ablation menu: named circuits (cell types / roles from data/world) ----
CIRCUITS = [
    # (key, label, behaviour, match)  match: {"types": [...]} or {"roles": [...]} or {"prefix": "..."}
    ("gf", "Giant Fiber (DNp01)", "escape", {"types": ["DNp01"]}),
    ("loom", "Looming eyes (LC4, LPLC2, LC6)", "escape", {"roles": ["sense_looming"]}),
    ("escdn", "Escape DNs (DNp02/04/06/11)", "escape", {"types": ["DNp02", "DNp04", "DNp06", "DNp11"]}),
    ("gfmid", "Looming->GF middle cells", "escape", {"roles": ["middle_gf"]}),
    ("ears", "Ears (Johnston's organ)", "escape", {"roles": ["sense_hearing"]}),
    ("lc10a", "Female-chase eyes (LC10a)", "courtship", {"types": ["LC10a"]}),
    ("aotu", "Chase relay (AOTU019/012/015/025..)", "courtship", {"roles": ["middle_steer"]}),
    ("p1", "P1/pC1 arousal (156 cells)", "courtship", {"roles": ["court_p1"]}),
    ("pip10", "Song neuron pIP10", "courtship", {"types": ["pIP10"]}),
    ("vab3", "Pheromone relay vAB3", "courtship", {"roles": ["relay_vab3"]}),
    ("mal", "mAL brake (GABA)", "courtship", {"prefix": "mAL"}),
    ("dna02", "Steering DNa02", "steering", {"types": ["DNa02"]}),
    ("eyeL", "Left eye: all visual cells (lobula L)", "vision", {"roles": ["sense_looming", "sense_pursuit"], "side": "L"}),
    ("eyeR", "Right eye: all visual cells (lobula R)", "vision", {"roles": ["sense_looming", "sense_pursuit"], "side": "R"}),
    ("mn9", "Proboscis motor neuron MN9", "feeding", {"types": ["MN9"]}),
    ("tastemid", "Taste relay cells (GNG)", "feeding", {"roles": ["middle_taste"]}),
    ("orn", "Food-smell receptors (ORNs)", "feeding", {"roles": ["sense_smell"]}),
    ("pn", "Smell relays (projection neurons)", "feeding", {"roles": ["smell_pn"]}),
]


def rot(v, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def right_of(h):
    return np.array([-h[1], h[0]])   # screen y points down, so this is the fly's right


@dataclass(eq=False)       # identity comparison (fields hold arrays)
class Entity:
    kind: str                # "food" | "female" | "predator"
    pos: np.ndarray
    heading: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    amount: float = 1.0      # food left
    home: np.ndarray = None  # predator spawn point
    rest_until: float = 0.0
    prev_dist: float = None
    uid: int = 0


@dataclass
class Stats:
    time: float = 0.0
    jumps: int = 0
    catches: int = 0
    eaten: float = 0.0
    song_time: float = 0.0
    chase_time: float = 0.0


class FlyWorld:
    def __init__(self, seed=0, brain=None):
        self.rng = np.random.default_rng(seed)
        self.brain = brain or WorldBrain(seed=seed)
        n = self.brain.neurons
        self.types, self.roles = n.type.fillna("").to_numpy(), n.role.to_numpy()
        self.sides = n.side.fillna("").to_numpy()
        self.circuit_idx = {key: self._match(m) for key, _, _, m in CIRCUITS}
        # injury lab state (3D strokes at a clicked brain spot: flypac/fly3d_server.py Brain3DMap)
        self.superclass = n.superclass.fillna("").to_numpy() if "superclass" in n else np.array([""] * len(n))
        self.not_sensory = ~np.char.startswith(self.roles.astype(str), "sense")
        self.dn = {s: np.flatnonzero((self.superclass == "descending_neuron") & (self.sides == s)) for s in "LR"}
        self.neck_cut = set()                  # sides whose descending neurons no longer reach the body
        self.lost = set()                      # neurons killed by strokes / random loss / decline
        self.decline = False
        self.syn_gain, self.inh_gain = 1.0, 1.0
        self.ablated_circuits, self.ablated_types = set(), set()
        self.loom_cos = np.cos(np.radians(self.brain.body_az[self.brain.loom]))
        self.side_idx = {s: [self.brain.out[f"{t}_{s}"] for t in ["DNp01", "DNp02", "DNp04", "DNp11"]] for s in "LR"}
        self.pn = np.flatnonzero(self.roles == "smell_pn")
        self.entities = []
        self._uid = 0
        self.stats = Stats()
        self.fly_pos = np.array([WORLD_W / 2, WORLD_H / 2])
        self.heading = np.array([1.0, 0.0])
        self.hist = deque(maxlen=int(DIR_WINDOW / FRAME))
        self.counts = np.zeros(self.brain.N, dtype=np.int32)
        self.motor = self._readout()
        self.escape_until = -1.0
        self.last_gf = -10.0
        self.event, self.event_time = "", -10.0
        self.feeding = self.singing = False
        self.log = []

    # ------------------------------------------------------------------ ablation
    def _match(self, m):
        if "types" in m:
            hit = np.isin(self.types, m["types"])
        elif "roles" in m:
            hit = np.isin(self.roles, m["roles"])
        else:
            hit = np.char.startswith(self.types.astype(str), m["prefix"])
        if "side" in m:
            hit &= self.sides == m["side"]
        return np.flatnonzero(hit)

    def type_index(self, name):
        """Neurons for a cell type name, or a single bodyId given as digits."""
        if name.lstrip("-").isdigit():
            bid = int(name)
            return np.array([self.brain.idx[bid]]) if bid in self.brain.idx.index else np.array([], int)
        return np.flatnonzero(self.types == name)

    def ablated_index(self):
        parts = [self.circuit_idx[k] for k in self.ablated_circuits] + [self.type_index(t) for t in self.ablated_types]
        parts.append(np.fromiter(self.lost, int, len(self.lost)))
        return np.unique(np.concatenate(parts)).astype(int)

    def _apply_ablation(self):
        self.brain.set_network(self.ablated_index(), self.syn_gain, self.inh_gain)

    # ------------------------------------------------------------------ injury lab
    def cut_neck(self, sides):
        """Sever the descending neurons of these sides ('L', 'R'): the brain still fires them,
        but the message never reaches the body (like a spinal cord injury). Toggles."""
        self.neck_cut ^= set(sides)

    def lose_random(self, fraction):
        """Diffuse damage: kill this fraction of the currently living neurons, chosen at random."""
        alive = np.flatnonzero(self.brain.alive)
        k = int(round(fraction * len(alive)))
        self.lost |= set(self.rng.choice(alive, k, replace=False).tolist())
        self._apply_ablation()
        return k

    def stroke(self, idx):
        """Focal damage: kill the given neurons (e.g. everything at one spot in the brain)."""
        new = set(int(i) for i in idx) - self.lost
        self.lost |= new
        self._apply_ablation()
        return len(new)

    def set_gains(self, syn_gain=None, inh_gain=None):
        if syn_gain is not None:
            self.syn_gain = syn_gain
        if inh_gain is not None:
            self.inh_gain = inh_gain
        self._apply_ablation()

    def seizure_level(self):
        keep = self.brain.alive & self.not_sensory
        r = self.brain.rates[keep]
        return float((r >= SEIZURE_HZ).mean()) if len(r) else 0.0

    def _readout(self):
        """Motor readout from neurons, with severed descending neurons silent to the body."""
        b, o = self.brain, self.brain.out
        r, c = b.rates.copy(), self.counts.copy()
        for side in self.neck_cut:
            r[self.dn[side]] = 0.0
            c[self.dn[side]] = 0
        self.body_counts = c
        return dict(jump=bool(c[o["DNp01_L"]] + c[o["DNp01_R"]] > 0),
                    turn_right=float(r[o["DNa02_R"]] - r[o["DNa02_L"]]),
                    walk=float(r[o["DNa02_L"]] + r[o["DNa02_R"]]),
                    song=float(r[b.song].sum()), feed=float(r[b.feed].sum()),
                    arousal=float(b.rates[b.p1].mean()), gf_rate=float(r[b.gf].sum()))

    def toggle_circuit(self, key):
        self.ablated_circuits ^= {key}
        self._apply_ablation()

    def toggle_type(self, name):
        if not len(self.type_index(name)):
            return False
        self.ablated_types ^= {name}
        self._apply_ablation()
        return True

    def clear_ablation(self):
        """Restore everything: removed circuits, lost cells, severed neck, synapse strengths."""
        self.ablated_circuits.clear()
        self.ablated_types.clear()
        self.lost.clear()
        self.neck_cut.clear()
        self.decline = False
        self.syn_gain = self.inh_gain = 1.0
        self._apply_ablation()

    # ------------------------------------------------------------------ entities
    def spawn(self, kind, pos):
        self._uid += 1
        pos = np.clip(np.asarray(pos, float), 0.3, [WORLD_W - 0.3, WORLD_H - 0.3])
        e = Entity(kind, pos, heading=rot(np.array([1.0, 0.0]), self.rng.uniform(0, 360)), uid=self._uid)
        if kind == "predator":
            e.home = pos.copy()
            e.rest_until = self.stats.time + 0.5
        self.entities.append(e)
        return e

    def remove_near(self, pos, radius=1.2):
        if not self.entities:
            return None
        d = [np.hypot(*(e.pos - pos)) for e in self.entities]
        k = int(np.argmin(d))
        return self.entities.pop(k) if d[k] < radius else None

    def of(self, kind):
        return [e for e in self.entities if e.kind == kind]

    # ------------------------------------------------------------------ senses
    def _rel(self, p):
        rel = p - self.fly_pos
        return rel, float(np.hypot(*rel)), float(np.degrees(np.arctan2(rel @ right_of(self.heading), rel @ self.heading)))

    def sense(self):
        preds, fems, foods = [], [], []
        for e in self.of("predator"):
            rel, d, bearing = self._rel(e.pos)
            closing = 0.0 if e.prev_dist is None else (e.prev_dist - d) / FRAME
            e.prev_dist = d
            moving = self.stats.time >= e.rest_until
            preds.append(dict(bearing=bearing, dist=d, size=PREDATOR_SIZE, closing=closing,
                              speed=PREDATOR_SPEED if moving else 0.0, visible=True, hear_dist=d))
        for e in self.of("female"):
            _, d, bearing = self._rel(e.pos)
            fems.append(dict(bearing=bearing, dist=d, size=FEMALE_SIZE, visible=True))
        for e in self.of("food"):
            _, d, bearing = self._rel(e.pos)
            foods.append(dict(bearing=bearing, dist=d))
        self.pred_inputs = preds
        self.brain.sense_world(predators=preds, females=fems, foods=foods)

    # ------------------------------------------------------------------ motor
    def escape_vector(self):
        """Body-frame (fwd, right) takeoff direction from neurons (Fly-Pacman's 'eyes' readout)."""
        w = np.sum(self.hist, axis=0) if self.hist else self.body_counts
        side_l, side_r = w[self.side_idx["L"]].sum(), w[self.side_idx["R"]].sum()
        right = (side_l - side_r) / (side_l + side_r + 1.0)
        lc = w[self.brain.loom]
        fwd = -(lc @ self.loom_cos) / (lc.sum() + 1.0)
        return float(fwd), float(right)

    def takeoff(self):
        fwd, right = self.escape_vector()
        want = fwd * self.heading + right * right_of(self.heading)
        if np.hypot(*want) > 0.05:
            self.heading = want / np.hypot(*want)
        self.escape_until = self.stats.time + JUMP_TIME
        self.stats.jumps += 1
        self.flash("GIANT FIBER FIRED - TAKEOFF")
        self.log.append(dict(kind="takeoff", t=self.stats.time))

    def flash(self, text):
        self.event, self.event_time = text, self.stats.time

    def circuit_levels(self):
        """How strongly each behaviour's real circuit is active right now (0..1, display only)."""
        b, r = self.brain, self.brain.rates
        recent_gf = self.stats.time - self.last_gf < 0.6
        return {
            "escape": 1.0 if recent_gf else min(float(r[b.loom].mean()) / 8.0, 0.99),
            "chase": min(float(r[b.purs].mean()) / 3.0, 1.0),
            "song": min(self.motor["song"] / 40.0, 1.0),
            "feeding": min(self.motor["feed"] / 40.0, 1.0),
            "smell": min(float(r[self.pn].mean()) / 40.0, 1.0),
        }

    LABELS = {"escape": "ALARMED (looming cells)", "chase": "CHASING (LC10a)", "song": "SINGING (pIP10)",
              "smell": "SMELLS FOOD"}

    def behaviour(self):
        """Name of the most active circuit. Display only: movement is read from neurons directly."""
        lv = self.circuit_levels()
        if self.seizure_level() >= SEIZURE_FRAC:
            return "SEIZURE (runaway activity)", lv
        if self.neck_cut == {"L", "R"}:
            return ("FEEDING (head still works)" if self.feeding else "PARALYSED (neck cut)"), lv
        if self.stats.time < self.escape_until or lv["escape"] >= 1.0:
            return "ESCAPING (Giant Fiber)", lv
        if self.feeding:
            return "FEEDING (MN9)", lv
        k = max(self.LABELS, key=lambda k: lv[k])
        return (self.LABELS[k] if lv[k] >= 0.15 else "EXPLORING"), lv

    # ------------------------------------------------------------------ tick
    def step(self):
        dt = FRAME
        self.stats.time += dt
        t = self.stats.time
        self.sense()
        if self.decline and round(t / dt) % round(1.0 / dt) == 0:
            self.lose_random(DECLINE_RATE)
        self.counts = self.brain.step(dt)
        self.motor = m = self._readout()
        self.hist.append(self.body_counts)

        if m["jump"]:
            self.last_gf = t
            if t >= self.escape_until:
                self.takeoff()
        escaping = t < self.escape_until
        on_food = [e for e in self.of("food") if np.hypot(*(e.pos - self.fly_pos)) < 0.7]
        self.feeding = bool(on_food) and m["feed"] >= FEED_ON and not escaping
        self.singing = m["song"] >= SONG_ON
        if self.singing:
            self.stats.song_time += dt
        if self.brain.rates[self.brain.purs].mean() > 1.0:
            self.stats.chase_time += dt

        # fly movement: turning from DNa02 R-L, speed from state
        if not escaping:
            turn = float(np.clip(TURN_GAIN * m["turn_right"], -MAX_TURN, MAX_TURN))
            self.heading = rot(self.heading, turn * dt)
        walk = FLY_WALK * min(m["walk"] / WALK_REF, WALK_MAX)       # walking needs DNa02 drive
        speed = FLY_JUMP if escaping else (0.0 if self.feeding else walk)
        self.fly_pos = self._move(self.fly_pos, self.heading, speed * dt, is_fly=True)
        if self.feeding:
            food = on_food[0]
            food.amount -= EAT_RATE * dt
            self.stats.eaten += EAT_RATE * dt
            if food.amount <= 0:
                self.entities.remove(food)
                self.flash("FOOD FINISHED")

        for e in list(self.entities):
            if e.kind == "female":
                e.heading = rot(e.heading, self.rng.normal(0, 90) * dt * 3)
                e.pos = self._move(e.pos, e.heading, FEMALE_SPEED * dt, entity=e)
            elif e.kind == "predator" and t >= e.rest_until:
                to = self.fly_pos - e.pos
                d = np.hypot(*to)
                e.heading = to / max(d, 1e-6)
                e.pos = e.pos + e.heading * min(PREDATOR_SPEED * dt, d)
                if d < CATCH_DIST:
                    self.stats.catches += 1
                    self.flash("CAUGHT BY PREDATOR")
                    self.log.append(dict(kind="catch", t=t))
                    e.pos, e.prev_dist, e.rest_until = e.home.copy(), None, t + PREDATOR_REST

    def _move(self, pos, heading, dist, is_fly=False, entity=None):
        """Walk, bouncing off the arena edge (DESIGN: the arena has walls)."""
        new = pos + heading * dist
        h = heading.copy()
        for k, hi in ((0, WORLD_W), (1, WORLD_H)):
            if new[k] < 0.3 or new[k] > hi - 0.3:
                h[k] = -h[k]
                new[k] = np.clip(new[k], 0.3, hi - 0.3)
        if is_fly:
            self.heading = h
        elif entity is not None:
            entity.heading = h
        return new
