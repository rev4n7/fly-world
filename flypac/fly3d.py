"""Fly 3D logic (no graphics): Fly World in three dimensions, with flight.

The fly flies or walks. Everything that moves it is read from neurons:
  wing power (lift)   <- wing power motor neurons DLMn/DVMn, driven by DNg02 through real synapses
  turning             <- DNa02 R - L (flight and walking)
  walking speed       <- DNa02 L + R
  escape burst        <- Giant Fiber spike; direction from which looming cells fired (their real
                         azimuth AND elevation, so a threat from above sends the fly down and away)
  climb/descend while chasing <- elevation of the active LC10a cells (female below -> descend)
  eating              <- MN9
A neck cut here is a real cut: the synapses from severed descending neurons onto body (VNC) neurons,
such as DNg02 -> wing motor neurons, are removed from the network.
"""
import numpy as np
from scipy import sparse

from . import brain as B
from . import config as C
from .flyworld import (CATCH_DIST, DIR_WINDOW, EAT_RATE, FEED_ON, FRAME, PREDATOR_REST, SONG_ON,
                       Entity, FlyWorld, Stats, rot)
from .worldbrain import WorldBrain

FLY3D_DIR = C.DATA_DIR / "fly3d"
WORLD = (40.0, 40.0)            # floor size (units ~ fly body lengths)
BOX_H = 12.0                    # the world is a closed box: 4 walls and a ceiling this high

# ---- DESIGN: flight / body constants (not connectome data) ----
FLIGHT_DRIVE = 40.0     # Hz Poisson drive to DNg02 while airborne ("wants to keep flying")
FLY_SPEED = 5.0         # units/s cruising airspeed at normal wing power
WALK_SPEED = 2.5        # units/s at 10 Hz of DNa02 L+R
TURN_GAIN, MAX_TURN = 2.5, 400.0
CRUISE_Z = 3.0          # units; preferred flight height (real flies hold height with optic flow: not simulated)
HOLD_GAIN = 0.8         # 1/s, pull toward CRUISE_Z when wing power is normal
SINK = 3.0              # units/s falling speed with no wing power
PITCH_GAIN = 1.2        # vertical speed per unit of LC10a elevation signal
ESCAPE_SPEED, ESCAPE_TIME = 9.0, 0.35
LIFT_TAU = 0.6          # s, smoothing of wing power
LAUNCH_TIME = 0.8       # s after a takeoff during which the jump itself carries the fly while wing power builds up
LAND_URGE = 0.04        # per second: chance of deciding to land somewhere (flies land and walk)
WALK_BORED = 7.0        # s on the ground with nothing to do -> takes off again
PRED_SPEED, PRED_SIZE = 2.6, 1.3
FEMALE_SPEED, FEMALE_SIZE = 0.7, 0.6


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


class Brain3D(WorldBrain):
    def __init__(self, seed=0):
        super().__init__(seed=seed, data_dir=WorldBrain._prepared(FLY3D_DIR))
        roles, n = self.neurons.role.to_numpy(), self.neurons
        self.flight_dn = np.flatnonzero(roles == "out_flight")
        self.power = np.flatnonzero(roles == "out_wing_power")
        self.steer_mn = np.flatnonzero(roles == "out_wing_steer")
        sc = n.superclass.fillna("").to_numpy()
        self.is_dn = sc == "descending_neuron"
        self.in_body = np.char.startswith(sc.astype(str), "vnc")    # cells below the neck (VNC)
        self.side = n.side.fillna("").to_numpy()
        self.neck_cut = set()
        self.flying = True

    def set_network(self, removed=(), syn_gain=1.0, inh_gain=1.0):
        super().set_network(removed, syn_gain, inh_gain)
        if self.neck_cut:   # remove synapses from severed descending neurons onto body cells
            cut_pre = self.is_dn & np.isin(self.side, list(self.neck_cut))
            W = self.W.tocoo()
            keep = ~(cut_pre[W.col] & self.in_body[W.row])
            self.W = sparse.csc_matrix((W.data[keep], (W.row[keep], W.col[keep])), shape=W.shape)

    def sense_3d(self, predators=(), females=(), foods=()):
        """Like WorldBrain.sense_world, with elevation: every object has bearing (deg, +right),
        elev (deg, +up), dist. Looming and LC10a cells respond by their real azimuth AND elevation."""
        S = B.SENSE
        self.sense_world(predators=[], females=[], foods=foods)   # smell, taste, touch (touch needs dist only)
        r = self.input_rate
        touching = any(f["dist"] < 0.7 for f in females)
        r[self.touch] = WSENSE_TOUCH if touching else 0.0
        loom = np.zeros(len(self.loom))
        purs = np.zeros(len(self.purs))
        ear = np.zeros(2)
        for p in predators:
            d = max(p["dist"], 0.2)
            theta = np.degrees(2 * np.arctan(p["size"] / (2 * d)))
            if p["closing"] > 0:
                dtheta = np.degrees(p["size"] * p["closing"] / (d ** 2 + p["size"] ** 2 / 4))
                loom += dtheta * self._rf_overlap(self.loom, p["bearing"], theta, p["elev"])
            a = p["speed"] * (S["hear_r0"] / max(d, 0.3)) ** 3
            lat = np.sin(np.radians(p["bearing"])) * S["ear_directionality"]
            ear += a * np.array([0.5 - lat, 0.5 + lat]) * 2
        for f in females:
            theta = np.degrees(2 * np.arctan(f["size"] / (2 * max(f["dist"], 0.2))))
            sal = theta * np.exp(-theta / S["pursuit_pref_size"])
            purs += sal * self._rf_overlap(self.purs, f["bearing"], theta, f["elev"])
        r[self.loom] = S["loom_rate_max"] * np.tanh(loom / S["loom_half"])
        r[self.purs] = S["pursuit_rate_max"] * np.tanh(purs / S["pursuit_half"])
        ear = np.maximum(ear - S["hear_threshold"], 0.0)
        r[self.hear] = S["hear_rate_max"] * np.tanh(np.where(self.hear_right, ear[1], ear[0]) / S["hear_half"])
        if self.flying:
            r[self.flight_dn] += FLIGHT_DRIVE
        r[~self.alive] = 0.0
        self.touching = touching


from .worldbrain import WSENSE  # noqa: E402
WSENSE_TOUCH = WSENSE["touch_rate"]


class Fly3D(FlyWorld):
    """Same circuits, ablation and injury lab as FlyWorld; 3D bodies and flight."""

    def __init__(self, seed=0):
        self.pos = np.array([WORLD[0] / 2, WORLD[1] / 2, CRUISE_Z])   # before FlyWorld sets fly_pos
        super().__init__(seed=seed, brain=Brain3D(seed=seed))
        self.pos = np.array([WORLD[0] / 2, WORLD[1] / 2, CRUISE_Z])
        self.heading = np.array([1.0, 0.0])
        self.flying = True
        self.vz = 0.0
        self.escape_dir = None
        self.landing = False
        self.ground_time = 0.0
        self.power_ref = None
        self.power_ema, self.calib = 0.0, []
        self.takeoff_time = -10.0
        self.lift = 1.0
        self.hist_len = int(DIR_WINDOW / FRAME)

    # fly_pos (2D, used by FlyWorld helpers) mirrors pos[:2]
    @property
    def fly_pos(self):
        return self.pos[:2]

    @fly_pos.setter
    def fly_pos(self, v):
        self.pos[:2] = v

    # ------------------------------------------------------------------ neck cut acts on the network
    def _apply_ablation(self):
        self.brain.neck_cut = set(self.neck_cut)
        super()._apply_ablation()

    def cut_neck(self, sides):
        self.neck_cut ^= set(sides)
        self._apply_ablation()

    def set_neck(self, sides):
        self.neck_cut = set(sides)
        self._apply_ablation()

    # ------------------------------------------------------------------ entities
    def spawn(self, kind, pos):
        pos = np.asarray(pos, float)
        xy = np.clip(pos[:2], 0.5, [WORLD[0] - 0.5, WORLD[1] - 0.5])
        z = 0.0 if kind in ("food", "female") else min(max(pos[2] if len(pos) > 2 else 6.0, 0.5), BOX_H - 1.0)
        self._uid += 1
        e = Entity(kind, np.array([xy[0], xy[1], z]), heading=rot(np.array([1.0, 0.0]), self.rng.uniform(0, 360)),
                   uid=self._uid)
        if kind == "predator":
            e.home = e.pos.copy()
            e.rest_until = self.stats.time + 1.0
        self.entities.append(e)
        return e

    def remove_near(self, pos, radius=2.0):
        if not self.entities:
            return None
        d = [np.hypot(*(e.pos[:2] - np.asarray(pos)[:2])) for e in self.entities]
        k = int(np.argmin(d))
        return self.entities.pop(k) if d[k] < radius else None

    # ------------------------------------------------------------------ senses
    def _rel3(self, p):
        rel = p - self.pos
        h = np.array([self.heading[0], self.heading[1], 0.0])
        r = np.array([-self.heading[1], self.heading[0], 0.0])
        horiz = np.hypot(rel[0], rel[1])
        return (float(np.linalg.norm(rel)), float(np.degrees(np.arctan2(rel @ r, rel @ h))),
                float(np.degrees(np.arctan2(rel[2], horiz))))

    def sense(self):
        preds, fems, foods = [], [], []
        for e in self.of("predator"):
            d, bearing, elev = self._rel3(e.pos)
            closing = 0.0 if e.prev_dist is None else (e.prev_dist - d) / FRAME
            e.prev_dist = d
            moving = self.stats.time >= e.rest_until
            preds.append(dict(bearing=bearing, elev=elev, dist=d, size=PRED_SIZE, closing=closing,
                              speed=PRED_SPEED if moving else 0.0))
        for e in self.of("female"):
            d, bearing, elev = self._rel3(e.pos)
            fems.append(dict(bearing=bearing, elev=elev, dist=d, size=FEMALE_SIZE))
        for e in self.of("food"):
            d, bearing, _ = self._rel3(e.pos)
            foods.append(dict(bearing=bearing, dist=d if not self.flying else max(d, 1.0)))
        self.pred_inputs = preds
        self.brain.flying = self.flying
        self.brain.sense_3d(predators=preds, females=fems, foods=foods)

    # ------------------------------------------------------------------ motor
    def escape_vector3(self):
        """Escape direction from neurons: horizontal as in 2D; vertical = away from the elevation
        at which looming cells fired (their real retinotopic elevation)."""
        fwd, right = self.escape_vector()
        w = np.sum(self.hist, axis=0) if self.hist else self.body_counts
        lc = w[self.brain.loom]
        # elevation relative to the population average (looming cells over-represent the upper visual field)
        sel = np.sin(np.radians(self.brain.el[self.brain.loom]))
        up = -(lc @ (sel - sel.mean())) / (lc.sum() + 1.0)
        h = np.array([self.heading[0], self.heading[1]])
        horiz = fwd * h + right * np.array([-h[1], h[0]])
        return unit(np.array([horiz[0], horiz[1], 2.0 * up]))

    def _readout(self):
        m = super()._readout()
        b = self.brain
        m["power"] = float(b.rates[b.power].mean()) if len(b.power) else 0.0
        pr = b.rates[b.purs]
        sel = np.sin(np.radians(b.el[b.purs]))
        m["chase_elev"] = float((pr @ (sel - sel.mean())) / (pr.sum() + 1.0))
        m["chase_level"] = float(pr.mean())
        return m

    def circuit_levels(self):
        lv = super().circuit_levels()
        lv["flight"] = min(self.lift, 1.5) / 1.5
        return lv

    def behaviour(self):
        label, lv = super().behaviour()
        if label == "EXPLORING":
            label = "FLYING" if self.flying else "WALKING"
        if self.flying and self.lift < 0.3 and self.pos[2] > 0.1:
            label = "FALLING (no wing power)"
        return label, lv

    def takeoff(self):
        if not self.flying or self.pos[2] < 0.05:
            self.takeoff_time = self.stats.time
        self.escape_dir = self.escape_vector3()
        if self.pos[2] < 0.05:                     # from the ground the Giant Fiber jump goes up and away
            self.escape_dir = unit(np.array([*self.escape_dir[:2], max(self.escape_dir[2], 0.7)]))
        self.flying, self.landing = True, False
        self.escape_until = self.stats.time + ESCAPE_TIME
        self.stats.jumps += 1
        self.flash("GIANT FIBER FIRED - ESCAPE")
        self.log.append(dict(kind="takeoff", t=self.stats.time))

    # ------------------------------------------------------------------ tick
    def step(self):
        dt = FRAME
        self.stats.time += dt
        t = self.stats.time
        if self.decline and round(t / dt) % round(1.0 / dt) == 0:
            self.lose_random(0.004)
        self.sense()
        self.counts = self.brain.step(dt)
        self.motor = m = self._readout()
        self.hist.append(self.body_counts)
        # wing power: power-MN rate smoothed over ~0.6 s; the healthy average over the first 2 s of
        # flight defines lift = 1 (DESIGN: real power muscles are stretch-activated, MNs set their level)
        a = 1 - np.exp(-dt / LIFT_TAU)
        self.power_ema += a * (m["power"] - self.power_ema)
        if self.power_ref is None:
            if self.flying and t > 0.3:
                self.calib.append(m["power"])
            if len(self.calib) >= int(2.0 / dt):
                self.power_ref = max(float(np.mean(self.calib)), 0.5)
        self.lift = self.power_ema / self.power_ref if self.power_ref else 1.0
        launching = t - self.takeoff_time < LAUNCH_TIME
        if launching:
            self.lift = max(self.lift, 1.0)

        if m["jump"]:
            self.last_gf = t
            if t >= self.escape_until:
                self.takeoff()
        escaping = t < self.escape_until
        self.singing = m["song"] >= SONG_ON
        if self.singing:
            self.stats.song_time += dt
        if m["chase_level"] > 1.0:
            self.stats.chase_time += dt

        turn = float(np.clip(TURN_GAIN * m["turn_right"], -MAX_TURN, MAX_TURN))
        if escaping and self.escape_dir is not None:
            v = self.escape_dir * ESCAPE_SPEED
            self.heading = unit(self.escape_dir[:2]) if np.hypot(*self.escape_dir[:2]) > 0.1 else self.heading
        elif self.flying:
            self.heading = rot(self.heading, turn * dt)
            lift = min(self.lift, 1.2)
            target = 0.0 if self.landing else CRUISE_Z
            vz = HOLD_GAIN * (target - self.pos[2]) * min(lift, 1.0) - SINK * max(0.0, 1.0 - lift)
            if m["chase_level"] > 0.5:                     # female seen: climb/descend toward where she appears
                vz += PITCH_GAIN * FLY_SPEED * m["chase_elev"]
            if self.landing:
                vz = min(vz, -1.0)
            v = np.array([*(self.heading * FLY_SPEED * min(lift, 1.2)), vz])
            if self.rng.random() < LAND_URGE * dt:
                self.landing = True
        else:
            self.heading = rot(self.heading, turn * dt)
            walk = WALK_SPEED * min(m["walk"] / 10.0, 1.3)
            on_food = [e for e in self.of("food") if np.hypot(*(e.pos[:2] - self.pos[:2])) < 0.7]
            self.feeding = bool(on_food) and m["feed"] >= FEED_ON
            v = np.array([*(self.heading * (0.0 if self.feeding else walk)), 0.0])
            busy = self.feeding or self.brain.touching or m["chase_level"] > 0.5
            self.ground_time = 0.0 if busy else self.ground_time + dt
            if self.ground_time > WALK_BORED and self.neck_cut != {"L", "R"} and self.lift > 0.5:
                self.flying, self.landing, self.ground_time = True, False, 0.0
                self.takeoff_time = t
                v[2] = 2.0
            if self.feeding:
                food = on_food[0]
                food.amount -= EAT_RATE * dt
                self.stats.eaten += EAT_RATE * dt
                if food.amount <= 0:
                    self.entities.remove(food)
                    self.flash("FOOD FINISHED")
        if self.flying:
            self.feeding = False

        self.pos = self.pos + v * dt
        for k in (0, 1):                               # arena walls (DESIGN)
            if self.pos[k] < 0.5 or self.pos[k] > WORLD[k] - 0.5:
                self.heading[k] = -self.heading[k]
                self.pos[k] = np.clip(self.pos[k], 0.5, WORLD[k] - 0.5)
        self.pos[2] = min(self.pos[2], BOX_H - 0.5)            # ceiling
        if self.pos[2] <= 0.0:                         # touched down
            self.pos[2] = 0.0
            if self.flying and not escaping:
                self.flying, self.landing, self.ground_time = False, False, 0.0
                if self.lift < 0.3:
                    self.flash("CRASHED - NO WING POWER")

        self._move_entities(t, dt)

    def _move_entities(self, t, dt):
        for e in list(self.entities):
            if e.kind == "female":
                e.heading = rot(e.heading, self.rng.normal(0, 90) * dt * 3)
                e.pos[:2] = e.pos[:2] + e.heading * FEMALE_SPEED * dt
                for k in (0, 1):
                    if e.pos[k] < 0.5 or e.pos[k] > WORLD[k] - 0.5:
                        e.heading[k] = -e.heading[k]
                        e.pos[k] = np.clip(e.pos[k], 0.5, WORLD[k] - 0.5)
            elif e.kind == "predator" and t >= e.rest_until:
                to = self.pos - e.pos
                d = np.linalg.norm(to)
                e.heading = unit(to[:2]) if np.hypot(*to[:2]) > 1e-6 else e.heading
                e.pos = e.pos + unit(to) * min(PRED_SPEED * dt, d)
                e.pos = np.clip(e.pos, [0.5, 0.5, 0.3], [WORLD[0] - 0.5, WORLD[1] - 0.5, BOX_H - 0.5])
                if d < CATCH_DIST + 0.2:
                    self.stats.catches += 1
                    self.flash("CAUGHT BY PREDATOR")
                    self.log.append(dict(kind="catch", t=t))
                    e.pos, e.prev_dist, e.rest_until = e.home.copy(), None, t + PREDATOR_REST
