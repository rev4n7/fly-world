"""Fly World: spawn food, a female and predators; ablate real neurons; watch the connectome-driven fly.

Run:  .venv\\Scripts\\python -m flypac.flyworld_game
Mouse: left click in the arena = place the selected thing (1 food, 2 female, 3 predator, 4 erase)
       right click = remove the nearest thing
       click a circuit in the side panel = remove it from the brain (click again to restore)
       INJURY LAB (bottom right): neck cuts, random cell loss, slow decline, weaker synapses/brakes;
       click the brain picture = stroke at that spot ([ and ] change the stroke size)
Keys:  1-4 tool | T type a cell type or bodyId to remove (Enter) | C restore everything (heal)
       SPACE pause | F fast-forward | R clear the arena | W fullscreen | S screenshot | Esc quit
"""
import argparse
import math
import time

import numpy as np

from . import config as C
from .brainview import BrainView
from .flyworld import CIRCUITS, FRAME, SEIZURE_FRAC, WORLD_H, WORLD_W, FlyWorld
from .injury import BrainMap
from .game import Display, enable_dpi_awareness, lerp_arr

CANVAS = (1920, 1080)
TOP = 56
UNIT = 36                                   # px per world unit -> arena 1080 x 684
ARENA = (12, TOP + 10)
PANEL_X = ARENA[0] + int(WORLD_W * UNIT) + 16
BG, PANEL_BG = (8, 8, 16), (16, 17, 28)
TEXT, DIM, FAINT = (225, 228, 240), (130, 134, 158), (60, 62, 84)
RED = (255, 80, 80)
BEHAV_COLORS = {"escape": (255, 230, 90), "courtship": (255, 110, 190), "steering": (120, 255, 200),
                "feeding": (255, 170, 70), "vision": (140, 200, 255)}
STROKE_SIZES = [10, 20, 30]           # brain-image pixels (1 px = 3.2 um)
SYN_STEPS = [1.0, 0.7, 0.4]           # "all synapses" strength
INH_STEPS = [1.0, 0.5, 0.2]           # "brakes" (inhibitory synapses) strength
LEVEL_ROWS = [("escape", "Escape (looming -> Giant Fiber)", (255, 230, 90)),
              ("chase", "Chase (LC10a -> AOTU -> DNa02)", (255, 110, 190)),
              ("song", "Song (pIP10)", (255, 60, 150)),
              ("feeding", "Eating (taste -> MN9)", (255, 170, 70)),
              ("smell", "Smell (food ORNs -> PNs)", (190, 150, 255))]
TOOLS = [("food", "1 FOOD", (255, 150, 60)), ("female", "2 FEMALE", (255, 140, 200)),
         ("predator", "3 PREDATOR", (230, 70, 70)), ("erase", "4 ERASE", (160, 160, 180))]


class Renderer:
    def __init__(self, world, pygame, screen):
        self.pg, self.w, self.s = pygame, world, screen
        f = pygame.font.SysFont
        mono = "consolas,dejavusansmono,monospace"
        self.f_title, self.f_big = f(mono, 28, bold=True), f(mono, 20, bold=True)
        self.f, self.f_small = f(mono, 17), f(mono, 14)
        self.tool = "food"
        self.typing, self.typed, self.type_msg = False, "", ""
        self.frame = 0
        self.gf_glow = 0.0
        self.circuit_rects, self.tool_rects = [], []
        self.brainview = BrainView(world.brain, pygame, scale=1.7,
                                   footprints=C.DATA_DIR / "world" / "anatomy_footprints.csv")
        self.ablated_overlay = None
        self._overlay_for = None
        self.brainmap = BrainMap(world.brain)
        self.stroke_r = 20
        self.strokes = []                 # (px, py, r) in brain-image pixels, drawn as scars
        self.injury_rects = []
        self.brain_rect = None
        self.floor = self._floor()

    # ------------------------------------------------------------------ helpers
    def to_px(self, p):
        return int(ARENA[0] + p[0] * UNIT), int(ARENA[1] + p[1] * UNIT)

    def to_world(self, xy):
        return np.array([(xy[0] - ARENA[0]) / UNIT, (xy[1] - ARENA[1]) / UNIT])

    def in_arena(self, xy):
        return 0 <= xy[0] - ARENA[0] < WORLD_W * UNIT and 0 <= xy[1] - ARENA[1] < WORLD_H * UNIT

    def text(self, font, s, col, pos):
        self.s.blit(font.render(s, True, col), pos)

    def _floor(self):
        pg = self.pg
        surf = pg.Surface((int(WORLD_W * UNIT), int(WORLD_H * UNIT)))
        surf.fill((20, 24, 20))
        for x in range(0, int(WORLD_W) + 1, 2):
            pg.draw.line(surf, (28, 34, 28), (x * UNIT, 0), (x * UNIT, WORLD_H * UNIT))
        for y in range(0, int(WORLD_H) + 1, 2):
            pg.draw.line(surf, (28, 34, 28), (0, y * UNIT), (WORLD_W * UNIT, y * UNIT))
        pg.draw.rect(surf, (70, 80, 70), surf.get_rect(), 2)
        return surf

    # ------------------------------------------------------------------ arena
    def draw_fly(self, pos, heading, body=(120, 80, 40), eyes=(210, 40, 30), scale=1.0, wing_out=False,
                 proboscis=False, blind_spot=False):
        pg, s = self.pg, self.s
        a = math.atan2(heading[1], heading[0])
        fwd = np.array([math.cos(a), math.sin(a)])
        rt = np.array([-fwd[1], fwd[0]])
        cx, cy = self.to_px(pos)
        k = scale * UNIT / 28

        def P(f, r):
            v = np.array([cx, cy]) + (fwd * f + rt * r) * k
            return int(v[0]), int(v[1])

        if blind_spot:
            wedge = [P(0, 0)] + [P(60 * math.cos(math.radians(180 + d)), 60 * math.sin(math.radians(180 + d)))
                                 for d in np.linspace(-20, 20, 7)]
            xs, ys = [p[0] for p in wedge], [p[1] for p in wedge]
            ov = pg.Surface((max(xs) - min(xs) + 1, max(ys) - min(ys) + 1), pg.SRCALPHA)
            pg.draw.polygon(ov, (150, 150, 190, 30), [(x - min(xs), y - min(ys)) for x, y in wedge])
            s.blit(ov, (min(xs), min(ys)))
        for side in (-1, 1):
            if wing_out and side == 1:     # one wing extended sideways = courtship song
                wing = [P(1, 2), P(-2, 16), P(-7, 17), P(-6, 2)]
            else:
                wing = [P(1, side * 2), P(-11, side * 9), P(-15, side * 5), P(-6, side * 1)]
            pg.draw.polygon(s, (200, 215, 235), wing)
        pg.draw.polygon(s, body, [P(9, 0), P(5, 4), P(-9, 3), P(-12, 0), P(-9, -3), P(5, -4)])
        pg.draw.circle(s, eyes, P(7, 3), max(2, int(3 * k)))
        pg.draw.circle(s, eyes, P(7, -3), max(2, int(3 * k)))
        if proboscis:
            pg.draw.line(s, (255, 220, 120), P(9, 0), P(15, 0), max(2, int(2 * k)))

    def draw_predator(self, e, resting):
        pg, s = self.pg, self.s
        cx, cy = self.to_px(e.pos)
        r = int(0.65 * UNIT)
        a = math.atan2(e.heading[1], e.heading[0])
        for k in range(4):                      # legs
            for side in (-1, 1):
                ang = a + side * (0.6 + 0.45 * k)
                pg.draw.line(s, (90, 30, 30), (cx, cy), (int(cx + 1.5 * r * math.cos(ang)), int(cy + 1.5 * r * math.sin(ang))), 3)
        pg.draw.circle(s, (70, 20, 24) if resting else (140, 30, 36), (cx, cy), r)
        for side in (-1, 1):
            ex, ey = cx + 0.45 * r * math.cos(a + side * 0.5), cy + 0.45 * r * math.sin(a + side * 0.5)
            pg.draw.circle(s, (255, 220, 60), (int(ex), int(ey)), max(3, r // 5))

    def draw_arena(self):
        pg, s, w = self.pg, self.s, self.w
        s.blit(self.floor, ARENA)
        for e in w.of("food"):
            r = max(4, int(0.35 * UNIT * (0.4 + 0.6 * e.amount)))
            cx, cy = self.to_px(e.pos)
            pg.draw.circle(s, (230, 110, 40), (cx, cy), r)
            pg.draw.circle(s, (255, 190, 90), (cx - r // 3, cy - r // 3), max(2, r // 3))
            pg.draw.line(s, (90, 160, 60), (cx, cy - r), (cx + 3, cy - r - 6), 3)
        for e in w.of("female"):
            self.draw_fly(e.pos, e.heading, body=(170, 120, 90), eyes=(200, 60, 60), scale=1.15)
        for e in w.of("predator"):
            self.draw_predator(e, w.stats.time < e.rest_until)
        escaping = w.stats.time < w.escape_until
        if escaping:
            pg.draw.circle(s, (255, 230, 90), self.to_px(w.fly_pos), int(0.8 * UNIT), 3)
        self.draw_fly(w.fly_pos, w.heading, wing_out=w.singing, proboscis=w.feeding, blind_spot=True)
        if w.singing:
            fx, fy = self.to_px(w.fly_pos)
            for k in range(2):
                ph = (w.stats.time * 1.5 + k * 0.5) % 1.0
                self.text(self.f_big, "♪", (255, 110, 190), (fx + 14 + k * 12, int(fy - 20 - 30 * ph)))
        if w.event and w.stats.time - w.event_time < 0.9:
            col = RED if "CAUGHT" in w.event else (255, 230, 90)
            img = self.f_big.render(w.event, True, col)
            box = img.get_rect(midtop=(ARENA[0] + int(WORLD_W * UNIT / 2), ARENA[1] + 10)).inflate(16, 8)
            pg.draw.rect(s, (0, 0, 0), box)
            s.blit(img, img.get_rect(center=box.center))

    def draw_below_arena(self):
        pg, s, w = self.pg, self.s, self.w
        y0 = ARENA[1] + int(WORLD_H * UNIT) + 10
        self.tool_rects = []
        x = ARENA[0]
        self.text(self.f_big, "PLACE:", TEXT, (x, y0 + 8))
        x += 90
        for key, label, col in TOOLS:
            r = pg.Rect(x, y0, 150, 38)
            sel = self.tool == key
            pg.draw.rect(s, col if sel else (30, 32, 46), r, border_radius=6)
            pg.draw.rect(s, col, r, 2, border_radius=6)
            img = self.f.render(label, True, BG if sel else col)
            s.blit(img, img.get_rect(center=r.center))
            self.tool_rects.append((r, key))
            x += 162
        self.text(self.f_small, "left click: place   right click: remove", DIM, (x + 6, y0 + 3))
        self.text(self.f_small, "R clear arena   SPACE pause   F fast", DIM, (x + 6, y0 + 21))

        # which real circuit is active right now
        y = y0 + 54
        label, lv = w.behaviour()
        self.text(self.f_big, "CIRCUIT ACTIVITY NOW", TEXT, (ARENA[0], y))
        self.text(self.f_small, "the fly's state = its most active real circuit (movement comes from neurons, not these bars)",
                  DIM, (ARENA[0] + 260, y + 4))
        for k, (key, name, col) in enumerate(LEVEL_ROWS):
            yy = y + 30 + k * 24
            self.text(self.f, name, DIM, (ARENA[0], yy))
            bx = ARENA[0] + 360
            pg.draw.rect(s, (36, 38, 54), pg.Rect(bx, yy + 4, 300, 14))
            pg.draw.rect(s, col, pg.Rect(bx, yy + 4, int(300 * lv[key]), 14))
        m = w.motor
        seiz = w.seizure_level()
        info = [f"Giant Fiber spikes: {w.stats.jumps} takeoffs", f"DNa02 steer  L {w.brain.rates[w.brain.out['DNa02_L']]:4.0f}"
                f" / R {w.brain.rates[w.brain.out['DNa02_R']]:4.0f} Hz", f"pIP10 song   {m['song']:5.0f} Hz",
                f"MN9 proboscis {m['feed']:5.0f} Hz", f"P1/pC1 mean  {m['arousal']:5.1f} Hz",
                f"cells > 80 Hz {100 * seiz:4.1f}%" + ("  SEIZURE" if seiz >= SEIZURE_FRAC else "")]
        for k, line in enumerate(info):
            self.text(self.f, line, DIM, (ARENA[0] + 700, y + 30 + k * 24))

    # ------------------------------------------------------------------ side panel
    def _ablated_overlay(self):
        key = (frozenset(self.w.ablated_circuits), frozenset(self.w.ablated_types), len(self.w.lost))
        if key != self._overlay_for:
            self._overlay_for = key
            dead = ~self.w.brain.alive
            if dead.any():
                bv = self.brainview
                img = np.asarray(bv.F_mono @ dead.astype(float)).reshape(bv.H, bv.W)
                a = 1 - np.exp(-img / 3.0)
                rgb = np.stack([a * 0.9, a * 0.05, a * 0.1], -1)
                self.ablated_overlay = bv._surface(rgb)
            else:
                self.ablated_overlay = None
        return self.ablated_overlay

    def draw_panel(self):
        pg, s, w, b = self.pg, self.s, self.w, self.w.brain
        x0 = PANEL_X
        pg.draw.rect(s, PANEL_BG, pg.Rect(x0 - 6, TOP, CANVAS[0] - x0 + 6, CANVAS[1] - TOP))
        bv = self.brainview
        self.text(self.f_big, "FLY BRAIN (live, real neuron positions)", TEXT, (x0 + 8, TOP + 8))
        self.text(self.f_small, "seen from behind; red = neurons you removed", DIM, (x0 + 8, TOP + 32))
        top = TOP + 52
        self.brain_rect = pg.Rect(x0 + 8, top, bv.base.get_width(), bv.base.get_height())
        s.blit(bv.base, (x0 + 8, top))
        s.blit(bv.bloom, (x0 + 8, top), special_flags=pg.BLEND_ADD)
        s.blit(bv.glow, (x0 + 8, top), special_flags=pg.BLEND_ADD)
        ov = self._ablated_overlay()
        if ov is not None:
            s.blit(ov, (x0 + 8, top), special_flags=pg.BLEND_ADD)
        for px, py, r in self.strokes:
            pg.draw.circle(s, (255, 70, 70), (x0 + 8 + int(px * bv.scale), top + int(py * bv.scale)),
                           int(r * bv.scale), 2)
        if self.brain_rect.collidepoint(self.mouse):
            pg.draw.circle(s, (255, 150, 150), self.mouse, int(self.stroke_r * bv.scale), 1)
            self.text(self.f_small, "click = stroke here", (255, 150, 150), (self.mouse[0] + 8, self.mouse[1] + 8))
        for label, (px, py) in bv.labels.items():
            sx, sy = x0 + 8 + int(px * bv.scale), top + int(py * bv.scale)
            pg.draw.circle(s, (170, 175, 205), (sx, sy), 3)
            self.text(self.f_small, label, (170, 175, 205), (sx + 6, sy - 8))
        spk = w.counts[b.gf].sum() > 0
        self.gf_glow = 1.0 if spk else self.gf_glow * 0.85
        gx, gy = x0 + 8 + bv.base.get_width() + 20, top + 20
        pg.draw.circle(s, (int(50 + 205 * self.gf_glow), int(50 + 190 * self.gf_glow), int(70 + 50 * self.gf_glow)),
                       (gx + 40, gy + 40), 34)
        self.text(self.f_small, "Giant", BG if self.gf_glow > 0.5 else TEXT, (gx + 20, gy + 26))
        self.text(self.f_small, "Fiber", BG if self.gf_glow > 0.5 else TEXT, (gx + 20, gy + 42))
        if w.neck_cut:
            self.text(self.f_small, "NECK CUT: " + "+".join(sorted(w.neck_cut)), RED, (gx, gy + 212))
        if not b.alive[b.gf].any():
            pg.draw.line(s, RED, (gx + 10, gy + 10), (gx + 70, gy + 70), 4)
            self.text(self.f_small, "REMOVED", RED, (gx + 12, gy + 78))
        legend = [("looming", (255, 120, 40)), ("female-chase", (90, 230, 120)), ("ears", (110, 160, 255)),
                  ("courtship P1/song", (255, 90, 170)), ("taste/eating", (255, 210, 110)), ("smell", (190, 150, 255))]
        for k, (name, col) in enumerate(legend):
            ly = gy + 96 + k * 19
            pg.draw.rect(s, col, pg.Rect(gx, ly + 3, 10, 10))
            self.text(self.f_small, name, DIM, (gx + 16, ly))

        # ablation list
        y = top + bv.base.get_height() + 14
        n_dead = int((~b.alive).sum())
        self.text(self.f_big, "BRAIN CONTROL: click a circuit to remove it", TEXT, (x0 + 8, y))
        self.text(self.f_small, f"{n_dead} of {b.N:,} neurons removed (cells + all their synapses)   C: heal all",
                  RED if n_dead else DIM, (x0 + 8, y + 22))
        y += 42
        self.circuit_rects = []
        rates = b.rates
        for key, label, behav, _ in CIRCUITS:
            idx = w.circuit_idx[key]
            dead = key in w.ablated_circuits
            r = pg.Rect(x0 + 4, y, CANVAS[0] - x0 - 16, 20)
            if r.collidepoint(self.mouse):
                pg.draw.rect(s, (30, 32, 48), r)
            col = BEHAV_COLORS[behav]
            pg.draw.rect(s, RED if dead else col, pg.Rect(x0 + 10, y + 5, 12, 12), 0 if dead else 2)
            self.text(self.f_small, f"{label}  [{len(idx)}]", RED if dead else TEXT, (x0 + 30, y + 3))
            if dead:
                self.text(self.f_small, "REMOVED", RED, (x0 + 610, y + 3))
            else:
                hz = float(rates[idx].mean()) if len(idx) else 0.0
                pg.draw.rect(s, (36, 38, 54), pg.Rect(x0 + 520, y + 6, 160, 10))
                pg.draw.rect(s, col, pg.Rect(x0 + 520, y + 6, int(160 * min(hz / 30.0, 1.0)), 10))
                self.text(self.f_small, f"{hz:5.1f} Hz", DIM, (x0 + 690, y + 3))
            self.circuit_rects.append((r, key))
            y += 21

        # free-text ablation
        y += 4
        box = pg.Rect(x0 + 8, y, 330, 26)
        self.type_box = box
        pg.draw.rect(s, (30, 32, 48), box)
        pg.draw.rect(s, (255, 230, 90) if self.typing else FAINT, box, 2)
        shown = self.typed + ("_" if self.typing and self.frame % 40 < 20 else "")
        self.text(self.f, shown or "T: type cell type / bodyId", TEXT if self.typed else DIM, (box.x + 8, box.y + 4))
        self.text(self.f_small, self.type_msg, DIM, (box.right + 12, y + 6))
        if w.ablated_types:
            self.text(self.f_small, "removed: " + ", ".join(sorted(w.ablated_types))[:40], RED, (box.right + 12, y + 20))
        self.draw_injury_lab(x0, y + 34)
        self.text(self.f_small, f"Real wiring: male-cns v1.0 (neuPrint), {b.N:,} neurons, "
                  f"{int(b.edges.weight.sum()) / 1e6:.2f}M synapses.",
                  DIM, (x0 + 8, CANVAS[1] - 20))

    def draw_injury_lab(self, x0, y):
        pg, s, w = self.pg, self.s, self.w
        self.text(self.f_big, "INJURY LAB", (255, 120, 120), (x0 + 8, y))
        self.text(self.f_small, "click the brain picture = stroke there   [ ] stroke size", DIM, (x0 + 150, y + 4))
        pct = lambda g: f"{int(round(g * 100))}%"
        rows = [
            [("neck_LR", "Neck cut: both", w.neck_cut == {"L", "R"}),
             ("neck_L", "Left half", w.neck_cut == {"L"}), ("neck_R", "Right half", w.neck_cut == {"R"})],
            [("lose10", "Lose 10% cells", False), ("lose30", "Lose 30% cells", False),
             ("decline", "Slow decline: " + ("ON" if w.decline else "off"), w.decline)],
            [("syn", "All synapses " + pct(w.syn_gain), w.syn_gain < 1),
             ("inh", "Brakes " + pct(w.inh_gain), w.inh_gain < 1),
             ("size", f"Stroke size {self.stroke_r * 3.2:.0f} um", False)],
        ]
        self.injury_rects = []
        bw, bh = 250, 28
        for k, row in enumerate(rows):
            for j, (key, label, on) in enumerate(row):
                r = pg.Rect(x0 + 8 + j * (bw + 8), y + 28 + k * (bh + 6), bw, bh)
                hover = r.collidepoint(self.mouse)
                pg.draw.rect(s, (120, 30, 36) if on else (40, 30, 40) if hover else (28, 26, 38), r, border_radius=5)
                pg.draw.rect(s, (255, 110, 110) if on else (110, 70, 80), r, 1, border_radius=5)
                img = self.f_small.render(label, True, TEXT if on else (230, 190, 195))
                s.blit(img, img.get_rect(center=r.center))
                self.injury_rects.append((r, key))

    def injury_click(self, key):
        w = self.w
        if key == "neck_LR":
            w.neck_cut = set() if w.neck_cut == {"L", "R"} else {"L", "R"}
        elif key in ("neck_L", "neck_R"):
            side = {key[-1]}
            w.neck_cut = set() if w.neck_cut == side else side
        elif key in ("lose10", "lose30"):
            k = w.lose_random(0.10 if key == "lose10" else 0.30)
            w.flash(f"{k} NEURONS DESTROYED AT RANDOM")
        elif key == "decline":
            w.decline = not w.decline
        elif key == "syn":
            w.set_gains(syn_gain=SYN_STEPS[(SYN_STEPS.index(w.syn_gain) + 1) % len(SYN_STEPS)]
                        if w.syn_gain in SYN_STEPS else 1.0)
        elif key == "inh":
            w.set_gains(inh_gain=INH_STEPS[(INH_STEPS.index(w.inh_gain) + 1) % len(INH_STEPS)]
                        if w.inh_gain in INH_STEPS else 1.0)
        elif key == "size":
            self.stroke_r = STROKE_SIZES[(STROKE_SIZES.index(self.stroke_r) + 1) % len(STROKE_SIZES)]

    def brain_click(self, pos):
        bv = self.brainview
        px, py = (pos[0] - self.brain_rect.x) / bv.scale, (pos[1] - self.brain_rect.y) / bv.scale
        k = self.w.stroke(self.brainmap.cells_in_disc(px, py, self.stroke_r))
        self.strokes.append((px, py, self.stroke_r))
        self.w.flash(f"STROKE: {k} NEURONS DESTROYED")

    # ------------------------------------------------------------------ top bar
    def draw_top(self):
        pg, s, w = self.pg, self.s, self.w
        pg.draw.rect(s, (0, 0, 0), pg.Rect(0, 0, CANVAS[0], TOP))
        self.text(self.f_title, "FLY WORLD", (255, 220, 80), (14, 12))
        label, _ = w.behaviour()
        col = RED if ("SEIZ" in label or "PARAL" in label) else (255, 230, 90) if "ESCAP" in label else (255, 170, 70) if "FEED" in label else \
            (255, 110, 190) if ("CHAS" in label or "SING" in label) else (190, 150, 255) if "SMELL" in label else (110, 230, 130)
        self.text(self.f_title, label, col, (200, 12))
        st = w.stats
        self.text(self.f, f"t {st.time:5.0f}s  takeoffs {st.jumps}  caught {st.catches}  food eaten {st.eaten:4.2f}"
                  f"  song {st.song_time:4.0f}s", TEXT, (720, 18))
        sub = self.f_small.render("a male fruit fly brain from the real connectome", True, DIM)
        s.blit(sub, (CANVAS[0] - sub.get_width() - 16, 22))

    def draw(self, mouse):
        self.mouse = mouse
        self.s.fill(BG)
        self.frame += 1
        if self.frame % 2 or not hasattr(self.brainview, "glow"):
            self.brainview.update(self.w.counts, self.w)
        self.draw_top()
        self.draw_arena()
        self.draw_below_arena()
        self.draw_panel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--demo", action="store_true", help="start with food, a female and a predator placed")
    ap.add_argument("--ablate", nargs="*", default=[], help="circuit keys to remove at start (see flyworld.CIRCUITS)")
    ap.add_argument("--neck", default="", help="cut the neck at start: L, R or LR")
    ap.add_argument("--inh", type=float, default=1.0, help="brake (inhibitory synapse) strength at start")
    ap.add_argument("--stroke", default=None, help="stroke at a labelled brain spot at start, e.g. lobula")
    ap.add_argument("--screenshot-at", type=float, nargs="*", default=[], help="sim seconds; PNG saved to data/")
    ap.add_argument("--quit-after", type=float, default=None, help="sim seconds, then exit")
    args = ap.parse_args()

    enable_dpi_awareness()
    import pygame
    pygame.init()
    display = Display(pygame, fullscreen=args.fullscreen, title="Fly World - connectome-driven fly (male-cns v1.0)")
    screen = display.canvas
    world = FlyWorld(seed=args.seed)
    for k in args.ablate:
        world.toggle_circuit(k)
    if args.demo:
        for p in [(6, 5), (24, 14), (9, 15), (22, 4)]:
            world.spawn("food", p)
        world.spawn("female", (20, 9))
        world.spawn("predator", (3, 3))
    rend = Renderer(world, pygame, screen)
    world.neck_cut = set(args.neck)
    if args.inh != 1.0:
        world.set_gains(inh_gain=args.inh)
    if args.stroke:
        px, py = rend.brainmap.labels[args.stroke]
        world.stroke(rend.brainmap.cells_in_disc(px, py, rend.stroke_r))
        rend.strokes.append((px, py, rend.stroke_r))
    clock = pygame.time.Clock()
    paused, fast = False, False
    shots = sorted(args.screenshot_at)
    owed, last = 0.0, time.perf_counter()
    running = True
    mouse = (0, 0)
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.MOUSEMOTION:
                mouse = ev.pos        # event positions are in canvas coordinates (renderer logical size)
            if ev.type in (pygame.QUIT, getattr(pygame, "WINDOWCLOSE", pygame.QUIT)):
                running = False
            elif ev.type == pygame.KEYDOWN and rend.typing:
                if ev.key == pygame.K_RETURN:
                    name = rend.typed.strip()
                    if name:
                        ok = world.toggle_type(name)
                        rend.type_msg = (("removed " if name in world.ablated_types else "restored ") + name) if ok \
                            else f"no neuron type '{name}' in the simulated set"
                    rend.typed, rend.typing = "", False
                elif ev.key == pygame.K_ESCAPE:
                    rend.typed, rend.typing = "", False
                elif ev.key == pygame.K_BACKSPACE:
                    rend.typed = rend.typed[:-1]
                elif ev.unicode and ev.unicode.isprintable() and len(rend.typed) < 24:
                    rend.typed += ev.unicode
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k == pygame.K_f:
                    fast = not fast
                elif k == pygame.K_t:
                    rend.typing, rend.typed = True, ""
                elif k == pygame.K_c:
                    world.clear_ablation()
                    rend.strokes.clear()
                elif k in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                    i = STROKE_SIZES.index(rend.stroke_r) + (1 if k == pygame.K_RIGHTBRACKET else -1)
                    rend.stroke_r = STROKE_SIZES[max(0, min(i, len(STROKE_SIZES) - 1))]
                elif k == pygame.K_r:
                    world.entities.clear()
                elif k in (pygame.K_w, pygame.K_F11):
                    display.toggle_fullscreen()
                elif pygame.K_1 <= k <= pygame.K_4:
                    rend.tool = TOOLS[k - pygame.K_1][0]
                elif k == pygame.K_s:
                    path = C.DATA_DIR / f"screenshot_world_{int(time.time())}.png"
                    pygame.image.save(screen, str(path))
                    print("saved", path)
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                pos = ev.pos
                if rend.in_arena(pos):
                    wp = rend.to_world(pos)
                    if ev.button == 3 or rend.tool == "erase":
                        world.remove_near(wp)
                    elif ev.button == 1:
                        world.spawn(rend.tool, wp)
                elif ev.button == 1 and rend.brain_rect and rend.brain_rect.collidepoint(pos):
                    rend.brain_click(pos)
                elif ev.button == 1:
                    for r, key in rend.injury_rects:
                        if r.collidepoint(pos):
                            rend.injury_click(key)
                    for r, key in rend.tool_rects:
                        if r.collidepoint(pos):
                            rend.tool = key
                    for r, key in rend.circuit_rects:
                        if r.collidepoint(pos):
                            world.toggle_circuit(key)
                    if getattr(rend, "type_box", None) and rend.type_box.collidepoint(pos):
                        rend.typing, rend.typed = True, ""
        now = time.perf_counter()
        owed = min(owed + now - last, 5 * FRAME)
        last = now
        if args.quit_after is not None:
            owed = FRAME
        if not paused:
            steps = 0
            while owed >= FRAME and steps < 3:
                for _ in range(4 if fast else 1):
                    world.step()
                owed -= FRAME
                steps += 1
        else:
            owed = 0.0
        rend.draw(mouse)
        display.present()
        while shots and world.stats.time >= shots[0]:
            path = C.DATA_DIR / f"world_shot_{shots.pop(0):.0f}s.png"
            pygame.image.save(screen, str(path))
            print("saved", path)
        if args.quit_after is not None and world.stats.time >= args.quit_after:
            running = False
        if args.quit_after is None:
            clock.tick(int(1 / FRAME))
    pygame.quit()
    print(world.stats)


if __name__ == "__main__":
    main()
