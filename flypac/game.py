"""Fly-Pacman: watch a connectome-driven fly play Pac-Man.

Run:  .venv\\Scripts\\python -m flypac.game
Keys: M escape readout (circuit/eyes) | H hearing on/off | +/- ghost speed | 1-4 ghosts
      F fast-forward | W fullscreen | SPACE pause | R restart | S screenshot | Esc quit

Everything is drawn on a fixed 1920x1080 (16:9) canvas; the window is scaled to fit the screen.
"""
import argparse
import math
import os
import time

import numpy as np

from . import config as C
from .brainview import BrainView
from .world import FRAME, World

CANVAS = (1920, 1080)
TOP = 60                       # top bar height
TILE = 42                      # maze tile size -> maze 798 x 882
BRAIN_W = 598                  # middle column (live brain)
PANEL_W = 524                  # right column (eyes / ears / escape)
S = TILE / 32                  # sprite scale relative to the original 32 px tiles

WALL = (33, 60, 190)
BG = (8, 8, 16)
PANEL_BG = (16, 17, 28)
TEXT = (225, 228, 240)
DIM = (130, 134, 158)
GHOST_COLORS = [(235, 60, 60), (250, 150, 210), (70, 220, 230), (250, 170, 60)]
LOOM_COLOR = (255, 120, 40)
PURSUIT_COLOR = (90, 230, 120)
HEAR_COLOR = (120, 170, 255)
EXC, INH = (255, 170, 60), (90, 150, 255)


def lerp_arr(c0, c1, a):
    """Vectorised colour blend: one RGB row per value in `a`."""
    a = np.clip(np.asarray(a, float), 0.0, 1.0)[:, None]
    return (np.asarray(c0, float) + (np.asarray(c1, float) - np.asarray(c0, float)) * a).astype(int)


def lerp(c0, c1, a):
    a = max(0.0, min(1.0, a))
    return tuple(int(c0[i] + (c1[i] - c0[i]) * a) for i in range(3))


class Display:
    """Draw on a fixed 16:9 canvas; the GPU scales it into a window sized to fit the screen.

    Uses SDL2's renderer directly so the window size can be set (pygame.SCALED ignores it).
    Falls back to pygame.SCALED if the SDL2 video module is unavailable (e.g. headless runs).
    """

    def __init__(self, pygame, fullscreen=False, title="Fly-Pacman - connectome-driven fly (male-cns v1.0)"):
        self.pg = pygame
        self.fullscreen = False
        try:
            if os.environ.get("SDL_VIDEODRIVER") == "dummy":
                raise RuntimeError("headless")
            from pygame._sdl2.video import WINDOWPOS_CENTERED, Renderer, Texture, Window
            dw, dh = pygame.display.get_desktop_sizes()[0]
            k = min(dw * 0.96 / CANVAS[0], dh * 0.86 / CANVAS[1], 1.0)
            self.win = Window(title, size=(int(CANVAS[0] * k), int(CANVAS[1] * k)), resizable=True,
                              position=WINDOWPOS_CENTERED)
            self.ren = Renderer(self.win, accelerated=1, vsync=False)
            self.ren.logical_size = CANVAS                      # letterboxed GPU scaling
            self.tex = Texture(self.ren, CANVAS, streaming=True)
            self.tex.blend_mode = 0
            # same pixel layout as the texture, so uploading each frame is a plain copy
            self.canvas = pygame.Surface(CANVAS, pygame.SRCALPHA, 32)
            self.mode = "sdl2"
        except Exception:
            self.canvas = pygame.display.set_mode(CANVAS, pygame.SCALED | pygame.RESIZABLE)
            pygame.display.set_caption(title)
            self.mode = "scaled"
        if fullscreen:
            self.toggle_fullscreen()

    def present(self):
        if self.mode == "sdl2":
            self.tex.update(self.canvas)
            self.ren.draw_color = (0, 0, 0, 255)
            self.ren.clear()
            self.tex.draw()
            self.ren.present()
        else:
            self.pg.display.flip()

    def toggle_fullscreen(self):
        if self.mode == "sdl2":
            if self.fullscreen:
                self.win.set_windowed()
            else:
                self.win.set_fullscreen(desktop=True)
            self.fullscreen = not self.fullscreen
        else:
            self.pg.display.toggle_fullscreen()


class Renderer:
    def __init__(self, world, pygame, screen):
        self.pg, self.w, self.screen = pygame, world, screen
        m = world.maze
        self.maze_w, self.maze_h = m.w * TILE, m.h * TILE
        self.brain_x = self.maze_w
        self.panel_x = self.maze_w + BRAIN_W
        self.size = CANVAS
        f = pygame.font.SysFont
        mono = "consolas,dejavusansmono,monospace"
        self.f_title = f(mono, 30, bold=True)
        self.f_big = f(mono, 22, bold=True)
        self.f = f(mono, 18)
        self.f_small = f(mono, 15)
        self.visual_heading = math.atan2(world.heading[1], world.heading[0])
        self.gf_glow = [0.0, 0.0]
        self.frame = 0
        self.set_world(world)

    def set_world(self, world):
        self.w = world
        self.brainview = BrainView(world.brain, self.pg, scale=2.2)
        self._prep_brain_layout()
        self.walls = self._draw_walls()

    # ------------------------------------------------------------------ static layers
    def _draw_walls(self):
        pg, m = self.pg, self.w.maze
        surf = pg.Surface((self.maze_w, self.maze_h))
        surf.fill(BG)
        t = max(3, int(3 * S))
        for y in range(m.h):
            for x in range(m.w):
                c = m.grid[y][x]
                r = pg.Rect(x * TILE, y * TILE, TILE, TILE)
                if c == "#":
                    pg.draw.rect(surf, (14, 22, 70), r)
                    for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < m.w and 0 <= ny < m.h and m.grid[ny][nx] != "#":
                            a = (r.left if dx <= 0 else r.right - t, r.top if dy <= 0 else r.bottom - t)
                            size = (TILE if dy else t, t if dy else TILE)
                            pg.draw.rect(surf, WALL, pg.Rect(a, size))
                elif c == "-":
                    pg.draw.rect(surf, (230, 150, 200), pg.Rect(x * TILE, y * TILE + TILE // 2 - 3, TILE, 5))
        return surf

    def _prep_brain_layout(self):
        b = self.w.brain
        n = b.neurons
        self.eye_r = 168
        self.eye_c = (self.panel_x + 200, TOP + 250)
        ring = {"LC4": 70, "LPLC2": 96, "LC6": 122, "LC10a": 152}
        types = n.type.to_numpy()
        self.vis_idx = np.concatenate([b.loom, b.purs])
        rad = np.array([ring[types[i]] + b.el[i] * 0.14 for i in self.vis_idx])
        ang = np.radians(b.body_az[self.vis_idx])
        self.vis_xy = np.stack([self.eye_c[0] + rad * np.sin(ang), self.eye_c[1] - rad * np.cos(ang)], 1).astype(int)
        self.vis_is_loom = np.isin(self.vis_idx, b.loom)
        roles = n.role.to_numpy()
        self.mid_idx = np.flatnonzero(np.isin(roles, ["middle_gf", "middle_feedback", "middle_steer"]))
        self.mid_sign = n.sign.to_numpy()[self.mid_idx]
        order = np.argsort(-self.mid_sign, kind="stable")
        self.mid_idx, self.mid_sign = self.mid_idx[order], self.mid_sign[order]

    # ------------------------------------------------------------------ maze column
    def to_px(self, pos):
        return (int(pos[0] * TILE + TILE / 2), int(pos[1] * TILE + TILE / 2 + TOP))

    def draw_maze(self):
        pg, w, s = self.pg, self.w, self.screen
        s.blit(self.walls, (0, TOP))
        for p in w.maze.pellets:
            pg.draw.circle(s, (245, 215, 170), self.to_px(p), max(3, int(3 * S)))
        self.draw_fly()
        for i, g in enumerate(w.ghosts):
            self.draw_ghost(g, GHOST_COLORS[i % 4])
        t = w.stats.time
        if w.event and t - w.event_time < 0.8:
            col = (255, 80, 80) if w.event == "CAUGHT" else (255, 230, 90)
            txt = self.f_big.render(w.event, True, col)
            box = txt.get_rect(center=(self.maze_w // 2, TOP + self.maze_h // 2)).inflate(16, 8)
            pg.draw.rect(s, (0, 0, 0), box)
            s.blit(txt, txt.get_rect(center=box.center))

        # explainer for viewers, under the maze
        y = TOP + self.maze_h + 12
        s.blit(self.f_big.render("WHAT YOU'RE WATCHING", True, (255, 220, 80)), (14, y))
        lines = [
            "A fruit fly plays Pac-Man by itself. Nobody controls it.",
            "Its brain: 1,887 spiking neurons wired with 1.15M real synapses",
            "from the male fruit fly connectome (MaleCNS v1.0).",
            "No 'if ghost near, run' rule: it escapes only when its",
            "simulated Giant Fiber neuron fires.",
        ]
        for k, line in enumerate(lines):
            s.blit(self.f_small.render(line, True, DIM), (14, y + 30 + k * 18))

    def draw_ghost(self, g, col):
        pg = self.pg
        cx, cy = self.to_px(g.pos)
        r = int(TILE * 0.45)
        pg.draw.circle(self.screen, col, (cx, cy - 2), r)
        pg.draw.rect(self.screen, col, pg.Rect(cx - r, cy - 2, 2 * r, r))
        for k in range(3):
            pg.draw.circle(self.screen, BG, (cx - r + 5 + k * (2 * r - 10) // 2, cy + r), int(4 * S))
        dx, dy = g.dir
        for ex in (-int(6 * S), int(6 * S)):
            pg.draw.circle(self.screen, (255, 255, 255), (cx + ex, cy - int(5 * S)), int(5 * S))
            pg.draw.circle(self.screen, (30, 40, 140), (cx + ex + 2 * dx, cy - int(5 * S) + 2 * dy), int(2.5 * S))

    def draw_fly(self):
        pg, w, s = self.pg, self.w, self.screen
        target = math.atan2(w.heading[1], w.heading[0])
        diff = (target - self.visual_heading + math.pi) % (2 * math.pi) - math.pi
        self.visual_heading += diff * 0.35
        a = self.visual_heading
        cx, cy = self.to_px(w.fly.pos)
        fwd = np.array([math.cos(a), math.sin(a)])
        rt = np.array([-fwd[1], fwd[0]])

        def P(f, r):
            v = np.array([cx, cy]) + (fwd * f + rt * r) * S
            return int(v[0]), int(v[1])

        # blind spot wedge behind the fly (eyes cover about +/-160 deg)
        wedge = [P(0, 0)]
        for k in np.linspace(-20, 20, 7):
            ang = math.radians(180 + k)
            wedge.append(P(56 * math.cos(ang), 56 * math.sin(ang)))
        xs, ys = [p[0] for p in wedge], [p[1] for p in wedge]
        ox, oy = min(xs), min(ys)
        overlay = pg.Surface((max(xs) - ox + 1, max(ys) - oy + 1), pg.SRCALPHA)
        pg.draw.polygon(overlay, (120, 120, 160, 26), [(px - ox, py - oy) for px, py in wedge])
        s.blit(overlay, (ox, oy))

        if w.stats.time < w.escape_until:
            pg.draw.circle(s, (255, 230, 90), (cx, cy), int(17 * S), 3)
        for side in (-1, 1):
            wing = [P(1, side * 2), P(-11, side * 9), P(-15, side * 5), P(-6, side * 1)]
            pg.draw.polygon(s, (200, 215, 235), wing)
        body = [P(9, 0), P(5, 4), P(-9, 3), P(-12, 0), P(-9, -3), P(5, -4)]
        pg.draw.polygon(s, (120, 80, 40), body)
        pg.draw.circle(s, (210, 40, 30), P(7, 3), int(3 * S))
        pg.draw.circle(s, (210, 40, 30), P(7, -3), int(3 * S))

    # ------------------------------------------------------------------ brain column
    def draw_brain(self):
        pg, s, b = self.pg, self.screen, self.w.brain
        x0 = self.brain_x
        pg.draw.rect(s, (11, 12, 20), pg.Rect(x0, TOP, BRAIN_W, CANVAS[1] - TOP))
        pg.draw.line(s, (40, 42, 60), (x0, TOP), (x0, CANVAS[1]), 1)
        y = self.brainview.draw(s, x0 + 24, TOP + 14, (self.f_big, self.f, self.f_small), trace_h=130) + 16

        s.blit(self.f_small.render("COLOURS", True, TEXT), (x0 + 24, y))
        items = [("looming detectors", (255, 120, 40)), ("pellet detectors", (90, 230, 120)),
                 ("ear neurons", (110, 160, 255)), ("escape (Giant Fiber..)", (255, 245, 140)),
                 ("steering (DNa02..)", (120, 255, 200)), ("middle: excitatory", (255, 190, 90)),
                 ("middle: inhibitory", (100, 150, 255))]
        for k, (name, col) in enumerate(items):
            cx, cy = x0 + 24 + (k % 2) * 280, y + 22 + (k // 2) * 20
            pg.draw.rect(s, col, pg.Rect(cx, cy + 4, 11, 11))
            s.blit(self.f_small.render(name, True, DIM), (cx + 18, cy))

        y += 22 + 4 * 20 + 14
        s.blit(self.f_small.render(f"MIDDLE LAYER: {len(self.mid_idx)} neurons (orange excit. / blue inhib.)",
                                   True, DIM), (x0 + 24, y))
        ncol = 78
        mr = b.rates[self.mid_idx] / 60.0
        mcols = np.where((self.mid_sign < 0)[:, None], lerp_arr((34, 36, 52), INH, mr), lerp_arr((34, 36, 52), EXC, mr))
        gy = y + 22
        for k, col in enumerate(mcols.tolist()):
            s.fill(col, (x0 + 24 + (k % ncol) * 7, gy + (k // ncol) * 7, 6, 6))
        s.blit(self.f_small.render("Glow = firing rate (feedback cells dimmer).", True, DIM), (x0 + 24, CANVAS[1] - 50))
        s.blit(self.f_small.render("Positions: 12.7M real synapse locations.", True, DIM), (x0 + 24, CANVAS[1] - 30))

    # ------------------------------------------------------------------ right column
    def bar(self, x, y, label, value, vmax, col, width=140, label_w=112):
        pg, s = self.pg, self.screen
        s.blit(self.f_small.render(label, True, DIM), (x, y))
        pg.draw.rect(s, (40, 42, 60), pg.Rect(x + label_w, y + 3, width, 12))
        pg.draw.rect(s, col, pg.Rect(x + label_w, y + 3, int(width * min(value / vmax, 1.0)), 12))
        s.blit(self.f_small.render(f"{value:4.0f} Hz", True, DIM), (x + label_w + width + 8, y))

    def draw_panel(self):
        pg, s, w, b = self.pg, self.screen, self.w, self.w.brain
        x0 = self.panel_x
        pg.draw.rect(s, PANEL_BG, pg.Rect(x0, TOP, PANEL_W, CANVAS[1] - TOP))
        rates, counts, o = b.rates, w.counts, b.out

        s.blit(self.f_big.render("EYES", True, TEXT), (x0 + 20, TOP + 14))
        s.blit(self.f_small.render("dot = one real neuron, placed where it looks", True, DIM), (x0 + 90, TOP + 19))
        cx, cy = self.eye_c
        R = self.eye_r
        pg.draw.circle(s, (30, 32, 48), (cx, cy), R)
        blind = [(cx, cy)] + [(int(cx + R * math.sin(math.radians(180 + k))), int(cy - R * math.cos(math.radians(180 + k))))
                              for k in np.linspace(-20, 20, 9)]
        pg.draw.polygon(s, (22, 22, 34), blind)
        rv = rates[self.vis_idx]
        cols = np.where(self.vis_is_loom[:, None], lerp_arr((70, 55, 50), LOOM_COLOR, rv / 40.0),
                        lerp_arr((45, 70, 55), PURSUIT_COLOR, rv / 40.0))
        for (px, py), col, r in zip(self.vis_xy, cols.tolist(), rv):
            if r < 1:
                s.set_at((px, py), col)
            else:
                pg.draw.circle(s, col, (px, py), 3)
        for gin, g_col in zip(w.ghost_inputs, GHOST_COLORS):
            if gin["visible"] and gin["dist"] < 12:
                a = math.radians(gin["bearing"])
                pg.draw.circle(s, g_col, (int(cx + (R + 8) * math.sin(a)), int(cy - (R + 8) * math.cos(a))), 6)
        pg.draw.polygon(s, (150, 110, 60), [(cx, cy - 15), (cx + 8, cy + 10), (cx - 8, cy + 10)])
        front = self.f_small.render("front", True, DIM)
        s.blit(front, (cx - front.get_width() // 2, cy - R - 26))
        blind_t = self.f_small.render("blind spot", True, DIM)
        s.blit(blind_t, (cx - blind_t.get_width() // 2, cy + R + 8))
        ly = TOP + 110
        for name, col in [("LC4 (inner)", LOOM_COLOR), ("LPLC2", LOOM_COLOR), ("LC6", LOOM_COLOR),
                          ("LC10a (outer)", PURSUIT_COLOR), ("", DIM), ("orange:", DIM), (" looming", DIM),
                          ("green:", DIM), (" pellets", DIM), ("rim dots:", DIM), (" ghosts", DIM)]:
            s.blit(self.f_small.render(name, True, col), (x0 + 392, ly))
            ly += 19

        y = TOP + 470
        s.blit(self.f_big.render("EARS" + ("" if w.hearing else " (off)"), True, TEXT), (x0 + 20, y))
        hl = rates[b.hear[~b.hear_right]].mean()
        hr = rates[b.hear[b.hear_right]].mean()
        self.bar(x0 + 150, y + 2, "JO left", hl, 40, HEAR_COLOR)
        self.bar(x0 + 150, y + 24, "JO right*", hr, 40, HEAR_COLOR)

        y = TOP + 540
        s.blit(self.f_big.render("ESCAPE", True, TEXT), (x0 + 20, y))
        for k, side in enumerate("LR"):
            spiked = counts[o[f"DNp01_{side}"]] > 0
            self.gf_glow[k] = 1.0 if spiked else self.gf_glow[k] * 0.85
            gx = x0 + 50 + k * 76
            pg.draw.circle(s, lerp((50, 50, 70), (255, 240, 120), self.gf_glow[k]), (gx, y + 70), 28)
            t = self.f_small.render(f"GF {side}", True, BG if self.gf_glow[k] > 0.5 else TEXT)
            s.blit(t, (gx - t.get_width() // 2, y + 62))
        s.blit(self.f_small.render("Giant Fibers", True, DIM), (x0 + 38, y + 104))
        bx = x0 + 180
        self.bar(bx, y + 4, "DNp02 front", rates[o["DNp02_L"]] + rates[o["DNp02_R"]], 150, (255, 110, 90))
        self.bar(bx, y + 26, "DNp11 back", rates[o["DNp11_L"]] + rates[o["DNp11_R"]], 150, (255, 110, 90))
        self.bar(bx, y + 48, "DNp04 low", rates[o["DNp04_L"]] + rates[o["DNp04_R"]], 150, (255, 110, 90))
        self.bar(bx, y + 70, "TTMn jump", rates[o["TTMn_L"]] + rates[o["TTMn_R"]], 150, (255, 110, 90))

        y = TOP + 680
        s.blit(self.f_big.render("STEER", True, TEXT), (x0 + 20, y))
        self.bar(bx, y + 4, "DNa02 L", rates[o["DNa02_L"]], 80, PURSUIT_COLOR)
        self.bar(bx, y + 26, "DNa02 R", rates[o["DNa02_R"]], 80, PURSUIT_COLOR)

        y = TOP + 760
        s.blit(self.f_big.render("SETTINGS", True, TEXT), (x0 + 20, y))
        for k, line in enumerate([f"escape readout [M]: {w.readout}", f"ears [H]: {'on' if w.hearing else 'off'}",
                                  f"ghosts [1-4]: {w.n_ghosts}   speed [+/-]: {w.ghost_speed:g}",
                                  "pause [Space]  fast [F]  fullscreen [W]"]):
            s.blit(self.f_small.render(line, True, DIM), (x0 + 20, y + 30 + k * 20))

        y = CANVAS[1] - 96
        for k, line in enumerate(["Real wiring: male-cns v1.0 via neuPrint", f"{w.n_neurons:,} neurons, "
                                  f"{w.n_synapses / 1e6:.2f}M synapses",
                                  "Design choices: eye geometry, gains, motor map,",
                                  "ghost sound.  *right ear mirrored from left"]):
            s.blit(self.f_small.render(line, True, DIM), (x0 + 20, y + k * 20))

    # ------------------------------------------------------------------ top bar
    def draw_top(self):
        pg, s, w = self.pg, self.screen, self.w
        pg.draw.rect(s, (0, 0, 0), pg.Rect(0, 0, CANVAS[0], TOP))
        st = w.stats
        s.blit(self.f_title.render("FLY-PACMAN", True, (255, 220, 80)), (14, 12))
        s.blit(self.f.render(f"t {st.time:5.0f}s  pellets {st.pellets}  jumps {st.jumps}  "
                             f"survived {st.escapes}  caught {st.catches}", True, TEXT), (220, 20))
        escaping = st.time < w.escape_until or w.pending_takeoff is not None
        state = ("CAUGHT", (255, 80, 80)) if w.pause > 0 else \
                ("ESCAPE! (Giant Fiber fired)", (255, 230, 90)) if escaping else ("FORAGING", (110, 230, 130))
        s.blit(self.f_title.render(state[0], True, state[1]), (self.brain_x + 24, 12))
        sub = self.f_small.render("a fruit fly brain from the real connectome, playing Pac-Man", True, DIM)
        s.blit(sub, (CANVAS[0] - sub.get_width() - 20, 22))

    def draw(self):
        self.screen.fill(BG)
        self.frame += 1
        if self.frame % 2 or not hasattr(self.brainview, "glow"):   # glow image at 25 fps
            self.brainview.update(self.w.counts, self.w)
        self.draw_top()
        self.draw_maze()
        self.draw_brain()
        self.draw_panel()


def enable_dpi_awareness():
    """Ask Windows for real pixels so 125%/150% display scaling doesn't blow the window up."""
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ghosts", type=int, default=4)
    ap.add_argument("--ghost-speed", type=float, default=2.0, help="tiles/s (fly walks 3.5)")
    ap.add_argument("--readout", choices=["circuit", "eyes"], default="eyes")
    ap.add_argument("--no-hearing", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--screenshot-at", type=float, nargs="*", default=[],
                    help="sim seconds at which to save a PNG to data/ (use with --quit-after)")
    ap.add_argument("--quit-after", type=float, default=None, help="sim seconds, then exit")
    ap.add_argument("--record", default=None, help="folder to save every frame as PNG")
    ap.add_argument("--shot-on-jump", type=int, default=0, help="save a PNG just after each of the first N takeoffs")
    args = ap.parse_args()

    # if the window ever dies without a Python error, the low-level cause lands here
    import faulthandler
    crash_log = open(C.DATA_DIR / "crash_log.txt", "w")
    faulthandler.enable(crash_log)

    enable_dpi_awareness()
    import pygame
    pygame.init()
    display = Display(pygame, fullscreen=args.fullscreen)
    screen = display.canvas
    world = World(n_ghosts=args.ghosts, ghost_speed=args.ghost_speed, readout=args.readout,
                  hearing=not args.no_hearing, seed=args.seed)
    rend = Renderer(world, pygame, screen)
    clock = pygame.time.Clock()
    paused, fast = False, False
    shots = sorted(args.screenshot_at)
    frame_no = 0
    jump_shots, last_jumps, shot_due = 0, 0, None
    if args.record:
        os.makedirs(args.record, exist_ok=True)

    def restart(**kw):
        nonlocal world
        opts = dict(n_ghosts=world.n_ghosts, ghost_speed=world.ghost_speed, readout=world.readout,
                    hearing=world.hearing, seed=world.stats.rounds + args.seed)
        opts.update(kw)
        world = World(**opts)
        rend.set_world(world)

    running = True
    owed, last_tick = 0.0, time.perf_counter()
    while running:
        for ev in pygame.event.get():
            if ev.type in (pygame.QUIT, getattr(pygame, "WINDOWCLOSE", pygame.QUIT)):
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k == pygame.K_m:
                    world.readout = "eyes" if world.readout == "circuit" else "circuit"
                elif k == pygame.K_h:
                    world.set_hearing(not world.hearing)
                elif k == pygame.K_f:
                    fast = not fast
                elif k in (pygame.K_w, pygame.K_F11):
                    display.toggle_fullscreen()
                elif k == pygame.K_r:
                    restart()
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    restart(ghost_speed=round(world.ghost_speed + 0.25, 2))
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    restart(ghost_speed=max(0.5, round(world.ghost_speed - 0.25, 2)))
                elif pygame.K_1 <= k <= pygame.K_4:
                    restart(n_ghosts=k - pygame.K_0)
                elif k == pygame.K_s:
                    path = C.DATA_DIR / f"screenshot_{int(time.time())}.png"
                    pygame.image.save(rend.screen, str(path))
                    print("saved", path)
        # keep the simulation in real time even if drawing a frame takes longer than 20 ms
        now = time.perf_counter()
        owed = min(owed + (now - last_tick), 5 * FRAME)
        last_tick = now
        if args.record or args.quit_after is not None:
            owed = FRAME                     # offline runs: exactly one step per drawn frame
        if not paused:
            steps = 0
            while owed >= FRAME and steps < 3:
                for _ in range(4 if fast else 1):
                    world.step()
                owed -= FRAME
                steps += 1
        else:
            owed = 0.0
        rend.draw()
        display.present()
        if args.record:
            pygame.image.save(rend.screen, os.path.join(args.record, f"frame_{frame_no:06d}.png"))
        frame_no += 1
        while shots and world.stats.time >= shots[0]:
            path = C.DATA_DIR / f"shot_{shots.pop(0):.0f}s.png"
            pygame.image.save(rend.screen, str(path))
            print("saved", path)
        if args.shot_on_jump and world.stats.jumps != last_jumps:
            last_jumps, shot_due = world.stats.jumps, frame_no + 3
        if shot_due == frame_no and jump_shots < args.shot_on_jump:
            jump_shots += 1
            path = C.DATA_DIR / f"jump_{jump_shots}.png"
            pygame.image.save(rend.screen, str(path))
            print("saved", path)
            if jump_shots >= args.shot_on_jump and args.quit_after is None:
                running = False
        if args.quit_after is not None and world.stats.time >= args.quit_after:
            running = False
        if not args.record and args.quit_after is None:
            clock.tick(int(1 / FRAME))
    pygame.quit()
    print(world.stats)
    crash_log.close()


if __name__ == "__main__":
    main()
