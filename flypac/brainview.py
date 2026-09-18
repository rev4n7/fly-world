"""Live anatomical brain view: every simulated neuron glows where it really sits.

Positions come from data/anatomy_* (flypac.fetch_anatomy): each neuron's real synapse
locations from male-cns:v1.0, binned to pixels, and silhouettes of real region meshes.
View is from behind the head, so the fly's left is on the left. Mirrored right-ear
cells (negative bodyIds) use the mirror image of their left-ear source.
"""
import json
from collections import deque

import numpy as np
import pandas as pd
from scipy import ndimage, sparse

from . import config as C

GROUP_COLORS = {
    "sense_looming": (255, 120, 40),
    "sense_pursuit": (90, 230, 120),
    "sense_hearing": (110, 160, 255),
    "out_escape": (255, 245, 140),
    "out_steer": (120, 255, 200),
    "middle_steer": (120, 255, 200),
    "middle_gf": (255, 190, 90),
    "middle_feedback+": (255, 190, 90),
    "middle_feedback-": (100, 150, 255),
    # Fly World circuits
    "sense_touch": (255, 120, 200), "relay_vab3": (255, 120, 200), "court_p1": (255, 90, 170),
    "middle_court": (255, 150, 210), "out_song": (255, 60, 150),
    "sense_smell": (190, 150, 255), "smell_pn": (190, 150, 255),
    "sense_taste": (255, 200, 80), "middle_taste": (255, 210, 110), "out_feed": (255, 230, 60),
}
DEFAULT_COLOR = (200, 200, 200)
RATE_REF = 30.0     # Hz that counts as "fully lit" for one neuron
GLOW_GAIN = 0.25    # tone-mapping strength (higher saturates the centre to white)
WEIGHT_POW = 0.7    # footprint softening: 1 = only dense synapse clusters glow, 0.5 = whole arbor
WIRING_DIM = 0.10   # brightness of the static map of all simulated neurons
# display-only: the 590 feedback neurons overlap everything, so they are drawn dimmer
GROUP_BRIGHTNESS = {"middle_feedback+": 0.45, "middle_feedback-": 0.45}
HISTORY = 300       # frames of trace (6 s at 50 fps)


class BrainView:
    def __init__(self, brain, pygame, scale=2.1, footprints=None):
        self.pg, self.brain, self.scale = pygame, brain, scale
        meta = json.loads((C.DATA_DIR / "anatomy_meta.json").read_text())
        self.W, self.H = meta["width"], meta["height"]
        self.labels = meta["labels"]
        masks = np.load(C.DATA_DIR / "anatomy_masks.npz")
        fp = pd.read_csv(footprints or C.DATA_DIR / "anatomy_footprints.csv")
        n = brain.neurons

        # mirrored right-ear cells reuse their left-ear source footprint, reflected at the midline
        main_meta = json.loads((C.DATA_DIR / "meta.json").read_text())
        mid_px = (meta["x_range"][1] - main_meta["midline_x"]) / meta["bin_voxels"]
        mirrored_ids = [b for b in n.index if b < 0]
        mir = fp[fp.bodyId.isin([-b for b in mirrored_ids])].copy()
        mir["bodyId"] = -mir["bodyId"]
        mir["px"] = np.rint(2 * mid_px - mir["px"]).astype(int)
        fp = pd.concat([fp, mir])
        fp = fp[fp.bodyId.isin(n.index) & (fp.px >= 0) & (fp.px < self.W)]

        col = brain.idx[fp.bodyId].to_numpy()
        pix = (fp.py * self.W + fp.px).to_numpy()
        weight = fp["count"].to_numpy(float)
        per_neuron_max = np.zeros(brain.N)
        np.maximum.at(per_neuron_max, col, weight)
        weight = (weight / per_neuron_max[col]) ** WEIGHT_POW

        roles = n.role.to_numpy().astype(object)
        sign = n.sign.to_numpy()
        keys = np.where(roles == "middle_feedback", np.where(sign < 0, "middle_feedback-", "middle_feedback+"), roles)
        rgb = np.array([GROUP_COLORS.get(k, DEFAULT_COLOR) for k in keys], float) / 255.0
        rgb *= np.array([GROUP_BRIGHTNESS.get(k, 1.0) for k in keys])[:, None]
        shape = (self.H * self.W, brain.N)
        self.F = [sparse.csr_matrix((weight * rgb[col, ch], (pix, col)), shape=shape) for ch in range(3)]
        n_pix = self.H * self.W
        self.F_all = sparse.vstack(self.F).tocsr()   # one multiply for all three colour channels
        self.F_mono = sparse.csr_matrix((weight, (pix, col)), shape=shape)   # footprint only, for overlays

        # static base: real silhouettes + faint map of all simulated neurons
        base = np.zeros((self.H, self.W, 3))
        for name in ["optic_L", "optic_R", "brain"]:
            m = masks[name]
            edge = m & ~ndimage.binary_erosion(m, iterations=1)
            base[m] = np.maximum(base[m], [0.10, 0.11, 0.17])
            base[edge] = [0.30, 0.33, 0.48]
        wiring = np.asarray(sparse.csr_matrix((weight, (pix, col)), shape=shape).sum(axis=1)).reshape(self.H, self.W)
        wiring = 1 - np.exp(-wiring / 6.0)
        base += wiring[..., None] * np.array([1.0, 1.0, 1.375]) * WIRING_DIM
        self.base = self._surface(np.clip(base, 0, 1))

        # anchor for the Giant Fiber callout: weighted centre of both GF footprints in the brain
        gf = [brain.out["DNp01_L"], brain.out["DNp01_R"]]
        sel = np.isin(col, gf)
        self.gf_xy = (float(np.average(fp.px.to_numpy()[sel], weights=weight[sel])),
                      float(np.average(fp.py.to_numpy()[sel], weights=weight[sel])))
        self.gf_flash = 0.0
        self.trace = {k: deque([0.0] * HISTORY, maxlen=HISTORY) for k in ["loom", "pursuit", "hear", "gf"]}
        self.caption, self.caption_age = [], 0.0

    def _surface(self, rgb01):
        arr = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8).transpose(1, 0, 2)
        surf = self.pg.surfarray.make_surface(arr)
        return self.pg.transform.smoothscale(surf, (int(self.W * self.scale), int(self.H * self.scale)))

    # ------------------------------------------------------------------ per frame
    def update(self, counts, world):
        b = self.brain
        act = np.clip(b.rates / RATE_REF, 0, 2.0)
        img = (self.F_all @ act).reshape(3, self.H, self.W).transpose(1, 2, 0)
        self.glow = self._surface(1 - np.exp(-GLOW_GAIN * img))
        blur = self.pg.transform.smoothscale(self.pg.transform.smoothscale(self.glow, (self.W // 2, self.H // 2)),
                                             self.glow.get_size())
        self.bloom = blur

        gf_spk = int(counts[b.out["DNp01_L"]] + counts[b.out["DNp01_R"]])
        self.gf_flash = 1.0 if gf_spk else self.gf_flash * 0.9
        loom = float(b.rates[b.loom].mean())
        purs = float(b.rates[b.purs].mean())
        hear = float(b.rates[b.hear].mean())
        for k, v in [("loom", loom), ("pursuit", purs), ("hear", hear), ("gf", float(gf_spk))]:
            self.trace[k].append(v)

        lines = []
        if gf_spk or self.gf_flash > 0.3:
            lines.append(("GIANT FIBER FIRED  ->  TAKEOFF", (255, 240, 120)))
        if loom > 1.0:
            lines.append((f"Looming detectors firing ({loom:.0f} Hz avg): something is coming at me", (255, 140, 60)))
        if hear > 1.0:
            lines.append((f"Antennae feel air vibration ({hear:.0f} Hz avg)", (130, 170, 255)))
        if purs > 1.0 and not lines:
            turn = world.motor["turn_right"] if hasattr(world, "motor") else 0.0
            side = "right" if turn > 3 else "left" if turn < -3 else "ahead"
            lines.append((f"Pellet detectors firing -> steering {side}", (110, 230, 140)))
        if not lines:
            lines.append(("Quiet: wandering", (140, 145, 170)))
        self.caption = lines

    def _tag(self, surf, font, text, color, pos):
        img = font.render(text, True, color)
        box = self.pg.Surface((img.get_width() + 6, img.get_height() + 2), self.pg.SRCALPHA)
        box.fill((6, 7, 14, 170))
        surf.blit(box, (pos[0] - 3, pos[1] - 1))
        surf.blit(img, pos)

    def draw(self, surf, x, y, fonts, trace_h=140):
        """Draw title, brain image, caption and activity trace; returns the bottom y."""
        pg = self.pg
        f_big, f, f_small = fonts
        w, h = self.base.get_size()
        surf.blit(f_big.render("FLY BRAIN  (live, real neuron positions)", True, (225, 228, 240)), (x, y))
        surf.blit(f_small.render("seen from behind the head: the fly's left is on the left", True, (120, 124, 145)),
                  (x, y + f_big.get_height() + 2))
        top = y + f_big.get_height() + f_small.get_height() + 12
        surf.blit(self.base, (x, top))
        surf.blit(self.bloom, (x, top), special_flags=pg.BLEND_ADD)
        surf.blit(self.glow, (x, top), special_flags=pg.BLEND_ADD)
        for label, (px, py) in self.labels.items():
            sx, sy = x + int(px * self.scale), top + int(py * self.scale)
            pg.draw.circle(surf, (170, 175, 205), (sx, sy), 3)
            self._tag(surf, f_small, label, (170, 175, 205), (sx + 6, sy - f_small.get_height() // 2))
        if self.gf_flash > 0.05:
            gx, gy = x + int(self.gf_xy[0] * self.scale), top + int(self.gf_xy[1] * self.scale)
            r = int(12 + 30 * (1 - self.gf_flash))
            pg.draw.circle(surf, (255, 240, 120), (gx, gy), r, 3)
            self._tag(surf, f, "Giant Fiber", (255, 240, 120), (gx + r + 6, gy - 30))

        cy = top + h + 10
        line_h = f.get_height() + 4
        for text, color in self.caption[:3]:
            surf.blit(f.render(text, True, color), (x, cy))
            cy += line_h

        ty = top + h + 12 + 3 * line_h
        pg.draw.rect(surf, (22, 23, 36), pg.Rect(x, ty, w, trace_h))
        legend = [("last 6 s:", (120, 124, 145)), ("looming", (255, 140, 60)), ("pellet", (110, 230, 140)),
                  ("ears", (130, 170, 255)), ("GF spike", (255, 240, 120))]
        lx = x + 6
        for name, color in legend:
            img = f_small.render(name, True, color)
            surf.blit(img, (lx, ty + 4))
            lx += img.get_width() + 14
        plot_top = ty + f_small.get_height() + 8
        plot_h = trace_h - (plot_top - ty) - 6
        for k, color in [("loom", (255, 140, 60)), ("pursuit", (110, 230, 140)), ("hear", (130, 170, 255))]:
            vals = np.array(self.trace[k])
            pts = [(x + int(i * w / HISTORY), plot_top + plot_h - int(min(v / 20.0, 1) * plot_h))
                   for i, v in enumerate(vals)]
            pg.draw.lines(surf, color, False, pts, 2)
        for i, v in enumerate(self.trace["gf"]):
            if v:
                sx = x + int(i * w / HISTORY)
                pg.draw.line(surf, (255, 240, 120), (sx, plot_top), (sx, plot_top + plot_h), 2)
        return ty + trace_h
