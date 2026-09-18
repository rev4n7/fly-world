"""Where each simulated neuron sits in the brain image, for "stroke" lesions at a clicked spot.

Uses the real synapse locations in data/world/anatomy_footprints.csv (flypac.fetch_anatomy --world),
on the same pixel grid as the brain view. A neuron is destroyed by a lesion if at least `frac` of its
synapses lie inside the damaged disc (default 15%: e.g. LC10a keeps most synapses in its output area,
so a lobula stroke must count damage to its input arbour); cells merely passing through keep working.
"""
import json

import numpy as np
import pandas as pd
from scipy import sparse

from . import config as C


class BrainMap:
    def __init__(self, brain):
        meta = json.loads((C.DATA_DIR / "anatomy_meta.json").read_text())
        self.W, self.H = meta["width"], meta["height"]
        self.labels = meta["labels"]
        fp = pd.read_csv(C.DATA_DIR / "world" / "anatomy_footprints.csv")
        n = brain.neurons
        # mirrored right-ear cells: their left-ear source reflected at the midline (as in the brain view)
        main_meta = json.loads((C.DATA_DIR / "meta.json").read_text())
        mid_px = (meta["x_range"][1] - main_meta["midline_x"]) / meta["bin_voxels"]
        mir = fp[fp.bodyId.isin([-b for b in n.index if b < 0])].copy()
        mir["bodyId"] = -mir["bodyId"]
        mir["px"] = np.rint(2 * mid_px - mir["px"]).astype(int)
        fp = pd.concat([fp, mir])
        fp = fp[fp.bodyId.isin(n.index) & (fp.px >= 0) & (fp.px < self.W)]
        col = brain.idx[fp.bodyId].to_numpy()
        self.px, self.py = fp.px.to_numpy(), fp.py.to_numpy()
        self.col, self.count = col, fp["count"].to_numpy(float)
        self.total = np.bincount(col, weights=self.count, minlength=brain.N)

    def cells_in_disc(self, cx, cy, r, frac=0.15):
        """Neuron indices with >= frac of their synapses within r pixels of (cx, cy)."""
        inside = (self.px - cx) ** 2 + (self.py - cy) ** 2 <= r * r
        hit = np.bincount(self.col[inside], weights=self.count[inside], minlength=len(self.total))
        return np.flatnonzero((self.total > 0) & (hit >= frac * self.total))
