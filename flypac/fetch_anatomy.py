"""Anatomy layer: real positions for the brain view.

Run:  .venv\\Scripts\\python -m flypac.fetch_anatomy      (needs the neuPrint token)
      .venv\\Scripts\\python -m flypac.fetch_anatomy --world   (Fly World: data/world/anatomy_footprints.csv)

Writes to data/:
  anatomy_footprints.csv  bodyId, px, py, count: every synapse location of each simulated
                          neuron, binned into image pixels
  anatomy_masks.npz       filled silhouettes of real brain-region meshes (same pixel grid)
  anatomy_meta.json       grid definition + label positions of named regions

View: looking at the fly's head from BEHIND, so the fly's left is on the image's left
(dataset x grows toward the fly's left, so screen x = -dataset x). Screen y = dataset y
(dataset y points ventral). Depth (z, front-back) is flattened.
"""
import json
import time

import numpy as np
import pandas as pd
from neuprint import Client
from scipy import ndimage

from . import config as C
from .fetch import load_token

BIN = 400             # dataset voxels (8 nm) per image pixel -> 3.2 um per pixel
X_MIN, X_MAX = 0, 100_000
Y_MIN, Y_MAX = 2_000, 58_000
W = (X_MAX - X_MIN) // BIN
H = (Y_MAX - Y_MIN) // BIN

SILHOUETTES = {"brain": ["CentralBrain"],
               "optic_R": ["LO(R)", "LOP(R)", "ME(R)", "LA(R)"],
               "optic_L": ["LO(L)", "LOP(L)", "ME(L)", "LA(L)"]}
LABELS = {"LO(L)": "lobula", "AOTU(L)": "AOTU", "PVLP(R)": "PVLP", "AMMC(R)": "AMMC (hearing)",
          "GNG": "GNG", "ME(R)": "medulla", "LAL(L)": "LAL"}


def to_px(x, y):
    px = ((X_MAX - np.asarray(x, float)) / BIN).astype(int)   # mirror: fly's left on the left
    py = ((np.asarray(y, float) - Y_MIN) / BIN).astype(int)
    return px, py


def mesh_vertices(c, roi):
    raw = c.fetch_roi_mesh(roi).decode(errors="ignore")
    return np.array([l.split()[1:4] for l in raw.splitlines() if l.startswith("v ")], float)


def silhouette(verts):
    px, py = to_px(verts[:, 0], verts[:, 1])
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    m = np.zeros((H, W), bool)
    m[py[ok], px[ok]] = True
    m = ndimage.binary_closing(m, structure=np.ones((5, 5)), iterations=2)
    return ndimage.binary_fill_holes(m)


def fetch_footprints(c, ids, t0):
    parts = []
    for i in range(0, len(ids), 40):
        batch = ids[i:i + 40]
        df = c.fetch_custom(f"""
            MATCH (n:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]->(s:Synapse)
            WHERE n.bodyId IN {batch}
            RETURN n.bodyId AS bodyId, toInteger(({X_MAX} - s.location.x) / {BIN}) AS px,
                   toInteger((s.location.y - {Y_MIN}) / {BIN}) AS py, count(*) AS count""")
        parts.append(df)
        print(f"  {min(i + 40, len(ids))}/{len(ids)} neurons  ({time.time() - t0:.0f}s)")
    fp = pd.concat(parts)
    return fp[(fp.px >= 0) & (fp.px < W) & (fp.py >= 0) & (fp.py < H)]


def main():
    t0 = time.time()
    c = Client(C.SERVER, dataset=C.DATASET, token=load_token())
    meta = {"dataset": C.DATASET, "bin_voxels": BIN, "width": W, "height": H,
            "x_range": [X_MIN, X_MAX], "y_range": [Y_MIN, Y_MAX],
            "view": "from behind the head; fly's left on image left", "labels": {}, "missing_meshes": []}

    print("fetching region meshes ...")
    masks = {}
    for name, rois in SILHOUETTES.items():
        m = np.zeros((H, W), bool)
        for roi in rois:
            try:
                m |= silhouette(mesh_vertices(c, roi))
            except Exception as e:  # some ROI meshes are not served
                meta["missing_meshes"].append(roi)
                print(f"  no mesh for {roi}: {str(e)[:60]}")
        masks[name] = m
    for roi, label in LABELS.items():
        try:
            v = mesh_vertices(c, roi)
            px, py = to_px(v[:, 0].mean(), v[:, 1].mean())
            meta["labels"][label] = [int(px), int(py)]
        except Exception:
            meta["missing_meshes"].append(roi)
    np.savez_compressed(C.DATA_DIR / "anatomy_masks.npz", **masks)

    print("fetching neuron footprints (all synapse positions, binned) ...")
    ids = [int(b) for b in pd.read_csv(C.DATA_DIR / "neurons.csv").bodyId]
    fp = fetch_footprints(c, ids, t0)
    fp.to_csv(C.DATA_DIR / "anatomy_footprints.csv", index=False)

    meta["neurons_with_footprint"] = int(fp.bodyId.nunique())
    meta["synapses_placed"] = int(fp["count"].sum())
    meta["seconds"] = round(time.time() - t0, 1)
    (C.DATA_DIR / "anatomy_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ["neurons_with_footprint", "synapses_placed", "labels",
                                           "missing_meshes", "seconds"]}, indent=2))


if __name__ == "__main__":
    main()
