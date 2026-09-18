"""Fly World self-test: spawn combinations x ablations, headless. Prints behaviour metrics."""
import sys, time
import numpy as np
from flypac.flyworld import FlyWorld, WORLD_W, WORLD_H

def scenario(spawns, ablate=(), T=30.0, seeds=(0, 1, 2)):
    rows = []
    for s in seeds:
        w = FlyWorld(seed=s)
        for k in ablate:
            w.toggle_circuit(k)
        for kind, pos in spawns:
            w.spawn(kind, pos)
        fem_d, labels = [], {}
        for _ in range(int(T / 0.02)):
            w.step()
            fs = w.of("female")
            if fs:
                fem_d.append(np.hypot(*(fs[0].pos - w.fly_pos)))
            lab = w.behaviour()[0]
            labels[lab] = labels.get(lab, 0) + 0.02
        st = w.stats
        rows.append(dict(jumps=st.jumps, caught=st.catches, eaten=st.eaten, song=st.song_time,
                         near_female=(np.array(fem_d) < 2.0).mean() if fem_d else np.nan,
                         fem_dist=np.mean(fem_d) if fem_d else np.nan, labels=labels))
    agg = {k: np.nanmean([r[k] for r in rows]) for k in ["jumps", "caught", "eaten", "song", "near_female", "fem_dist"]}
    labs = {}
    for r in rows:
        for k, v in r["labels"].items():
            labs[k] = labs.get(k, 0) + v / len(rows)
    return agg, labs

C = (WORLD_W / 2, WORLD_H / 2)
FOODS = [("food", (C[0] + dx, C[1] + dy)) for dx, dy in [(4, 0), (-5, 3), (0, -5), (7, 5), (-8, -4), (3, 6)]]
TESTS = [
    ("female only", [("female", (C[0] + 6, C[1] + 3))], ()),
    ("female only | LC10a ablated", [("female", (C[0] + 6, C[1] + 3))], ("lc10a",)),
    ("female only | P1 ablated", [("female", (C[0] + 6, C[1] + 3))], ("p1",)),
    ("predator only", [("predator", (C[0] + 8, C[1]))], ()),
    ("predator only | GF ablated", [("predator", (C[0] + 8, C[1]))], ("gf",)),
    ("6 foods", FOODS, ()),
    ("6 foods | MN9 ablated", FOODS, ("mn9",)),
    ("female + 6 foods | LC10a ablated", [("female", (C[0] + 6, C[1] + 3))] + FOODS, ("lc10a",)),
    ("6 foods + predator", FOODS + [("predator", (C[0] - 9, C[1] - 6))], ()),
    ("6 foods + predator | GF ablated", FOODS + [("predator", (C[0] - 9, C[1] - 6))], ("gf",)),
    ("female + predator", [("female", (C[0] + 6, C[1] + 3)), ("predator", (C[0] - 9, C[1]))], ()),
    ("predator only | P1 ablated", [("predator", (C[0] + 8, C[1]))], ("p1",)),
    ("female + predator | GF ablated", [("female", (C[0] + 6, C[1] + 3)), ("predator", (C[0] - 9, C[1]))], ("gf",)),
]
only = sys.argv[1:]
for name, spawns, abl in TESTS:
    if only and not any(o in name for o in only):
        continue
    t0 = time.time()
    a, labs = scenario(spawns, abl)
    top = ", ".join(f"{k} {v:.0f}s" for k, v in sorted(labs.items(), key=lambda kv: -kv[1])[:3])
    print(f"{name:36s} jumps {a['jumps']:5.1f} caught {a['caught']:4.1f} eaten {a['eaten']:4.2f} song {a['song']:4.1f}s "
          f"near-female {a['near_female']:.2f} (mean d {a['fem_dist']:4.1f}) | {top}  [{time.time()-t0:.0f}s]", flush=True)
