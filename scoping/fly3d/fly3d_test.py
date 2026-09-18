"""Fly 3D self-test: flight, injuries and threats in 3D (headless)."""
import numpy as np
from flypac.fly3d import Fly3D, WORLD

def run(label, setup=lambda w: None, spawns=(), T=15.0, seeds=(0, 1)):
    rows = []
    for s in seeds:
        w = Fly3D(seed=s)
        for k, p in spawns:
            w.spawn(k, p)
        z, lifts, fem, esc_dz = [], [], [], []
        for i in range(int(T / 0.02)):
            if i == 150:
                setup(w)                   # injure after 3 s of healthy flight (wing power calibrated)
            w.step()
            z.append(w.pos[2]); lifts.append(w.lift)
            if w.of("female"):
                fem.append(np.linalg.norm(w.of("female")[0].pos - w.pos) < 2.5)
            if w.stats.time < w.escape_until and w.escape_dir is not None:
                esc_dz.append(w.escape_dir[2])
        rows.append([np.mean(z[150:]), np.mean(z[-50:]), np.mean(lifts[150:]), w.stats.jumps, w.stats.catches,
                     np.mean(fem) if fem else np.nan, np.mean(esc_dz) if esc_dz else np.nan, w.stats.eaten])
    m = np.nanmean(rows, 0)
    print(f"{label:40s} height avg {m[0]:4.2f} end {m[1]:4.2f} | wing power {m[2]:4.2f} | escapes {m[3]:4.1f} caught {m[4]:3.1f} "
          f"| near female {m[5]:.2f} | escape up/down {m[6]:+.2f} | eaten {m[7]:.2f}", flush=True)

C = (WORLD[0] / 2, WORLD[1] / 2)
DNG02 = ["DNg02_a", "DNg02_b", "DNg02_c", "DNg02_d", "DNg02_e", "DNg02_f", "DNg02_g"]
TESTS = [
    ("healthy flight, empty", lambda w: None, ()),
    ("neck cut both (after 3 s)", lambda w: w.set_neck("LR"), ()),
    ("flight command cells DNg02 removed", lambda w: [w.toggle_type(t) for t in DNG02], ()),
    ("30% random cell loss", lambda w: w.lose_random(0.3), ()),
    ("predator diving from ABOVE", lambda w: None, [("predator", (C[0] + 6, C[1], 9))]),
    ("predator coming from BELOW", lambda w: None, [("predator", (C[0] + 8, C[1], 0.3))]),
    ("predator above | GF removed", lambda w: w.toggle_circuit("gf"), [("predator", (C[0] + 6, C[1], 9))]),
    ("female on the ground", lambda w: None, [("female", (C[0] + 6, C[1] + 2))]),
    ("female on the ground | LC10a removed", lambda w: w.toggle_circuit("lc10a"), [("female", (C[0] + 6, C[1] + 2))]),
]
import sys
for name, setup, spawns in TESTS:
    if sys.argv[1:] and not any(a in name for a in sys.argv[1:]):
        continue
    run(name, setup, spawns, T=25 if "female" in name else 15)
