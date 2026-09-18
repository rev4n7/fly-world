"""Stimulate each sense in isolation and read the circuit outputs (worldbrain, no world)."""
import sys
import numpy as np
from flypac.worldbrain import WorldBrain

def run(label, T=1.5, seeds=(0, 1, 2), ablate_types=(), **scene):
    out = []
    for s in seeds:
        b = WorldBrain(seed=s)
        if ablate_types:
            b.set_ablated(np.flatnonzero(b.neurons.type.isin(ablate_types).to_numpy() |
                                         b.neurons.role.isin(ablate_types).to_numpy()))
        b.sense_world(**scene)
        tot = np.zeros(b.N)
        for _ in range(int(T / 0.02)):
            c = b.step(0.02); tot += c
        hz = tot / T
        o = b.out
        out.append([hz[o["DNa02_L"]], hz[o["DNa02_R"]], hz[b.gf].sum(), hz[b.p1].mean(), hz[b.song].sum(),
                    hz[b.feed].sum(), hz[b.smell].mean(), (hz[b.neurons.role.str.startswith("sense").to_numpy() == False] > 50).sum()])
    m = np.mean(out, 0)
    print(f"{label:38s} DNa02 L {m[0]:5.1f} R {m[1]:5.1f} | GF {m[2]:5.1f} | P1 {m[3]:5.2f} | pIP10 {m[4]:5.1f} | MN9 {m[5]:5.1f} | ORN {m[6]:4.1f} | hot {m[7]:.0f}")

pred = lambda b: dict(bearing=b, dist=2.0, size=0.9, closing=3.0, speed=3.0, visible=True)
fem = lambda b, d=3.0: dict(bearing=b, dist=d, size=0.6, visible=True)
food = lambda b, d=3.0: dict(bearing=b, dist=d)
run("nothing")
run("predator front-right", predators=[pred(30)])
run("female left 3u", females=[fem(-40)])
run("female right 3u", females=[fem(40)])
run("female touching (front)", females=[fem(0, 0.5)])
run("food smell left 3u", foods=[food(-60)])
run("food smell right 3u", foods=[food(60)])
run("food smell left 1.5u", foods=[food(-60, 1.5)])
run("food smell right 1.5u", foods=[food(60, 1.5)])
run("standing on food", foods=[food(0, 0.3)])

if "--chain" in sys.argv:
    for label, scene in [("touch", dict(females=[fem(0, 0.5)])), ("see female", dict(females=[fem(20, 2.0)]))]:
        b = WorldBrain(seed=0); b.sense_world(**scene)
        tot = np.zeros(b.N)
        for _ in range(50): tot += b.step(0.02)
        hz = tot / 1.0
        import pandas as pd
        df = pd.DataFrame(dict(role=b.neurons.role.values, type=b.neurons.type.values, hz=hz))
        print("==", label); print(df[df.role.isin(["sense_touch","relay_vab3","court_p1","middle_court","out_song","sense_pursuit"])]
              .groupby(["role"]).hz.agg(["mean","max"]).round(1).to_string())
        print(df[df.hz > 0].groupby(["role","type"]).hz.mean().sort_values(ascending=False).head(15).round(1).to_string())

if "--ablate" in sys.argv:
    mal = ["mAL_m1","mAL_m2b","mAL_m3a","mAL_m3c","mAL_m4","mAL_m5a","mAL_m5b","mAL_m5c","mAL_m7","mAL_m8","mAL_m9"]
    run("touching, intact", females=[fem(0, 0.5)])
    run("touching, mAL brake ablated", ablate_types=mal, females=[fem(0, 0.5)])
    run("see+touch female, mAL ablated", ablate_types=mal, females=[fem(0, 0.5)], predators=[])
    run("female right, LC10a ablated", ablate_types=["LC10a"], females=[fem(40)])
    run("female right, AOTU019 ablated", ablate_types=["AOTU019"], females=[fem(40)])
    run("predator, GF ablated", ablate_types=["DNp01"], predators=[pred(30)])
    run("food + taste, MN9 route", foods=[food(0, 0.3)])
