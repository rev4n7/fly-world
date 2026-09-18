"""Injury lab self-test: each injury vs healthy, headless (3 seeds x 20 s)."""
import numpy as np
from flypac.flyworld import FlyWorld, WORLD_W, WORLD_H, SEIZURE_FRAC
from flypac.injury import BrainMap

C = (WORLD_W / 2, WORLD_H / 2)
_map = {}

def run(label, spawns, injure=lambda w: None, T=20.0, seeds=(0, 1, 2)):
    res = []
    for s in seeds:
        w = FlyWorld(seed=s)
        injure(w)
        for k, p in spawns:
            w.spawn(k, p)
        dist, turn, fem, seiz = 0.0, [], [], 0.0
        prev = w.fly_pos.copy()
        for _ in range(int(T / 0.02)):
            w.step()
            dist += np.hypot(*(w.fly_pos - prev)); prev = w.fly_pos.copy()
            turn.append(w.motor["turn_right"])
            if w.of("female"):
                fem.append(np.hypot(*(w.of("female")[0].pos - w.fly_pos)) < 2.5)
            seiz += (w.seizure_level() >= SEIZURE_FRAC) * 0.02
        res.append([dist / T, np.mean(turn), w.stats.jumps, w.stats.catches, w.stats.eaten,
                    np.mean(fem) if fem else np.nan, seiz, int((~w.brain.alive).sum())])
    m = np.nanmean(res, 0)
    print(f"{label:44s} speed {m[0]:4.2f} turnbias {m[1]:+6.1f} jumps {m[2]:4.1f} caught {m[3]:3.1f} "
          f"eaten {m[4]:4.2f} near-fem {m[5]:.2f} seizure {m[6]:4.1f}s lost {m[7]:.0f}", flush=True)

def stroke_at(label_name, r=12, dx=0):
    def f(w):
        bm = _map.setdefault("m", BrainMap(w.brain))
        px, py = bm.labels[label_name]
        w.stroke(bm.cells_in_disc(px + dx, py, r))
    return f

fem_L = [("female", (C[0] + 4, C[1] - 4))]   # fly starts facing +x; -y is its left
fem_R = [("female", (C[0] + 4, C[1] + 4))]
pred = [("predator", (C[0] + 8, C[1]))]
food_under = [("food", C)]
run("healthy, empty arena", [])
run("neck cut BOTH (spinal injury)", [], lambda w: w.cut_neck("LR"))
run("neck cut LEFT half", [], lambda w: w.cut_neck("L"))
run("healthy, predator", pred)
run("neck cut BOTH, predator", pred, lambda w: w.cut_neck("LR"))
run("neck cut BOTH, standing on food", food_under, lambda w: w.cut_neck("LR"))
run("healthy, female", fem_R)
run("30% random cell loss, female", fem_R, lambda w: w.lose_random(0.30))
run("60% random cell loss, female", fem_R, lambda w: w.lose_random(0.60))
run("30% random cell loss, predator", pred, lambda w: w.lose_random(0.30))
run("all synapses 50% weaker, female", fem_R, lambda w: w.set_gains(syn_gain=0.5))
run("brakes 50% weaker, empty", [], lambda w: w.set_gains(inh_gain=0.5))
run("brakes 20% strength, empty", [], lambda w: w.set_gains(inh_gain=0.2))
run("brakes 20% strength, female", fem_R, lambda w: w.set_gains(inh_gain=0.2))
run("healthy, female on LEFT", fem_L)
run("left eye removed, female on LEFT", fem_L, lambda w: w.toggle_circuit("eyeL"))
run("stroke at left lobula, female on LEFT", fem_L, stroke_at("lobula"))
run("left eye removed, female on RIGHT", fem_R, lambda w: w.toggle_circuit("eyeL"))
