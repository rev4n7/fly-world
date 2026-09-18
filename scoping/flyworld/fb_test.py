"""Does a per-circuit feedback layer (flypac.fetch rule) stop the taste / P1 persistent loops without runaway?"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
from neuprint import Client, fetch_adjacencies
from flypac import config as C
from flypac.fetch import load_token
from flypac import worldbrain as WB

W = C.DATA_DIR / "world"
tmp = Path(os.environ["TEMP"]) / "fw_fb"; tmp.mkdir(exist_ok=True)
n = pd.read_csv(W / "sim" / "neurons.csv")
if not (tmp / "all_edges.csv").exists():
    c = Client(C.SERVER, dataset=C.DATASET, token=load_token())
    fb = pd.concat([pd.read_csv(W / "fb_taste.csv").assign(circ="taste"), pd.read_csv(W / "fb_court.csv").assign(circ="court")])
    new = sorted(set(int(b) for b in fb.bodyId) - set(n.bodyId))
    props = c.fetch_custom(f"""MATCH (n:Neuron) WHERE n.bodyId IN {new} RETURN n.bodyId AS bodyId, n.type AS type,
        n.instance AS instance, n.somaSide AS side, n.consensusNt AS nt""")
    props.to_csv(tmp / "fb_props.csv", index=False)
    fb[["bodyId", "circ"]].to_csv(tmp / "fb_circ.csv", index=False)
    ids = sorted(set(int(b) for b in n.bodyId if b > 0) | set(new))
    _, rc = fetch_adjacencies(sources=ids, targets=ids, client=c)
    rc.groupby(["bodyId_pre", "bodyId_post"], as_index=False).weight.sum().rename(
        columns={"bodyId_pre": "pre", "bodyId_post": "post"}).to_csv(tmp / "all_edges.csv", index=False)

props = pd.read_csv(tmp / "fb_props.csv"); circ = pd.read_csv(tmp / "fb_circ.csv")
all_e = pd.read_csv(tmp / "all_edges.csv")
base_e = pd.read_csv(W / "sim" / "edges.csv")
def build(which):
    d = tmp / "_".join(which or ["none"]); d.mkdir(exist_ok=True)
    add = props[props.bodyId.isin(circ.bodyId[circ.circ.isin(which)])].copy()
    add["sign"] = add.nt.map(C.NT_SIGN).fillna(0).astype(int); add["role"] = "middle_feedback"
    nn = pd.concat([n, add], ignore_index=True).drop_duplicates("bodyId")
    ids = set(nn.bodyId)
    extra = all_e[all_e.pre.isin(ids) & all_e.post.isin(ids) & (all_e.pre.isin(add.bodyId) | all_e.post.isin(add.bodyId))]
    ee = pd.concat([base_e, extra]).drop_duplicates(["pre", "post"])
    smell = set(nn.bodyId[nn.role.isin(WB.SMELL_ROLES)])
    ee = ee[~(ee.pre.isin(smell) & ~ee.post.isin(smell))]
    nn.to_csv(d / "neurons.csv", index=False); ee.to_csv(d / "edges.csv", index=False)
    return d

def test(d, label):
    out = []
    for s in range(2):
        b = WB.WorldBrain(seed=s, data_dir=d); tr = []
        for k in range(150):
            b.sense_world(foods=[dict(bearing=0, dist=0.2 if k < 25 else 5)]); c = b.step(0.02); tr.append(c[b.feed].sum() / 0.02)
        tr = np.array(tr)
        b = WB.WorldBrain(seed=s, data_dir=d); song = []
        for k in range(150):
            b.sense_world(predators=[dict(bearing=20, dist=2.0, size=1.3, closing=3.0 if k < 25 else 0, speed=2, visible=True)] if k < 25 else [])
            c = b.step(0.02); song.append(c[b.song].sum() / 0.02)
        b = WB.WorldBrain(seed=s, data_dir=d); b.sense_world(females=[dict(bearing=40, dist=3, size=0.6, visible=True)])
        tot = np.zeros(b.N)
        for k in range(50): tot += b.step(0.02)
        hot = (tot[~b.neurons.role.str.startswith("sense").to_numpy()] > 50).sum()
        out.append([tr[:25].mean(), tr[-50:].mean(), np.mean(song[:25]), np.mean(song[-50:]), tot[b.out["DNa02_R"]], hot])
    m = np.mean(out, 0)
    print(f"{label:14s} N={b.N}: MN9 on food {m[0]:5.0f} -> 2s after {m[1]:5.0f} | pIP10 during loom {m[2]:5.0f} -> after {m[3]:5.0f} "
          f"| female R: DNa02_R {m[4]:5.0f} hot {m[5]:.0f}", flush=True)

test(W / "sim", "current")
for which in (["taste"], ["court"], ["taste", "court"]):
    test(build(which), "+fb " + "+".join(which))
