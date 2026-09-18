"""Fly 3D data layer: Fly World's circuit + flight power/steering, from male-cns:v1.0.

Run:  .venv\\Scripts\\python -m flypac.fetch_fly3d      (needs the neuPrint token; run flypac.fetch_world first)

Adds to the simulated Fly World set (data/world/sim):
  * DNg02 (all subtypes): descending neurons that drive the wing power muscles (Namiki et al. 2022)
  * wing motor neurons (VNC motor neurons with subclass "wm"): power (DLMn, DVMn) and steering (b1-3, i1-2,
    iii1/3, hg1-4, tp1-2, tpn, ps1 ...)
and then all connections among the combined set (so DN -> wing MN synapses are real).
Also writes each neuron's synapse locations binned in 3D, for the rotatable 3D brain and 3D strokes.

Writes data/fly3d/: neurons.csv, edges.csv, anatomy3d.csv (bodyId, bx, by, bz, count), meta.json
"""
import json
import time

import pandas as pd
from neuprint import Client, fetch_adjacencies

from . import config as C
from .fetch import load_token

OUT = C.DATA_DIR / "fly3d"
FLIGHT_DN_PREFIX = "DNg02"
BIN3D = 1000            # dataset voxels (8 nm) per 3D bin -> 8 um


def main():
    t0 = time.time()
    c = Client(C.SERVER, dataset=C.DATASET, token=load_token())
    base = pd.read_csv(C.DATA_DIR / "world" / "sim" / "neurons.csv").set_index("bodyId")
    add = c.fetch_custom(f"""
        MATCH (n:Neuron)
        WHERE n.type STARTS WITH "{FLIGHT_DN_PREFIX}" OR (n.superclass = "vnc_motor" AND n.subclass = "wm")
        RETURN n.bodyId AS bodyId, n.type AS type, n.instance AS instance, n.somaSide AS side,
               n.consensusNt AS nt, n.predictedNtConfidence AS nt_conf, n.pre AS pre, n.post AS post,
               n.status AS status, n.group AS group, n.superclass AS superclass""").set_index("bodyId")
    add = add[~add.index.isin(base.index)]
    add["sign"] = add.nt.map(C.NT_SIGN).fillna(0).astype(int)
    power = add.type.str.startswith(("DLMn", "DVMn"))
    add["role"] = "out_wing_steer"
    add.loc[power, "role"] = "out_wing_power"
    add.loc[add.type.str.startswith(FLIGHT_DN_PREFIX), "role"] = "out_flight"
    neurons = pd.concat([base, add])
    print(f"adding {len(add)} neurons: {add.role.value_counts().to_dict()}")

    ids = sorted(int(i) for i in neurons.index if i > 0)
    print(f"fetching connectivity among {len(ids)} neurons ...")
    _, rc = fetch_adjacencies(sources=ids, targets=ids, client=c)
    edges = (rc.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()
             .rename(columns={"bodyId_pre": "pre", "bodyId_post": "post"}))
    # mirrored right-ear cells (negative ids) keep the edges flypac.worldbrain built for them
    old = pd.read_csv(C.DATA_DIR / "world" / "sim" / "edges.csv")
    edges = pd.concat([edges, old[(old.pre < 0) | (old.post < 0)]]).groupby(["pre", "post"], as_index=False).weight.sum()
    print(f"  {len(edges)} connections, {int(edges.weight.sum())} synapses")

    print("fetching 3D synapse positions ...")
    parts = []
    for i in range(0, len(ids), 40):
        batch = ids[i:i + 40]
        parts.append(c.fetch_custom(f"""
            MATCH (n:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]->(s:Synapse) WHERE n.bodyId IN {batch}
            RETURN n.bodyId AS bodyId, toInteger(s.location.x / {BIN3D}) AS bx, toInteger(s.location.y / {BIN3D}) AS by,
                   toInteger(s.location.z / {BIN3D}) AS bz, count(*) AS count"""))
        if i % 800 == 0:
            print(f"  {i + len(batch)}/{len(ids)}  ({time.time() - t0:.0f}s)")
    a3 = pd.concat(parts)

    OUT.mkdir(parents=True, exist_ok=True)
    neurons.reset_index().to_csv(OUT / "neurons.csv", index=False)
    edges.to_csv(OUT / "edges.csv", index=False)
    a3.to_csv(OUT / "anatomy3d.csv", index=False)
    meta = {"dataset": C.DATASET, "fetched": time.strftime("%Y-%m-%d %H:%M:%S"), "base": "data/world/sim",
            "added": add.groupby("type").size().to_dict(), "bin3d_voxels": BIN3D,
            "counts": {"neurons": len(neurons), "edges": len(edges), "synapses": int(edges.weight.sum()),
                       "bins3d": len(a3)}, "seconds": round(time.time() - t0, 1)}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps(meta["counts"]))


if __name__ == "__main__":
    main()
