"""Fly World data layer: Fly-Pacman's circuit + courtship + smell + eating, from male-cns:v1.0.

Run:  .venv\\Scripts\\python -m flypac.fetch_world        (needs the neuPrint token; run flypac.fetch first)

Starts from data/neurons.csv (escape, pursuit, hearing; retinotopy already computed) and adds
the circuits scoped in scoping/flyworld/FINDINGS.md. Named cell types are listed in config.py;
everything in between is selected by synapse-count thresholds, never hand-picked.

Writes to data/world/: neurons.csv, edges.csv, meta.json (same format as data/).
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from neuprint import Client, fetch_adjacencies

from . import config as C
from .fetch import cy_list, load_token, select_feedback


def ids_of(df):
    return set(int(b) for b in df.bodyId)


def select_courtship(c):
    """Leg taste -> vAB3 -> P1/pC1 (+ mAL brake) -> pIP10 song; LC10a -> pC1."""
    vab3 = c.fetch_custom(f'MATCH (n:Neuron) WHERE n.type STARTS WITH "{C.VAB3_PREFIX}" '
                          'RETURN n.bodyId AS bodyId, n.type AS type')
    p1 = c.fetch_custom(f'MATCH (n:Neuron) WHERE n.type STARTS WITH "{C.P1_PREFIX}" RETURN n.bodyId AS bodyId, n.type AS type')
    song = c.fetch_custom(f'MATCH (n:Neuron) WHERE n.type IN {cy_list(C.SONG_TYPES)} RETURN n.bodyId AS bodyId, n.type AS type')
    touch = c.fetch_custom(f"""
        MATCH (g:Neuron {{class:"gustatory"}})-[w:ConnectsTo]->(v:Neuron) WHERE v.bodyId IN {sorted(ids_of(vab3))}
        WITH g, sum(w.weight) AS w_out WHERE w_out >= {C.TOUCH_MIN_OUT}
        RETURN g.bodyId AS bodyId, g.type AS type, w_out""")
    src = sorted(ids_of(vab3) | ids_of(c.fetch_custom(
        f'MATCH (n:Neuron) WHERE n.type IN {cy_list(C.PURSUIT_TYPES)} RETURN n.bodyId AS bodyId')))
    core = ids_of(vab3) | ids_of(p1) | ids_of(song)
    mid = c.fetch_custom(f"""
        MATCH (m:Neuron)-[w2:ConnectsTo]->(p:Neuron) WHERE p.bodyId IN {sorted(ids_of(p1))} AND NOT m.bodyId IN {sorted(core)}
        WITH m, sum(w2.weight) AS w_out WHERE w_out >= {C.COURT_MID_MIN_OUT}
        MATCH (s:Neuron)-[w1:ConnectsTo]->(m) WHERE s.bodyId IN {src}
        WITH m, w_out, sum(w1.weight) AS w_in WHERE w_in >= {C.COURT_MID_MIN_IN}
        RETURN m.bodyId AS bodyId, m.type AS type, w_in, w_out""")
    song_mid = c.fetch_custom(f"""
        MATCH (m:Neuron)-[w2:ConnectsTo]->(p:Neuron) WHERE p.bodyId IN {sorted(ids_of(song))} AND NOT m.bodyId IN {sorted(core)}
        WITH m, sum(w2.weight) AS w_out WHERE w_out >= {C.COURT_MID_MIN_OUT}
        MATCH (s:Neuron)-[w1:ConnectsTo]->(m) WHERE s.bodyId IN {sorted(ids_of(p1))}
        WITH m, w_out, sum(w1.weight) AS w_in WHERE w_in >= {C.COURT_MID_MIN_IN}
        RETURN m.bodyId AS bodyId, m.type AS type, w_in, w_out""")
    return dict(touch=touch, vab3=vab3, p1=p1, song=song, court_mid=mid, song_mid=song_mid)


def select_smell(c, steer_ids):
    """Food-odour ORNs -> their PNs -> every neuron on a PN->A->B->steering route (each hop >= threshold)."""
    orn = c.fetch_custom(f'MATCH (n:Neuron) WHERE n.type IN {cy_list(["ORN_" + g for g in C.FOOD_GLOMERULI])} '
                         'RETURN n.bodyId AS bodyId, n.type AS type, n.instance AS instance')
    pn = c.fetch_custom(f"""
        MATCH (o:Neuron)-[w:ConnectsTo]->(p:Neuron {{class:"ALPN"}}) WHERE o.bodyId IN {sorted(ids_of(orn))}
        WITH p, sum(w.weight) AS w_in WHERE w_in >= {C.SMELL_PN_MIN_IN}
        RETURN p.bodyId AS bodyId, p.type AS type, w_in""")
    h = C.SMELL_HOP_MIN
    routes = c.fetch_custom(f"""
        MATCH (p:Neuron)-[w1:ConnectsTo]->(a:Neuron)-[w2:ConnectsTo]->(b:Neuron)-[w3:ConnectsTo]->(d:Neuron)
        WHERE p.bodyId IN {sorted(ids_of(pn))} AND d.bodyId IN {sorted(steer_ids)}
          AND w1.weight >= {h} AND w2.weight >= {h} AND w3.weight >= {h}
          AND NOT coalesce(a.class, "") IN ["Kenyon_Cell", "ALLN"] AND NOT coalesce(b.class, "") IN ["Kenyon_Cell", "ALLN"]
        RETURN DISTINCT a.bodyId AS a, a.type AS a_type, b.bodyId AS b, b.type AS b_type""")
    two = c.fetch_custom(f"""
        MATCH (p:Neuron)-[w1:ConnectsTo]->(a:Neuron)-[w2:ConnectsTo]->(d:Neuron)
        WHERE p.bodyId IN {sorted(ids_of(pn))} AND d.bodyId IN {sorted(steer_ids)} AND w1.weight >= {h} AND w2.weight >= {h}
          AND NOT coalesce(a.class, "") IN ["Kenyon_Cell", "ALLN"]
        RETURN DISTINCT a.bodyId AS a, a.type AS a_type""")
    mid = pd.concat([routes[["a", "a_type"]].set_axis(["bodyId", "type"], axis=1),
                     routes[["b", "b_type"]].set_axis(["bodyId", "type"], axis=1),
                     two.set_axis(["bodyId", "type"], axis=1)]).drop_duplicates("bodyId")
    mid = mid[~mid.bodyId.isin(steer_ids)]
    return dict(orn=orn, pn=pn, smell_mid=mid, smell_routes=routes)


def select_taste(c):
    """Mouthpart taste neurons -> A -> (B ->) MN9, the proboscis-extension motor neuron (each hop >= threshold)."""
    mn9 = c.fetch_custom(f'MATCH (n:Neuron) WHERE n.type IN {cy_list(C.FEED_TYPES)} RETURN n.bodyId AS bodyId, n.type AS type')
    grn_where = " OR ".join(f'g.type STARTS WITH "{p}"' for p in C.MOUTH_GRN_PREFIXES)
    h, mn = C.TASTE_HOP_MIN, sorted(ids_of(mn9))
    three = c.fetch_custom(f"""
        MATCH (g:Neuron {{class:"gustatory"}})-[w1:ConnectsTo]->(a:Neuron)-[w2:ConnectsTo]->(b:Neuron)-[w3:ConnectsTo]->(t:Neuron)
        WHERE ({grn_where}) AND t.bodyId IN {mn} AND w1.weight >= {h} AND w2.weight >= {h} AND w3.weight >= {h}
        RETURN DISTINCT g.bodyId AS g, g.type AS g_type, a.bodyId AS a, a.type AS a_type, b.bodyId AS b, b.type AS b_type""")
    two = c.fetch_custom(f"""
        MATCH (g:Neuron {{class:"gustatory"}})-[w1:ConnectsTo]->(a:Neuron)-[w2:ConnectsTo]->(t:Neuron)
        WHERE ({grn_where}) AND t.bodyId IN {mn} AND w1.weight >= {h} AND w2.weight >= {h}
        RETURN DISTINCT g.bodyId AS g, g.type AS g_type, a.bodyId AS a, a.type AS a_type""")
    cols = ["bodyId", "type"]
    grn = pd.concat([three[["g", "g_type"]].set_axis(cols, axis=1), two[["g", "g_type"]].set_axis(cols, axis=1)])
    mid = pd.concat([three[["a", "a_type"]].set_axis(cols, axis=1), three[["b", "b_type"]].set_axis(cols, axis=1),
                     two[["a", "a_type"]].set_axis(cols, axis=1)])
    return dict(taste=grn.drop_duplicates("bodyId"), taste_mid=mid.drop_duplicates("bodyId"), feed=mn9)


ROLE_OF = {  # later entries win when a neuron is picked by several rules
    "feedback": "middle_feedback", "smell_mid": "middle_smell", "taste_mid": "middle_taste",
    "court_mid": "middle_court", "song_mid": "middle_court",
    "touch": "sense_touch", "vab3": "relay_vab3", "orn": "sense_smell", "pn": "smell_pn",
    "taste": "sense_taste", "p1": "court_p1", "song": "out_song", "feed": "out_feed",
}


def main():
    t0 = time.time()
    c = Client(C.SERVER, dataset=C.DATASET, token=load_token())
    base = pd.read_csv(C.DATA_DIR / "neurons.csv").set_index("bodyId")
    base_meta = json.loads((C.DATA_DIR / "meta.json").read_text())
    meta = {"dataset": C.DATASET, "server": C.SERVER, "fetched": time.strftime("%Y-%m-%d %H:%M:%S"),
            "neuprint_python": __import__("neuprint").__version__, "base": "data/neurons.csv (flypac.fetch)",
            "thresholds": {k: getattr(C, k) for k in dir(C) if "_MIN" in k},
            "food_glomeruli": C.FOOD_GLOMERULI, "mouth_grn_prefixes": C.MOUTH_GRN_PREFIXES}
    steer_ids = set(base.index[base.type.isin(["DNa02", "DNa03"])])

    print("selecting courtship circuit ...")
    sel = select_courtship(c)
    print("selecting smell circuit ...")
    sel.update(select_smell(c, steer_ids))
    print("selecting taste -> proboscis circuit ...")
    sel.update(select_taste(c))
    new_ids = set().union(*[ids_of(sel[k]) for k in ROLE_OF if k in sel]) - set(base.index)
    ids = set(base.index) | new_ids

    # Feedback layer PER CIRCUIT (same >= 100 / >= 100 rule as flypac.fetch), for taste and courtship.
    # Applied to all new circuits at once it selects ~3,400 cells (mostly premotor hubs) and a weak LC10a
    # stimulus ignites ~800 cells; per circuit it is stable, and the taste layer is what switches MN9
    # off again after the fly leaves food (without it MN9 stays at ~480 Hz forever). No layer for smell,
    # whose onward wiring is left out anyway (flypac/worldbrain.py). See scoping/flyworld/FINDINGS.md.
    fb_parts = []
    for circ, members, targets in [
            ("taste", ["taste", "taste_mid", "feed"], ["taste_mid", "feed"]),
            ("court", ["touch", "vab3", "p1", "court_mid", "song_mid", "song"], ["p1", "court_mid", "song_mid", "song"])]:
        src = set().union(*[ids_of(sel[k]) for k in members])
        tgt = set().union(*[ids_of(sel[k]) for k in targets])
        fb = select_feedback(c, sorted(src), sorted(tgt))
        print(f"  feedback onto {circ}: {len(fb)} cells")
        fb_parts.append(fb)
    sel["feedback"] = pd.concat(fb_parts).drop_duplicates("bodyId")
    new_ids |= ids_of(sel["feedback"]) - set(base.index)
    ids = set(base.index) | new_ids
    print(f"  {len(new_ids)} new neurons")
    meta["selection"] = {k: v.groupby("type").size().to_dict() for k, v in sel.items() if k != "smell_routes"}
    meta["selection"]["smell_routes"] = {f"{a} -> {b}": int(k) for (a, b), k in
                                         sel["smell_routes"].groupby(["a_type", "b_type"]).size().items()}
    if "--select-only" in sys.argv:
        print(json.dumps({k: v if len(v) < 40 else f"{len(v)} types, {sum(v.values())} cells"
                          for k, v in meta["selection"].items()}, indent=1, default=str))
        return

    print(f"fetching properties of {len(new_ids)} new neurons ...")
    new = c.fetch_custom(f"""
        MATCH (n:Neuron) WHERE n.bodyId IN {sorted(new_ids)}
        RETURN n.bodyId AS bodyId, n.type AS type, n.instance AS instance, n.somaSide AS side,
               n.consensusNt AS nt, n.predictedNtConfidence AS nt_conf, n.pre AS pre, n.post AS post,
               n.status AS status, n.group AS group""").set_index("bodyId")
    new["sign"] = new.nt.map(C.NT_SIGN).fillna(0).astype(int)
    new["role"] = ""
    neurons = pd.concat([base, new])
    for key, role in ROLE_OF.items():   # Fly-Pacman neurons keep their original roles
        neurons.loc[neurons.index.isin(ids_of(sel[key])) & neurons.index.isin(new_ids), "role"] = role
    # sensory cells without a soma in the volume: side from their instance name (e.g. ORN_DM1_L)
    inst_side = neurons.instance.astype(str).str.extract(r"_([LR])$")[0]
    fill = neurons.side.isna() & neurons.role.isin(["sense_smell", "sense_taste", "sense_touch"])
    neurons.loc[fill, "side"] = inst_side[fill]

    # superclass (e.g. descending_neuron = brain-to-body command cells) for every neuron; used by the injury lab
    sc = c.fetch_custom(f"MATCH (n:Neuron) WHERE n.bodyId IN {sorted(int(i) for i in neurons.index if i > 0)} "
                        "RETURN n.bodyId AS bodyId, n.superclass AS superclass").set_index("bodyId").superclass
    neurons["superclass"] = sc.reindex(neurons.index)
    neurons.loc[neurons.index < 0, "superclass"] = "cb_sensory"     # mirrored hearing cells

    all_ids = sorted(int(i) for i in neurons.index)
    print(f"fetching connectivity among {len(all_ids)} neurons ...")
    _, roi_conn = fetch_adjacencies(sources=all_ids, targets=all_ids, client=c)
    edges = (roi_conn.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()
             .rename(columns={"bodyId_pre": "pre", "bodyId_post": "post"}))
    print(f"  {len(edges)} connections, {edges.weight.sum()} synapses")

    out = C.DATA_DIR / "world"
    out.mkdir(parents=True, exist_ok=True)
    neurons.reset_index().to_csv(out / "neurons.csv", index=False)
    edges.to_csv(out / "edges.csv", index=False)
    meta["midline_x"] = base_meta["midline_x"]
    meta["counts"] = {"neurons": len(neurons), "new_neurons": len(new_ids), "edges": len(edges),
                      "synapses": int(edges.weight.sum()), "by_role": neurons.role.value_counts().to_dict()}
    meta["seconds"] = round(time.time() - t0, 1)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps({k: meta[k] for k in ["counts", "selection"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
