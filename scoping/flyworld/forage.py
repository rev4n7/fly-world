"""Fly World scoping: food odour -> steering, and taste -> proboscis (MN9)."""
from neuprint import Client
from flypac import config as C
from flypac.fetch import load_token
c = Client(C.SERVER, dataset=C.DATASET, token=load_token())
q = c.fetch_custom
FOOD = ["DM1", "VA2", "DM4", "DP1m", "DM2", "DM5"]
print(q('MATCH (n:Neuron) WHERE n.type IN ["ORN_DM1","ORN_VA2"] RETURN n.type AS t, n.instance AS inst, count(*) AS n').to_string())
print(q('MATCH (n:Neuron) WHERE n.type STARTS WITH "mAL_m" RETURN n.type AS t, count(*) AS n, collect(DISTINCT n.consensusNt) AS nt ORDER BY t').to_string())
pn = f'a.class = "ALPN" AND a.type =~ "({"|".join(FOOD)})_.*"'
steer = '["DNa02","DNa03","DNa01","DNp09","DNa04","DNb05","DNb06","DNa11"]'
# 1 hop and 2 hops (through any middle neuron) from food PNs to steering DNs
print("\n== food PN -> steer DN direct")
print(q(f'MATCH (a:Neuron)-[w:ConnectsTo]->(b:Neuron) WHERE {pn} AND b.type IN {steer} RETURN a.type, b.type, sum(w.weight) AS syn').to_string())
print("\n== food PN -> X -> steer DN (min weight path strength, top 20)")
df = q(f'''MATCH (a:Neuron)-[w1:ConnectsTo]->(m:Neuron)-[w2:ConnectsTo]->(b:Neuron)
          WHERE {pn} AND b.type IN {steer} AND w1.weight >= 5 AND w2.weight >= 5
          RETURN m.type AS mid, b.type AS dn, sum(w1.weight) AS w_in, sum(w2.weight) AS w_out, count(*) AS paths
          ORDER BY w_in * w_out DESC LIMIT 20''')
print(df.to_string())
print("\n== what do food PNs mainly target (top 15 types)")
print(q(f'MATCH (a:Neuron)-[w:ConnectsTo]->(b:Neuron) WHERE {pn} RETURN b.type AS t, b.class AS cls, sum(w.weight) AS syn ORDER BY syn DESC LIMIT 15').to_string())
print("\n== food ORN -> PN (uni-glomerular check)")
print(q(f'MATCH (a:Neuron)-[w:ConnectsTo]->(b:Neuron) WHERE a.type IN {["ORN_"+f for f in FOOD]} AND b.class="ALPN" RETURN a.type AS orn, b.type AS pn, sum(w.weight) AS syn ORDER BY syn DESC LIMIT 15').to_string())
print("\n== taste GRN -> X -> MN9 (proboscis extension), top 15")
df = q('''MATCH (a:Neuron)-[w1:ConnectsTo]->(m:Neuron)-[w2:ConnectsTo]->(b:Neuron {type:"MN9"})
          WHERE a.class = "gustatory" AND w1.weight >= 3 AND w2.weight >= 5
          RETURN a.type AS grn, m.type AS mid, sum(w1.weight) AS w_in, sum(w2.weight) AS w_out ORDER BY w_in*w_out DESC LIMIT 15''')
print(df.to_string())
print("\n== GRN types by total 2-hop drive to MN9")
df = q('''MATCH (a:Neuron)-[w1:ConnectsTo]->(m:Neuron)-[w2:ConnectsTo]->(b:Neuron {type:"MN9"})
          WHERE a.class = "gustatory" AND w1.weight >= 3 AND w2.weight >= 5
          RETURN a.type AS grn, a.subclass AS sub, count(DISTINCT a) AS cells, sum(w1.weight) AS w_in ORDER BY w_in DESC LIMIT 12''')
print(df.to_string())
