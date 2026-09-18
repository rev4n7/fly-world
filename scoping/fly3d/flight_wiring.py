"""Are flight DNs wired to wing motor neurons (direct / 2-hop), and what drives them?"""
from neuprint import Client
from flypac import config as C
from flypac.fetch import load_token
c = Client(C.SERVER, dataset=C.DATASET, token=load_token())
q = c.fetch_custom
WMN = 'm.superclass = "vnc_motor" AND m.subclass = "wm"'
DN = 'd.superclass = "descending_neuron"'
print("== DN -> wing MN direct, top DN types")
print(q(f'''MATCH (d:Neuron)-[w:ConnectsTo]->(m:Neuron) WHERE {DN} AND {WMN}
  RETURN d.type AS dn, sum(w.weight) AS syn, collect(DISTINCT m.type)[0..6] AS mns ORDER BY syn DESC LIMIT 15''').to_string(max_colwidth=70))
print("\n== DN -> X -> wing MN (each hop >= 10), top DN types by path strength")
print(q(f'''MATCH (d:Neuron)-[w1:ConnectsTo]->(x:Neuron)-[w2:ConnectsTo]->(m:Neuron) WHERE {DN} AND {WMN} AND w1.weight >= 10 AND w2.weight >= 10
  RETURN d.type AS dn, sum(w1.weight) AS w_in, sum(w2.weight) AS w_out, count(DISTINCT x) AS mids ORDER BY w_in*w_out DESC LIMIT 20''').to_string())
print("\n== wing MN side vs DN side (DNa02, DNb01, DNg02_a, DNa01): same-side vs opposite 2-hop weight")
for t in ["DNa02", "DNa01", "DNb01", "DNg02_a", "DNp15", "DNa03"]:
    df = q(f'''MATCH (d:Neuron {{type:"{t}"}})-[w1:ConnectsTo]->(x:Neuron)-[w2:ConnectsTo]->(m:Neuron) WHERE {WMN} AND w1.weight >= 5 AND w2.weight >= 5
      RETURN d.somaSide AS dside, m.somaSide AS mside, m.type AS mn, sum(w1.weight*w2.weight) AS s''')
    if len(df):
        g = df.groupby(["dside", "mside"]).s.sum().to_dict()
        print(t, g, "top MNs:", df.groupby("mn").s.sum().sort_values(ascending=False).head(5).index.tolist())
    else:
        print(t, "none")
print("\n== optic flow (HS/VS/H2) -> DN, top")
print(q('''MATCH (a:Neuron)-[w:ConnectsTo]->(d:Neuron) WHERE a.type =~ "HS.|H2|VS.*" AND d.superclass = "descending_neuron"
  RETURN a.type AS src, d.type AS dn, sum(w.weight) AS syn ORDER BY syn DESC LIMIT 12''').to_string())
print("\n== looming (LC4/LPLC2/LPLC1) -> flight DNs")
print(q('''MATCH (a:Neuron)-[w:ConnectsTo]->(d:Neuron) WHERE a.type IN ["LC4","LPLC2","LPLC1"] AND d.superclass = "descending_neuron"
  RETURN a.type AS src, d.type AS dn, sum(w.weight) AS syn ORDER BY syn DESC LIMIT 12''').to_string())
