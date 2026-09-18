"""3-hop check: food-odour PNs -> A -> B -> steering DNs, vs LC10a -> ... for scale."""
from neuprint import Client
from flypac import config as C
from flypac.fetch import load_token
c = Client(C.SERVER, dataset=C.DATASET, token=load_token())
q = c.fetch_custom
FOOD = ["DM1", "VA2", "DM4", "DP1m", "DM2", "DM5"]
pn = f'p.class = "ALPN" AND p.type =~ "({"|".join(FOOD)})_.*"'
for dn in [["DNa02"], ["DNa03"], ["DNa01", "DNp09", "DNa04"]]:
    df = q(f'''MATCH (p:Neuron)-[w1:ConnectsTo]->(a:Neuron)-[w2:ConnectsTo]->(b:Neuron)-[w3:ConnectsTo]->(d:Neuron)
              WHERE {pn} AND d.type IN {dn} AND w1.weight >= 20 AND w2.weight >= 20 AND w3.weight >= 20
                AND NOT a.class IN ["Kenyon_Cell","ALLN"]
              WITH a.type AS A, b.type AS B, d.type AS D, sum(w1.weight) AS s1, sum(w2.weight) AS s2, sum(w3.weight) AS s3
              RETURN A, B, D, s1, s2, s3 ORDER BY s1*s2*s3 DESC LIMIT 8''')
    print(f"\n== food PN ->A->B-> {dn} (each hop >=20 syn): {len(df)} routes shown"); print(df.to_string() if len(df) else " none")
# does the sugar/taste side reach steering at all? and the MB output side
df = q('''MATCH (m:Neuron)-[w:ConnectsTo]->(d:Neuron) WHERE m.class = "MBON" AND d.type IN ["DNa02","DNa03"]
          RETURN m.type AS mbon, d.type AS dn, sum(w.weight) AS syn ORDER BY syn DESC LIMIT 8''')
print("\n== MBON -> DNa02/03 direct"); print(df.to_string() if len(df) else " none")
