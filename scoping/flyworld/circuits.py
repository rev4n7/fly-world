"""Fly World scoping: are courtship + foraging pathways wired in male-cns:v1.0?"""
from neuprint import Client
from flypac import config as C
from flypac.fetch import load_token
c = Client(C.SERVER, dataset=C.DATASET, token=load_token())
q = c.fetch_custom
pd_opts = dict(max_colwidth=80)

print("== pC1 types: dimorphism / fruDsx")
p = q('MATCH (n:Neuron) WHERE n.type STARTS WITH "pC1" RETURN n.type AS t, count(*) AS n, collect(DISTINCT n.dimorphism) AS dim, '
      'collect(DISTINCT n.fruDsx) AS fd, collect(DISTINCT n.consensusNt) AS nt, collect(DISTINCT n.synonyms)[0] AS syn ORDER BY t')
print(p.to_string(**pd_opts))

def conn(pre_where, post_where, label, top=15):
    df = q(f"""MATCH (a:Neuron)-[w:ConnectsTo]->(b:Neuron) WHERE {pre_where} AND {post_where}
               RETURN a.type AS pre, b.type AS post, sum(w.weight) AS syn, count(*) AS pairs ORDER BY syn DESC LIMIT {top}""")
    print(f"\n== {label}: total {df.syn.sum() if len(df) else 0}"); print(df.to_string() if len(df) else "  none")

conn('a.type STARTS WITH "AN09B017"', 'b.type STARTS WITH "pC1"', "vAB3 -> pC1")
conn('a.type = "AN05B102a"', 'b.type STARTS WITH "pC1"', "PPN1 -> pC1")
conn('a.type STARTS WITH "AN09B017"', 'true', "vAB3 -> anything")
conn('a.class = "gustatory"', 'b.type STARTS WITH "AN09B017"', "gustatory -> vAB3")
conn('a.class = "gustatory"', 'b.type = "AN05B102a"', "gustatory -> PPN1")
conn('a.type STARTS WITH "mAL"', 'b.type STARTS WITH "pC1"', "mAL -> pC1")
conn('a.type STARTS WITH "pC1"', 'b.type IN ["pIP10","DNa02","DNa03","DNp13","AOTU019","AOTU012","AOTU015","AOTU025"]', "pC1 -> song/steer/AOTU")
conn('a.type = "LC10a"', 'b.type STARTS WITH "pC1"', "LC10a -> pC1")
conn('a.type STARTS WITH "pC1"', 'b.type STARTS WITH "pC1"', "pC1 -> pC1 (recurrence)", 8)
conn('true', 'b.type = "pIP10"', "inputs to pIP10 (song DN)")
conn('a.type = "pIP10"', 'true', "pIP10 outputs")
conn('true', 'b.type = "MN9"', "inputs to MN9 (proboscis extension MN)")
conn('a.class = "gustatory"', 'b.type = "MN9"', "gustatory -> MN9 direct")
