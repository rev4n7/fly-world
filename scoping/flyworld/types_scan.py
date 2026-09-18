"""Fly World scoping: which foraging / courtship neuron types exist in male-cns:v1.0."""
import re, sys
from neuprint import Client
from flypac import config as C
from flypac.fetch import load_token

c = Client(C.SERVER, dataset=C.DATASET, token=load_token())
pats = sys.argv[1:] or [
    "^P1", "^pC1", "^pIP10", "^vPR", "^vAB3", "^PPN1", "^mAL", "^aSP", "^pMP", "^DNp13", "^vpoDN",
    "^ORN_", "^Gr", "^ppk", "^GRN", "^M_", "^LC10", "^LC9$", "^DNa02", "^DNp11", "^MN9", "^Fdg", "^DNg", "^aIP",
    "^pCd", "^LHAD", "^CL062", "^AVLP5", "^SMP", "^TPN", "^mcAL", "^lvPN", "^vPN", "^ALPN", "^adPN", "^lPN",
]
q = "MATCH (n:Neuron) WHERE n.type =~ $p RETURN n.type AS type, count(*) AS n, collect(DISTINCT n.consensusNt)[0..3] AS nt ORDER BY type"
for p in pats:
    df = c.fetch_custom(q.replace("$p", '"' + p + '.*"'))
    if len(df) > 25:
        print(f"{p}: {len(df)} types, {df.n.sum()} cells; e.g. {', '.join(df.type.head(25))}")
    else:
        print(f"{p}: " + (", ".join(f"{r.type}({r.n},{'/'.join(map(str, r.nt))})" for r in df.itertuples()) or "NONE"))
