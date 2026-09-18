"""3D / flight scoping in male-cns:v1.0: flight DNs, wing motor neurons, optic-flow cells, and their wiring."""
from neuprint import Client
from flypac import config as C
from flypac.fetch import load_token
c = Client(C.SERVER, dataset=C.DATASET, token=load_token())
q = c.fetch_custom
def types(where, label):
    df = q(f"MATCH (n:Neuron) WHERE {where} RETURN n.type AS t, count(*) AS n, collect(DISTINCT n.consensusNt)[0] AS nt, "
           f"collect(DISTINCT n.subclass)[0] AS sub, collect(DISTINCT n.synonyms)[0] AS syn ORDER BY t")
    print(f"\n== {label}: {len(df)} types, {df.n.sum() if len(df) else 0} cells")
    print(df.head(60).to_string(max_colwidth=50) if len(df) else " none")
    return df
types('n.type =~ "DNg02.*|DNb01|DNb06|DNa01|DNa04|DNa05|DNa06|DNa09|DNa10|DNp03|DNp15|DNp18|DNp19|DNp20|DNp22|DNp26|DNa15|DNp06"', "known flight DNs")
types('n.superclass = "vnc_motor" AND (n.subclass CONTAINS "wing" OR n.type =~ "(?i).*(b1|b2|b3|i1|i2|iii1|iii3|hg1|hg2|hg3|hg4|tp1|tp2|ps1|DLM|DVM|tpn).*")', "wing motor neurons")
types('n.superclass = "vnc_motor"', "all VNC motor (sample)")
types('n.type =~ "HS.*|VS.*|H2|LPLC1|LPLC4|LC12|CH|FD.*" AND n.superclass = "visual_projection"', "optic-flow / lobula plate")
