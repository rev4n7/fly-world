from neuprint import Client
from flypac import config as C
from flypac.fetch import load_token
c = Client(C.SERVER, dataset=C.DATASET, token=load_token())
def q(cy):
    return c.fetch_custom(cy)
for key in ["P1", "vAB3", "PPN1", "ppk", "Gr", "sugar", "vPR6", "pIP10", "P2b", "aSP-g", "M-cell", "LC1", "vMS", "PPK", "LC10", "sweet", "Ir", "Fru", "aDN", "mAL", "DN1", "Kimura"]:
    df = q(f'MATCH (n:Neuron) WHERE n.synonyms CONTAINS "{key}" RETURN n.type AS type, n.synonyms AS syn, count(*) AS n ORDER BY n DESC LIMIT 12')
    print(f"\n== synonyms ~ {key}: {len(df)} types"); print(df.to_string(max_colwidth=70) if len(df) else "")
print("\n== gustatory class types by subclass")
print(q('MATCH (n:Neuron {class:"gustatory"}) RETURN n.subclass AS sub, n.superclass AS sup, n.entryNerve AS nerve, count(*) AS n, collect(DISTINCT n.type)[0..8] AS types ORDER BY n DESC').to_string(max_colwidth=90))
print("\n== olfactory ORN attractive glomeruli")
print(q('MATCH (n:Neuron) WHERE n.type IN ["ORN_DM1","ORN_DM2","ORN_DM4","ORN_DM5","ORN_VA2","ORN_DP1m","ORN_DM3","ORN_VM7d","ORN_VA6","ORN_DA2","ORN_V","ORN_DA1","ORN_VA1v","ORN_VL2a","ORN_VA1d"] RETURN n.type AS t, n.somaSide AS side, count(*) AS n ORDER BY t').to_string())
print("\n== PN types for DM1/DM4/VA2/DP1m")
print(q('MATCH (n:Neuron {class:"ALPN"}) WHERE n.type =~ "(DM1|DM2|DM4|DM5|VA2|DP1m|DA1|VA1v|VL2a)_.*" RETURN n.type AS t, count(*) AS n, collect(DISTINCT n.consensusNt) AS nt').to_string())
