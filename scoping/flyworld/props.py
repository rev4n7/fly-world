from neuprint import Client
from flypac import config as C
from flypac.fetch import load_token
c = Client(C.SERVER, dataset=C.DATASET, token=load_token())
n = c.fetch_custom('MATCH (n:Neuron {type:"pC1_1a"}) RETURN properties(n) AS p LIMIT 1').p[0]
print(sorted(k for k in n if not k.startswith(("AL","AME","AOTU","ANm","CA","CRE","EB","FB","GNG","LAL","LH","LO","ME","MB","NO","PB","PLP","PVLP","SAD","SCL","SIP","SLP","SMP","SPS","VES","WED","ICL","IB","IPS","GOR","ATL","AVLP","BU","EPA","FLA","PRW","PED","CAN","AMMC","LegNp","WTct","HTct","IntTct","LTct","NTct","Ov","VNC","mVAC","CV","LA","CentralBrain","OL","Leg","dPR","ADMN","ProAN","LegN"))))
print({k:n[k] for k in n if not isinstance(n[k], dict) and not k[:2].isupper() or k in ("type","instance")})
for prop in ["synonyms", "fruDsx", "superclass", "class", "subclass", "somaNeuromere", "entryNerve", "hemilineage", "modality"]:
    try:
        df = c.fetch_custom(f'MATCH (n:Neuron) WHERE n.{prop} IS NOT NULL RETURN n.{prop} AS v, count(*) AS k ORDER BY k DESC LIMIT 40')
        print("\n==", prop, len(df)); print(df.to_string(max_colwidth=60))
    except Exception as e:
        print(prop, "ERR", e)
