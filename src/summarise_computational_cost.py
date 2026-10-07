import json,pandas as pd
from src.project_utils import ensure_parent
def main():
    m=json.load(open("results/representation_metadata.json")); rows=[]
    for name,v in m.items(): rows.append(dict(representation=name,feature_dimension=int(v["dimension"]),segments=int(v["segments"]),device=v.get("device","CPU"),extraction_seconds=float(v["extraction_seconds"]),seconds_per_five_second_segment=float(v["seconds_per_segment"])))
    d=pd.DataFrame(rows); base=d.loc[d.representation=="Classical","extraction_seconds"].iloc[0]; d["time_relative_to_classical"]=d.extraction_seconds/base; d.to_csv(ensure_parent("results/computational_cost.csv"),index=False); print(d)
if __name__=="__main__": main()
