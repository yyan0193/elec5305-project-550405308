"""Summarise important representation-level confusion pairs."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.project_utils import INSTRUMENTS,ensure_parent
def count(d,rep,a,b):
    s=d[d.representation==rep]; return int((((s.instrument_id==a)&(s.prediction==b))|((s.instrument_id==b)&(s.prediction==a))).sum())
def main():
    d=pd.read_csv("results/full_data_segment_predictions.csv"); ids={v:k for k,v in INSTRUMENTS.items()}; rows=[]
    for first,second,role in [("Sheng","Dizi","Improved pair: free-reed and wind instruments"),("Yangqin","Guzheng","Persistent pair: related plucked/struck strings")]:
        c=count(d,"Classical",ids[first],ids[second]); m=count(d,"MERT-v1-95M",ids[first],ids[second]); rows.append(dict(pair=f"{first}-{second}",interpretation_role=role,classical_symmetric_segment_errors=c,mert_symmetric_segment_errors=m,error_reduction=c-m))
    out=pd.DataFrame(rows); out.to_csv(ensure_parent("results/confusion_pair_analysis.csv"),index=False)
    x=np.arange(len(out)); width=.34; fig,ax=plt.subplots(figsize=(7.2,4.6)); ax.bar(x-width/2,out.classical_symmetric_segment_errors,width,label="Classical"); ax.bar(x+width/2,out.mert_symmetric_segment_errors,width,label="MERT-v1-95M"); ax.set_xticks(x,out.pair); ax.set_ylabel("Symmetric five-second segment errors"); ax.set_title("Selected confusion-pair analysis"); ax.legend(); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(ensure_parent("figures/confusion_pair_analysis.png"),dpi=220); plt.close(fig); print(out)
if __name__=="__main__": main()
