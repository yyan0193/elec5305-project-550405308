"""Compare classical and MERT representations under source-aware evaluation."""
import argparse,itertools,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.evaluation import evaluate_fold
from src.project_utils import INSTRUMENTS,ensure_parent,load_config,load_feature_cache,load_manifest
def merge_features(manifest,path):
    f,m=load_feature_cache(path); d=manifest.merge(f,on="segment_id",validate="one_to_one"); return d,[c for c in f if c!="segment_id"],m
def combinations_for_budget(nums,budget): return list(itertools.combinations(sorted(nums),budget))
def plot_curve(results,output,metric,ylabel):
    s=results.groupby(["representation","label_budget"])[metric].agg(["mean","std"]).reset_index(); fig,ax=plt.subplots(figsize=(7.2,4.8))
    for name,g in s.groupby("representation"): ax.errorbar(g.label_budget,g["mean"],yerr=g["std"].fillna(0),marker="o",capsize=4,linewidth=2,label=name)
    ax.set(xlabel="Labelled source recordings per instrument",ylabel=ylabel,ylim=(0,1.02)); ax.set_xticks([1,2,3,4]); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); ensure_parent(output); fig.savefig(output,dpi=220); plt.close(fig)
def plot_confusion(matrix,output,title):
    names=[INSTRUMENTS[i] for i in sorted(INSTRUMENTS)]; fig,ax=plt.subplots(figsize=(9,7.5)); sns.heatmap(matrix,annot=True,fmt="d",cmap="Blues",xticklabels=names,yticklabels=names,ax=ax); ax.set(xlabel="Predicted instrument",ylabel="True instrument",title=title); fig.tight_layout(); fig.savefig(ensure_parent(output),dpi=220); plt.close(fig)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config/experiment.yaml"); a=p.parse_args(); c=load_config(a.config); man=load_manifest(c["data"]["manifest_path"]); labels=sorted(INSTRUMENTS); reps={"Classical":c["classical"]["output_path"],"MERT-v1-95M":c["mert"]["output_path"]}; rows=[]; rec_pred=[]; seg_pred=[]; rec_m={k:np.zeros((11,11),int) for k in reps}; seg_m={k:np.zeros((11,11),int) for k in reps}; meta={}
    for rep,path in reps.items():
        data,features,metadata=merge_features(man,path); meta[rep]=metadata
        for fold in range(1,6):
            test=data[data.recording_number==fold]; train_nums=[n for n in range(1,6) if n!=fold]
            for budget in c["evaluation"]["label_budgets"]:
                for ci,combo in enumerate(combinations_for_budget(train_nums,budget),1):
                    result=evaluate_fold(data[data.recording_number.isin(combo)],test,features,float(c["evaluation"]["c"]),int(c["evaluation"]["max_iter"]),labels); rows.append(dict(representation=rep,feature_dimension=len(features),fold=fold,test_recording_number=fold,label_budget=budget,training_recording_numbers="+".join(map(str,combo)),combination_index=ci,**result.metrics))
                    if budget==4:
                        rec_m[rep]+=result.confusion; ct=pd.crosstab(pd.Categorical(result.segment_predictions.instrument_id,categories=labels),pd.Categorical(result.segment_predictions.prediction,categories=labels),dropna=False).to_numpy(); seg_m[rep]+=ct; rp=result.recording_predictions.copy(); rp["representation"]=rep; rp["fold"]=fold; rec_pred.append(rp); sp=result.segment_predictions.copy(); sp["representation"]=rep; sp["fold"]=fold; seg_pred.append(sp)
    results=pd.DataFrame(rows); results.to_csv(ensure_parent("results/label_efficiency_results.csv"),index=False); summary=results.groupby(["representation","label_budget"]).agg(recording_macro_f1_mean=("recording_macro_f1","mean"),recording_macro_f1_std=("recording_macro_f1","std"),recording_accuracy_mean=("recording_accuracy","mean"),segment_macro_f1_mean=("segment_macro_f1","mean"),segment_accuracy_mean=("segment_accuracy","mean"),training_seconds_mean=("training_seconds","mean"),inference_seconds_mean=("inference_seconds","mean"),runs=("recording_macro_f1","size")).reset_index(); summary.to_csv(ensure_parent("results/label_efficiency_summary.csv"),index=False); summary[summary.label_budget==4].to_csv(ensure_parent("results/full_data_representation_comparison.csv"),index=False); pd.concat(rec_pred).to_csv(ensure_parent("results/full_data_recording_predictions.csv"),index=False); pd.concat(seg_pred).to_csv(ensure_parent("results/full_data_segment_predictions.csv"),index=False); plot_curve(results,"figures/label_efficiency_recording_macro_f1.png","recording_macro_f1","Recording-level macro-F1"); plot_curve(results,"figures/label_efficiency_segment_macro_f1.png","segment_macro_f1","Segment-level macro-F1")
    for rep in reps:
        slug=rep.lower().replace("-","_").replace(" ","_"); np.savetxt(ensure_parent(f"results/{slug}_recording_confusion.csv"),rec_m[rep],fmt="%d",delimiter=","); np.savetxt(ensure_parent(f"results/{slug}_segment_confusion.csv"),seg_m[rep],fmt="%d",delimiter=","); plot_confusion(rec_m[rep],f"figures/{slug}_recording_confusion.png",f"{rep}: source-level five-fold confusion matrix"); plot_confusion(seg_m[rep],f"figures/{slug}_segment_confusion.png",f"{rep}: five-second segment confusion matrix")
    with ensure_parent("results/representation_metadata.json").open("w") as f: json.dump(meta,f,indent=2,sort_keys=True)
    print(summary.to_string(index=False))
if __name__=="__main__": main()
