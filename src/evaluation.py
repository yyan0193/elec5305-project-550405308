"""Shared source-aware evaluation logic."""
import time
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
@dataclass
class FoldResult:
    metrics:dict; segment_predictions:pd.DataFrame; recording_predictions:pd.DataFrame; confusion:np.ndarray
def recording_equal_weights(rows):
    counts=rows.groupby("recording_id")["segment_id"].transform("count").to_numpy(); w=1.0/counts; return w/w.mean()
def aggregate_recording_probabilities(rows,probabilities,classes):
    p=pd.DataFrame(probabilities,columns=[str(c) for c in classes]); meta=rows[["recording_id","instrument_id","instrument"]].reset_index(drop=True); joined=pd.concat([meta,p],axis=1); cols=[str(c) for c in classes]; agg=joined.groupby(["recording_id","instrument_id","instrument"],as_index=False)[cols].mean().reset_index(drop=True); agg["prediction"]=classes[np.argmax(agg[cols].to_numpy(),axis=1)]; return agg
def evaluate_fold(train,test,features,c,max_iter,labels):
    pipe=Pipeline([("scaler",StandardScaler()),("classifier",LogisticRegression(C=c,max_iter=max_iter,solver="lbfgs",random_state=0))]); start=time.perf_counter(); pipe.fit(train[features].to_numpy(),train.instrument_id.to_numpy(),classifier__sample_weight=recording_equal_weights(train)); training=time.perf_counter()-start; start=time.perf_counter(); probs=pipe.predict_proba(test[features].to_numpy()); inference=time.perf_counter()-start
    seg=test[["segment_id","recording_id","instrument_id","instrument"]].copy(); seg["prediction"]=pipe.classes_[np.argmax(probs,axis=1)]; rec=aggregate_recording_probabilities(test,probs,pipe.classes_)
    metrics=dict(segment_accuracy=accuracy_score(seg.instrument_id,seg.prediction),segment_macro_f1=f1_score(seg.instrument_id,seg.prediction,average="macro",labels=labels),recording_accuracy=accuracy_score(rec.instrument_id,rec.prediction),recording_macro_f1=f1_score(rec.instrument_id,rec.prediction,average="macro",labels=labels),training_seconds=training,inference_seconds=inference,train_segments=len(train),test_segments=len(test),train_recordings=train.recording_id.nunique(),test_recordings=test.recording_id.nunique())
    return FoldResult(metrics,seg,rec,confusion_matrix(rec.instrument_id,rec.prediction,labels=labels))
