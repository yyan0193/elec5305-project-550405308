"""Shared utilities for the ELEC5305 ChMusic experiments."""
from __future__ import annotations
import json, random
from pathlib import Path
from typing import Iterable
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import yaml

INSTRUMENTS={1:"Erhu",2:"Pipa",3:"Sanxian",4:"Dizi",5:"Suona",6:"Zhuiqin",7:"Zhongruan",8:"Liuqin",9:"Guzheng",10:"Yangqin",11:"Sheng"}

def load_config(path="config/experiment.yaml"):
    with Path(path).open(encoding="utf-8") as f: return yaml.safe_load(f)
def set_reproducible_seed(seed):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch; torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except ImportError: pass
def ensure_parent(path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); return p
def load_manifest(path):
    d=pd.read_csv(path)
    required={"segment_id","instrument_id","instrument","recording_number","recording_id","source_path","start_seconds","duration_seconds","fold"}
    if required-set(d.columns): raise ValueError(f"Manifest is missing columns: {sorted(required-set(d.columns))}")
    return d
def read_segment(row,target_sr):
    info=sf.info(row["source_path"]); start=int(round(float(row["start_seconds"])*info.samplerate)); frames=int(round(float(row["duration_seconds"])*info.samplerate))
    audio,sr=sf.read(row["source_path"],start=start,frames=frames,dtype="float32",always_2d=True); audio=audio.mean(axis=1)
    if sr!=target_sr: audio=librosa.resample(audio,orig_sr=sr,target_sr=target_sr)
    expected=int(round(float(row["duration_seconds"])*target_sr))
    if len(audio)<expected: audio=np.pad(audio,(0,expected-len(audio)))
    return np.asarray(audio[:expected],dtype=np.float32)
def save_feature_cache(path,segment_ids,features,feature_names=None,metadata=None):
    np.savez_compressed(ensure_parent(path),segment_ids=np.asarray(list(segment_ids),dtype=str),features=np.asarray(features,dtype=np.float32),feature_names=np.asarray(list(feature_names or []),dtype=str),metadata_json=np.asarray(json.dumps(metadata or {},sort_keys=True)))
def load_feature_cache(path):
    c=np.load(path,allow_pickle=False); names=c["feature_names"].astype(str).tolist(); x=c["features"].astype(np.float32)
    if not names: names=[f"feature_{i:04d}" for i in range(x.shape[1])]
    d=pd.DataFrame(x,columns=names); d.insert(0,"segment_id",c["segment_ids"].astype(str)); return d,json.loads(str(c["metadata_json"].item()))
def validate_source_separation(manifest):
    for fold in sorted(manifest.fold.unique()):
        test=set(manifest.loc[manifest.fold==fold,"recording_id"]); train=set(manifest.loc[manifest.fold!=fold,"recording_id"])
        if test&train: raise AssertionError(f"Fold {fold} leaks recordings")
        counts=manifest.loc[manifest.fold==fold].groupby("instrument")["recording_id"].nunique()
        if not (counts==1).all(): raise AssertionError(f"Fold {fold} does not hold out one source per class")
