"""Extract interpretable classical timbral features for each segment."""
import argparse,time
import librosa
import numpy as np
from tqdm import tqdm
from src.project_utils import load_config,load_manifest,read_segment,save_feature_cache

def stats(x): return np.column_stack([np.mean(x,axis=1),np.std(x,axis=1)]).ravel()
def feature_names(n_mfcc,n_contrast):
    names=[]
    for i in range(n_mfcc): names += [f"mfcc_{i+1:02d}_mean",f"mfcc_{i+1:02d}_std"]
    for name in ["spectral_centroid","spectral_bandwidth","spectral_rolloff"]: names += [f"{name}_mean",f"{name}_std"]
    for i in range(n_contrast+1): names += [f"spectral_contrast_{i+1:02d}_mean",f"spectral_contrast_{i+1:02d}_std"]
    names += ["rms_mean","rms_std","zcr_mean","zcr_std"]; return names
def extract(y,sr,c):
    kw=dict(n_fft=int(c["n_fft"]),hop_length=int(c["hop_length"])); parts=[librosa.feature.mfcc(y=y,sr=sr,n_mfcc=int(c["n_mfcc"]),**kw),librosa.feature.spectral_centroid(y=y,sr=sr,**kw),librosa.feature.spectral_bandwidth(y=y,sr=sr,**kw),librosa.feature.spectral_rolloff(y=y,sr=sr,roll_percent=float(c["roll_percent"]),**kw),librosa.feature.spectral_contrast(y=y,sr=sr,n_bands=int(c["n_contrast_bands"]),**kw),librosa.feature.rms(y=y,frame_length=int(c["n_fft"]),hop_length=int(c["hop_length"])),librosa.feature.zero_crossing_rate(y,frame_length=int(c["n_fft"]),hop_length=int(c["hop_length"]))]
    return np.concatenate([stats(x) for x in parts]).astype(np.float32)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config/experiment.yaml"); args=p.parse_args(); cfg=load_config(args.config); man=load_manifest(cfg["data"]["manifest_path"]); sr=int(cfg["data"]["analysis_sample_rate"]); start=time.perf_counter(); vectors=[]
    for _,row in tqdm(man.iterrows(),total=len(man),desc="Classical features"): vectors.append(extract(read_segment(row,sr),sr,cfg["classical"]))
    elapsed=time.perf_counter()-start; x=np.vstack(vectors); save_feature_cache(cfg["classical"]["output_path"],man.segment_id,x,feature_names(int(cfg["classical"]["n_mfcc"]),int(cfg["classical"]["n_contrast_bands"])),dict(representation="classical_timbral",sample_rate=sr,segments=len(man),dimension=x.shape[1],extraction_seconds=elapsed,seconds_per_segment=elapsed/len(man)))
    print(x.shape,elapsed)
if __name__=="__main__": main()
