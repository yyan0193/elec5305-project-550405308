"""Extract and cache frozen MERT-v1-95M segment embeddings."""
import argparse,time
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor,AutoModel
from src.project_utils import load_config,load_manifest,read_segment,save_feature_cache,set_reproducible_seed
def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config/experiment.yaml"); p.add_argument("--limit",type=int); p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto"); a=p.parse_args(); c=load_config(a.config); set_reproducible_seed(int(c["project"]["random_seed"])); man=load_manifest(c["data"]["manifest_path"]); man=man.head(a.limit) if a.limit else man
    mc=c["mert"]; extractor=AutoFeatureExtractor.from_pretrained(mc["model_name"],revision=mc.get("revision","main"),trust_remote_code=True); model=AutoModel.from_pretrained(mc["model_name"],revision=mc.get("revision","main"),trust_remote_code=True); device=torch.device("cuda" if a.device=="auto" and torch.cuda.is_available() else ("cpu" if a.device=="auto" else a.device)); model.to(device).eval(); sr=int(extractor.sampling_rate); layer=int(mc.get("layer",-1)); vectors=[]; start=time.perf_counter()
    with torch.inference_mode():
        for _,row in tqdm(man.iterrows(),total=len(man),desc=f"MERT on {device}"):
            audio=read_segment(row,sr); inputs=extractor(audio,sampling_rate=sr,return_tensors="pt"); out=model(input_values=inputs["input_values"].to(device),output_hidden_states=layer!=-1); hidden=out.last_hidden_state if layer==-1 else out.hidden_states[layer]; vectors.append(hidden.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32))
    elapsed=time.perf_counter()-start; x=np.vstack(vectors); save_feature_cache(mc["output_path"],man.segment_id,x,[f"mert_{i:04d}" for i in range(x.shape[1])],dict(representation="MERT-v1-95M",model_name=mc["model_name"],revision=mc.get("revision","main"),sample_rate=sr,layer=layer,pooling="temporal_mean",device=str(device),segments=len(man),dimension=x.shape[1],extraction_seconds=elapsed,seconds_per_segment=elapsed/len(man))); print(x.shape,elapsed)
if __name__=="__main__": main()
