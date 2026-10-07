"""Audit ChMusic and create a source-aware five-second segment manifest."""
import argparse, math
from pathlib import Path
import pandas as pd
import soundfile as sf
from src.project_utils import INSTRUMENTS,ensure_parent,load_config,validate_source_separation

def parse_filename(path):
    parts=path.stem.split(".")
    if len(parts)!=2: raise ValueError(path.name)
    return tuple(map(int,parts))
def build_manifest(config):
    audio_dir=Path(config["data"]["audio_dir"]); seconds=float(config["data"]["segment_seconds"]); files=sorted(audio_dir.glob("*.wav"),key=parse_filename)
    if len(files)!=55: raise ValueError(f"Expected 55 WAV files in {audio_dir}, found {len(files)}")
    recordings=[]; segments=[]
    for path in files:
        iid,rnum=parse_filename(path); name=INSTRUMENTS[iid]; rid=f"{name.lower()}_r{rnum}"; info=sf.info(path); duration=info.frames/info.samplerate; n=math.floor(duration/seconds)
        recordings.append(dict(recording_id=rid,instrument_id=iid,instrument=name,recording_number=rnum,source_path=path.as_posix(),sample_rate=info.samplerate,channels=info.channels,frames=info.frames,duration_seconds=duration,segment_count=n,status="ok" if n else "too_short"))
        for i in range(n): segments.append(dict(segment_id=f"{rid}_s{i+1:03d}",instrument_id=iid,instrument=name,recording_number=rnum,recording_id=rid,source_path=path.as_posix(),start_seconds=i*seconds,duration_seconds=seconds,fold=rnum,published_split="test" if rnum==5 else "train"))
    rec=pd.DataFrame(recordings).sort_values(["instrument_id","recording_number"]); man=pd.DataFrame(segments).sort_values(["instrument_id","recording_number","start_seconds"])
    validate_source_separation(man); return rec,man
def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config/experiment.yaml"); c=load_config(p.parse_args().config); rec,man=build_manifest(c)
    rec.to_csv(ensure_parent(c["data"]["recording_audit_path"]),index=False); man.to_csv(ensure_parent(c["data"]["manifest_path"]),index=False)
    print(f"Audited {len(rec)} recordings; created {len(man)} segments")
if __name__=="__main__": main()
