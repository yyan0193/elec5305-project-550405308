from pathlib import Path
import numpy as np,pandas as pd,pytest
from src.evaluation import recording_equal_weights
from src.prepare_dataset import build_manifest
from src.project_utils import load_config,load_manifest,validate_source_separation
from src.run_representation_experiments import combinations_for_budget
def test_manifest_has_expected_sources_and_no_leakage():
    c=load_config();
    if not Path(c["data"]["audio_dir"]).exists(): pytest.skip("Raw ChMusic audio is not stored in Git")
    rec,man=build_manifest(c); assert len(rec)==55; assert rec.instrument.nunique()==11; validate_source_separation(man)
def test_expected_data_location_is_documented(): assert load_config()["data"]["audio_dir"]=="data/raw/ChMusic/Musics"
def test_tracked_manifest_is_source_aware():
    man=load_manifest("metadata/data_manifest.csv"); assert len(man)==988; assert man.recording_id.nunique()==55; validate_source_separation(man)
def test_label_budget_combinations(): assert [len(combinations_for_budget([1,2,3,4],b)) for b in range(1,5)]==[4,6,4,1]
def test_recording_weights():
    d=pd.DataFrame({"recording_id":["s","l","l","l"],"segment_id":["s1","l1","l2","l3"]}); d["w"]=recording_equal_weights(d); t=d.groupby("recording_id").w.sum().to_numpy(); assert np.allclose(t,t[0])
