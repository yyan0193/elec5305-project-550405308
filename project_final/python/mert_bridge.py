"""Frozen MERT extraction only. Classification/evaluation belongs to MATLAB.

extract: all manifest clips -> MAT file; export: validated Stage 2 NPZ -> MAT;
single: new audio -> JSON embedding for MATLAB's live demonstration.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from scipy.io import savemat, loadmat

MODEL = "m-a-p/MERT-v1-95M"
REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"


def manifest(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [r["segment_id"] for r in rows]
    if len(ids) != 988 or len(set(ids)) != 988:
        raise ValueError("Expected exactly 988 unique frozen manifest segments")
    for r in rows:
        if float(r["duration_seconds"]) != 5 or int(r["fold"]) != int(r["recording_number"]):
            raise ValueError("Invalid manifest segment or fold")
    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


def source_path(root, row):
    name = f'{int(row["instrument_id"])}.{int(row["recording_number"])}.wav'
    for p in [Path(root) / name, Path(root) / row["instrument_id"] / name]:
        if p.is_file():
            return p
    raise FileNotFoundError(f"Missing {name} below {root}")


def audio_fingerprints(root, rows):
    result = {}
    for row in rows:
        rid = row["recording_id"]
        if rid in result:
            continue
        h = hashlib.sha256()
        with source_path(root, row).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1048576), b""):
                h.update(chunk)
        result[rid] = h.hexdigest()
    return result


def read_audio(path, start, seconds):
    import librosa
    import soundfile as sf
    info = sf.info(path)
    first = round(start * info.samplerate)
    n = round(seconds * info.samplerate)
    if start < 0 or first + n > info.frames:
        raise ValueError("Input must contain a complete five-second window")
    y, sr = sf.read(path, start=first, frames=n, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != 24000:
        y = librosa.resample(y, orig_sr=sr, target_sr=24000)
    y = np.pad(y, (0, max(0, 120000-len(y))))[:120000]
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite audio")
    return y


def load_encoder(device):
    import torch
    from transformers import AutoFeatureExtractor, AutoModel
    torch.manual_seed(550405308)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable")
    opts = dict(revision=REVISION, trust_remote_code=True)
    extractor = AutoFeatureExtractor.from_pretrained(MODEL, **opts)
    encoder = AutoModel.from_pretrained(MODEL, **opts).to(device).eval()
    if int(extractor.sampling_rate) != 24000:
        raise ValueError("Unexpected model sample rate")
    return extractor, encoder


def embed(y, extractor, encoder, device):
    import torch
    inputs = extractor(y, sampling_rate=24000, return_tensors="pt")
    with torch.inference_mode():
        out = encoder(input_values=inputs["input_values"].to(device))
        x = out.last_hidden_state.mean(1).squeeze(0).cpu().numpy().astype(np.float32)
    if x.shape != (768,) or not np.isfinite(x).all():
        raise ValueError("Invalid MERT embedding")
    return x


def save_mat(output, ids, X, metadata):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    savemat(output, {"X": np.asarray(X, dtype=np.float32),
                    "segment_ids": np.asarray(ids, dtype=object).reshape(-1, 1),
                    "metadata": metadata}, do_compression=True)
    # Verify the actual serialised numeric matrix before returning to MATLAB.
    actual = loadmat(output, simplify_cells=True)
    if not np.array_equal(actual["X"], X):
        raise RuntimeError("MAT roundtrip changed embeddings")


def extract(args):
    rows, digest = manifest(args.manifest)
    fingerprints = audio_fingerprints(args.audio_root, rows)
    output = Path(args.output)
    if output.exists():
        old = loadmat(output, simplify_cells=True)
        meta = old.get("metadata", {})
        ids = [str(v) for v in np.asarray(old.get("segment_ids", [])).reshape(-1)]
        valid = (meta.get("manifest_sha256") == digest
                 and meta.get("revision") == REVISION
                 and meta.get("audio_sha256_json") == json.dumps(fingerprints, sort_keys=True)
                 and ids == [r["segment_id"] for r in rows]
                 and np.asarray(old.get("X", [])).shape == (988, 768)
                 and np.isfinite(old["X"]).all())
        if valid:
            print("Reusing verified frozen MERT feature cache; no classifier or predictions loaded.")
            return
        if not args.overwrite:
            raise ValueError("Existing MERT cache is stale or unverifiable. Move it aside or explicitly use --overwrite.")
    load_start = time.perf_counter()
    extractor, encoder = load_encoder(args.device)
    load_seconds = time.perf_counter()-load_start
    start = time.perf_counter()
    vectors = []
    for k, row in enumerate(rows):
        y = read_audio(source_path(args.audio_root, row), float(row["start_seconds"]), 5)
        vectors.append(embed(y, extractor, encoder, args.device))
        if (k+1) % 25 == 0:
            print(f"MERT extraction {k+1}/988", flush=True)
    metadata = dict(representation="frozen_MERT", model_name=MODEL, revision=REVISION,
                    layer=-1, pooling="temporal_mean", dimension=768, sample_rate=24000,
                    extraction_seconds=time.perf_counter()-start,
                    model_loading_seconds=load_seconds, includes_audio_io=True,
                    device=args.device, manifest_sha256=digest,
                    audio_sha256_json=json.dumps(fingerprints, sort_keys=True),
                    origin="new Python frozen-encoder extraction; MATLAB downstream evaluation",
                    python_version=platform.python_version())
    save_mat(output, [r["segment_id"] for r in rows], np.stack(vectors), metadata)
    print(f"Saved 988 x 768 features to {output}")


def export(args):
    rows, digest = manifest(args.manifest)
    c = np.load(args.cache, allow_pickle=False)
    ids = c["segment_ids"].astype(str).tolist()
    X = c["features"]
    meta = json.loads(str(c["metadata_json"].item()))
    if len(set(ids)) != len(ids) or set(ids) != {r["segment_id"] for r in rows}:
        raise ValueError("Stage 2 cache does not exactly match frozen manifest IDs")
    for key, value in [("model_name", MODEL), ("revision", REVISION), ("layer", -1),
                       ("pooling", "temporal_mean"), ("sample_rate", 24000)]:
        if meta.get(key) != value:
            raise ValueError(f"Missing/incompatible Stage 2 provenance: {key}")
    if X.shape != (988, 768) or not np.isfinite(X).all():
        raise ValueError("Invalid Stage 2 embedding shape or values")
    if Path(args.output).exists():
        raise FileExistsError("Export will not overwrite an existing file")
    by_id = {sid:i for i,sid in enumerate(ids)}
    X = X[[by_id[r["segment_id"]] for r in rows]].astype(np.float32)
    meta.update(manifest_sha256=digest, origin="Stage 2 NPZ export, NOT newly extracted",
                audio_verification="IDs and recorded metadata checked; raw audio identity not reverified")
    save_mat(args.output, [r["segment_id"] for r in rows], X, meta)


def single(args):
    if Path(args.output).exists():
        raise FileExistsError("Single-audio output must be a new temporary file")
    t = time.perf_counter()
    ex, enc = load_encoder(args.device)
    load_seconds = time.perf_counter()-t
    t = time.perf_counter()
    y = read_audio(args.audio, args.start, 5)
    x = embed(y, ex, enc, args.device)
    result = dict(embedding=x.tolist(), model=MODEL, revision=REVISION,
                  extraction_seconds=time.perf_counter()-t,
                  loading_seconds=load_seconds, mode="fresh_single_audio_embedding")
    Path(args.output).write_text(json.dumps(result), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    for mode in ["extract", "export", "single"]:
        q = sub.add_parser(mode)
        q.add_argument("--output", required=True)
        if mode != "single":
            q.add_argument("--manifest", required=True)
        if mode == "extract":
            q.add_argument("--audio-root", required=True)
            q.add_argument("--overwrite", action="store_true")
        if mode == "export":
            q.add_argument("--cache", required=True)
        else:
            q.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        if mode == "single":
            q.add_argument("--audio", required=True)
            q.add_argument("--start", type=float, default=0)
    a = p.parse_args()
    {"extract": extract, "export": export, "single": single}[a.mode](a)


if __name__ == "__main__":
    main()
