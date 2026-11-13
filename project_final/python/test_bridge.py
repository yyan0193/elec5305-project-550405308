"""Synthetic transport tests only; no claim of MERT inference or MATLAB execution."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.io import loadmat
import mert_bridge as b


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest = Path(__file__).resolve().parents[1]/"metadata/data_manifest.csv"
        self.rows, self.digest = b.manifest(self.manifest)
        self.ids = [r["segment_id"] for r in self.rows]
        self.X = np.arange(988*768, dtype=np.float32).reshape(988,768)/1000
        self.meta = dict(model_name=b.MODEL,revision=b.REVISION,layer=-1,
                         pooling="temporal_mean",sample_rate=24000,extraction_seconds=1.)

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self, ids=None, X=None, meta=None):
        path=self.root/"input.npz"
        np.savez(path,segment_ids=np.array(self.ids if ids is None else ids),
                 features=self.X if X is None else X,
                 metadata_json=np.array(json.dumps(self.meta if meta is None else meta)))
        return argparse.Namespace(manifest=str(self.manifest),cache=str(path),output=str(self.root/"out.mat"))

    def test_manifest(self):
        self.assertEqual(len(self.rows),988)
        self.assertEqual(len({r["recording_id"] for r in self.rows}),55)
        self.assertEqual(len(self.digest),64)

    def test_export_reorders_and_roundtrips(self):
        a=self.fixture(self.ids[::-1],self.X[::-1]); b.export(a)
        out=loadmat(a.output,simplify_cells=True)
        np.testing.assert_array_equal(out["X"],self.X)
        self.assertIn("NOT newly extracted",out["metadata"]["origin"])

    def test_duplicate_rejected(self):
        ids=self.ids.copy(); ids[-1]=ids[0]
        with self.assertRaises(ValueError): b.export(self.fixture(ids))

    def test_wrong_revision_rejected(self):
        meta=self.meta|{"revision":"wrong"}
        with self.assertRaises(ValueError): b.export(self.fixture(meta=meta))

    def test_wrong_dimension_rejected(self):
        with self.assertRaises(ValueError): b.export(self.fixture(X=self.X[:,:10]))

    def test_nonfinite_rejected(self):
        X=self.X.copy(); X[0,0]=np.nan
        with self.assertRaises(ValueError): b.export(self.fixture(X=X))

    def test_export_refuses_overwrite(self):
        a=self.fixture(); b.export(a)
        with self.assertRaises(FileExistsError): b.export(a)

    def test_source_lookup(self):
        row=self.rows[0]; p=self.root/"1"; p.mkdir(); (p/"1.1.wav").touch()
        self.assertEqual(b.source_path(self.root,row),p/"1.1.wav")


if __name__ == "__main__":
    unittest.main()
