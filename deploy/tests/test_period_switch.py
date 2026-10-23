import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / name) for name in ("app", "tools")]
from night_lamp.evidence import analyse_region
from server.contracts import PeriodicitySpec
from night_intermediates import write_plot_index, _lamp_ids


class PeriodSwitchTests(unittest.TestCase):
    def test_only_poly_points_are_selected(self):
        from types import SimpleNamespace
        checks = SimpleNamespace(polys={"lamp01":np.array([[1,1],[5,1],[5,5],[1,5]],np.int32)})
        self.assertEqual(_lamp_ids(checks,dict(x=3,y=3)),["lamp01"])
        self.assertEqual(_lamp_ids(checks,dict(x=1,y=3)),["lamp01"])
        self.assertEqual(_lamp_ids(checks,dict(x=10,y=10)),[])
        self.assertIsNone(_lamp_ids(None,dict(x=10,y=10)))

    def test_only_period_filter_is_bypassed(self):
        rng = np.random.default_rng(7)
        on = rng.random(1000) > .5
        bgr = 20 + on[:,None] * np.array([10,90,90]) + rng.normal(0,.5,(1000,3))
        period = PeriodicitySpec(onset_min=20, peak_floor=1.1)
        self.assertFalse(analyse_region(bgr,period,2,44,force_stutter=False)["flashing"])
        period.enabled = False
        result = analyse_region(bgr,period,2,44,force_stutter=False)
        self.assertTrue(result["flashing"])
        self.assertEqual(result["reason"], "periodicity_disabled")
        self.assertFalse(analyse_region(np.full((1000,3),20),period,2,44,
                                        force_stutter=False)["flashing"])

    def test_onset_count_still_required(self):
        signal = np.full((1000,3),20.)
        signal[500:] += [10,90,90]
        result = analyse_region(signal,PeriodicitySpec(enabled=False,onset_min=20),2,44,
                                force_stutter=False)
        self.assertFalse(result["flashing"])

    def test_plot_index_includes_failed_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = root / "2000" / "intermediate"
            out.mkdir(parents=True)
            (out / "manifest.json").write_text(json.dumps({"candidates":[
                dict(candidate_id=0,x=10,y=20,reason="not_periodic",flashing=False)]}))
            (out / "candidate_000.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
            self.assertEqual(write_plot_index(root),1)
            text = (root / "index.html").read_text(encoding="utf-8")
            self.assertIn("not_periodic",text)
            self.assertIn("2000/intermediate/candidate_000.svg",text)
