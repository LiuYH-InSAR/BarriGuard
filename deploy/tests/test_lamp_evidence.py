"""Behaviour checks: two states alone must not be labelled periodic lamps."""
import sys
import unittest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from night_lamp.evidence import analyse_region
from night_lamp.session import TemporalSeries
from night_lamp.session import NightSession
from night_intermediates import save_intermediates
from server.contracts import (NightLampSpec, BaselineSpec, MemorySpec,
                              SamplingSpec, PeriodicitySpec, AlignmentSpec,
                              OvernightSpec)

PERIOD = SimpleNamespace(onset_min=50, peak_floor=0.2, min_peaks=3, gap_tol=1)


class EvidenceTests(unittest.TestCase):
    def signal(self, on, delta):
        noise = np.random.default_rng(42).normal(0, 1, (len(on), 3))
        return 20 + noise + np.asarray(on)[:, None] * np.asarray(delta)

    def analyse(self, on, delta):
        return analyse_region(self.signal(on, delta), PERIOD, 2, 44)

    def test_yellow_periodic(self):
        on = np.arange(1000) % 6 < 3
        result = self.analyse(on, [5, 70, 80])
        self.assertTrue(result["flashing"])
        self.assertGreater(result["yellow_score"], 0.8)
        self.assertAlmostEqual(result["duty"], 0.5, delta=0.01)

    def test_white_periodic_survives(self):
        result = self.analyse(np.arange(1000) % 6 < 3, [70, 70, 70])
        self.assertTrue(result["flashing"])
        self.assertLess(result["yellow_score"], 0.1)

    def test_short_on_state(self):
        result = self.analyse(np.arange(1000) % 10 == 0, [5, 70, 80])
        self.assertTrue(result["flashing"])
        self.assertAlmostEqual(result["duty"], 0.1, delta=0.01)

    def test_single_exposure_switch(self):
        result = self.analyse(np.arange(1000) >= 500, [5, 70, 80])
        self.assertFalse(result["flashing"])
        self.assertGreater(result['disturbance']['masked_samples'],0)
        self.assertEqual(result["reason"], "states_not_separated")

    def test_noise_is_not_lamp(self):
        result = self.analyse(np.zeros(1000), [5, 70, 80])
        self.assertFalse(result["flashing"])

    def test_random_two_states(self):
        on = np.random.default_rng(9).random(1000) < 0.5
        self.assertFalse(self.analyse(on, [5, 70, 80])["flashing"])

    def test_region_and_tail(self):
        series = TemporalSeries(2, cap=4)
        row = series.track(2, 2)
        mask = np.zeros((5, 5), bool)
        for i in range(8):
            frame = np.full((5, 5, 3), i, np.uint8)
            frame[2, 2] = 100
            series.append(frame[..., 0], mask, 0, frame)
        expected = [(24 * i + 100) / 25 for i in range(4, 8)]
        np.testing.assert_allclose(series.region_series(row)[:, 0], expected)

    def test_late_discovery(self):
        series = TemporalSeries(2)
        frame = np.full((5, 5, 3), 20, np.uint8)
        mask = np.zeros((5, 5), bool)
        for _ in range(1100):
            series.append(frame[..., 0], mask, 0, frame)
        row = series.track(2, 2)
        series.append(frame[..., 0], mask, 0, frame)
        np.testing.assert_allclose(series.region_series(row), [[20, 20, 20]])

    def test_export_raw_time_and_missing(self):
        series = TemporalSeries(2)
        mask = np.zeros((5, 5), bool)
        for i in range(3):
            frame = np.full((5, 5, 3), [10, 20, 30], np.uint8)
            series.append(frame[..., 0], mask, 7, frame, ts=i * 0.16)
        row = series.track(2, 2)
        for i in range(3, 83):
            frame = np.full((5, 5, 3), [10, 20, 30], np.uint8)
            series.append(frame[..., 0], mask, 7, frame, ts=i * 0.16)
        arrays = series.export_arrays()
        self.assertTrue(np.isnan(arrays["gray_raw"][row, :3]).all())
        self.assertNotIn("rgb_raw", arrays)
        np.testing.assert_allclose(arrays["gray_raw"][row, 3:], np.full(80, 21.85))
        np.testing.assert_allclose(arrays["gray_corrected"][row, 3:], np.full(80, 14.85))
        np.testing.assert_allclose(arrays["time_s"], np.arange(83) * 0.16)
        session = SimpleNamespace(intermediate_arrays=series.export_arrays)
        report = SimpleNamespace(image_jpeg=None, metadata={
            "lamp_method": {"evidence_scope": "current_window"},
            "lamp_evidence": [{"series_row": row, "x": 2, "y": 2,
                               "reason": "states_not_separated", "flashing": False}]})
        root = Path(__file__).resolve().parents[2] / "tmp"
        with tempfile.TemporaryDirectory(dir=root) as folder:
            save_intermediates(session, report, folder)
            with np.load(Path(folder) / "series.npz", allow_pickle=False) as saved:
                np.testing.assert_allclose(saved["gray_raw"], arrays["gray_raw"])
                self.assertTrue(np.isnan(saved["on_state"][row, :3]).all())
            self.assertTrue((Path(folder) / "candidate_000.svg").exists())
            self.assertTrue((Path(folder) / "candidate_000.csv").exists())
            manifest = json.loads((Path(folder) / "manifest.json").read_text())
            self.assertEqual(manifest["channel_order"], "Gray")

    def test_export_tail_timestamps(self):
        series = TemporalSeries(1, cap=4)
        mask = np.zeros((5, 5), bool)
        row = series.track(2, 2)
        for i in range(8):
            frame = np.full((5, 5, 3), i, np.uint8)
            series.append(frame[..., 0], mask, 0, frame, ts=i * 0.16)
        arrays = series.export_arrays()
        np.testing.assert_array_equal(arrays["sample_index"], [4, 5, 6, 7])
        np.testing.assert_allclose(arrays["time_s"], np.arange(4, 8) * 0.16)
        np.testing.assert_allclose(arrays["gray_raw"][row, :], [4, 5, 6, 7])

    def test_continuous_snapshot_keeps_fit(self):
        spec = NightLampSpec(
            sampling=SamplingSpec(interval_ms=160),
            memory=MemorySpec(max_candidates=20, series_cap=None),
            baseline=BaselineSpec(frames=60, warmup_s=600),
            periodicity=PeriodicitySpec(period_step=2, lag_lo_s=0.3,
                lag_hi_s=7, peak_floor=0.2, min_peaks=3, gap_tol=1, onset_min=50),
            alignment=AlignmentSpec(min_cc=0.5),
            overnight=OvernightSpec(enabled=False, cadence_minutes=60,
                                   burst_seconds=120, anchor="wall"))
        session = NightSession("test", spec)
        for i in range(1000):
            frame = np.full((64, 64, 3), 20, np.uint8)
            if i % 6 < 3:
                frame[28:35, 28:35] = [25, 110, 120]
            session.accumulate(frame, i * 0.16)
        snapshot = session.snapshot("test")
        final = session.finalize(None, "test")
        self.assertEqual(final.metadata["flash_bucket_count"], 0)
        self.assertEqual(final.metadata["lamp_method"]["evidence_scope"], "current_window")
        self.assertTrue(final.metadata["lamp_evidence"])
        self.assertGreater(final.metadata["n_flash"], 0)
        session.release()


if __name__ == "__main__":
    unittest.main()
