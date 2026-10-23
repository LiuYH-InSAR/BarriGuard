import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from night_lamp.session import NightSession


class ManualLampRoiTests(unittest.TestCase):
    def session(self, fixed=True, legacy=True):
        session = NightSession.__new__(NightSession)
        session._h, session._w = 20, 30
        session._mask = np.zeros((20, 30), bool)
        session._lamp_roi = {"calibration": "old.yaml"} if legacy else {}
        session._lamp_roi_frac = None
        polys = {"lamp01": np.array([[2, 2], [8, 2], [8, 8], [2, 8]], np.int32),
                 "lamp02": np.array([[20, 10], [25, 10], [25, 15], [20, 15]], np.int32)}
        session._lamp_checks = SimpleNamespace(size=(20, 30), polys=polys) if fixed else None
        return session

    def test_manual_polys_replace_old_band(self):
        session = self.session()
        with patch("night_lamp.session.nm.lamp_roi_mask", return_value=np.ones((20, 30), bool)) as old:
            session._merge_lamp_roi()
            old.assert_not_called()
        self.assertFalse(session._mask[4, 4])
        self.assertFalse(session._mask[12, 22])
        self.assertTrue(session._mask[18, 28])
        expected = np.zeros((20, 30), np.uint8)
        cv.fillPoly(expected, list(session._lamp_checks.polys.values()), 1)
        np.testing.assert_array_equal(~session._mask, expected > 0)
        self.assertEqual(session._lamp_roi_frac, round(float(expected.mean()), 4))

    def test_manual_polys_work_without_old_band(self):
        session = self.session(legacy=False)
        session._merge_lamp_roi()
        self.assertFalse(session._mask[4, 4])
        self.assertTrue(session._mask[18, 28])

    def test_osd_remains_excluded_inside_manual_poly(self):
        session = self.session()
        session._mask[4, 4] = True
        session._merge_lamp_roi()
        self.assertTrue(session._mask[4, 4])
        self.assertFalse(session._mask[4, 5])

    def test_legacy_band_unchanged_without_manual_polys(self):
        session = self.session(fixed=False)
        roi = np.zeros((20, 30), bool)
        roi[10:15, 10:15] = True
        with patch("night_lamp.session.nm.lamp_roi_mask", return_value=roi) as old:
            session._merge_lamp_roi()
            old.assert_called_once()
        np.testing.assert_array_equal(session._mask, ~roi)

    def test_no_roi_keeps_original_mask(self):
        session = self.session(fixed=False, legacy=False)
        session._mask[4, 4] = True
        before = session._mask.copy()
        session._merge_lamp_roi()
        np.testing.assert_array_equal(session._mask, before)

    def test_wrong_stream_size_fails_before_sampling(self):
        session = self.session()
        session._lamp_checks.size = (10, 10)
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            session._merge_lamp_roi()


if __name__ == "__main__":
    unittest.main()
