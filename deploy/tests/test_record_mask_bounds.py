import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from night_lamp.session import NightSession


class RecordMaskBoundsTests(unittest.TestCase):
    def test_saved_crop_bounds(self):
        mask = np.array([True, False, True, False, False], bool)
        packed = np.zeros((2, 3, 1), np.uint8)
        packed[1, 2] = np.packbits(mask, bitorder='little')
        session = SimpleNamespace(
            _disturbance_packed=packed, _disturbance_origin=(54, 10),
            _n_seen=5,
            _series=SimpleNamespace(
                _pys=np.array([55, 53, 56, 54, 54]),
                _pxs=np.array([12, 10, 10, 9, 13]),
                region_series=lambda row: np.zeros((3, 3))))
        np.testing.assert_array_equal(
            NightSession._record_disturbance_mask(session, 0), mask[-3:])
        for row in range(1, 5):
            with self.subTest(row=row):
                self.assertIsNone(NightSession._record_disturbance_mask(session, row))


if __name__ == '__main__':
    unittest.main()
