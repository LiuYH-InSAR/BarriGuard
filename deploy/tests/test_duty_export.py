"""Duty diagnostics preserve accumulated counts independently of lamp verdicts."""
import sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'deploy/app'))
sys.path.insert(0, str(ROOT / 'deploy/tools'))
from night_lamp.session import NightSession
from night_lamp.evidence import analyse_region
from night_intermediates import save_duty_maps
from server.contracts import NightLampSpec, BaselineSpec, OvernightSpec


class DutyTests(unittest.TestCase):
    def test_duty_is_count_ratio_and_images_are_written(self):
        session = NightSession('test', NightLampSpec(
            baseline=BaselineSpec(frames=60), overnight=OvernightSpec(enabled=False)))
        for i in range(120):
            image = np.full((32, 32, 3), 20, np.uint8)
            image[12:17, 12:17] = 80 if i % 4 == 0 else 20
            session.accumulate(image, i * .16)
        arrays = session.intermediate_arrays()
        self.assertAlmostEqual(float(arrays['duty'][14, 14]), .25)
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as out:
            save_duty_maps(out, arrays['duty'], arrays['valid_mask'])
            np.testing.assert_array_equal(np.load(Path(out) / 'duty.npy'), arrays['duty'])
            for name in ('duty_map.jpg', 'duty_candidate_mask.jpg'):
                image = cv.imread(str(Path(out) / name))
                self.assertEqual(image.shape[:2], (97, 32))
        session.release()

    @patch.dict('night_lamp.disturbance.PARAMETERS', enabled=False)
    def test_25_repeated_flashes_pass_20_but_fail_50(self):
        index = np.arange(200)
        on = (index < 150) & (index % 6 < 2)
        bgr = 20 + on[:, None] * np.array([5, 70, 80])
        period = SimpleNamespace(onset_min=20, peak_floor=.2, min_peaks=3, gap_tol=1)
        result = analyse_region(bgr, period, 2, 44)
        self.assertEqual(result['onsets'], 25)
        self.assertTrue(result['flashing'])
        period.onset_min = 50
        self.assertEqual(analyse_region(bgr, period, 2, 44)['reason'], 'too_few_onsets')


if __name__ == '__main__':
    unittest.main()
