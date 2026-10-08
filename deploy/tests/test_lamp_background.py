"""Restored v2 discovery retains the original median background."""
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from night_lamp.session import NightSession, TemporalSeries
from server.contracts import NightLampSpec, BaselineSpec, OvernightSpec


class BackgroundTests(unittest.TestCase):
    def test_lookup_radius_separate_from_dedup(self):
        series = TemporalSeries(2)
        row = series.track(563,283)
        self.assertEqual(series.lookup(560,280),row)
        self.assertIsNone(series.lookup(560,280,radius=4))
        self.assertIsNone(series.lookup(560,277))

    def test_jump_reset_switch(self):
        for enabled in (False,True):
            with patch('night_lamp.session._CHANGE_POINT_RESET_ENABLED',enabled):
                spec = NightLampSpec(baseline=BaselineSpec(frames=60),overnight=OvernightSpec(enabled=False))
                session = NightSession('test',spec)
                for i in range(60):
                    session.accumulate(np.full((32,32,3),20,np.uint8),i*.16)
                frame = np.full((32,32,3),60,np.uint8)
                frame[12:17,12:17]=150
                session.accumulate(frame,60*.16)
                # Global correction subtracts 40; lit pixel becomes 110.
                self.assertAlmostEqual(float(session.dump_state()['base'][14,14]),110 if enabled else 20,delta=.1)
                session.release()

    def test_bright_majority_uses_original_median(self):
        spec = NightLampSpec(baseline=BaselineSpec(frames=60),
                             overnight=OvernightSpec(enabled=False))
        session = NightSession('test', spec)
        for i in range(60):
            frame = np.full((32, 32, 3), 20, np.uint8)
            if i % 4 != 0:
                frame[12:17, 12:17] = 80
            session.accumulate(frame, i * .16)
        self.assertAlmostEqual(float(session.dump_state()['base'][14, 14]), 80, delta=.1)
        self.assertEqual(float(session.dump_state()['on'][14, 14]), 0)
        for i in range(60, 80):
            frame = np.full((32, 32, 3), 20, np.uint8)
            frame[12:17, 12:17] = 80
            session.accumulate(frame, i * .16)
        self.assertAlmostEqual(float(session.dump_state()['base'][14, 14]), 80, delta=.1)
        session.release()

    def test_constant_background_uses_its_mean(self):
        spec = NightLampSpec(baseline=BaselineSpec(frames=60),
                             overnight=OvernightSpec(enabled=False))
        session = NightSession('test', spec)
        for i in range(60):
            session.accumulate(np.full((32, 32, 3), 35, np.uint8), i * .16)
        self.assertAlmostEqual(float(session.dump_state()['base'][14, 14]), 35, delta=.1)
        session.release()


if __name__ == '__main__':
    unittest.main()
