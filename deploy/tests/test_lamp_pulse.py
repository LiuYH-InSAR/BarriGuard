"""Returned repeated pulses are distinct from steady light and isolated bursts."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from night_lamp.pulse import analyse_pulses
from server.contracts import PulseSpec, PeriodicitySpec


class PulseTests(unittest.TestCase):
    def analyse(self, values, spec=None):
        return analyse_pulses(values, 6.25, spec or PulseSpec(), PeriodicitySpec())

    def test_repeated_narrow_peaks(self):
        values = np.full(1000, 30.0)
        values[3::6] = 100
        result, mask = self.analyse(values)
        self.assertTrue(result["flashing"])
        self.assertGreater(result["pulse_count"], 100)
        self.assertAlmostEqual(result["pulses"][0]["width_s"], 0.32)
        self.assertEqual(mask[3], 1)

    def test_exposure_step_does_not_return(self):
        values = np.full(1000, 30.0)
        values[500:] = 100
        self.assertFalse(self.analyse(values)[0]["flashing"])
        self.assertEqual(self.analyse(values)[0]["pulse_count"], 0)

    def test_long_plateaus(self):
        values = np.where(np.arange(2000) % 40 < 20, 100, 30)
        self.assertFalse(self.analyse(values)[0]["flashing"])

    def test_short_burst(self):
        values = np.full(1000, 30.0)
        values[3:123:2] = 100
        self.assertEqual(self.analyse(values)[0]["reason"], "not_sustained")

    def test_separated_bursts(self):
        values = np.full(5000, 30.0)
        for start in range(100, 4100, 200):
            values[start:start + 30:6] = 100
        self.assertEqual(self.analyse(values)[0]["reason"], "not_sustained")

    def test_threshold_changes_verdict(self):
        values = np.full(1000, 30.0)
        values[3::6] = 45
        self.assertFalse(self.analyse(values)[0]["flashing"])
        self.assertTrue(self.analyse(values, PulseSpec(amplitude_min=10))[0]["flashing"])


if __name__ == "__main__":
    unittest.main()
