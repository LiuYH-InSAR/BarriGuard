"""Replay the occlusion and incomplete baseline failures with a controlled clock."""
from dataclasses import replace
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from water_barrier.yolo_params import PARAMS
from water_barrier.yolo_pipeline import YoloWaterAlgorithm
from water_barrier.yolo_store import Store
from water_barrier.yolo_vision import Vision


class WaterGapRegression(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[2] / 'tmp'
        scratch.mkdir(exist_ok=True)
        tmp = TemporaryDirectory(dir=scratch)
        self.addCleanup(tmp.cleanup)
        self.stamp = 1000000.
        self.frame = np.full((100, 120, 3), 150, np.uint8)
        self.box = [5., 12., 17., 36.]
        self.neighbor = [20., 12., 32., 36., .9]
        self.params = replace(
            PARAMS, storage=replace(PARAMS.storage, root=Path(tmp.name)),
            debug=replace(PARAMS.debug, enabled=False),
            statistics=replace(PARAMS.statistics, window_s=90., cooldown_s=900.),
            policy=replace(PARAMS.policy, day_confirm_samples=1, suspect_day_s=120.,
                           night_credit_s=0.))
        self.detector = Mock()
        self.vision = Mock()
        self.algo = self.make_algorithm()

    def make_algorithm(self):
        algo = YoloWaterAlgorithm(self.frame.shape, camera_id='1750', params=self.params,
                                  detector=self.detector, vision=self.vision,
                                  wall_clock=lambda: self.stamp)
        algo._quality = lambda *args: dict(ok=True, light='day')
        algo._daylight.candidate_for = lambda *args: 'day'
        algo._alignment_ok = True
        algo._matrix = np.eye(3)
        algo._refs.expired = lambda stamp: False
        algo._refs.view = 'test'
        algo._refs.boxes = [self.box, self.neighbor[:4]]
        algo._refs.identities = ['test:B6', 'test:B7']
        algo._refs.rows = [[0, 1]]
        algo._refs.gap_scan_complete = True
        return algo

    def tick(self, detections):
        self.stamp += 30.
        self.frame[-1, -1] += 1
        self.detector.detect.return_value = detections
        return self.algo._detect(self.frame, self.stamp, self.stamp)

    def test_occlusion_can_clear_into_confirmed_missing(self):
        self.vision.road_evidence.return_value = False, dict(reason='occluded')
        for _ in range(8):
            self.assertEqual(self.tick([self.neighbor]), [])
        self.assertEqual(self.algo._accumulation.records['test:B6']['day_s'], 0)
        self.vision.road_evidence.return_value = True, dict(reason='road_exposed')
        self.assertFalse(self.tick([self.neighbor]))
        self.assertFalse(self.tick([self.neighbor]))
        events = self.tick([self.neighbor])
        self.assertEqual(events[0].payload['error_reason'], 'confirmed_missing')
        self.vision.road_evidence.return_value = False, dict(reason='occluded')
        self.assertFalse(self.tick([self.neighbor]))
        self.assertFalse(any(a.level == 'alarm' for a in self.algo._annots))

    def test_old_occlusion_timeout_does_not_survive_upgrade(self):
        Store(self.params, '1750').snapshot(self.stamp, dict(
            policy_version=3, model_sha256='old',
            episodes=dict(records={
                'test:B6': dict(active=True, notified=True, error_reason='suspect_timeout'),
                'test:B7': dict(active=True, notified=True, last_alarm=self.stamp,
                                error_reason='confirmed_missing')})))
        restored = self.make_algorithm()
        self.assertNotIn('test:B6', restored._episodes.records)
        self.assertTrue(restored._episodes.records['test:B7']['active'])

    def test_stable_new_slot_enters_baseline_and_can_alarm(self):
        added = [35., 12., 47., 36., .9]
        self.vision.road_evidence.return_value = False, dict(reason='occluded')
        self.tick([self.neighbor, added])
        self.tick([self.neighbor])
        for _ in range(self.params.reference.min_baseline_observations - 1):
            self.tick([self.neighbor, added])
        self.assertEqual(len(self.algo._refs.boxes), 2)
        self.tick([self.neighbor, added])
        self.assertEqual(len(self.algo._refs.boxes), 3)
        def evidence(frame, boxes, index, *args):
            return (True, dict(reason='road_exposed')) if index == 2 else (False, dict(reason='occluded'))
        self.vision.road_evidence.side_effect = evidence
        self.tick([self.neighbor])
        self.tick([self.neighbor])
        events = self.tick([self.neighbor])
        self.assertEqual(events[0].payload['barrier_ids'], ['test:B2'])

    def test_inferred_slot_requires_road_evidence(self):
        self.algo._refs.inferred_indices = [0]
        self.vision.road_evidence.return_value = False, dict(reason='occluded')
        for _ in range(5):
            self.assertFalse(self.tick([self.neighbor]))
        self.vision.road_evidence.return_value = True, dict(reason='road_exposed')
        self.assertFalse(self.tick([self.neighbor]))
        self.assertFalse(self.tick([self.neighbor]))
        self.assertEqual(self.tick([self.neighbor])[0].payload['error_reason'], 'confirmed_missing')

    def test_small_position_can_sample_exposed_road(self):
        box = [5., 12., 15., 36.]
        road, details = Vision(self.params).road_evidence(
            self.frame, [box, self.neighbor[:4]], 0, {1: 0}, [[0, 1]], {0})
        self.assertTrue(road, details)

    def test_row_with_several_losses_rechecks_after_first_sample(self):
        boxes = [[x, 12., x+10., 36.] for x in (5., 20., 35., 50., 65., 80.)]
        vision = Vision(self.params)
        matches = {0: 0, 1: 1}
        rows = [list(range(len(boxes)))]
        first, details = vision.road_evidence(self.frame, boxes, 5, matches, rows, set())
        self.assertFalse(first)
        self.assertEqual(details['reason'], 'simultaneous_disappearance')
        road, details = vision.road_evidence(self.frame, boxes, 5, matches, rows, {2, 3, 4, 5})
        self.assertTrue(road, details)
