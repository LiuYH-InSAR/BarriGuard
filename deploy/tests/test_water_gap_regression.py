"""Replay the occlusion and incomplete baseline failures with a controlled clock."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from camera_motion.contracts import MotionResult
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
        self.vision.road_profile.return_value = None
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
        def evidence(frame, boxes, index, *args, **kwargs):
            return (True, dict(reason='road_exposed')) if index == 2 else (False, dict(reason='occluded'))
        self.vision.road_evidence.side_effect = evidence
        self.tick([self.neighbor])
        self.tick([self.neighbor])
        events = self.tick([self.neighbor])
        self.assertEqual(events[0].payload['barrier_ids'], ['test:B2'])

    def test_inferred_slot_needs_a_real_detection_before_it_can_alarm(self):
        self.algo._refs.inferred_indices = [0]
        self.vision.road_evidence.return_value = False, dict(reason='occluded')
        for _ in range(5):
            self.assertFalse(self.tick([self.neighbor]))
        self.vision.road_evidence.return_value = True, dict(reason='road_exposed')
        for _ in range(5):
            self.assertFalse(self.tick([self.neighbor]))
        for _ in range(self.params.reference.min_baseline_observations - 1):
            self.tick([self.neighbor, self.box + [.9]])
        self.assertIn(0, self.algo._refs.inferred_indices)
        self.tick([self.neighbor, self.box + [.9]])
        self.assertNotIn(0, self.algo._refs.inferred_indices)
        self.assertFalse(self.tick([self.neighbor]))
        self.assertFalse(self.tick([self.neighbor]))
        self.assertEqual(self.tick([self.neighbor])[0].payload['error_reason'], 'confirmed_missing')

    def test_unaligned_detections_remain_visible_without_updating_baseline(self):
        self.algo._alignment_ok = False
        self.assertFalse(self.tick([self.neighbor]))
        self.assertEqual(len(self.algo._refs.boxes), 2)
        self.assertEqual(len(self.algo._annots), 1)
        self.assertTrue(self.algo._annots[0].label.startswith('Suspect - Uncalibrated'))
        self.vision.road_evidence.assert_not_called()

    def test_reconnect_discards_pre_loss_missing_evidence(self):
        self.vision.road_evidence.return_value = True, dict(reason='road_exposed')
        self.assertFalse(self.tick([self.neighbor]))
        self.assertFalse(self.tick([self.neighbor]))
        self.algo.resume_stream()
        self.assertFalse(self.algo._alignment_ok)
        self.assertEqual(self.algo._histories, {})
        self.assertEqual(self.algo._refs.boxes, [self.box, self.neighbor[:4]])
        self.algo._alignment_ok = True
        for _ in range(3):
            self.assertFalse(self.tick([self.neighbor]))
        self.assertEqual(self.tick([self.neighbor])[0].payload['error_reason'], 'confirmed_missing')

    def test_migrated_baseline_is_replaced_by_independent_observations(self):
        self.algo._refs.rebuilding = True
        self.algo._refs.carried_indices = [0, 1]
        new = [80., 12., 94., 36., .9]
        moved_neighbor = [22., 12., 34., 36., .9]
        self.vision.road_evidence.return_value = False, dict(reason='occluded')
        for _ in range(self.params.reference.min_baseline_observations - 1):
            self.tick([moved_neighbor, new])
            self.assertTrue(self.algo._refs.rebuilding)
            self.assertEqual(self.algo._refs.boxes[1], self.neighbor[:4])
        self.tick([moved_neighbor, new])
        self.assertTrue(self.algo._refs.rebuilding)
        self.assertEqual(self.algo._refs.identities[:2], ['test:B6', 'test:B7'])
        self.assertEqual(self.algo._refs.boxes[:2], [self.box, moved_neighbor[:4]])
        self.assertEqual(self.algo._refs.carried_indices, [0])
        self.assertEqual(self.algo.status()['baseline_coverage'],
                         dict(left='partial', right='calibrated'))
        self.tick([self.box + [.9], moved_neighbor, new])
        self.assertEqual(self.algo._refs.carried_indices, [0])
        for _ in range(self.params.reference.min_baseline_observations - 1):
            self.tick([self.box + [.9], moved_neighbor, new])
        self.assertEqual(self.algo._refs.carried_indices, [])
        self.assertFalse(self.algo._refs.rebuilding)

    def test_unmapped_rebuild_reports_far_and_right_independently(self):
        refs = self.algo._refs
        refs.boxes, refs.identities, refs.rows = [], [], []
        refs.rebuilding = refs.rebuild_unmapped = True
        left_near = [[5., 62., 17., 86., .9], [20., 62., 32., 86., .9]]
        for _ in range(5):
            self.tick(left_near)
        self.assertTrue(refs.rebuilding)
        self.assertEqual(self.algo.status()['baseline_segments'], dict(
            left_far='uncalibrated', left_near='calibrated', right_far='uncalibrated', right_near='uncalibrated'))
        remaining = [self.box + [.9], [80., 12., 94., 36., .9], [80., 62., 94., 86., .9]]
        for _ in range(4):
            self.tick(left_near + remaining)
        self.assertEqual(len(refs.boxes), 2)
        self.tick(left_near + remaining)
        self.assertFalse(refs.rebuilding)
        self.assertEqual(len(refs.boxes), 5)

    def test_entire_row_requires_historical_road_and_can_alarm(self):
        self.algo._vision = Vision(self.params)
        for box in self.algo._refs.boxes:
            x0,y0,x1,y1 = map(int,box)
            self.frame[y0:y1,x0:x1] = (30,30,210)
        self.tick([self.box + [.9], self.neighbor])
        self.assertTrue(all(p['road'] for p in self.algo._presence_patches.values()))
        for box in self.algo._refs.boxes:
            x0,y0,x1,y1 = map(int,box)
            self.frame[y0:y1,x0:x1] = (30,160,30)
        for _ in range(5):
            self.assertFalse(self.tick([]))
        self.frame.fill(150)
        self.assertFalse(self.tick([]))  # The first cleared image is still changing.
        self.assertFalse(self.tick([]))
        self.assertFalse(self.tick([]))
        events = self.tick([])
        self.assertEqual(events[0].payload['barrier_ids'], ['test:B6', 'test:B7'])
        self.assertEqual(events[0].payload['error_reason'], 'confirmed_missing')

    def test_migration_feedback_requires_multiple_aligned_samples(self):
        algo = self.algo
        algo._refs.boxes = [[x, 12., x+10., 36.] for x in (5., 20., 35., 50.)]
        self.detector.detect.return_value = [box + [.9] for box in algo._refs.boxes]
        result = MotionResult('1750','move','1',self.stamp,'test','new',1,'migration_pending',
                              np.eye(3).tolist(),self.frame.shape[:2],1.)
        self.vision.align.return_value = SimpleNamespace(valid=True,matrix=np.eye(3),displacement=0.)
        reference = self.frame.copy()
        for index in (1, 2, 3):
            self.frame[-1,-1] += 1
            algo.observe_offset_candidate(result,reference,self.frame,self.stamp,index*2.,index)
            if index < 3:
                self.assertFalse(algo.offset_feedback_ready(result,self.stamp))
                self.assertIsNone(algo.offset_feedback(result,reference))
        self.assertTrue(algo.offset_feedback_ready(result,self.stamp))
        feedback = algo.offset_feedback(result,reference)
        self.assertEqual(len(feedback.points()[0]), 4)
        self.assertEqual(feedback.evidence['samples'], 3)

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
