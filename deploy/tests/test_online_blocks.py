"""Online block alarms: recovery episodes, selected evidence and legacy isolation."""
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2 as cv
import numpy as np

DEPLOY = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(DEPLOY/'app'),str(DEPLOY/'tools')]
from water_barrier.online_blocks import BlockAlgorithm
from server.algo import Event
from server.reporting import Reporter, _target


class FakeDetector:
    def __init__(self, labels, reference, spec):
        self.spec = spec
        self.rows = {0:np.array([[0,0],[90,0],[90,90],[0,90]])}
        self.blocks = [dict(label='row0block01',row=0,number=1,poly=np.array([[10,10],[40,10],[40,70],[10,70]]))]
        self.vote, self.status = 1,'*'

    def step(self, frame, now):
        return [dict(label='row0block01',row=0,number=1,status=self.status,
                     evidence=self.vote,valid=1.,accum=10.,reason='test missing')]


class OnlineTests(unittest.TestCase):
    def make_algo(self):
        from water_manual import ManualSpec
        core=SimpleNamespace(ManualDetector=FakeDetector,ManualSpec=ManualSpec)
        with patch('water_barrier.online_blocks._core',return_value=core):
            return BlockAlgorithm((96,96,3),dict(labels='unused',reference=np.zeros((96,96,3),np.uint8),
                parameters=dict(reset_s=1.,max_observation_gap_s=1.),detect_hz=2.,alarm_interval=3600),'1749')

    def test_alarm_recovery_and_unknown(self):
        a=self.make_algo();f=np.full((96,96,3),100,np.uint8)
        self.assertFalse(a.step(f,0).events)
        a._detector.status='error'
        first=a.step(f,.5).events[0]
        self.assertEqual(_target(first),'row0block01')
        self.assertTrue(first.image_jpeg.startswith(b'\xff\xd8'))
        self.assertEqual(cv.imdecode(np.frombuffer(first.image_jpeg,np.uint8),cv.IMREAD_COLOR).shape,f.shape)
        self.assertFalse(a.step(f,.6).events)  # skipped input cannot resend
        self.assertFalse(a.step(f,1.).events)
        a._detector.status='?';a._detector.vote=-1
        self.assertFalse(a.step(f,1.5).events)
        a._detector.status='error';a._detector.vote=1
        self.assertFalse(a.step(f,2.).events)  # unknown does not open a new episode
        a._detector.status='normal';a._detector.vote=0
        for t in (2.5,3.,3.5): self.assertFalse(a.step(f,t).events)
        a._detector.status='error';a._detector.vote=1
        second=a.step(f,4.).events[0]
        self.assertNotEqual(first.payload['episode_id'],second.payload['episode_id'])
        self.assertLessEqual(len(a._records),5)

    def test_reporter_episode_and_block_keys(self):
        r=Reporter.__new__(Reporter)
        r._closed=False;r._enabled=True;r._lock=threading.Lock()
        r._last={};r._episodes={};r._alarm_interval={('1749','water_gap'):3600}
        accepted=[];r._enqueue=accepted.append;r._persist_alarm=lambda e,j:'test.jpg'
        with patch('server.reporting.time.monotonic',return_value=100.):
            for block,episode in [('row0block01','one'),('row0block01','one'),
                                  ('row0block02','one'),('row0block01','two')]:
                r.submit_event(Event('1749','water_gap',100.,'alarm',
                    dict(block_id=block,row_id=0,episode_id=episode)),b'image')
        self.assertEqual(len(accepted),3)
        legacy=Event('1749','water_gap',100.,'alarm',dict(row_id='row0'))
        self.assertEqual(_target(legacy),'row0')

    def test_reminder_keeps_episode(self):
        a=self.make_algo();f=np.full((96,96,3),100,np.uint8)
        a._detector.status='error'
        first=a.step(f,0.).events[0]
        self.assertFalse(a.step(f,3599.5).events)
        reminder=a.step(f,3600.).events[0]
        self.assertEqual(first.payload['episode_id'],reminder.payload['episode_id'])

    def test_evidence_excludes_normal_frames(self):
        a=self.make_algo();f=np.full((96,96,3),100,np.uint8)
        a._detector.status='normal';a._detector.vote=0
        a.step(f,0.)
        a._detector.status='error';a._detector.vote=1
        event=a.step(f,.5).events[0]
        self.assertEqual(event.payload['evidence_at_monotonic_s'],.5)

    def test_realtime_has_no_frequency_gate(self):
        a=self.make_algo();a._config['detect_hz']=0.
        frame=np.full((96,96,3),100,np.uint8)
        with patch.object(a._detector,'step',wraps=a._detector.step) as detect:
            a.step(frame,0.)
            a.step(frame,.1)
            self.assertEqual(detect.call_count,2)

    def test_thirty_second_window_with_slow_frames(self):
        from water_manual import ManualSpec
        from water_manual_recent import RecentMissing
        recent=RecentMissing(ManualSpec(hold_s=30.,max_observation_gap_s=3.))
        for i in range(26):
            total,confirmed=recent.update('gap',1,round(i*1.2,1))
            self.assertEqual(confirmed,i==25)
        self.assertAlmostEqual(total,30.)
        _,confirmed=recent.update('gap',1,34.)
        self.assertFalse(confirmed)  # four-second interruption clears evidence
        masked=RecentMissing(ManualSpec(hold_s=30.,max_observation_gap_s=3.))
        for i in range(26):
            _,confirmed=masked.update('gap',-1 if i==10 else 1,round(i*1.2,1))
        self.assertFalse(confirmed)  # unknown time is not missing support

    def test_configs_and_build(self):
        from server.config import load_runtime
        from server.worker import _load_calibration
        from server.registry import create
        original=load_runtime(DEPLOY/'config.yaml')
        self.assertTrue(all(c.algorithms['water_gap'].spec.backend=='legacy' for c in original.cameras))
        config=load_runtime(DEPLOY/'config_blocks.yaml')
        for camera in config.cameras:
            spec=camera.algorithms['water_gap'].spec
            self.assertEqual(spec.detect_hz,0.)
            self.assertEqual(spec.max_observation_gap_s,3.)
            self.assertEqual(spec.alarm_hold_s,30.)
            calibration=_load_calibration('water_gap',spec)
            algo=create('water_gap',calibration['reference'].shape,calibration,camera.id)
            self.assertIsInstance(algo,BlockAlgorithm)
            result=algo.step(calibration['reference'],100.)
            self.assertFalse(result.reports)
            self.assertEqual(result.debug['backend'],'blocks')


if __name__=='__main__':
    unittest.main()
