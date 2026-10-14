import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import Mock
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from night_lamp.local_recording import RecordedAdapter, _Recorder
from server.contracts import NightLampSpec
from server.maintenance import Sweeper


class RecordingTests(unittest.TestCase):
    def test_record_then_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            spec = NightLampSpec()
            spec.overnight.record_root = folder
            session = Mock(camera_id="1750")
            session.snapshot.return_value = None
            recorder = _Recorder(session, spec.overnight, .16)
            recorder.begin_burst(datetime.now(timezone.utc))
            for stamp in (10.0, 10.16, 12.0):
                recorder.accumulate(np.full((32, 40, 3), 90, np.uint8), stamp)
            session.accumulate.assert_not_called()
            recorder.snapshot(datetime.now(timezone.utc), force=True)
            self.assertEqual([c.args[1] for c in session.accumulate.call_args_list],
                             [10.0, 10.16, 12.0])
            self.assertTrue(recorder.path.exists())

    def test_ttl(self):
        import os
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "night_offline"
            root.mkdir()
            old = root / "old.mp4"
            old.write_bytes(b"old")
            fresh = root / "fresh.mp4"
            fresh.write_bytes(b"new")
            os.utime(old, (time.time()-4*86400,)*2)
            self.assertEqual(Sweeper([root], 72, skip_names=()).sweep(), 1)
            self.assertTrue(fresh.exists())

    def test_hourly_not_pushed(self):
        from server.worker import CameraWorker
        with tempfile.TemporaryDirectory() as folder:
            spec = NightLampSpec()
            spec.overnight.record_local = True
            spec.overnight.record_root = folder
            worker = CameraWorker.__new__(CameraWorker)
            worker._camera = SimpleNamespace(algorithms={"night_lamp": SimpleNamespace(spec=spec)})
            worker._store = Mock()
            worker._reporter = Mock()
            report = Mock(algorithm="night_lamp", report_type="night_heatmap_burst",
                          camera="1750", created_at="2026-10-14T20:05:00", status="ok",
                          image_jpeg=b"jpeg", metadata={})
            worker.publish(report)
            worker._reporter.submit.assert_not_called()
            self.assertEqual(len(list(Path(folder).rglob("*.jpg"))), 1)
            report.report_type = "night_heatmap"
            worker.publish(report)
            worker._reporter.submit.assert_called_once_with(report)

    def test_background_settlement(self):
        import threading
        with tempfile.TemporaryDirectory() as folder:
            spec = NightLampSpec()
            spec.overnight.enabled = True
            spec.overnight.anchor = "wall"
            spec.overnight.burst_seconds = 1
            spec.overnight.record_root = folder
            gate = threading.Event()
            session = Mock(camera_id="1750")
            session.snapshot.side_effect = lambda *a, **k: (gate.wait(3), None)[1]
            session.dump_state.return_value = None
            wall = datetime(2026, 10, 14, 20, tzinfo=timezone.utc)
            clock = Mock()
            clock.wall.return_value = wall
            clock.monotonic.return_value = 10.0
            adapter = RecordedAdapter(session, spec, clock)
            try:
                adapter.tick(wall)
                adapter.tick(wall + timedelta(seconds=1))
                started = time.monotonic()
                adapter.tick(wall + timedelta(seconds=2))
                self.assertLess(time.monotonic()-started, .5)
                gate.set()
                adapter.freeze()
                self.assertEqual(len(adapter.take_reports()), 1)
            finally:
                gate.set()
                adapter.release()
