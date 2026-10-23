import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from server.algo import Report
from server.contracts import RuntimeConfig, ScheduleSpec, NightLampSpec
from server.output import ReportStore
from server.reporting import Reporter
from night_lamp.replay_night import night_bounds, recordings, recompute, digest
from night_lamp.publish_night import publish


class ReplayTests(unittest.TestCase):
    def test_morning_date_crosses_year(self):
        start, end, morning = night_bounds(date(2026, 1, 1), ScheduleSpec())
        self.assertEqual(start.isoformat(), "2025-12-31T20:00:00+08:00")
        self.assertEqual(end.isoformat(), "2026-01-01T03:00:00+08:00")
        self.assertEqual(morning.isoformat(), "2026-01-01T07:00:00+08:00")

    def test_select_night_and_reject_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in ("20261022_195900_a", "20261022_200000_a", "20261023_020000_a", "20261023_030000_a"):
                (folder / (name + ".mp4")).touch()
                (folder / (name + ".jsonl")).write_text("10.0\n10.16\n", encoding="utf-8")
            start, end, _ = night_bounds(date(2026, 10, 23), ScheduleSpec())
            self.assertEqual(len(recordings(folder, start, end, 3600)), 2)
            (folder / "20261022_200100_b.mp4").touch()
            with self.assertRaisesRegex(ValueError, "Multiple recordings"):
                recordings(folder, start, end, 3600)

    def test_missing_or_empty_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "20261022_200000_a.mp4"
            path.touch()
            start, end, _ = night_bounds(date(2026, 10, 23), ScheduleSpec())
            with self.assertRaisesRegex(ValueError, "Missing capture"):
                recordings(tmp, start, end, 3600)
            path.with_suffix(".jsonl").touch()
            with self.assertRaisesRegex(ValueError, "invalid capture"):
                recordings(tmp, start, end, 3600)

    def test_isolated_recompute_and_no_send(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = NightLampSpec()
            spec.overnight.enabled = spec.overnight.record_local = True
            spec.overnight.record_root = tmp
            camera = SimpleNamespace(id="1750", algorithms={"night_lamp": SimpleNamespace(spec=spec)})
            cfg = RuntimeConfig(cameras=[camera])
            video = root / "videos/1750/20261022_200000_a.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"original")
            video.with_suffix(".jsonl").write_text("10\n10.16\n", encoding="utf-8")
            session = Mock()
            session.finalize.return_value = Report("1750", "night_lamp", "night_heatmap", "2026-10-23T07:00:00+08:00", b"jpeg")
            recorder = Mock()
            recorder.snapshot.return_value = Report("1750", "night_lamp", "night_heatmap_burst", "2026-10-22T20:05:00+08:00", b"hour")
            out = root / "review"
            with patch("night_lamp.replay_night.NightSession", return_value=session), patch("night_lamp.replay_night._Recorder", return_value=recorder), patch("urllib.request.urlopen") as send:
                recompute(cfg, date(2026, 10, 23), ["1750"], out)
                send.assert_not_called()
            self.assertEqual(video.read_bytes(), b"original")
            session.begin_burst.assert_called_once()
            session.release.assert_called_once()
            manifest = json.loads((out / "run.json").read_text())
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["finals"][0]["camera"], "1750")
            with self.assertRaises(FileExistsError):
                recompute(cfg, date(2026, 10, 23), ["1750"], out)


class PublishTests(unittest.TestCase):
    def bundle(self, folder):
        cfg = RuntimeConfig()
        cfg.callback.url = "http://example.invalid/callback"
        report = Report("1750", "night_lamp", "night_heatmap", "2026-10-23T07:00:00+08:00", b"jpeg", {"fixed_lamps": []})
        path = Path(ReportStore(folder).save(report))
        image = path.with_suffix(".jpg")
        manifest = {"version": 1, "complete": True, "run_id": "test", "finals": [
            {"camera": "1750", "report": str(path.relative_to(folder)), "sha256": digest(path), "image_sha256": digest(image)}]}
        (folder / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
        return cfg, report, path

    def test_production_payload_receipt_and_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            cfg, report, _ = self.bundle(folder)
            response = Mock()
            response.__enter__ = Mock(return_value=SimpleNamespace(status=200))
            response.__exit__ = Mock(return_value=False)
            with patch("urllib.request.urlopen", return_value=response) as send:
                publish(cfg, folder)
                payload = json.loads(send.call_args.args[0].data)
                self.assertEqual(payload["alert_type"], "night_heatmap")
                self.assertEqual(payload["timestamp"], "2026-10-22T23:00:00+00:00")
                self.assertEqual(payload["image_base64"], "anBlZw==")
                self.assertEqual(payload["metadata"], report.metadata)
                publish(cfg, folder)
                self.assertEqual(send.call_count, 1)
            self.assertTrue((folder / "published_1750.json").is_file())
            self.assertFalse((folder / ".publish.lock").exists())

    def test_tampered_result_never_sent(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            cfg, _, path = self.bundle(folder)
            path.with_suffix(".jpg").write_bytes(b"changed")
            with patch("urllib.request.urlopen") as send:
                with self.assertRaisesRegex(ValueError, "changed"):
                    publish(cfg, folder)
                send.assert_not_called()

    def test_failed_send_has_no_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            cfg, _, _ = self.bundle(folder)
            with patch("urllib.request.urlopen", side_effect=OSError("offline")):
                with self.assertRaises(OSError):
                    publish(cfg, folder)
            self.assertFalse((folder / "published_1750.json").exists())
            self.assertFalse((folder / ".publish.lock").exists())

    def test_incomplete_run_never_sent(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            cfg, _, _ = self.bundle(folder)
            manifest = json.loads((folder / "run.json").read_text())
            manifest["complete"] = False
            (folder / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch("urllib.request.urlopen") as send:
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    publish(cfg, folder)
                send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
