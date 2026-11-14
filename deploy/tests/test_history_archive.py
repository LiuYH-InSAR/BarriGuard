"""Exercise archiving against the on-disk formats produced by the service."""
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from server.history_archive import HK, HistoryArchive


class HistoryArchiveTest(unittest.TestCase):
    def test_copies_completed_history_and_preserves_live_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'archive'
            water = root / 'output/water_yolo/1749'
            motion = root / 'output/camera_motion/1749'
            lamps = root / 'output/lamp_state/1749'
            data = root / 'data'
            organizer = HistoryArchive(archive, water.parent, data, data / 'logs',
                                       {'1749': motion}, {'1749': lamps})
            day = datetime(2026, 11, 13, 8, tzinfo=HK)
            morning = datetime(2026, 11, 14, 7, tzinfo=HK)
            record = f'{day.timestamp():.6f}_1'
            folder = f'{morning.timestamp():.6f}_report'
            report_name = '1749_20261114070000_lamp_status_degraded.json'
            files = {
                water / 'current.json': b'{}',
                water / f'suspect/{record}.jpg': b'image',
                water / f'suspect/{record}.json': b'{}',
                water / f'debug/{day.timestamp():.6f}_2.json': b'{}',
                water / f'debug/{day.timestamp()-1:.6f}_3.json': b'{}',
                water / f'observations/{morning.timestamp():.6f}_1.json': b'{}',
                motion / f'references/{record}.jpg': b'reference',
                motion / f'references/{record}.json': b'{}',
                motion / 'current.json': b'{}',
                lamps / f'archive/{folder}/state.json': b'{"night_key":"2026-11-13"}',
                lamps / f'archive/{folder}/reference_0.png': b'png',
                lamps / 'current/HEAD': b'active',
                data / '1749' / report_name: b'{"report_type":"lamp_status"}',
                data / 'logs/barriguard_2026-11-13.log': b'log',
            }
            for path, payload in files.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                if path.suffix == '.log':
                    os.utime(path, (day.timestamp(), day.timestamp()))

            organizer.copy_completed(morning)
            self.assertFalse(archive.exists())
            organizer.copy_completed(morning.replace(hour=8, minute=5))
            dest = archive / '2026-11-13/1749'
            self.assertEqual((dest / f'water/suspects/{record}.jpg').read_bytes(), b'image')
            self.assertTrue((dest / f'camera_motion/references/{record}.json').is_file())
            self.assertFalse((dest / 'lamps/final').exists())
            self.assertEqual(len(list((dest / 'water/debug').glob('*.json'))), 1)
            self.assertTrue((archive / '2026-11-13/logs').exists())
            self.assertFalse((archive / '2026-11-14').exists())
            self.assertFalse(list(archive.rglob('current.json')))
            self.assertFalse(list(archive.rglob('HEAD')))

            next_run = morning.replace(hour=8, minute=5) + timedelta(days=1)
            organizer.copy_completed(next_run)
            organizer.copy_completed(next_run)
            self.assertTrue((dest / f'lamps/final/{folder}/reference_0.png').is_file())
            self.assertTrue((dest / 'lamps/final' / report_name).is_file())
            self.assertEqual(len(list(dest.rglob('*.png'))), 1)
            self.assertTrue((archive / '2026-11-13/logs/barriguard_2026-11-13.log').is_file())
            for path, payload in files.items():
                self.assertEqual(path.read_bytes(), payload)

    def test_daily_marker_and_seven_day_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'archive'
            water = root / 'output/water_yolo'
            organizer = HistoryArchive(archive, water, None, None, {}, {})
            scheduled = datetime(2026, 11, 14, 8, 5, tzinfo=HK)
            cutoff = (scheduled - timedelta(days=7)).timestamp()
            expired = archive / '2026-11-06/1749/lamps/final/old.json'
            recent = archive / '2026-11-06/1749/lamps/videos/recent.mp4'
            original = water / '1749/observations' / f'{cutoff-86400:.6f}_1.json'
            for path, stamp in ((expired, cutoff), (recent, cutoff + 1), (original, cutoff - 86400)):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'example')
                os.utime(path, (stamp, stamp))
            with patch.object(organizer, 'copy_completed', wraps=organizer.copy_completed) as copy:
                organizer._run_daily(scheduled)
                restarted = HistoryArchive(archive, water, None, None, {}, {})
                with patch.object(restarted, 'copy_completed') as copy_after_restart:
                    restarted._run_daily(scheduled)
                copy.assert_called_once_with(scheduled)
                copy_after_restart.assert_not_called()
            self.assertFalse(expired.parent.exists())
            self.assertTrue(recent.is_file())
            self.assertTrue(original.is_file())
            self.assertFalse(list(archive.rglob(original.name)))
