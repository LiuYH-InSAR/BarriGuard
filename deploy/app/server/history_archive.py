"""Copy completed BarriGuard records into the project archive."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import copy2
from tempfile import NamedTemporaryFile
import os
import re
import threading

from loguru import logger
from .logging_setup import ROTATION


HK = timezone(timedelta(hours=8))
_LOG_DATE = re.compile(r'barriguard_(\d{4}-\d{2}-\d{2})(?:_.*)?\.log')
_WATER_GROUPS = dict(baselines='baselines', observations='observations', suspect='suspects',
                     alarms='alarms', movement='movement',
                     alignment_recovery='alignment_recovery', reconstruction='reconstruction')
_LAMP_REPORT_SHIFT = timedelta(hours=12)
_MIN_SOURCE_AGE = timedelta(hours=24)
_RETENTION = timedelta(days=7)
_DAILY_RUN = ROTATION.replace(minute=5)


def _record_time(path):
    return datetime.fromtimestamp(float(path.name.split('_', 1)[0]), HK)


def _eligible(written, now):
    # Older originals may still exist after their archived copies expire.
    return _MIN_SOURCE_AGE <= now - written < _MIN_SOURCE_AGE + _RETENTION


class HistoryArchive:
    def __init__(self, root, water, reports, logs, motion, lamps):
        self.root = root
        self.water = water
        self.reports = reports
        self.logs = logs
        self.motion = motion
        self.lamps = lamps
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, name='history-archive', daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self):
        while not self._stop.is_set():
            now = datetime.now(HK)
            scheduled = datetime.combine(now.date(), _DAILY_RUN)
            try:
                if now >= scheduled:
                    self._run_daily(scheduled)
            except OSError:
                logger.exception('[archive] Could not scan history')
            next_run = scheduled if now < scheduled else scheduled + timedelta(days=1)
            self._stop.wait(max(0, (next_run - datetime.now(HK)).total_seconds()))

    def _run_daily(self, scheduled):
        marker = self.root / '.last_daily_run'
        today = scheduled.date().isoformat()
        try:
            if marker.read_text(encoding='ascii') == today:
                return
        except FileNotFoundError:
            pass
        self.copy_completed(scheduled)
        self._remove_expired(scheduled)
        if not self._stop.is_set():
            self.root.mkdir(parents=True, exist_ok=True)
            temporary = marker.with_suffix('.tmp')
            temporary.write_text(today, encoding='ascii')
            os.replace(temporary, marker)

    def copy_completed(self, now):
        now = now.astimezone(HK)
        if self.water is not None:
            for source in self.water.glob('*'):
                if source.is_dir() and not source.is_symlink():
                    self._copy_records(source, now, 'water', _WATER_GROUPS)
                    self._copy_water_debug(source, now)
                    previous = source / 'previous_baseline.json'
                    try:
                        written = datetime.fromtimestamp(previous.stat().st_mtime, HK)
                    except FileNotFoundError:
                        continue
                    if _eligible(written, now):
                        destination = self.root / str(written.date()) / source.name / 'water/baselines'
                        self._copy(previous, destination / f'previous_{written:%H%M%S_%f}.json')
        for source in self.motion.values():
            self._copy_records(source, now, 'camera_motion',
                               dict(references='references', migrations='migrations'))
        for source in self.lamps.values():
            for folder in (source / 'archive').glob('*_*'):
                if not folder.is_dir() or folder.is_symlink():
                    continue
                try:
                    written = _record_time(folder)
                except (ValueError, OverflowError, OSError):
                    continue
                if _eligible(written, now):
                    date = (written - _LAMP_REPORT_SHIFT).date()
                    dest = self.root / str(date) / source.name / 'lamps/final' / folder.name
                    for reference in folder.glob('reference_*.png'):
                        self._copy(reference, dest / reference.name)
                    self._copy(folder / 'state.json', dest / 'state.json')
        if self.reports is not None:
            self._copy_reports(now)
        if self.logs is not None:
            today = now.date()
            last_closed = today if now.timetz() >= ROTATION else today - timedelta(days=1)
            for path in self.logs.glob('barriguard_*.log'):
                match = _LOG_DATE.fullmatch(path.name)
                try:
                    written = datetime.fromtimestamp(path.stat().st_mtime, HK)
                except FileNotFoundError:
                    continue
                if match and match[1] < last_closed.isoformat() and _eligible(written, now):
                    self._copy(path, self.root / match[1] / 'logs' / path.name)

    def _copy_records(self, source, now, module, groups):
        for group, category in groups.items():
            for path in (source / group).glob('*.json'):
                try:
                    written = _record_time(path)
                except (ValueError, OverflowError, OSError):
                    continue
                if _eligible(written, now):
                    dest = self.root / str(written.date()) / source.name / module / category
                    self._copy_pair(path, dest)

    def _copy_reports(self, now):
        for camera in self.reports.iterdir() if self.reports.is_dir() else ():
            if not camera.is_dir() or camera.is_symlink():
                continue
            for path in camera.glob(f'{camera.name}_*.json'):
                name = path.name.removeprefix(f'{camera.name}_')
                try:
                    written = datetime.strptime(name.split('_', 1)[0], '%Y%m%d%H%M%S').replace(tzinfo=HK)
                except ValueError:
                    continue
                if '_lamp_status_' in name:
                    date = (written - _LAMP_REPORT_SHIFT).date()
                    category = 'lamps/final'
                elif '_gap_overlay_' in name:
                    date = written.date()
                    category = 'water/reports'
                else:
                    continue
                if _eligible(written, now):
                    self._copy_pair(path, self.root / str(date) / camera.name / category)

    def _copy_water_debug(self, source, now):
        incident_times = {path.name.split('_', 1)[0] for group in ('suspect', 'alarms')
                          for path in (source / group).glob('*.json')}
        for path in (source / 'debug').glob('*.json'):
            if path.name.split('_', 1)[0] not in incident_times:
                continue
            try:
                written = _record_time(path)
            except (ValueError, OverflowError, OSError):
                continue
            if _eligible(written, now):
                self._copy_pair(path, self.root / str(written.date()) / source.name / 'water/debug')

    def _remove_expired(self, now):
        cutoff = (now - _RETENTION).timestamp()
        root = self.root.resolve()
        for day in self.root.glob('????-??-??'):
            if not day.is_dir() or day.is_symlink() or day.is_junction():
                continue
            try:
                datetime.strptime(day.name, '%Y-%m-%d')
            except ValueError:
                continue
            paths = sorted(day.rglob('*'), key=lambda path: len(path.parts), reverse=True)
            for path in (*paths, day):
                if self._stop.is_set():
                    return
                if path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(root):
                    continue
                try:
                    if path.is_file() and path.stat().st_mtime <= cutoff:
                        path.unlink()
                    elif path.is_dir() and not any(path.iterdir()):
                        path.rmdir()
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.exception('[archive] Could not clean {}', path)

    def _copy_pair(self, path, destination):
        if not path.is_file() or path.is_symlink():
            return
        image = path.with_suffix('.jpg')
        if image.is_file():
            self._copy(image, destination / image.name)
        self._copy(path, destination / path.name)

    def _copy(self, source, destination):
        if self._stop.is_set() or source.is_symlink():
            return
        temporary = None
        try:
            original = source.stat()
            if destination.exists():
                return
            destination.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=destination.parent, suffix='.part', delete=False) as handle:
                temporary = Path(handle.name)
            copy2(source, temporary)
            current = source.stat()
            if (original.st_size, original.st_mtime_ns) == (current.st_size, current.st_mtime_ns):
                os.utime(temporary, None)
                os.replace(temporary, destination)
        except FileNotFoundError:
            # A rolling source can expire while a scan is in progress.
            pass
        except OSError:
            logger.exception('[archive] Could not copy {}', source)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
