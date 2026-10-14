"""Hourly local recordings; replay uses capture timestamps and the existing session."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import threading

import cv2 as cv
from loguru import logger

from night_lamp.adapter import NightAdapter
from server.maintenance import Sweeper
from server.output import ReportStore


def save_report(report, spec):
    ReportStore(Path(spec.record_root) / "results", persist_images=True).save(report)


class _Recorder:
    def __init__(self, session, spec, interval):
        self.session = session
        self.spec = spec
        self.interval = interval
        self.writer = None
        self.times = None
        self.path = None

    def __getattr__(self, name):
        return getattr(self.session, name)

    def begin_burst(self, wall):
        self.close()
        self.session.begin_burst(wall)
        folder = Path(self.spec.record_root) / "videos" / self.session.camera_id
        folder.mkdir(parents=True, exist_ok=True)
        from uuid import uuid4
        self.path = folder / (wall.strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8] + ".mp4")
        self.times = self.path.with_suffix(".jsonl").open("w", encoding="utf-8")

    def accumulate(self, frame, stamp):
        if self.writer is None:
            h, w = frame.shape[:2]
            self.writer = cv.VideoWriter(str(self.path), cv.VideoWriter_fourcc(*"mp4v"),
                                         1.0 / self.interval, (w, h))
            if not self.writer.isOpened():
                self.close()
                raise RuntimeError(f"Cannot record {self.path}")
        self.writer.write(frame)
        self.times.write(json.dumps(float(stamp)) + "\n")

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.times is not None:
            self.times.close()
            self.times = None

    def snapshot(self, wall, force=False):
        self.close()
        count = 0
        if self.path and self.path.exists():
            cap = cv.VideoCapture(str(self.path))
            try:
                with self.path.with_suffix(".jsonl").open(encoding="utf-8") as stamps:
                    for line in stamps:
                        ok, frame = cap.read()
                        if not ok:
                            raise RuntimeError(f"Recording truncated at frame {count}: {self.path}")
                        self.session.accumulate(frame, float(line))
                        count += 1
            finally:
                cap.release()
        logger.info("[night {}] local replay: {} frames from {}",
                    self.session.camera_id, count, self.path)
        report = self.session.snapshot(wall, force=force)
        if report is not None:
            report.metadata["local_recording"] = str(self.path)
            report.metadata["replayed_frames"] = count
            report.metadata["record_fps"] = 1.0 / self.interval
        return report


class RecordedAdapter(NightAdapter):
    """Record on the existing sampling grid, compute only after closing the clip."""
    def __init__(self, session, spec, clock):
        if not (spec.overnight.enabled and spec.overnight.anchor == "wall"):
            raise ValueError("record_local requires overnight enabled with anchor=wall")
        interval = max(int(spec.sampling.interval_ms), 1) / 1000.0
        super().__init__(_Recorder(session, spec.overnight, interval), spec, clock)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"replay-{session.camera_id}")
        self._job = None
        self._record_lock = threading.RLock()
        root = Path(spec.overnight.record_root)
        self._cleaners = [Sweeper([root / "videos"], spec.overnight.video_days * 24, skip_names=()),
                          Sweeper([root / "results"], spec.overnight.result_days * 24, skip_names=())]
        for cleaner in self._cleaners:
            cleaner.start()

    def tick(self, wall):
        with self._record_lock:
            if self._job is not None:
                if not self._job.done():
                    return
                self._job.result()
                self._job = None
            super().tick(wall)

    def on_frame(self, *args, **kwargs):
        with self._record_lock:
            if self._job is None:
                super().on_frame(*args, **kwargs)

    def _finalize_bucket(self, wall):
        if not self._window_open:
            return
        self._window_open = False
        self._session.close()
        self._job = self._executor.submit(self._replay, wall)

    def _replay(self, wall):
        # The sampler stays closed while the original bucket settlement runs.
        self._window_open = True
        try:
            super()._finalize_bucket(wall)
        finally:
            self._window_open = False

    def _wait(self):
        if self._job is not None:
            self._job.result()
            self._job = None

    def freeze(self):
        with self._record_lock:
            if self._window_open and self._job is None:
                self._finalize_bucket(self._clock.wall())
        self._wait()
        super().freeze()

    def finalize(self, base, wall):
        self._wait()
        return super().finalize(base, wall)

    def release(self):
        self._wait()
        self._session.close()
        self._executor.shutdown(wait=True)
        for cleaner in self._cleaners:
            cleaner.stop()
        super().release()
