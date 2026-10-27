"""摄像头状态机。重构不包含目标检测，稳定容差允许正常晃动。"""
from collections import deque

import numpy as np


class CameraState:
    NORMAL = 'normal'
    DETECT = 'offset_detection'
    RECOVER = 'alignment_recovery'
    REBUILD = 'rebuilding'

    def __init__(self, params):
        self._p = params
        self.state = self.NORMAL
        self.jitter = deque(maxlen=params.motion.jitter_samples)
        self.confirmed = 0
        self.returned = 0
        self.unavailable = 0
        self.stable_since = None
        self.origin = None
        self.previous = None

    @property
    def tolerance(self):
        cfg = self._p.motion
        learned = (float(np.median(self.jitter))*cfg.jitter_factor
                   if len(self.jitter) >= cfg.jitter_min_samples else cfg.jitter_floor)
        return float(np.clip(learned,cfg.jitter_floor,cfg.jitter_ceiling))

    def check(self, alignment):
        cfg = self._p.motion
        sparse_motion = bool(alignment and alignment.valid
            and alignment.details.get('stationary_fallback')
            and alignment.displacement > self.tolerance*cfg.exit_factor)
        if alignment is None or not alignment.valid or sparse_motion:
            self.confirmed = self.returned = 0
            self.unavailable += 1
            if self.state == self.NORMAL and self.unavailable >= cfg.unavailable_checks:
                self.state = self.RECOVER
                return 'alignment_recovery'
            return 'unavailable'
        self.unavailable = 0
        if self.state == self.NORMAL:
            if alignment.details.get('method') == 'background_phase':
                self.state = self.RECOVER
                self.confirmed = self.returned = 0
                return 'alignment_recovery'
            if alignment.displacement > self.tolerance*cfg.enter_factor:
                self.state = self.DETECT
                self.confirmed = self.returned = 0
                return 'suspected_move'
            if alignment.displacement <= self.tolerance*cfg.exit_factor:
                self.jitter.append(alignment.displacement)
            return 'aligned'
        if alignment.displacement > self.tolerance*cfg.enter_factor:
            if self.state == self.RECOVER:
                self.state = self.DETECT
                self.confirmed = self.returned = 0
                return 'suspected_move'
            self.confirmed += 1
            self.returned = 0
            if self.confirmed >= cfg.confirm_samples:
                self.state = self.REBUILD
                return 'confirmed_move'
            return 'confirming'
        self.confirmed = 0
        if alignment.displacement <= self.tolerance*cfg.exit_factor:
            self.returned += 1
            if self.returned >= cfg.return_samples:
                self.state = self.NORMAL
                return 'returned'
        else:
            self.returned = 0
        return 'uncertain'

    def begin_rebuild(self, frame, now):
        self.state = self.REBUILD
        self.origin = frame.copy()
        self.previous = frame.copy()
        self.stable_since = now

    def stable(self, frame, now, vision, quality):
        if self.origin is None or self.previous is None:
            self.begin_rebuild(frame,now)
            return False,'start_window'
        origin = vision.align(self.origin,frame)
        recent = vision.align(self.previous,frame)
        self.previous = frame.copy()
        limit = self.tolerance*self._p.motion.stable_factor
        if not (quality['ok'] and origin.valid and recent.valid
                and max(origin.displacement,recent.displacement) <= limit):
            self.begin_rebuild(frame,now)
            return False,'reset_unstable_or_unobservable'
        duration = now-self.stable_since
        return duration >= self._p.timing.period_s*self._p.timing.rebuild_periods,'stable'
