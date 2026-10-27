"""昼夜门控、可见事实与按有效YOLO次数累计；不识别遮挡物类别。"""
import math
from datetime import datetime, timedelta, timezone


class Daylight:
    def __init__(self, params):
        self._p = params
        self.streak = 0
        self.mode = 'pending'
        self.candidate = None
        self.last_confirmed_mode = None

    def candidate_for(self, stamp, quality):
        cfg = self._p.policy
        local = datetime.fromtimestamp(stamp, timezone(timedelta(hours=cfg.utc_offset_hours)))
        hour = local.hour + local.minute / 60
        scheduled = not cfg.use_clock or cfg.day_start_hour <= hour < cfg.day_end_hour
        deep_night = cfg.use_clock and (hour >= cfg.deep_night_start_hour or hour < cfg.deep_night_end_hour)
        light = quality['light']
        luma = quality.get('luma',self._p.reference.day_luma if light == 'day'
                           else self._p.reference.night_luma if light == 'night'
                           else (self._p.reference.day_luma+self._p.reference.night_luma)/2)
        upper = quality.get('upper_luma')
        dark_upper = upper is not None and (upper <= cfg.upper_night_luma
            or (not scheduled and upper <= cfg.upper_twilight_luma))
        if light == 'night' or deep_night or dark_upper:
            candidate = 'night'
        elif light == 'day' or scheduled:
            candidate = 'day'
        elif luma <= cfg.twilight_night_luma:
            candidate = 'night'
        else:
            candidate = None
        if candidate == 'day' and cfg.use_clock and not scheduled and self.last_confirmed_mode == 'night':
            candidate = None
        return candidate

    def observe(self, stamp, quality):
        cfg = self._p.policy
        candidate = self.candidate_for(stamp,quality)
        self.streak = self.streak+1 if candidate and candidate == self.candidate else int(candidate is not None)
        self.candidate = candidate
        required = cfg.day_confirm_samples if candidate == 'day' else cfg.night_confirm_samples
        self.mode = candidate if candidate and self.streak >= required else 'pending'
        if self.mode != 'pending':
            self.last_confirmed_mode = self.mode
        return self.mode == 'day'

    def dump(self):
        return dict(mode=self.mode,candidate=self.candidate,streak=self.streak,
                    last_confirmed_mode=self.last_confirmed_mode)

    def restore(self, data):
        self.mode = data.get('mode','pending')
        self.candidate = data.get('candidate')
        self.streak = data.get('streak',0)
        self.last_confirmed_mode = data.get('last_confirmed_mode')

    def pause(self):
        self.mode = 'pending'
        self.candidate = None
        self.streak = 0


class Accumulation:
    def __init__(self, params, records=None):
        self._p = params
        self.records = records or {}

    def observe(self, identity, stamp, value, daylight, hold=False):
        record = self.records.setdefault(identity, dict(day_s=0.0, night_s=0.0,
            present_run=0, last_observation=None, last_seen=None, night_last_seen=None,
            occlusion_held=False))
        last = record['last_observation']
        period = self._p.timing.period_s
        if last is not None and stamp-last < period:
            return self.result(identity, daylight)
        if last is not None and stamp-last > period*self._p.timing.max_gap_periods:
            record['present_run'] = 0
        record['last_observation'] = stamp
        if value == 'present':
            record['last_seen'] = stamp
            if daylight is False:
                record['night_last_seen'] = stamp
            record['present_run'] += 1
        else:
            record['present_run'] = 0
            if value == 'weak_present':
                record['last_seen'] = stamp
            if value == 'suspect' and daylight is not None:
                record['day_s' if daylight else 'night_s'] += period
                record['occlusion_held'] = record['occlusion_held'] or hold
        recovered = record['present_run'] >= math.ceil(self._p.statistics.recovery_s/period)
        if recovered:
            record.update(day_s=0.0, night_s=0.0, occlusion_held=False)
        return self.result(identity, daylight)

    def result(self, identity, daylight):
        r = self.records[identity]
        cfg = self._p.policy
        credit = cfg.night_credit_s if r['night_s'] >= cfg.night_credit_after_s else 0.0
        total = r['day_s'] + credit
        return dict(day_s=r['day_s'], night_s=r['night_s'], night_credit_s=credit,
            equivalent_day_s=total, required_day_s=cfg.suspect_day_s,
            occlusion_held=r['occlusion_held'], last_seen=r['last_seen'],
            night_last_seen=r['night_last_seen'],
            recovered=r['present_run'] >= math.ceil(self._p.statistics.recovery_s/self._p.timing.period_s),
            timeout=bool(daylight is True and total >= cfg.suspect_day_s))

    def pause(self):
        # 不清累计，但中断连续可见恢复证据。
        for r in self.records.values():
            r['present_run'] = 0

    def dump(self):
        return self.records
