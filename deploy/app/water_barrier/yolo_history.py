"""有限滑窗和持久事件状态，时间使用墙钟以支持重启续跑。"""
import math
from collections import deque


class History:
    def __init__(self, params, entries=None, started=None):
        self._p = params
        self.entries = deque(entries or [])
        self.started = started

    def observe(self, stamp, value):
        if self.started is None:
            self.started = stamp
        if self.entries and stamp <= self.entries[-1][0]:
            return
        self.entries.append([stamp,value])
        self.trim(stamp)

    def trim(self, stamp):
        while self.entries and self.entries[0][0] <= stamp-self._p.statistics.window_s:
            self.entries.popleft()

    def result(self, stamp):
        self.trim(stamp)
        cfg = self._p.statistics
        expected = math.ceil(cfg.window_s/self._p.timing.period_s)
        valid = [v for _,v in self.entries if v != 'unknown']
        missing = sum(v == 'missing' for v in valid)
        coverage = min(1.0,len(valid)/expected)
        ratio = missing/len(valid) if valid else 0.0
        mature = self.started is not None and stamp-self.started >= cfg.window_s
        alarm = bool(mature and coverage >= cfg.min_coverage and ratio >= cfg.missing_ratio
                     and self.entries and self.entries[-1][1] == 'missing')
        recovery = [(t,v) for t,v in self.entries if t > stamp-cfg.recovery_s]
        recover_expected = math.ceil(cfg.recovery_s/self._p.timing.period_s)
        recovered = (len(recovery) >= recover_expected*cfg.min_coverage
                     and sum(v=='present' for _,v in recovery)/recover_expected >= cfg.recovery_ratio
                     and bool(recovery) and recovery[-1][1]=='present')
        return dict(alarm=alarm,recovered=bool(recovered),coverage=coverage,missing_ratio=ratio,
                    expected=expected,valid=len(valid),missing=missing,mature=mature)

    def dump(self):
        return dict(started=self.started,entries=list(self.entries))


class Episodes:
    def __init__(self, params, records=None, guard_until=0.0):
        self._p = params
        self.records = records or {}
        self.guard_until = guard_until

    def decide(self, identities, results, stamp):
        notify = []
        for identity in identities:
            record = self.records.setdefault(identity,dict(active=False,last_alarm=None,notified=False))
            result = results[identity]
            if result['recovered']:
                record['active'] = False
                record['notified'] = False
                record['error_reason'] = None
            if not result['alarm']:
                continue
            record['active'] = True
            reason = result.get('error_reason','confirmed_missing')
            # 累计error升级成确认缺失只更新标识，不重复通知。
            if record.get('error_reason') != 'confirmed_missing' or reason == 'confirmed_missing':
                record['error_reason'] = reason
            last = record['last_alarm']
            cooldown = last is not None and stamp-last < self._p.statistics.cooldown_s
            if record['notified'] or cooldown or stamp < self.guard_until:
                continue
            record['last_alarm'] = stamp
            record['notified'] = True
            notify.append(identity)
        # 已恢复且冷却结束的历史不无限增长；持续事件永远保留去重标记。
        for key,record in list(self.records.items()):
            last = record['last_alarm']
            if (key not in identities and not record['active']
                    and (last is None or stamp-last >= self._p.statistics.cooldown_s)):
                del self.records[key]
        return notify

    def guard_new_view(self, stamp):
        """新旧区域无法对应时保守保留相机冷却，避免重构绕过24小时限制。"""
        last = [r['last_alarm'] for r in self.records.values() if r['last_alarm'] is not None]
        self.guard_until = max([self.guard_until,*[t+self._p.statistics.cooldown_s for t in last]])

    def dump(self):
        return dict(records=self.records,guard_until=self.guard_until)
