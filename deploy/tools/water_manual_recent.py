"""Optional rolling-time evidence; unknown intervals never vote missing."""
from collections import deque


class RecentMissing:
    def __init__(self, spec):
        self.spec = spec
        self.windows = {}

    def update(self, label, evidence, now):
        q = self.windows.setdefault(label,deque())
        if evidence == 0 or (q and now-q[-1][0] > self.spec.max_observation_gap_s):
            q.clear()
        q.append((now,evidence))
        cutoff = now-self.spec.hold_s
        while len(q)>1 and q[1][0]<=cutoff:
            q.popleft()
        missing = 0.
        for (start,a),(end,b) in zip(q,list(q)[1:]):
            if a == 1 and b == 1:
                missing += end-max(start,cutoff)
        ready = bool(q and now-q[0][0]>=self.spec.hold_s)
        confirmed = ready and missing >= self.spec.hold_s*self.spec.recent_missing_fraction
        return missing,confirmed
