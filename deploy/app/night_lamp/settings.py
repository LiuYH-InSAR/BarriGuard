"""Lamp observation policy, independent of model weights and vision imports."""
from dataclasses import dataclass, fields
from math import isfinite


@dataclass(frozen=True)
class LampParams:
    cadence_s: float = 1800.
    effective_s: float = 300.
    deadline_s: float = 1500.
    normal_pulses: int = 15
    failure_span_s: float = 10800.
    failure_windows: int = 6
    max_sample_gap_s: float = .5
    on_delta: float = 25.
    off_delta: float = 10.
    min_width_s: float = .05
    max_width_s: float = 3.
    discovery_hits: int = 3
    candidate_ttl_s: float = 60.
    min_iou: float = .2
    center_ratio: float = .35
    max_center_ratio: float = .6
    min_size_ratio: float = .45
    max_size_ratio: float = 2.2
    presence_period_s: float = 30.
    presence_min_body_px: float = 10.
    missing_windows: int = 2
    missing_min_checks: int = 10
    missing_max_age_s: float = 86400.
    migration_min_samples: int = 12
    migration_min_span_s: float = 3.
    migration_error_px: float = 3.

    def __post_init__(self):
        if not all(isfinite(getattr(self,f.name)) for f in fields(self)):
            raise ValueError('Lamp parameters must be finite')
        if any(isinstance(getattr(self,k),bool) or not isinstance(getattr(self,k),int)
               for k in ('normal_pulses','failure_windows','discovery_hits','missing_windows','missing_min_checks','migration_min_samples')):
            raise ValueError('Lamp counts must be integers')
        if not (self.candidate_ttl_s>0 and 0<=self.min_iou<=1
                and 0<self.center_ratio<=self.max_center_ratio
                and 0<self.min_size_ratio<=self.max_size_ratio):
            raise ValueError('Invalid lamp candidate or matching thresholds')
        if not (self.presence_period_s>0 and self.presence_min_body_px>=6
                and self.missing_windows>=2 and self.missing_min_checks>=3
                and self.missing_max_age_s>self.cadence_s and self.migration_min_samples>=3
                and self.migration_min_span_s>0 and self.migration_error_px>0):
            raise ValueError('Invalid lamp migration or physical-presence thresholds')
        if not (0 < self.effective_s <= self.deadline_s < self.cadence_s
                and self.normal_pulses > 0 and self.failure_windows > 0
                and self.failure_span_s == self.failure_windows*self.cadence_s
                and 0 <= self.off_delta < self.on_delta and self.discovery_hits > 0
                and 0 < self.max_sample_gap_s and 0 <= self.min_width_s < self.max_width_s):
            raise ValueError('Invalid lamp timing or pulse thresholds')
