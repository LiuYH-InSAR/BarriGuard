"""Serializable, versioned pixel-coordinate contracts for all consumers."""
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np


class MotionStatus(str, Enum):
    ALIGNED = 'aligned'
    SUSPECTED = 'suspected_move'
    MOVED = 'confirmed_move'
    UNAVAILABLE = 'unavailable'
    PENDING = 'migration_pending'
    COMMITTED = 'committed'
    FAILED = 'failed'


@dataclass(frozen=True)
class MotionResult:
    camera_id: str
    task_id: str
    frame_id: str
    checked_at: float
    old_reference: str
    new_reference: str
    version: int
    status: str
    matrix: list | None
    shape: tuple
    confidence: float
    details: dict = field(default_factory=dict)
    direction: str = 'old_reference_to_new_reference'
    coordinate_space: str = 'original_pixels'

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Feedback:
    camera_id: str
    task_id: str
    consumer: str
    old_reference: str
    new_reference: str
    version: int
    frame_id: str
    old_points: tuple
    observed_points: tuple
    confidence: float = 1.0
    independent: bool = True
    coordinate_space: str = 'new_reference_pixels'

    def points(self):
        a, b = np.asarray(self.old_points, float), np.asarray(self.observed_points, float)
        if (a.ndim != 2 or a.shape[1] != 2 or a.shape != b.shape
                or not np.isfinite(a).all() or not np.isfinite(b).all()
                or not 0 < self.confidence <= 1 or not self.independent
                or self.coordinate_space != 'new_reference_pixels'):
            raise ValueError('Feedback requires independent finite corresponding Nx2 points')
        if len(np.unique(a,axis=0)) != len(a) or len(np.unique(b,axis=0)) != len(b):
            raise ValueError('Correspondences must be unique')
        return a, b
