"""Independent camera defaults; no water detector or weight dependency."""
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class Timing:
    period_s: float = 30.0
    alignment_phase: float = 0.5
    fast_s: float = 1.0
    rebuild_periods: float = 4.0
    max_gap_periods: float = 2.0

@dataclass(frozen=True)
class Motion:
    width: int = 960
    features: int = 2400
    feature_margin: float = 0.12
    ratio_test: float = 0.75
    min_matches: int = 24
    min_inliers: int = 18
    min_inlier_ratio: float = 0.60
    grid: int = 3
    min_cells: int = 5
    stationary_min_cells: int = 3
    stationary_min_inliers: int = 60
    stationary_min_inlier_ratio: float = 0.90
    ransac_px: float = 2.5
    max_error_px: float = 2.0
    max_scale_error: float = 0.08
    jitter_floor: float = 0.004  # 画面对角线比例
    jitter_ceiling: float = 0.012
    jitter_factor: float = 3.0
    jitter_samples: int = 40
    jitter_min_samples: int = 8
    enter_factor: float = 2.0
    exit_factor: float = 1.0
    confirm_samples: int = 4
    return_samples: int = 3
    stable_factor: float = 1.0
    min_visible_ratio: float = 0.85
    unavailable_checks: int = 2
    overlay_exclusions: tuple = ((0.68,0.0,1.0,0.20),(0.0,0.88,0.13,1.0))

@dataclass(frozen=True)
class AlignmentRecovery:
    columns: int = 6
    rows: int = 4
    min_response: float = 0.18
    min_texture: float = 5.0
    max_excluded_fraction: float = 0.10
    min_support: int = 5
    min_columns: int = 3
    min_rows: int = 2
    min_span_x: float = 0.45
    min_span_y: float = 0.50
    consistency_px: float = 2.5
    blur_kernel: int = 3
    gradient_kernel: int = 3

@dataclass(frozen=True)
class Reference:
    refresh_s: float = 600.0
    per_light: int = 3
    blur_floor: float = 12.0
    contrast_floor: float = 12.0
    clipped_max: float = 0.35
    dark_pixel: int = 5
    bright_pixel: int = 250
    night_luma: float = 55.0
    day_luma: float = 115.0
    lighting_upper_fraction: float = 0.25
    adaptive_quality_ratio: float = 0.65
    quality_samples: int = 30
    light_confirm: int = 3
    min_features: int = 100

@dataclass(frozen=True)
class Storage:
    root: Path = Path('output/camera_motion')
    retention_s: float = 259200.0
    cleanup_s: float = 600.0
    jpeg_quality: int = 95


@dataclass(frozen=True)
class MotionParams:
    timing: Timing = field(default_factory=Timing)
    motion: Motion = field(default_factory=Motion)
    alignment_recovery: AlignmentRecovery = field(default_factory=AlignmentRecovery)
    reference: Reference = field(default_factory=Reference)
    storage: Storage = field(default_factory=Storage)
