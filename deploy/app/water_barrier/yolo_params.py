"""YOLO 水马算法的唯一调参入口。修改后重启服务生效；时间单位均为秒。

带边界的自适应只在可信的普通状态学习，不能用疑似移动/缺失降低门槛。
结构维度、数组索引、单位换算不属于可调算法参数。
"""
from dataclasses import dataclass, field
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]


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
    max_unresolved_fraction: float = 0.25
    min_baseline_observations: int = 5
    max_baseline_observations: int = 12
    baseline_presence: float = 0.8
    baseline_count_change: float = 0.10
    min_barriers: int = 2
    row_vertical_ratio: float = 0.75
    row_distance_ratio: float = 2.5


@dataclass(frozen=True)
class Inference:
    # 用户选定V6；与离线比较采用相同的方形输入和单尺度重叠分块。
    weights: Path = PROJECT / 'models/water_barrier_v6/best.pt'
    tile_px: int = 960
    overlap: float = 0.25
    imgsz: int = 960
    rect: bool = False
    confidence: float = 0.35
    nms_iou: float = 0.45
    merge_iou: float = 0.45
    class_id: int = 0
    device: str = 'cpu'  # 可改为 '0' 使用 CUDA；先在实际服务器计时
    max_detections: int = 300
    border_fraction: float = 0.025
    min_box_px: float = 2.0


@dataclass(frozen=True)
class Matching:
    min_iou: float = 0.20
    center_ratio: float = 0.35
    max_center_ratio: float = 0.60
    min_size_ratio: float = 0.45
    max_size_ratio: float = 2.20


@dataclass(frozen=True)
class Evidence:
    patch_px: int = 12
    band_height_ratio: float = 0.8
    min_road_patches: int = 6
    min_road_fraction: float = 0.65
    max_chroma: float = 24.0
    color_distance: float = 18.0
    max_texture_ratio: float = 3.0
    texture_floor: float = 5.0
    neighbor_ratio: float = 0.60
    history_motion_ratio: float = 0.12
    motion_luma_delta: float = 25.0
    simultaneous_min: int = 2
    suspect_save_s: float = 600.0
    appearance_patch_px: int = 32
    appearance_max_age_s: float = 600.0
    appearance_difference: float = 0.80


@dataclass(frozen=True)
class Statistics:
    window_s: float = 900.0
    min_coverage: float = 0.80
    missing_ratio: float = 0.90
    recovery_s: float = 180.0
    recovery_ratio: float = 0.90
    cooldown_s: float = 86400.0


@dataclass(frozen=True)
class DecisionPolicy:
    # 亮度主导，时段辅助；待确认不等于夜间，不计缺失时间。
    use_clock: bool = True
    utc_offset_hours: float = 8.0
    day_start_hour: float = 6.0
    day_end_hour: float = 18.0
    day_confirm_samples: int = 3
    night_confirm_samples: int = 3
    recovery_confidence: float = 0.65
    deep_night_start_hour: float = 20.0
    deep_night_end_hour: float = 5.0
    twilight_night_luma: float = 100.0
    upper_night_luma: float = 75.0
    upper_twilight_luma: float = 100.0
    suspect_day_s: float = 10800.0
    night_credit_after_s: float = 10800.0
    night_credit_s: float = 3600.0  # 单次未恢复事件最多折算一次
    replay_sources: tuple = (
        ('1749','1749_20261013_162822'),
        ('1749','1749_20261013_192734'),
        ('1750','1750_20261013_162822'),
        ('1750','1750_20261013_212734'),
    )
    replay_bridge_times: tuple = ('175734','182734','185734')


@dataclass(frozen=True)
class InitialGaps:
    min_row_observations: int = 5
    local_neighbors: int = 3
    min_neighbor_intervals: int = 2
    min_gap_ratio: float = 1.7
    max_gap_ratio: float = 4.5
    integer_tolerance: float = 0.3
    min_direction_cosine: float = 0.90
    duplicate_iou: float = 0.50
    max_size_ratio: float = 2.0
    max_inferred_per_gap: int = 3
    max_slots: int = 300


@dataclass(frozen=True)
class Storage:
    root: Path = PROJECT / 'output/water_yolo'
    retention_s: float = 259200.0
    cleanup_s: float = 600.0
    jpeg_quality: int = 95
    checkpoint_s: float = 30.0


@dataclass(frozen=True)
class Debug:
    enabled: bool = True  # 唯一调试开关：全开/全关；正式上线可关闭
    max_records: int = 10000
    line_width: int = 2
    font_scale: float = 0.5
    replay_margin_px: int = 20
    replay_line_px: int = 28
    probe_tiles: tuple = (192, 320, 480, 640)


@dataclass(frozen=True)
class Diagnosis:
    """离线排查参数；不参与实时算法。"""
    sample_per_camera_split: int = 8
    seed: int = 42
    low_confidence: float = 0.001
    label_iou: float = 0.5
    image_sizes: tuple = (640, 1280)
    background_luma: int = 128
    mosaic_columns: int = 3
    preview_width: int = 480
    context_padding: tuple = (0.0, 0.25, 0.5, 1.0, 2.0)
    rectangular_tiles: tuple = ((192,384), (320,640))
    audit_workers: int = 8
    sky_region: tuple = (0.17, 0.0, 0.35, 0.20)
    padding_values: tuple = (0, 114, 128, 255)
    devices: tuple = ('cpu', 'cuda:0')
    numerical_tolerance: float = 0.002


@dataclass(frozen=True)
class Preparation:
    """离线标注准备参数，不改变实时检测。"""
    pilot_per_scene: int = 10
    background_per_scene: int = 10
    background_tile_sizes: tuple = (320, 480, 640)
    background_stride_ratio: float = 0.5
    background_exclusion_margin: float = 0.15
    background_per_image_max: int = 2
    background_same_image_iou: float = 0.25
    background_vertical_bands: int = 3
    background_seed: int = 42
    background_contact_columns: int = 4
    background_preview_width: int = 320
    background_caption_height: int = 30
    dataset_val_fraction: float = 0.15
    dataset_test_fraction: float = 0.15
    dataset_seed: int = 42
    expected_background_exclusions: int = 2
    context_tile_shapes: tuple = ((960,960),(1280,960))
    context_stride_ratio: float = 0.5
    context_min_objects: int = 3
    context_min_retained_area: float = 0.5
    context_max_per_image: int = 4
    label_decimal_places: int = 8


@dataclass(frozen=True)
class FocusExperiment:
    """当前两视角的离线试验。范围外不作识别保证；不影响在线算法。"""
    focus_regions: tuple = (
        ('scene01', ((0,450,1920,1080),)),
        ('scene02', ((0,0,890,600),(1320,120,1760,330))),
    )
    forbidden_regions: tuple = (('scene02', ((960,0,1280,320),)),)
    min_box_side: float = 3.0
    min_retained_area: float = 0.20
    min_objects: int = 3
    context_shapes: tuple = ((640,480),(960,480))
    context_max_per_focus: int = 2
    degraded_fraction: float = 0.5
    degraded_scale: float = 0.5
    blur_kernel: int = 3
    blur_sigma: float = 0.45
    jpeg_quality: int = 75
    pretrained: Path = PROJECT / 'models/yolo11s.pt'
    vehicle_confidence: float = 0.35
    vehicle_classes: tuple = ('car','truck','bus','motorcycle')
    vehicle_body_inset: float = 0.15
    vehicle_limit: int = 8
    vehicle_min_side: int = 32
    reviewed_vehicle_indices: tuple = (2,4,7,11,12,15,17,19,21,23,24,25,28,29,32,33,34,37,39,40)
    vehicle_negative_selection: tuple = (
        (2,(0.0,0.4,1.0,1.0)),(24,(0.1,0.2,0.8,0.9)),
        (25,(0.0,0.0,1.0,1.0)),(33,(0.0,0.0,1.0,1.0)),
        (37,(0.0,0.0,1.0,1.0)),(39,(0.0,0.0,1.0,1.0)),
        (40,(0.0,0.0,1.0,1.0)),(32,(0.0,0.4,1.0,1.0)),
    )
    vehicle_cooccurrence_index: int = 27
    vehicle_overlap_guard: float = 0.85
    occlusion_fraction: float = 0.25
    occlusion_max_fraction: float = 0.35
    epochs: int = 12
    patience: int = 6
    batch: int = 2
    imgsz: int = 960
    device: str = '0'
    workers: int = 0
    learning_rate: float = 0.001
    seed: int = 42
    mosaic: float = 0.5
    close_mosaic: int = 3
    translate: float = 0.15
    scale: float = 0.5
    fliplr: float = 0.5
    hsv_h: float = 0.01
    hsv_s: float = 0.35
    hsv_v: float = 0.30
    degrees: float = 2.0
    evaluation_confidence: float = 0.35
    evaluation_iou: float = 0.5
    replay_seconds: int = 180
    replay_interval_s: int = 30
    replay_frames_per_camera: int = 6
    run_name: str = 'focus_yolo11s_pilot_20261026'


@dataclass(frozen=True)
class SmallTargetPreparation:
    """Offline multi-object crops from existing reviewed polygons; not runtime ROI."""
    regions: tuple = (
        ('scene01', ((1400,375,1640,490),)),
        ('scene02', ((600,90,890,220),)),
    )
    source_dataset: str = 'focus_dataset_20261026_v1'
    output_dataset: str = 'focus_dataset_20261026_v2_small'
    visual_occlusion_exclusions: tuple = ('1749_20261014_105732', '1749_20261014_155733')


@dataclass(frozen=True)
class HardNegativePreparation:
    """Mine train-only backgrounds; every selected image still needs visual review."""
    source_dataset: str = 'focus_dataset_20261026_v2_small'
    output_dataset: str = 'focus_dataset_20261026_v3_hardneg'
    review_batch: str = 'hard_negatives_20261026'
    mining_weights: Path = PROJECT / 'training/water_barrier_v2/05_weights/pilot_yolo11s_best.pt'
    regions: tuple = (
        ('scene01', ((0,0,320,320),(320,0,640,320),(640,0,960,320),
                     (800,0,1120,320),(960,0,1280,320),(1120,0,1440,320),
                     (1280,0,1600,320),(1600,0,1920,320))),
        ('scene02', ((0,640,640,1080),(640,640,1280,1080),
                     (1280,640,1920,1080),(0,0,640,80))),
    )
    per_camera_limits: tuple = (('1749',24),('1750',8))
    max_per_source: int = 2
    max_per_region: int = 6
    preview_per_page: int = 8
    preview_columns: int = 2
    run_name: str = 'focus_yolo11s_hardneg_20261026'
    control_source_dataset: str = 'focus_dataset_20261026_v1'
    control_output_dataset: str = 'focus_dataset_20261026_v3_negonly'
    control_run_name: str = 'focus_yolo11s_negonly_20261026'


@dataclass(frozen=True)
class DayNightPreparation:
    """Offline boundary-aware, hourly coverage data; unrelated to runtime ROI."""
    batch_name: str = 'daynight_boundary_20261026'
    dataset_name: str = 'dataset_20261026_v4_daynight'
    run_name: str = 'yolo11s_daynight_boundary_20261026'
    positive_repeat: int = 4
    evaluation_device: str = 'cpu'
    evaluation_threads: int = 2
    tile_trials: tuple = (
        ('input960',640,0.25,960,8.0),
        ('small2',640,0.25,960,2.0),
        ('overlap50',640,0.50,960,2.0),
        ('tile960',960,0.50,960,2.0),
    )
    square_tile_trials: tuple = (
        ('square640',640,0.50,960,2.0),
        ('square960',960,0.50,960,2.0),
    )
    multiscale_tiles: tuple = (480,960)
    multiscale_overlap: float = 0.50
    multiscale_imgsz: int = 960
    multiscale_min_box_px: float = 2.0
    positive_regions: tuple = (
        ('scene01', ((0,500,540,1080),(400,450,1100,920),(800,380,1640,720),(1320,380,1780,1080))),
        ('scene02', ((0,100,610,600),(560,80,900,270),(1300,100,1760,360))),
    )
    hours_per_day: int = 24
    frames_per_hour: int = 1
    feature_width: int = 960
    sift_features: int = 4000
    ratio_test: float = 0.70
    min_inliers: int = 18
    min_inlier_ratio: float = 0.50
    max_error_px: float = 1.5
    ransac_px: float = 2.0
    max_scale_error: float = 0.02
    max_displacement: float = 0.02
    min_cells: int = 3
    reference_attempts: int = 4
    boundary_step_px: int = 1
    boundary_max_steps: int = 512
    min_retained_area: float = 0.50
    min_partial_side_px: float = 8.0
    negative_guard_px: float = 16.0
    preview_width: int = 320
    preview_columns: int = 4
    preview_rows: int = 6
    night_start_hour: int = 18
    night_end_hour: int = 6
    new_val_groups: tuple = ('1749:20261013:19','1750:20261013:22','1750:20261014:04')
    new_test_groups: tuple = ('1749:20261013:20','1750:20261014:02','1750:20261014:07')
    epochs: int = 24
    patience: int = 8
    batch: int = 4
    mosaic: float = 0.0
    degrees: float = 0.0
    translate: float = 0.0
    scale: float = 0.0


@dataclass(frozen=True)
class DatasetOptimization:
    batch_name: str = 'optimized_20261027'
    dataset_name: str = 'dataset_20261027_v5_reviewed'
    run_name: str = 'yolo11s_v5_manual_20261027'
    context_regions: tuple = (
        ('scene01', ((0,500,960,1080),(1100,375,1740,720),(1320,380,1780,1080))),
        ('scene02', ((0,100,900,960),(550,90,900,400),(1300,140,1850,640))),
    )
    night_regions: tuple = (
        ('scene01', ((0,500,960,1080),)),
        ('scene02', ((0,100,610,960),(1500,150,1750,400))),
    )
    video_sources: tuple = ('1749_20261013_185734','1749_20261013_152823',
                            '1750_20261014_002734','1750_20261014_125733')
    video_seconds: tuple = (30,90)
    background_periods: tuple = ((6,11,'morning'),(11,15,'midday'),(15,18,'evening'))
    backgrounds_per_region_period: int = 1
    legacy_preview_count: int = 24
    epochs: int = 40
    patience: int = 10
    batch: int = 4
    positive_repeat: int = 1


@dataclass(frozen=True)
class TargetedOptimization:
    batch_name: str = 'targeted_v6_20261027'
    dataset_name: str = 'dataset_20261027_v6_targeted'
    run_name: str = 'yolo11s_v6_targeted_20261027'
    epochs: int = 24
    patience: int = 8
    batch: int = 4
    learning_rate: float = 0.0003
    targeted_repeat: int = 2
    video_seconds: tuple = (45, 120)
    video_sources: tuple = ('1749_20261013_182734', '1749_20261013_152823')
    grass_sources: tuple = ('1750_20261013_112823','1750_20261014_125733',
        '1750_20261014_172734','1750_20261013_192734','1750_20261013_212734',
        '1750_20261014_002734','1750_20261014_052734')
    grass_regions: tuple = ((1280,100,1920,740),(1280,100,1920,1080))
    umbrella_regions: tuple = ((1080,0,1300,150),(1080,0,1300,330))
    far_sources: tuple = ('1749_20261014_095732','1749_20261014_135732',
        '1749_20261014_142732','1750_20261014_082732','1750_20261014_125733',
        '1750_20261014_172734')
    far_regions: tuple = (('1749',((1000,375,1650,735),(1200,375,1650,650))),
        ('1750',((350,70,950,670),(500,70,950,520))))
    night_regions: tuple = ((750,390,1730,1080),(950,390,1730,1080),(1150,520,1730,1080))
    vehicle_regions: tuple = ((800,390,1730,1080),(1000,390,1730,1080))


@dataclass(frozen=True)
class SmallDarkOptimization:
    batch_name: str = 'small_dark_v7_20261027'
    dataset_name: str = 'dataset_20261027_v7_small_dark'
    run_name: str = 'yolo11s_v7_small_dark_20261027'
    epochs: int = 24
    patience: int = 8
    batch: int = 4
    learning_rate: float = 0.0002
    targeted_repeat: int = 2
    evaluation_tile_sizes: tuple = (480, 960)
    small_target_short_side_px: float = 20.0
    # Frozen diagnostic source indices; recall only, not exhaustive full-frame GT.
    diagnostic_cases: tuple = (
        ('night_middle', '1749_20261013_192734', tuple(range(7,18))),
        ('night_right', '1749_20261013_192734', tuple(range(51,57))),
        ('night_middle', '1749_20261013_202734', tuple(range(7,18))),
        ('night_right', '1749_20261013_202734', tuple(range(51,57))),
        ('night_dark_left', '1750_20261014_022734', tuple(range(3,10))),
        ('night_grass', '1750_20261014_022734', (23,24,25)),
        ('far_day', '1750_20261014_072733', tuple(range(12,23))),
        ('far_day', '1749_20261014_125733', tuple(range(30,51))),
    )
    video_seconds: tuple = (30, 120)
    video_sources: tuple = (
        ('1749_20261013_185734','1749_20261013_182734'),
        ('1749_20261014_135732','1749_20261014_135732'),
        ('1749_20261014_095732','1749_20261014_095732'),
        ('1750_20261014_082732','1750_20261014_082732'),
        ('1750_20261014_062733','1750_20261014_062733'),
        ('1750_20261014_175734','1750_20261014_175734'),
        ('1750_20261013_212734','1750_20261013_212734'),
    )
    far_regions: tuple = (
        ('1749',((1380,370,1640,472),(1450,370,1640,470),(1250,375,1600,520))),
        ('1750',((500,70,900,400),(650,70,900,320),(350,70,900,620))),
    )
    night_regions: tuple = (
        ('1749',((980,390,1640,720),(1380,370,1640,472),(1150,520,1730,1080))),
        ('1750',((330,100,610,500),(1280,100,1920,740))),
    )


@dataclass(frozen=True)
class Params:
    timing: Timing = field(default_factory=Timing)
    motion: Motion = field(default_factory=Motion)
    alignment_recovery: AlignmentRecovery = field(default_factory=AlignmentRecovery)
    reference: Reference = field(default_factory=Reference)
    inference: Inference = field(default_factory=Inference)
    matching: Matching = field(default_factory=Matching)
    evidence: Evidence = field(default_factory=Evidence)
    statistics: Statistics = field(default_factory=Statistics)
    policy: DecisionPolicy = field(default_factory=DecisionPolicy)
    initial_gaps: InitialGaps = field(default_factory=InitialGaps)
    storage: Storage = field(default_factory=Storage)
    debug: Debug = field(default_factory=Debug)
    diagnosis: Diagnosis = field(default_factory=Diagnosis)
    preparation: Preparation = field(default_factory=Preparation)
    focus_experiment: FocusExperiment = field(default_factory=FocusExperiment)
    small_target_preparation: SmallTargetPreparation = field(default_factory=SmallTargetPreparation)
    hard_negative_preparation: HardNegativePreparation = field(default_factory=HardNegativePreparation)
    daynight_preparation: DayNightPreparation = field(default_factory=DayNightPreparation)
    dataset_optimization: DatasetOptimization = field(default_factory=DatasetOptimization)
    targeted_optimization: TargetedOptimization = field(default_factory=TargetedOptimization)
    small_dark_optimization: SmallDarkOptimization = field(default_factory=SmallDarkOptimization)

    def validate(self):
        if not (0 <= self.policy.day_start_hour < self.policy.day_end_hour <= 24
                and -12 <= self.policy.utc_offset_hours <= 14
                and self.policy.day_confirm_samples > 0
                and self.policy.night_confirm_samples > 0
                and 0 < self.policy.recovery_confidence <= 1
                and 0 <= self.policy.deep_night_end_hour < self.policy.deep_night_start_hour <= 24
                and self.reference.night_luma < self.policy.twilight_night_luma < self.reference.day_luma
                and 0 < self.reference.lighting_upper_fraction < 1
                and 0 <= self.policy.upper_night_luma < self.policy.upper_twilight_luma <= 255
                and self.policy.suspect_day_s > 0 and self.policy.night_credit_after_s > 0
                and 0 <= self.policy.night_credit_s < self.policy.suspect_day_s):
            raise ValueError('Invalid daylight/accumulation policy')
        r = self.alignment_recovery
        if not (r.columns > 0 and r.rows > 0 and 0 < r.min_response <= 1
                and r.min_texture > 0 and 0 <= r.max_excluded_fraction < 1
                and 1 < r.min_support <= r.columns*r.rows
                and 1 < r.min_columns <= r.columns and 1 < r.min_rows <= r.rows
                and 0 < r.min_span_x <= 1 and 0 < r.min_span_y <= 1
                and r.consistency_px > 0 and r.blur_kernel > 0 and r.blur_kernel % 2 == 1
                and r.gradient_kernel in (1,3,5,7)
                and self.evidence.appearance_patch_px > 0
                and self.evidence.appearance_max_age_s > 0 and self.evidence.appearance_difference > 0):
            raise ValueError('Invalid background recovery/appearance parameters')
        if not (self.initial_gaps.min_row_observations >= 3
                and self.initial_gaps.local_neighbors > 0
                and self.initial_gaps.min_neighbor_intervals > 0
                and 1 < self.initial_gaps.min_gap_ratio < self.initial_gaps.max_gap_ratio
                and 0 < self.initial_gaps.integer_tolerance < 0.5
                and 0 < self.initial_gaps.min_direction_cosine <= 1
                and 0 < self.initial_gaps.duplicate_iou <= 1
                and self.initial_gaps.max_size_ratio >= 1
                and self.initial_gaps.max_inferred_per_gap > 0 and self.initial_gaps.max_slots > 0):
            raise ValueError('Invalid initial-gap parameters')
        if not (self.timing.period_s > 0 and 0 < self.timing.alignment_phase < 1
                and self.timing.fast_s > 0 and self.timing.rebuild_periods > 0):
            raise ValueError('Invalid timing parameters')
        if not (0 <= self.inference.overlap < 1 and self.inference.tile_px > 0
                and 0 < self.inference.confidence <= 1 and self.inference.imgsz > 0):
            raise ValueError('Invalid inference parameters')
        if not (0 < self.statistics.min_coverage <= 1
                and 0 < self.statistics.missing_ratio <= 1
                and self.statistics.window_s >= self.timing.period_s):
            raise ValueError('Invalid statistics parameters')
        if not (self.motion.enter_factor > self.motion.exit_factor > 0
                and self.motion.jitter_ceiling >= self.motion.jitter_floor > 0
                and self.storage.retention_s > self.reference.refresh_s > 0):
            raise ValueError('Invalid motion/retention parameters')
        if not (self.evidence.patch_px > 0 and self.reference.min_baseline_observations > 0
                and self.reference.max_baseline_observations >= self.reference.min_baseline_observations
                and self.reference.per_light > 0 and self.motion.grid > 0
                and 0 < self.motion.min_cells <= self.motion.grid**2
                and 0 < self.motion.stationary_min_cells <= self.motion.min_cells
                and self.motion.stationary_min_inliers >= self.motion.min_inliers
                and self.motion.min_inlier_ratio <= self.motion.stationary_min_inlier_ratio <= 1
                and self.motion.confirm_samples > 0 and self.motion.return_samples > 0
                and self.storage.cleanup_s > 0 and self.storage.checkpoint_s > 0
                and self.debug.max_records > 0):
            raise ValueError('Invalid counts/interval parameters')
        if not (0 < self.statistics.recovery_ratio <= 1 and self.statistics.recovery_s > 0
                and self.statistics.cooldown_s > 0 and self.matching.max_center_ratio >= self.matching.center_ratio > 0
                and 0 < self.matching.min_size_ratio <= self.matching.max_size_ratio
                and 0 < self.reference.baseline_presence <= 1
                and 0 <= self.reference.max_unresolved_fraction <= 1):
            raise ValueError('Invalid matching/recovery parameters')
        return self


PARAMS = Params().validate()
