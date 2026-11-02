"""Validated runtime contracts for cameras, shared offset and business modules."""
from dataclasses import dataclass, field
from enum import Enum
from night_lamp.settings import LampParams as LampObservationSpec

from .algo import Report  # noqa: F401  (re-export, 供 Runtime/Reporter 使用)

__all__ = [
    'Report','ScheduleState','CalibrationStatus','ScheduleSpec','CallbackSpec',
    'WaterGapSpec','DetectSpec','TrackSpec','OffsetSpec','LampDetectorSpec',
    'LampObservationSpec','NightLampSpec','AlgorithmBinding','CameraSpec','RuntimeConfig',
]


class ScheduleState(str, Enum):
    """Runtime 全局状态机。REPORT 不再是独立运行模式：它是 FINALIZING 的输出事件。"""
    OFF = "off"
    DAY = "day"            # 相机按各自 algo.schedule 活跃
    NIGHT = "night"
    FROZEN = "frozen"      # 夜间采样停止，保留灯具档案等待 07:00
    FINALIZING = "finalizing"


class CalibrationStatus(str, Enum):
    READY = "ready"
    PENDING = "calibration_pending"   # 打印但不禁用；不得静默关闭


@dataclass
class ScheduleSpec:
    timezone: str = "Asia/Shanghai"
    day_start: str = "07:00"
    day_end: str = "17:00"
    night_start: str = "20:00"
    night_end: str = "03:00"
    report_at: str = "07:00"


@dataclass
class CallbackSpec:
    url: str = ""
    timeout_s: float = 8.0
    queue_size: int = 64              # 必须有界，防止 callback 挂起导致内存泄漏
    image_field: str = "image_base64"
    enabled: bool = True
    retries: int = 0                  # 失败后额外重试次数（0 = 不重试）


@dataclass
class DetectSpec:
    """水马单帧检测参数（原标定文件 detect 段，上移为运行期选项）。"""
    trim: float = 0.04
    conf_th: float = 0.30
    support_th: float = 0.3
    min_box_width: int = 30
    floor_max: float = 0.24
    core_exit: float = 0.06
    min_width: int = 40
    core_min_width: int = 30


@dataclass
class TrackSpec:
    """水马时序跟踪参数（原标定文件 track 段非策略键，上移为运行期选项）。

    rer_threshold/alarm_hold_s/reconfirm_s 不在此处：它们是 WaterGapSpec 的
    策略字段（confidence/alarm_hold_s/reconfirm_s），注入时同名写入 track。
    """
    decay_rate: float = 0.5
    track_stale_s: float = 8.0
    match_iou: float = 0.25
    match_center_ratio: float = 0.5
    median_window: int = 5
    intact_reset_s: float = 5.0
    rer_purity: float = 0.70
    patch_size: int = 8


@dataclass
class WaterGapSpec:
    """水马：可切换原算法/逐block后端，配置业务策略与出图节奏。"""
    calibration: str = ""
    backend: str = "legacy"
    manual_blocks: str = ""
    manual_reference: str = ""
    manual_parameters: str = ""
    detect_hz: float = 0.0  # 0: process latest frames without a frequency limit.
    max_observation_gap_s: float = 3.0
    schedule: str = "day"             # 由 config 决定，不写死
    status: CalibrationStatus = CalibrationStatus.READY  # manifest-only, no runtime branch

    report_interval_s: int = 3600     # 每小时一张缺口图
    alarm_min_interval_s: int = 3600  # 最小告警间隔（与 report 独立）
    confidence: float = 0.25          # RER 门（track.rer_threshold）

    # 状态机（注入算法，算法内部不硬编码）
    alarm_hold_s: float = 10.0
    reconfirm_s: float = 5.0

    detect: DetectSpec = field(default_factory=DetectSpec)
    track: TrackSpec = field(default_factory=TrackSpec)


@dataclass
class OffsetSpec:
    enabled: bool = True
    root: str = ''
    period_s: float = 30.0
    fast_s: float = 1.0
    stable_s: float = 120.0
    refresh_s: float = 600.0
    reference_ttl_s: float = 259200.0
    feedback_timeout_s: float = 120.0
    max_iterations: int = 3
    error_px: float = 3.0


@dataclass
class LampDetectorSpec:
    weights: str = ''
    root: str = ''
    device: str = 'cpu'
    confidence: float = 0.35
    imgsz: int = 320
    detect_interval_s: float = 0.16
    visibility: str = 'background'


@dataclass
class NightLampSpec:
    """YOLO lamp discovery, identity migration and complete-pulse observation."""
    schedule: str = 'night'
    status: CalibrationStatus = CalibrationStatus.READY
    detector: LampDetectorSpec = field(default_factory=LampDetectorSpec)
    observation: LampObservationSpec = field(default_factory=LampObservationSpec)


@dataclass
class AlgorithmBinding:
    schedule: str                     # "day" | "night"
    spec: object                      # WaterGapSpec | NightLampSpec


@dataclass
class CameraSpec:
    id: str
    name: str
    rtsp_url: str
    enabled: bool = True
    algorithms: dict = field(default_factory=dict)   # name -> AlgorithmBinding
    offset: OffsetSpec = field(default_factory=OffsetSpec)


@dataclass
class RuntimeConfig:
    schedule: ScheduleSpec = field(default_factory=ScheduleSpec)
    callback: CallbackSpec = field(default_factory=CallbackSpec)
    cameras: list = field(default_factory=list)      # list[CameraSpec]
    server: dict = field(default_factory=dict)       # host/port/jpeg_quality/...
    memory_budget_mb_per_camera: int = 500           # manifest-only, not consumed at runtime
    output_dir: str = ""                             # 空 = 不落盘；host-mounted root
    alarm_dir: str = ""                              # 告警帧落盘根；空则回退 output_dir
    persist_images: bool = True                      # False = 不落图片，仅保留交换格式(JSON)
    retention_hours: float = 0.0                     # >0 时定时清理落盘目录中超过该时长的文件
