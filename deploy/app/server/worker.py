"""Camera runtime: one shared offset owner, water runner and YOLO lamp adapter."""
import dataclasses
import threading
import time
from collections import deque
from datetime import datetime

import cv2 as cv
from loguru import logger

from . import registry, render
from .contracts import ScheduleState
from .schedule import (
    Clock, RealClock, RuntimeEvent, RuntimeEventType, Scheduler,
)
from .source import Reader

LOG_EVERY = 100
EVENTS_KEPT = 200
PUMP_INTERVAL_S = 0.5
TICK_PERIOD_S = 0.02
STOP_JOIN_TIMEOUT_S = 5.0
STREAM_LOST_AFTER_S = 10.0
STREAM_HEARTBEAT_S = 600.0

DAY_DIAG_PERIOD_S = 300.0


def _default_day_runner(camera, spec, algo):
    from .agents.day import DayRunner
    return DayRunner(camera, spec, algo)


def _default_reporter(cfg):
    from .reporting import Reporter
    return Reporter(cfg)


def _default_store(cfg):
    """output_dir 为空时不落盘（测试/旧部署）；否则写 host-mounted 目录。"""
    out = str(getattr(cfg, "output_dir", "") or "").strip()
    if not out:
        return None
    from .output import ReportStore
    return ReportStore(out, persist_images=bool(
        getattr(cfg, "persist_images", True)))


def _default_sweeper(cfg):
    """按 retention_hours 定时清理落盘目录；<=0 或无可清目录时 no-op。"""
    from .maintenance import Sweeper
    hours = float(getattr(cfg, "retention_hours", 0.0) or 0.0)
    roots = (getattr(cfg, "output_dir", ""), getattr(cfg, "alarm_dir", ""))
    return Sweeper(roots, hours)


def _yolo_sweeper(cfg):
    """YOLO私有资产按自身保留期清理，不能套用旧服务的48小时策略。"""
    from .maintenance import Sweeper
    enabled = any(getattr(binding.spec, "backend", "") == "yolo"
                  for cam in cfg.cameras for binding in cam.algorithms.values())
    if not enabled:
        return Sweeper((), 0)
    from water_barrier.yolo_params import PARAMS
    return Sweeper((PARAMS.storage.root,), PARAMS.storage.retention_s / 3600,
                   interval_s=PARAMS.storage.cleanup_s)


# 业务策略字段 -> 算法 track 键（仅当 spec 暴露该字段时注入）。
_WATER_POLICY = (("confidence", "rer_threshold"),
                 ("alarm_hold_s", "alarm_hold_s"),
                 ("reconfirm_s", "reconfirm_s"))


def _apply_water_params(calib, spec):
    """把 WaterGapSpec 的 detect/track/策略合并进标定 dict（算法只读 calib）。

    水马数据文件只剩 rows/road_rois；detect、track 与 rer_threshold/
    alarm_hold_s/reconfirm_s 全由本函数从 spec 注入，保证单一来源。
    spec 无这些字段时（如夜灯）原样返回。
    """
    if not isinstance(calib, dict):
        return calib
    out = dict(calib)

    detect = getattr(spec, "detect", None)
    if dataclasses.is_dataclass(detect):
        out["detect"] = dataclasses.asdict(detect)

    track = dict(out.get("track") or {})
    track_spec = getattr(spec, "track", None)
    if dataclasses.is_dataclass(track_spec):
        track.update(dataclasses.asdict(track_spec))
    for attr, key in _WATER_POLICY:
        val = getattr(spec, attr, None)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            track[key] = val
    if track:
        out["track"] = track
    return out


def _load_calibration(name, spec):
    """spec.calibration 是（绝对）路径；也可能是已加载内容（dict），两者都兼容。"""
    if name == "water_gap" and getattr(spec, "backend", "legacy") == "yolo":
        return {"backend": "yolo"}  # YOLO 自动参照，不读取旧 poly 标定。
    if name == "water_gap" and getattr(spec,"backend","legacy") == "blocks":
        from water_barrier.online_blocks import load_config
        return load_config(spec)
    cal = getattr(spec, "calibration", "")
    if not isinstance(cal, str):
        return _apply_water_params(cal, spec)
    from .config import ROOT
    from .config_validate import resolve_resource
    resolved = resolve_resource(cal, ROOT, f"algorithm.{name}.calibration")
    return _apply_water_params(registry.load_calib(name, resolved), spec)


class CameraWorker:
    def __init__(self, camera, scheduler: Scheduler, reporter,
                 jpeg_quality: int = 80, evidence=None,
                 clock: Clock | None = None,
                 day_factory=None, store=None,
                 offset_factory=None, lamp_detector_factory=None):
        self._camera = camera
        self.id = camera.id
        self.name = getattr(camera, "name", None) or camera.id
        self._scheduler = scheduler
        self._reporter = reporter
        self._store = store
        self._clock = clock or RealClock()
        self._quality = jpeg_quality
        self._evidence = evidence
        self._day_factory = day_factory or _default_day_runner

        self._processing_lock=threading.RLock()
        self._offset_runtime=None
        self._lamp_adapter=None
        self._shared_last_frame=None
        from .contracts import OffsetSpec
        offset_spec=getattr(camera,'offset',OffsetSpec())
        if offset_spec.enabled:
            from camera_motion.runtime import CameraOffsetRuntime
            self._offset_runtime=(offset_factory or CameraOffsetRuntime)(self.id,offset_spec,self._clock)
        lamp_binding=(getattr(camera,'algorithms',None) or {}).get('night_lamp')
        if lamp_binding:
            if self._offset_runtime is None:raise ValueError('Lamp detection requires shared offset')
            from night_lamp.production import LampAdapter
            detector=lamp_detector_factory(lamp_binding.spec.detector) if lamp_detector_factory else None
            self._lamp_adapter=LampAdapter(self._offset_runtime,lamp_binding.spec,self._clock,detector)
            self._offset_runtime.attach('night_lamp',self._lamp_adapter)
        self._reader = Reader(camera.rtsp_url)
        self._lock = threading.Lock()
        self._vis = None
        self._debug = {}
        self._status = "OK"
        self._events = deque(maxlen=EVENTS_KEPT)
        self._n = 0
        self._built = False
        self._day_runners: dict = {}
        self._day_alarm_pending: dict = {}
        self._night: dict = {}
        self._latest_frame = None
        self._diag_last: dict = {}
        self._stop = threading.Event()
        self._stream_lost = False
        self._stream_last_emit = None
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name=f"cam-{self.id}")
        self._ticker = threading.Thread(target=self._tick_loop, daemon=True,
                                        name=f"tick-{self.id}")

    # ------------------------------------------------------------ 只读出口
    @property
    def camera(self):
        return self._camera

    @property
    def algo_names(self):
        return sorted((getattr(self._camera, "algorithms", None) or {}).keys())

    def start(self):
        self._stop.clear()
        self._reader.start()
        self._thread.start()
        self._ticker.start()
        return self

    def stop(self):
        self._stop.set()
        self._reader.stop()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=STOP_JOIN_TIMEOUT_S)
        if self._ticker.is_alive() and self._ticker is not threading.current_thread():
            self._ticker.join(timeout=STOP_JOIN_TIMEOUT_S)
        if self._lamp_adapter:self._lamp_adapter.save_state()
        if self._offset_runtime:self._offset_runtime.save()

    def _tick_loop(self):
        """Independent wall-clock ticker.

        Grid sampling must not share the frame loop's critical path (1080p
        preview encode + day algorithms): a slow iteration used to collapse
        several due grid points onto one stale frame and silently drop samples.
        """
        while not self._stop.wait(TICK_PERIOD_S):
            self._tick_night(self._clock.wall())

    def latest(self):
        with self._lock:
            return self._vis, dict(self._debug)

    def status(self):
        with self._lock:
            return self._status

    def debug_layer(self, layer):
        with self._lock:
            m = self._debug.get(layer)
            return None if m is None else m.copy()

    def events(self):
        with self._lock:
            return list(self._events)

    def publish(self, report) -> None:
        """落盘（若配置 output_dir）+ 异步 callback；任何失败都不冒泡。"""
        if report is None:
            return
        if self._store is not None:
            try:
                self._store.save(report)
            except Exception:
                logger.exception("[{}] report 落盘失败", self.id)
        try:
            self._reporter.submit(report)
        except Exception:
            logger.exception("[{}] report submit 失败", self.id)

    def _tick_night(self, now_wall):
        """墙钟推进所有夜间适配器并发布其结算的桶报告（断流时也调用）。"""
        with self._lock:
            adapters = list(self._night.items())
        for name, ad in adapters:
            if not self._scheduler.is_active(self.id, name):
                continue
            tick = getattr(ad, "tick", None)
            if callable(tick):
                try:
                    with self._processing_lock:tick(now_wall)
                except Exception:
                    logger.exception("[{}] night tick 失败: {}", self.id, name)


    # ------------------------------------------------------------ Runtime 回调
    def start_night(self):
        """Activate the sole lamp adapter without replacing persistent IDs."""
        if self._lamp_adapter is None:return
        with self._lock:
            if 'night_lamp' in self._night:return
            self._lamp_adapter.start_night()
            self._night['night_lamp']=self._lamp_adapter

    def flush_night(self):
        """停机前把在跑的夜间累计落盘；best-effort，不冒泡。"""
        with self._lock:
            adapters = list(self._night.values())
        if self._lamp_adapter and self._lamp_adapter not in adapters:
            adapters.append(self._lamp_adapter)
        if self._offset_runtime:
            with self._processing_lock:self._offset_runtime.save()
        for ad in adapters:
            save = getattr(ad, "save_state", None)
            if not callable(save):
                continue
            try:
                with self._processing_lock:save()
            except Exception:
                logger.exception("[{}] night flush 失败", self.id)

    def _check_stream_health(self, now_mono: float):
        """检测断流/恢复并推状态事件；Reader 无 health() 时静默跳过。"""
        health = getattr(self._reader, "health", None)
        if not callable(health):
            return
        info = health()
        last_frame = info.get("last_frame_mono")
        ref = last_frame if last_frame is not None else info.get("started_mono")
        stale = ref is not None and now_mono - ref >= STREAM_LOST_AFTER_S
        # A live decoder process/socket can stop delivering pixels. Connection
        # state alone must neither hide that outage nor announce recovery.
        if info.get("connected") and last_frame is not None and not stale:
            if self._stream_lost:
                self._stream_lost = False
                self._stream_last_emit = now_mono
                self._set_status("OK")
                self._emit_status("stream_restored", info)
            return
        if ref is None or not stale:
            return
        if not self._stream_lost:
            self._stream_lost = True
            self._stream_last_emit = now_mono
            self._set_status("STREAM_LOST")
            self._emit_status("stream_lost", info)
            return
        last_emit = self._stream_last_emit if self._stream_last_emit is not None else now_mono
        if now_mono - last_emit >= STREAM_HEARTBEAT_S:
            self._stream_last_emit = now_mono
            self._emit_status("stream_lost", info)

    def _set_status(self, status: str):
        with self._lock:
            self._status = status

    def _emit_status(self, status: str, health: dict):
        # 仅保留内部可观测（/events 与画面角标）；不对外推送 camera_status。
        with self._lock:
            self._events.append({"ts": datetime.now().astimezone().isoformat(),
                                 "algo": "stream", "kind": "status",
                                 "payload": {"status": status},
                                 "evidence_path": None})
        logger.warning("[{}] 流状态: {}", self.id, status)

    def freeze_night(self):
        """NIGHT_FREEZE：停止累积（各 session 冻结）。"""
        with self._lock:
            adapters = list(self._night.values())
        for ad in adapters:
            try:
                ad.freeze()
            except Exception:
                logger.exception("[{}] night freeze 失败", self.id)
        logger.info("[{}] 夜间采样冻结", self.id)


    def _select_day_base(self):
        if self._offset_runtime is not None:
            image=self._offset_runtime.day_reference()
            if image is not None:return image,'shared_day_reference'
        return (self._latest_frame.copy(),'live') if self._latest_frame is not None else (None,'none')

    def _log_day_diag(self, name, res):
        """白昼算法诊断（节流）：support 峰值 / 命中 run 数 / 轨道状态与 RER。"""
        diag = res.debug.get("diag") if isinstance(res.debug, dict) else None
        if not isinstance(diag, dict):
            return
        logger.info("[{}] {} diag sup_max={} runs={} tracks={}",
                    self.id, name, diag.get("sup_max"), diag.get("runs"),
                    diag.get("tracks"))

    def _log_night_report(self,name,report,base,base_src):
        metadata=getattr(report,'metadata',{}) or {}
        logger.info('[{}] lamp report status={} lamps={} reference={}',
            self.id,report.status,len(metadata.get('lamps',[])),metadata.get('reference'))

    def finalize_night(self, ts_wall: datetime):
        """NIGHT_FINALIZE：day_ref -> finalize -> submit -> release（按夜隔离）。"""
        with self._lock:
            adapters = list(self._night.items())
        base, base_src = self._select_day_base()
        for name, ad in adapters:
            try:
                with self._processing_lock:report = ad.finalize(base, ts_wall)
                if report is not None:
                    self._log_night_report(name, report, base, base_src)
                    self.publish(report)
            except Exception:
                logger.exception("[{}] night finalize 失败: {}", self.id, name)
            finally:
                try:
                    ad.release()
                except Exception:
                    logger.exception("[{}] night release 失败: {}", self.id, name)
                with self._lock:
                    self._night.pop(name, None)
        logger.info("[{}] 夜间 finalize 完成并 release", self.id)

    def snapshot_reports(self, ts_wall: datetime) -> list:
        """HOURLY_SNAPSHOT：本小时内出过真报警的白昼算法才出图。

        无真报警则整点不推送（告警本身由 submit_event 即时上报，不丢）。
        """
        out = []
        for name, runner in list(self._day_runners.items()):
            binding = (getattr(self._camera, "algorithms", None) or {}).get(name)
            spec = getattr(binding, "spec", None)
            if getattr(spec, "backend", "") == "yolo":
                continue  # YOLO已即时发送去重报警，整点不能再重复推送。
            if not getattr(spec, "report_interval_s", None):
                continue
            if not self._scheduler.is_active(self.id, name):
                self._day_alarm_pending[name] = False
                continue
            if not self._day_alarm_pending.get(name):
                continue
            try:
                rep = runner.snapshot(ts_wall)
            except Exception:
                logger.exception("[{}] snapshot 失败: {}", self.id, name)
                continue
            if rep is not None:
                self._day_alarm_pending[name] = False
                out.append(rep)
        return out

    # ------------------------------------------------------------ 工作循环
    def _build(self, shape):
        for name, binding in (getattr(self._camera, "algorithms", None) or {}).items():
            if getattr(binding, "schedule", None) != "day":
                continue
            try:
                kwargs={}
                if self._offset_runtime is not None and name=='water_gap' and getattr(binding.spec,'backend','')=='yolo':
                    kwargs=dict(offset_module=self._offset_runtime.offset,wall_clock=lambda:self._clock.wall().timestamp())
                algo = registry.create(name, shape,
                                       _load_calibration(name, binding.spec), self.id,**kwargs)
                if kwargs:self._offset_runtime.attach('water_gap',algo)
                self._day_runners[name] = self._day_factory(
                    self._camera, binding.spec, algo)
            except Exception:
                logger.exception("[{}] 构建 DayRunner 失败: {}", self.id, name)
        self._built = True
        logger.info("[{}] 已加载白昼算法: {}", self.id, sorted(self._day_runners))

    def _process_shared(self,frame,wall,mono,frame_id,source_ts=None):
        if self._offset_runtime is None:return
        with self._processing_lock:
            if frame_id is None or frame_id==self._shared_last_frame:return
            self._shared_last_frame=frame_id
            state=self._scheduler.state()
            water=(getattr(self._camera,'algorithms',None) or {}).get('water_gap')
            continuous=water is not None and getattr(water.spec,'backend','')=='yolo'
            latest=self._offset_runtime.offset.latest
            lamp=(getattr(self._camera,'algorithms',None) or {}).get('night_lamp')
            cadence=lamp.spec.observation.cadence_s if lamp else 1800.
            precheck=(wall.timestamp()%cadence)>=max(0.,cadence-300.)
            needs_recovery=(latest is None or latest.status!='aligned' or self._offset_runtime.offset.pending is not None)
            needs_recovery=needs_recovery or bool(self._lamp_adapter and self._lamp_adapter.needs_offset_recheck)
            if continuous or state!=ScheduleState.NIGHT or precheck or needs_recovery:
                self._offset_runtime.process(frame,wall.timestamp(),mono,frame_id,
                    light_hint=('night' if state in (ScheduleState.NIGHT,ScheduleState.FROZEN)
                                else 'transition' if state!=ScheduleState.DAY else None))
            if self._lamp_adapter:
                try:
                    self._lamp_adapter.on_shared_frame(frame,wall,source_ts if source_ts is not None else mono,
                        frame_id,self._scheduler.is_active(self.id,'night_lamp'),
                        self._scheduler.state()==ScheduleState.DAY)
                except Exception as error:
                    self._lamp_adapter.record_error(error)
                    logger.exception('[{}] lamp processing failed; shared offset remains available',self.id)

    def _loop(self):
        t0, n0 = time.time(), 0
        last_decoded_id = object()
        while not self._stop.is_set():
            sample=getattr(self._reader,'sample',None)
            if callable(sample):
                frame,frame_id,source_ts=sample()
            else:
                frame=self._reader.read();frame_id=None;source_ts=None
            self._check_stream_health(self._clock.monotonic())
            now_wall = self._clock.wall()
            now_mono = self._clock.monotonic()
            _seq_fn = getattr(self._reader, "seq", None)
            if not callable(sample):frame_id = _seq_fn() if callable(_seq_fn) else None
            if frame is None:
                time.sleep(0.05)
                continue
            if frame_id is not None and frame_id == last_decoded_id:
                # Reader keeps the latest image. Reusing it must not rerun
                # water evidence or JPEG encoding while another camera works.
                # Stream-health checks above and the independent ticker remain active.
                self._stop.wait(0.01)
                continue
            last_decoded_id = frame_id
            if not self._built:
                self._build(frame.shape)
            self._latest_frame = frame
            try:
                self._process_shared(frame,now_wall,now_mono,frame_id,source_ts)
            except Exception:
                logger.exception('[{}] shared offset/lamp processing failed',self.id)
                if self._offset_runtime:
                    self._offset_runtime.offset.latest=None  # No stale alignment may justify absence.

            annots, debug, events = [], {}, []
            status = "OK"
            for name, runner in list(self._day_runners.items()):
                continuous_water = (name == 'water_gap' and
                    getattr(getattr(runner,'spec',None),'backend',None) == 'yolo')
                if not continuous_water and not self._scheduler.is_active(self.id, name):
                    continue
                try:
                    res = runner.on_frame(frame, now_mono)
                except Exception:
                    logger.exception("[{}] day step 失败: {}", self.id, name)
                    continue
                annots += res.annots
                debug.update(res.debug)
                events += res.events
                if any(e.kind == "alarm" for e in res.events):
                    self._day_alarm_pending[name] = True
                for rep in getattr(res, "reports", None) or ():
                    try:
                        self.publish(rep)
                    except Exception:
                        logger.exception("[{}] publish report 失败: {}",
                                         self.id, name)
                if isinstance(res.debug.get("frame_status"), str):
                    status = res.debug["frame_status"]
                if now_mono - self._diag_last.get(name, 0.0) >= DAY_DIAG_PERIOD_S:
                    self._log_day_diag(name, res)
                    self._diag_last[name] = now_mono

            mask = debug.get("roi_mask")
            vis = render.draw_annots(
                render.overlay(frame, mask) if mask is not None else frame, annots)
            vis = render.draw_status(vis, status)
            alarming = any(e.kind == "alarm" for e in events)
            states = "+".join(sorted({a.level for a in annots}))
            path = (self._evidence.maybe_save(self.id, vis, alarming, states, now_mono)
                    if self._evidence is not None else None)
            if path:
                for e in events:
                    if e.kind == "alarm" and e.evidence_path is None and not getattr(e,"image_jpeg",None):
                        e.evidence_path = path
            ok, buf = cv.imencode(".jpg", vis, [cv.IMWRITE_JPEG_QUALITY, self._quality])
            jpeg_bytes = buf.tobytes() if ok else None
            for e in events:
                if e.kind != "alarm":
                    continue
                try:
                    self._reporter.submit_event(e, image_jpeg=getattr(e,"image_jpeg",None) or jpeg_bytes)
                except Exception:
                    logger.exception("[{}] submit_event 失败", self.id)
            with self._lock:
                self._vis = buf.tobytes() if ok else None
                self._debug = debug
                self._status = status
                self._events.extend({"ts": e.ts, "algo": e.algo, "kind": e.kind,
                                     "payload": e.payload,
                                     "evidence_path": e.evidence_path} for e in events)
                self._n += 1
            if self._n - n0 >= LOG_EVERY:
                elapsed = max(time.time() - t0, 1e-6)
                logger.info("[{}] 处理 {} 帧, {:.1f} fps", self.id, self._n,
                            (self._n - n0) / elapsed)
                t0, n0 = time.time(), self._n


class Runtime:
    """唯一 Scheduler 持有者：poll -> 分发事件到 workers / Reporter。"""

    def __init__(self, cfg, clock: Clock | None = None, reporter=None,
                 evidence=None,
                 day_factory=None):
        from .performance import configure_native_threads
        configure_native_threads()
        self.cfg = cfg
        self.clock = clock or RealClock()
        self.scheduler = Scheduler(cfg, self.clock)
        self.reporter = reporter if reporter is not None else _default_reporter(cfg)
        quality = 80
        if isinstance(getattr(cfg, "server", None), dict):
            quality = int(cfg.server.get("jpeg_quality", 80))
        self.store = _default_store(cfg)
        self.sweeper = _default_sweeper(cfg)
        self._water_sweeper = _yolo_sweeper(cfg)
        self.workers = {}
        for cam in cfg.cameras or []:
            if not getattr(cam, "enabled", True):
                continue
            self.workers[cam.id] = CameraWorker(
                cam, self.scheduler, self.reporter, quality, evidence=evidence,
                clock=self.clock, day_factory=day_factory,store=self.store)
        protected=[]
        for worker in self.workers.values():
            if worker._offset_runtime:protected.append(worker._offset_runtime.offset.store.root)
            if worker._lamp_adapter:protected.append(worker._lamp_adapter.controller.archive.root)
        self.sweeper.protect(protected)
        self._water_sweeper.protect(protected)
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        for w in self.workers.values():
            w.start()
        self.sweeper.start()
        self._water_sweeper.start()
        self._thread = threading.Thread(target=self._pump, daemon=True,
                                        name="runtime-pump")
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self.sweeper.stop()
        self._water_sweeper.stop()
        for w in self.workers.values():
            try:
                w.flush_night()
            except Exception:
                logger.exception("night flush 失败")
        for w in self.workers.values():
            w.stop()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=PUMP_INTERVAL_S * 4)
        try:
            self.reporter.close()
        except Exception:
            logger.exception("reporter.close 失败")

    close = stop

    def _pump(self):
        while not self._stop.wait(PUMP_INTERVAL_S):
            try:
                events = self.scheduler.poll()
            except Exception:
                logger.exception("scheduler.poll 失败")
                continue
            for ev in events:
                try:
                    self.dispatch(ev)
                except Exception:
                    logger.exception("runtime 事件处理失败: {}", ev.type)

    def dispatch(self, ev: RuntimeEvent):
        if ev.type == RuntimeEventType.NIGHT_START:
            for w in self.workers.values():
                w.start_night()
        elif ev.type == RuntimeEventType.NIGHT_FREEZE:
            for w in self.workers.values():
                w.freeze_night()
        elif ev.type == RuntimeEventType.NIGHT_FINALIZE:
            for w in self.workers.values():
                w.finalize_night(ev.at)
        elif ev.type == RuntimeEventType.HOURLY_SNAPSHOT:
            w = self.workers.get(ev.camera)
            if w is None:
                return
            for rep in w.snapshot_reports(ev.at):
                w.publish(rep)
