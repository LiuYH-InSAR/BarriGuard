"""Production YOLO lamp detection and observation; local weights are required."""
from pathlib import Path
import os
import threading
import cv2 as cv
from server.algo import Report
from .runtime import AdaptiveLampController
from .report_render import render_lamp_report


class LocalLampDetector:
    def __init__(self,spec):
        path=Path(spec.weights)
        if not path.is_file():raise ValueError('Lamp detection requires existing local lamp weights')
        self.spec=spec
        cache=Path(__file__).resolve().parents[3]/'output/model_settings'
        cache.mkdir(parents=True,exist_ok=True)
        os.environ.setdefault('YOLO_CONFIG_DIR',str(cache))
        from ultralytics import YOLO
        self.model=YOLO(str(path),task='detect')  # Existing path only: no auto-download.
        from server.performance import configure_native_threads
        configure_native_threads()

    def detect(self,frame):
        results=self.model.predict(frame,device=self.spec.device,conf=self.spec.confidence,
                                   imgsz=self.spec.imgsz,verbose=False)
        return results[0].boxes.xyxy.cpu().numpy().tolist() if results and results[0].boxes is not None else []

    def predict(self,frame,regions,imgsz,confidence):
        """Bounded crop batches; return original-pixel xyxy + confidence."""
        boxes=[]
        for start in range(0,len(regions),16):
            batch=regions[start:start+16]
            crops=[frame[y0:y1,x0:x1] for x0,y0,x1,y1 in batch]
            results=self.model.predict(crops,device=self.spec.device,conf=confidence,
                                       imgsz=imgsz,iou=.5,verbose=False)
            for (x,y,_,_),result in zip(batch,results):
                if result.boxes is None:continue
                for b in result.boxes.data.cpu().numpy():
                    boxes.append([float(b[0]+x),float(b[1]+y),float(b[2]+x),float(b[3]+y),float(b[4])])
        return boxes


class LampAdapter:
    def __init__(self,owner,spec,clock,detector=None):
        self.owner,self.offset,self.spec,self.clock=owner,owner.offset,spec,clock
        root=Path(spec.detector.root) if spec.detector.root else owner.offset.store.root.parent.parent/'lamp_state'
        raw=detector if detector is not None else LocalLampDetector(spec.detector)
        if not callable(getattr(raw,'predict',None)):
            raise TypeError('Lamp backend must provide predict(frame, regions, imgsz, confidence)')
        self.detector=raw
        self.controller=AdaptiveLampController(self.offset,raw,root,
            params=spec.observation,detector_spec=spec.detector)
        self._lock=threading.RLock();self._frozen=False;self._last_image=None
        self._last_image_reference=None
        self._last_raw_frame=None
        self.error=None

    def on_shared_frame(self,frame,wall,source_ts,frame_id,is_night,is_day):
        with self._lock:
            self._last_raw_frame=frame
            if not self._frozen or is_day:
                self.controller.on_frame(frame,wall.timestamp(),source_ts,frame_id,
                    is_night and not self._frozen,discovery_allowed=is_day)
                if self.needs_offset_recheck:self.owner.request_check(self.clock.monotonic())
            if is_night:
                self._last_image=frame.copy()
                self._last_image_reference=(self.controller.monitor.reference
                    if self.controller._position_ok(wall.timestamp()) else None)

    def record_error(self,error):
        with self._lock:
            self.error=type(error).__name__+': '+str(error)
            if self.controller._source_ts is not None:self.controller.pause(self.controller._source_ts)

    def tick(self,wall):
        with self._lock:self.controller.tick(wall.timestamp(),not self._frozen)
    def start_night(self):
        with self._lock:self._frozen=False
    def freeze(self):
        with self._lock:
            self._frozen=True;self.controller.tick(self.clock.wall().timestamp(),False)
    def save_state(self):
        with self._lock:self.controller.checkpoint(self.clock.wall().timestamp())
    def release(self):pass  # Persistent IDs survive a nightly report.

    def finalize(self,base,wall):
        with self._lock:
            data,path=self.controller.finalize(wall.timestamp())
            image=self._last_image if self._last_image is not None else base
            image_reference=(self._last_image_reference if self._last_image is not None
                else self.controller.monitor.reference)
            for lamp in data['lamps']:
                lamp['position_current']=bool(lamp['position_current'] and image_reference==self.controller.monitor.reference)
            jpeg=None;rendering=None
            if image is not None:
                vis,rendering=render_lamp_report(image,data['lamps'],self.offset.camera_id,wall.isoformat())
                ok,encoded=cv.imencode('.jpg',vis)
                if ok:jpeg=encoded.tobytes()
            return Report(self.offset.camera_id,'night_lamp','lamp_status',wall.isoformat(),jpeg,
                dict(detector='yolo_v9c',lamps=data['lamps'],archive=str(path),reference=self.offset.references.view,
                     quality_policy=self.spec.detector.visibility,processing_error=self.error,
                     image_rendering=rendering,image_reference=image_reference,
                     observation_policy=dict(normal_valid_cycles=self.spec.observation.normal_pulses,
                         normal_min_effective_s=0,negative_min_effective_s=self.spec.observation.effective_s,
                         deadline_s=self.spec.observation.deadline_s,cadence_s=self.spec.observation.cadence_s),
                     timing_source='decode_receive_monotonic'),
                status='degraded' if self.error or any(l['status'] in ('unobservable','insufficient_observation') for l in data['lamps']) or not data['lamps'] else 'ok')

    @property
    def needs_offset_recheck(self):
        return self.controller._geometry_blocked_at is not None

    @property
    def offset_reference(self):
        return self.controller.monitor.reference

    def offset_feedback(self,result,frame):
        with self._lock:return self.controller.spatial.feedback(result,self.clock.wall().timestamp())

    def observe_offset_candidate(self,result,reference_frame,frame,stamp,source_ts,frame_id,night=False):
        with self._lock:
            self.controller.spatial.observe_candidate(result,reference_frame,frame,stamp,source_ts,frame_id,night)

    def offset_feedback_ready(self,result,stamp):
        with self._lock:return self.controller.spatial.feedback_ready(result,stamp)

    def apply_shared_offset(self,result):
        with self._lock:
            self.controller.reset_reference(result)
