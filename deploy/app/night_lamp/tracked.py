"""Weight-independent lamp tracking and complete-pulse observation.

Detections are original-pixel xyxy boxes supplied by an injected detector.
Usability must be supplied per lamp: missing detection is NOT proof of occlusion.
Wall-clock timestamps define buckets; source timestamps define valid duration.
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Protocol
from uuid import uuid4
import cv2 as cv
import numpy as np
from camera_motion.geometry import warp_boxes, match_boxes
from .settings import LampParams


class Detector(Protocol):
    def detect(self, frame) -> list: ...


class Status(str, Enum):
    NORMAL = 'normal'
    BELOW = 'below_threshold'
    ZERO = 'no_flash'
    INSUFFICIENT = 'insufficient_observation'
    FAILED = 'not_flashing'
    UNKNOWN = 'unobservable'




@dataclass
class PulseCounter:
    valid_s: float = 0.
    count: int = 0
    previous_ts: float | None = None
    previous_valid: bool = False
    phase: str = 'unknown'
    rise: float | None = None
    last_flash: float | None = None

    def sample(self, ts, signal, valid, cfg):
        if not np.isfinite(ts):
            raise ValueError('Nonfinite source timestamp')
        if self.previous_ts is not None and ts <= self.previous_ts:
            return
        gap = None if self.previous_ts is None else ts-self.previous_ts
        connected = bool(valid and self.previous_valid and gap is not None
                         and gap <= cfg.max_sample_gap_s)
        if connected:
            self.valid_s += gap
        else:
            self.phase, self.rise = 'unknown', None
        self.previous_ts, self.previous_valid = ts, bool(valid)
        if not valid or not np.isfinite(signal):
            self.previous_valid = False
            return
        if signal <= cfg.off_delta:
            if self.phase == 'on' and self.rise is not None:
                if cfg.min_width_s <= ts-self.rise <= cfg.max_width_s:
                    self.count += 1
                    self.last_flash = ts
            self.phase, self.rise = 'off', None
        elif signal >= cfg.on_delta and self.phase == 'off':
            self.phase, self.rise = 'on', ts
        elif self.phase == 'on' and ts-self.rise > cfg.max_width_s:
            self.phase, self.rise = 'unknown', None


@dataclass
class Lamp:
    lamp_id: str
    box: list
    reference: str
    created_at: float
    evidence: str
    last_seen: float
    observable: bool = True
    last_flash: float | None = None
    history: list = field(default_factory=list)
    body_box: list | None = None
    aliases: list = field(default_factory=list)
    spatial: dict = field(default_factory=dict)
    presence: dict = field(default_factory=dict)


@dataclass
class Candidate:
    box: list
    created: float
    last_seen: float
    hits: int = 1
    pulses: PulseCounter = field(default_factory=PulseCounter)
    evidence_bucket: float | None = None


class LampMonitor:
    def __init__(self, camera_id, reference, params=None):
        self.camera_id, self.reference = str(camera_id), str(reference)
        self.params = params or LampParams()
        self.lamps = {}
        self.candidates = {}
        self.bucket = None
        self.observations = {}
        self.accepted = {}
        self.results = []
        self._last_frame = None
        self._last_source_ts = None
        self._applied = set()

    @staticmethod
    def signal(frame, box):
        h,w = frame.shape[:2]
        x0,y0,x1,y1 = map(int, box[:4])
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            return float('nan')
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        roi = gray[y0:y1,x0:x1]
        # Local background supports compensation for global exposure changes.
        margin = max(2, (x1-x0+y1-y0)//4)
        a,b,c,d = max(0,x0-margin), max(0,y0-margin), min(w,x1+margin), min(h,y1+margin)
        patch = gray[b:d,a:c]
        ring = np.ones(patch.shape,bool); ring[y0-b:y1-b,x0-a:x1-a] = False
        if not ring.any():
            return float('nan')
        return float(np.percentile(roi,90)-np.median(patch[ring]))

    def add_known(self, box, stamp, evidence='day_detection', lamp_id=None):
        box = list(map(float,box[:4]))
        if len(box)!=4 or not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:
            raise ValueError('Invalid lamp box')
        lamp_id = lamp_id or 'L'+uuid4().hex[:16]
        if lamp_id in self.lamps:
            raise ValueError('Duplicate persistent lamp ID')
        self.lamps[lamp_id] = Lamp(lamp_id,box,self.reference,stamp,evidence,stamp)
        return lamp_id

    def tick(self, wall_ts, active=True):
        if not active:
            if self.bucket is not None:
                self._finish(wall_ts)
            self.bucket = None
            return
        bucket = int(wall_ts//self.params.cadence_s)*self.params.cadence_s
        if self.bucket is not None and (bucket!=self.bucket or wall_ts>=self.bucket+self.params.deadline_s):
            self._finish(min(wall_ts,self.bucket+self.params.deadline_s))
            self.bucket = None
        if wall_ts < bucket+self.params.deadline_s and self.bucket is None:
            # Do not reopen a bucket already finalized at its deadline.
            if (not self.results or self.results[-1]['bucket'] != bucket
                or self.results[-1].get('reference',self.reference)!=self.reference):
                self.bucket = bucket
                self.observations = {key:PulseCounter() for key in self.lamps}
                self.accepted = {}

    def accept_normal(self,key,stamp):
        if self.bucket is None or key in self.accepted:return
        counter=self.observations.get(key)
        if counter is not None and counter.count>=self.params.normal_pulses:
            self.accepted[key]=dict(bucket=self.bucket,accepted_at=stamp,
                valid_s=counter.valid_s,pulses=counter.count,reference=self.lamps[key].reference)

    def on_frame(self, frame, wall_ts, source_ts, frame_id, boxes, *, usable,
                 is_night=True, discovery_usable=False, discovery_confirmed=True):
        self.tick(wall_ts,active=is_night)
        if frame_id is None or frame_id==self._last_frame:
            return
        if self._last_source_ts is not None and source_ts<=self._last_source_ts:
            return
        self._last_frame, self._last_source_ts = frame_id,source_ts
        boxes = [list(map(float,b[:4])) for b in boxes]
        if any(len(b)!=4 or not np.isfinite(b).all() or b[2]<=b[0] or b[3]<=b[1] for b in boxes):
            raise ValueError('Detector returned invalid original-pixel boxes')
        # Greedy detector duplicate suppression before one-to-one association.
        from camera_motion.geometry import iou
        unique=[]
        for box in boxes:
            if not any(iou(box,b)>.5 for b in unique): unique.append(box)
        boxes=unique
        keys=list(self.lamps)
        matches,_=match_boxes([self.lamps[k].box for k in keys],boxes,self.params)
        used=set(matches.values())
        for i,j in matches.items():
            lamp=self.lamps[keys[i]]
            lamp.last_seen=wall_ts
            # Keep the stable sampling region; do not chase every detector box.
        for key,lamp in self.lamps.items():
            if self.bucket is None: continue
            counter=self.observations.setdefault(key,PulseCounter())
            if key in self.accepted: continue
            ok=lamp.observable and bool(usable.get(key,False))
            counter.sample(source_ts,self.signal(frame,lamp.box) if ok else 0.,ok,self.params)
            if counter.last_flash == source_ts: lamp.last_flash=wall_ts
            if ok:self.accept_normal(key,wall_ts)
        if discovery_usable:
            self._discover(frame,wall_ts,source_ts,[b for j,b in enumerate(boxes) if j not in used],is_night,discovery_confirmed)

    def _discover(self,frame,wall_ts,source_ts,boxes,is_night,confirmed=True):
        for key in list(self.candidates):
            if wall_ts-self.candidates[key].last_seen>self.params.candidate_ttl_s:
                del self.candidates[key]
        keys=list(self.candidates)
        matches,_=match_boxes([self.candidates[k].box for k in keys],boxes,self.params)
        used=set(matches.values())
        for i,j in matches.items():
            c=self.candidates[keys[i]]; c.hits+=int(confirmed);c.last_seen=wall_ts
        for j,box in enumerate(boxes):
            if j not in used and confirmed:
                self.candidates[uuid4().hex]=Candidate(box,wall_ts,wall_ts)
        for key,c in list(self.candidates.items()):
            if is_night and c.evidence_bucket!=self.bucket:
                c.pulses=PulseCounter();c.evidence_bucket=self.bucket
            c.pulses.sample(source_ts,self.signal(frame,c.box),True,self.params)
            # Day observations admit a body; night candidates require independent pulses too.
            if c.hits>=self.params.discovery_hits and (not is_night or c.pulses.count>=self.params.normal_pulses):
                lamp_id=self.add_known(c.box,wall_ts,'night_flashes' if is_night else 'day_detection')
                if self.bucket is not None:
                    self.observations[lamp_id]=c.pulses if is_night and c.evidence_bucket==self.bucket else PulseCounter()
                    if is_night:self.accept_normal(lamp_id,wall_ts)
                del self.candidates[key]

    def _finish(self,stamp):
        rows=[]
        for key,lamp in self.lamps.items():
            obs=self.observations.get(key,PulseCounter())
            accepted=self.accepted.get(key)
            valid_s=accepted['valid_s'] if accepted else obs.valid_s
            pulses=accepted['pulses'] if accepted else obs.count
            status=(Status.NORMAL.value if pulses>=self.params.normal_pulses else
                    Status.INSUFFICIENT.value if valid_s<self.params.effective_s else
                    Status.BELOW.value if obs.count else Status.ZERO.value)
            row=dict(bucket=self.bucket,ended=stamp,valid_s=valid_s,pulses=pulses,status=status,
                     accepted_at=accepted['accepted_at'] if accepted else None,
                     reference=lamp.reference,lamp_id=key)
            lamp.history.append(row);lamp.history=lamp.history[-336:]
            rows.append(row)
        self.results.append(dict(bucket=self.bucket,ended=stamp,reference=self.reference,lamps=rows))
        self.results=self.results[-336:]

    def status(self,key,wall_ts,*,require_observable=True):
        lamp=self.lamps[key]
        if require_observable and not lamp.observable:return Status.UNKNOWN.value
        p=self.params
        end=int(wall_ts//p.cadence_s)*p.cadence_s
        expected=[end-p.cadence_s*i for i in range(1,p.failure_windows+1)]
        by_bucket={r['bucket']:r for r in lamp.history if r['ended']<=wall_ts}
        # A completed recovery in the current bucket supersedes the preceding
        # six failures even before the next half-hour boundary.
        current=self.observations.get(key) if self.bucket==end else None
        if current and (key in self.accepted or current.count>=p.normal_pulses):
            return Status.NORMAL.value
        if any(r['status']==Status.NORMAL.value and r['bucket']>=end for r in lamp.history if r['ended']<=wall_ts):
            return Status.NORMAL.value
        if all(b in by_bucket and by_bucket[b]['status']==Status.ZERO.value for b in expected):
            return Status.FAILED.value
        return lamp.history[-1]['status'] if lamp.history else Status.UNKNOWN.value

    def migrate(self,result, associations):
        token=(result.task_id,result.version)
        if token in self._applied:return False
        if result.camera_id!=self.camera_id or result.old_reference!=self.reference or result.status!='committed' or result.matrix is None:
            raise ValueError('Uncommitted or incompatible migration')
        if len({v for v in associations.values() if v is not None}) != sum(v is not None for v in associations.values()):
            raise ValueError('Candidate associations must be unique')
        # associations map persistent IDs to candidate keys; matched positions are independently observed.
        for key,candidate_id in associations.items():
            if key not in self.lamps or (candidate_id is not None and candidate_id not in self.candidates):
                raise ValueError('Unknown migration correspondence')
        if self.bucket is not None:
            self._finish(result.checked_at)
        predicted=warp_boxes([l.box for l in self.lamps.values()],np.asarray(result.matrix,float))
        h,w=result.shape
        for (key,lamp),box in zip(self.lamps.items(),predicted):
            candidate_id=associations.get(key)
            if candidate_id is not None:
                lamp.box=list(self.candidates[candidate_id].box)
                x0,y0,x1,y1=lamp.box
                lamp.observable=bool(0<=x0<x1<=w and 0<=y0<y1<=h)
            else:
                lamp.box=box
                # An unvalidated prediction must not become absence evidence.
                lamp.observable=False
            lamp.reference=result.new_reference
        self.reference=result.new_reference
        self._applied.add(token)
        self.bucket=None;self.observations={};self.accepted={};self.candidates={}
        return True

    def dump(self):
        return dict(schema=1,camera_id=self.camera_id,reference=self.reference,
            lamps={k:asdict(v) for k,v in self.lamps.items()},applied=[list(t) for t in self._applied],
            results=self.results)

    @classmethod
    def restore(cls,data,params=None):
        if data.get('schema')!=1:raise ValueError('Unsupported lamp state')
        monitor=cls(data['camera_id'],data['reference'],params)
        monitor.lamps={k:Lamp(**v) for k,v in data['lamps'].items()}
        monitor._applied={tuple(t) for t in data.get('applied',[])}
        monitor.results=data.get('results',[])
        # Partial windows intentionally do not resume as complete valid observations.
        return monitor
