"""Offline, bounded-cost multiscale discovery. No health decisions or production hooks.

No labels, historical annotation coordinates, or camera-specific lamp masks are read.
Registration/quality/pulse rules reuse the production modules without modifying them.
"""
from dataclasses import dataclass,field,replace
import time
import cv2 as cv
import numpy as np
from camera_motion.vision import Vision
from camera_motion.params import MotionParams
from camera_motion.geometry import warp_boxes
from night_lamp.quality import StreamQuality
from night_lamp.tracked import PulseCounter
from night_lamp.settings import LampParams
from .adaptive_flash_signal import AdaptiveFlashSignal,SignalParams,signal_plane

def iou(a,b):
    n=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));return n/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-n)
def center(b):return (np.asarray(b[:2])+np.asarray(b[2:4]))/2
def nms(boxes,threshold=.5):
    keep=[]
    for b in sorted(boxes,key=lambda b:b[4],reverse=True):
        if not any(iou(b,a)>threshold for a in keep):keep.append(b)
    return keep
def grid(shape,size,stride):
    h,w=shape[:2];size=min(size,h,w);xs=list(range(0,w-size+1,stride));ys=list(range(0,h-size+1,stride))
    if xs[-1]!=w-size:xs.append(w-size)
    if ys[-1]!=h-size:ys.append(h-size)
    return [[x,y,x+size,y+size] for y in ys for x in xs]
@dataclass(frozen=True)
class Params:
    coarse_period_s:float=5.
    fine_period_s:float=.4
    fine_budget:int=12
    fine_size:int=160
    survey_budget:int=2
    promotion_hits:int=3
    promotion_span_s:float=1.
    day_confidence:float=.65
    fine_confidence:float=.65
    proposal_hits:int=2
    ttl_s:float=30.
    max_gap_s:float=.5
    night_pulses:int=15
    move_fraction:float=.004
    # OSD masks differ from the larger masks used only for background registration.
    overlay_exclusions:tuple=((.68,0.,1.,.095),(.86,.095,1.,.19),(0.,.88,.13,1.))
    def __post_init__(self):
        if self.fine_budget<=0 or self.survey_budget<0 or self.survey_budget>self.fine_budget:raise ValueError('Invalid ROI budget')

@dataclass
class Track:
    id:int
    box:list
    created:float
    last_seen:float
    hits:int=1
    confidence:float=0.
    fine_hits:int=0
    confirmed:bool=False
    pulses:PulseCounter=field(default_factory=PulseCounter)
    flash:AdaptiveFlashSignal=field(default_factory=AdaptiveFlashSignal)
    ring:object=None
    source:str='yolo'
    position_observable:bool=False

class AdaptiveDetector:
    def __init__(self,backend,params=None,vision=None):
        self.backend=backend;self.p=params or Params();self.vision=vision or Vision(MotionParams())
        self.quality=StreamQuality(max_gap_s=self.p.max_gap_s)
        self.cfg=LampParams();self.tracks={};self.next_id=1;self.previous=None;self.previous_ts=None;self.previous_id=None
        signal_params=SignalParams(max_gap_s=self.p.max_gap_s)
        self.flash_cfg=replace(self.cfg,on_delta=signal_params.on_fraction,off_delta=signal_params.off_fraction,max_sample_gap_s=self.p.max_gap_s)
        self.last_coarse=-1e9;self.last_fine=-1e9;self.last_output=None;self.survey_cursor=0;self.proposals={};self.roi_tested={}
    def masked(self,c,shape):
        x,y=c;h,w=shape[:2]
        return any(a*w<=x<=d*w and b*h<=y<=e*h for a,b,d,e in self.p.overlay_exclusions)
    def roi(self,c,shape):
        h,w=shape[:2];s=min(self.p.fine_size,h,w);x=max(0,min(w-s,int(c[0])-s//2));y=max(0,min(h-s,int(c[1])-s//2));return [x,y,x+s,y+s]
    def proposals_from_image(self,frame,previous,night,ts):
        hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV);v=hsv[:,:,2]
        if night:
            if previous is None:return []
            old=cv.cvtColor(previous,cv.COLOR_BGR2HSV)[:,:,2]
            delta=cv.absdiff(v,old);mask=np.uint8((delta>28)&(np.maximum(v,old)>155))*255
        else:mask=cv.inRange(hsv,np.array([5,95,80]),np.array([43,255,255]))
        n,lab,stats,centers=cv.connectedComponentsWithStats(mask);found=[]
        for st,c in zip(stats[1:],centers[1:]):
            x,y,w,h,area=st
            if not 2<=area<=450 or max(w,h)>40 or min(w,h)==0 or max(w,h)/min(w,h)>4 or self.masked(c,frame.shape):continue
            key=(round(float(c[0])/8),round(float(c[1])/8));old=self.proposals.get(key)
            hits=(old['hits']+1 if old and ts-old['last']<1.1 else 1)
            self.proposals[key]=dict(center=c.tolist(),hits=hits,last=ts)
            if hits>=self.p.proposal_hits:found.append((1.+min(area,50)/50,c.tolist(),'night_change' if night else 'day_color'))
        self.proposals={k:v for k,v in self.proposals.items() if ts-v['last']<2.}
        return found
    def ring_sample(self,frame,b):
        c=center(b);s=int(max(16,min(48,max(b[2]-b[0],b[3]-b[1])*1.5)));x,y=np.rint(c).astype(int)
        if x-s<0 or y-s<0 or x+s>=frame.shape[1] or y+s>=frame.shape[0]:return None
        patch=cv.cvtColor(frame[y-s:y+s+1,x-s:x+s+1],cv.COLOR_BGR2GRAY);patch=cv.resize(patch,(32,32)).astype(float)
        mask=np.ones((32,32),bool);mask[10:22,10:22]=False
        return patch[mask]
    def associate(self,frame,detections,ts,night):
        pairs=[]
        for k,t in self.tracks.items():
            for j,b in enumerate(detections):
                distance=np.linalg.norm(center(t.box)-center(b));gate=max(4,min(14,.5*min(t.box[2]-t.box[0],t.box[3]-t.box[1])))
                if distance<=gate:pairs.append((distance,k,j))
        used=set();ids=set()
        for _,key,j in sorted(pairs):
            if key in ids or j in used:continue
            ids.add(key);used.add(j);t=self.tracks[key];b=detections[j];t.hits+=1;t.last_seen=ts;t.confidence=max(t.confidence,b[4]);t.fine_hits+=int(b[5]=='fine')
            # Fixed position by default; bound small, independently detected center corrections.
            delta=center(b)-center(t.box)
            if np.linalg.norm(delta)<=3:t.box=(np.array(t.box)+np.tile(delta*.15,2)).tolist()
        for j,b in enumerate(detections):
            if j in used:continue
            t=Track(self.next_id,b[:4],ts,ts,confidence=b[4],fine_hits=int(b[5]=='fine'));t.flash=AdaptiveFlashSignal(SignalParams(max_gap_s=self.p.max_gap_s));t.ring=self.ring_sample(frame,t.box);self.tracks[self.next_id]=t;self.next_id+=1
