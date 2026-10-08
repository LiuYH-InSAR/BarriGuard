"""Supplementary small-component candidates; original opened components stay intact."""
import cv2 as cv
import numpy as np
from . import evidence as ev
from .disturbance import detect
from .tools import nightly_map as nm

ENABLED = True
MIN_HITS = 20
MAX_AREA = 8
MAX_TRACKS = 64
MIN_EFFECTIVE_SECONDS = 20.0


def candidates(duty, on, excluded, opened):
    raw=((duty>=nm.CAND_DUTY_LO)&~excluded).astype(np.uint8)
    _,labels,stats,_=cv.connectedComponentsWithStats(raw,8)
    points=[]
    for label in range(1,len(stats)):
        if stats[label,cv.CC_STAT_AREA]>MAX_AREA:continue
        x,y,w,h=map(int,stats[label,:4])
        ys,xs=np.nonzero(labels[y:y+h,x:x+w]==label);ys+=y;xs+=x
        if np.any(opened[ys,xs]):continue
        eligible=on[ys,xs]>=MIN_HITS
        if not eligible.any():continue
        ys,xs=ys[eligible],xs[eligible];j=int(np.argmax(duty[ys,xs]))
        points.append((int(ys[j]),int(xs[j])))
    return sorted(points,key=lambda p:-float(duty[p]))


def analyse(gray,times,period,lag_lo,lag_hi,external=None):
    gray=np.asarray(gray,float);times=np.asarray(times,float)
    # Equal channels retain exactly the pixel Gray and provide no yellow relaxation.
    result=ev.analyse_region(np.repeat(gray[:,None],3,axis=1),period,lag_lo,lag_hi,times,external)
    mask,_,_=detect(gray,times)
    if external is not None:mask|=np.asarray(external,bool)
    dt=np.diff(times);valid=(~mask)&np.isfinite(gray)
    duration=float(dt[valid[:-1]&valid[1:]].sum()) if len(times)>1 else 0.
    result.update(effective_seconds=duration,min_effective_seconds=MIN_EFFECTIVE_SECONDS,
                  effective_duration_pass=duration>=MIN_EFFECTIVE_SECONDS,
                  evidence_signal='pixel_gray',yellow_auxiliary=False)
    if result['flashing'] and duration<MIN_EFFECTIVE_SECONDS:
        result.update(flashing=False,reason='insufficient_effective_duration')
    return result
