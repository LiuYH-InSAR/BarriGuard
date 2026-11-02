"""Evidence for one lamp body with one nested head; no annotation coordinates."""
from collections import deque
from dataclasses import dataclass,field
import cv2 as cv
import numpy as np
from .adaptive_prototype import center,iou
from .adaptive_tracking_v5 import BackgroundAnchor
from .adaptive_flash_signal import signal_plane


def area(box):return max(0,box[2]-box[0])*max(0,box[3]-box[1])
def containment(a,b):
    overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return overlap/max(1e-9,min(area(a),area(b)))

def nested(outer,inner):
    ratio=area(outer)/max(1e-9,area(inner))
    return (1.4<=ratio<=6 and containment(outer,inner)>=.9 and iou(outer,inner)<.5
            and min(inner[2]-inner[0],inner[3]-inner[1])>=12
            and np.linalg.norm(center(outer)-center(inner))<=.4*max(outer[2]-outer[0],outer[3]-outer[1]))

def children_for(parent,tracks):
    outer=parent.body_box;found=[t for t in tracks if t.id!=parent.id and nested(outer,t.body_box)]
    # Several overlapping head hypotheses at the same location are one cluster;
    # independently positioned children remain separate and block reconciliation.
    groups=[]
    for child in sorted(found,key=lambda t:(-t.hits,-t.confidence)):
        if not any(np.linalg.norm(center(child.body_box)-center(g.body_box))<=2.5 and iou(child.body_box,g.body_box)>=.5 for g in groups):groups.append(child)
    return groups


@dataclass
class PairEvidence:
    anchor:object
    samples:object=field(default_factory=lambda:deque(maxlen=32))
    last_probe:float=-1e9
    proof:dict=field(default_factory=dict)


class NestedResolver:
    def __init__(self):self.pairs={}

    def observe(self,frame,tracks,ts,frame_valid):
        gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY);offers=[];active=set()
        for parent in tracks:
            children=children_for(parent,tracks)
            if len(children)!=1:continue
            child=children[0];key=(parent.id,child.id);active.add(key)
            outer=parent.body_box;inner=child.body_box
            if parent.confidence<.65 or child.confidence<.65:continue
            state=self.pairs.setdefault(key,PairEvidence(BackgroundAnchor(gray,outer)))
            neighbors=[t.body_box for t in tracks if t.id not in key and containment(outer,t.body_box)<.9]
            stable=frame_valid and state.anchor.evaluate(gray,outer,neighbors)['ok']
            if not stable:state.samples.clear();continue
            if state.samples and ts-state.samples[-1][0]>.5:state.samples.clear()
            x0,y0,x1,y1=np.rint(outer).astype(int)
            if x0<0 or y0<0 or x1>frame.shape[1] or y1>frame.shape[0]:continue
            crop=frame[y0:y1,x0:x1]
            if not crop.size:continue
            hsv=cv.cvtColor(crop,cv.COLOR_BGR2HSV)
            mask=cv.inRange(hsv,np.array([5,70,60]),np.array([43,255,255]))
            count,labels,stats,_=cv.connectedComponentsWithStats(mask)
            bodies=[j for j in range(1,count) if stats[j,cv.CC_STAT_AREA]>=max(6,.06*mask.size)]
            body_ok=len(bodies)==1
            if body_ok:
                bx0,by0,bx1,by1=np.rint(np.asarray(inner)-[x0,y0,x0,y0]).astype(int)
                part=labels[max(0,by0):min(labels.shape[0],by1),max(0,bx0):min(labels.shape[1],bx1)]
                body_ok=bool(part.size and np.mean(part==bodies[0])>=.1)
            plane=cv.resize(signal_plane(crop),(32,32)).astype(float)
            state.samples.append((ts,plane,body_ok))
            while state.samples and ts-state.samples[0][0]>5.:state.samples.popleft()
            if (len(state.samples)<16 or ts-state.samples[0][0]<3 or ts-state.last_probe<.5
                    or parent.hits<3 or child.hits<2 or child.last_seen-child.created<1.):continue
            state.last_probe=ts
            fraction=float(np.mean([p[2] for p in state.samples]))
            if fraction<.6:continue
            stack=np.stack([p[1] for p in state.samples]);lo,hi=np.percentile(stack,[20,90],axis=0);variation=hi-lo
            py,px=np.unravel_index(np.argmax(variation),variation.shape)
            peak=np.array([x0+(px+.5)*(x1-x0)/32,y0+(py+.5)*(y1-y0)/32])
            if not(inner[0]-1<=peak[0]<=inner[2]+1 and inner[1]-1<=peak[1]<=inner[3]+1):continue
            ax=max(0,int((inner[0]-x0)*32/(x1-x0)));bx=min(32,int(np.ceil((inner[2]-x0)*32/(x1-x0))))
            ay=max(0,int((inner[1]-y0)*32/(y1-y0)));by=min(32,int(np.ceil((inner[3]-y0)*32/(y1-y0))))
            residual=stack[:,ay:by,ax:bx]-lo[ay:by,ax:bx]
            flat=residual.reshape(len(stack),-1);k=max(1,flat.shape[1]//4)
            series=np.partition(flat,-k,axis=1)[:,-k:].mean(1);low,high=np.percentile(series,[10,90]);amplitude=float(high-low)
            if amplitude<18:continue
            phase='unknown';cycles=0
            for v in series:
                if v<=low+.3*amplitude:
                    cycles+=phase=='on';phase='off'
                elif v>=low+.65*amplitude and phase=='off':phase='on'
            if cycles<2:continue
            proof=dict(samples=len(state.samples),span_s=ts-state.samples[0][0],single_body_fraction=fraction,
                       cycles=int(cycles),amplitude=amplitude,unique_child=True,peak_in_child=True,
                       parent_hits=parent.hits,child_hits=child.hits,containment=containment(outer,inner))
            state.proof=proof;offers.append((parent,child,proof))
        self.pairs={k:v for k,v in self.pairs.items() if k in active}
        return offers
