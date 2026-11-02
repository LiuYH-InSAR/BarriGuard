"""Resolve a non-lamp group box only after proving separate temporal sources."""
from collections import deque
import cv2 as cv
import numpy as np
from .adaptive_prototype import center,iou
from .optical_fixture_v9 import potential_fixture
from .adaptive_flash_signal import signal_plane


def temporal_core(samples,box):
    if len(samples)<16 or samples[-1][0]-samples[0][0]<3:return None
    stack=np.stack([s[1] for s in samples]);lo,hi=np.percentile(stack,[20,90],axis=0);spread=hi-lo
    mask=np.uint8(spread>=max(25.,float(spread.max())*.5))
    count,labels,stats,_=cv.connectedComponentsWithStats(mask)
    components=[j for j in range(1,count) if stats[j,cv.CC_STAT_AREA]>=3]
    if len(components)!=1:return None
    weights=spread*(labels==components[0]);total=float(weights.sum())
    if total<=0:return None
    yy,xx=np.indices(spread.shape);point=np.array([box[0]+(float((weights*xx).sum()/total)+.5)*(box[2]-box[0])/24,
        box[1]+(float((weights*yy).sum()/total)+.5)*(box[3]-box[1])/24])
    residual=(stack-lo)[:,labels==components[0]];series=residual.mean(1)
    low,high=np.percentile(series,[10,90]);amplitude=float(high-low)
    if amplitude<18:return None
    phase='unknown';cycles=0
    for value in series:
        if value<=low+.3*amplitude:cycles+=phase=='on';phase='off'
        elif value>=low+.65*amplitude and phase=='off':phase='on'
    if cycles<2:return None
    return dict(samples=len(samples),span_s=samples[-1][0]-samples[0][0],cycles=int(cycles),amplitude=amplitude,
        unique_core=True,point=point.tolist())


def separation_verified(proof,records):
    children=proof.get('children',[])
    if len(children)<2 or len({c['id'] for c in children})!=len(children):return False
    if proof.get('group_core_count')!=len(children) or len({c.get('component') for c in children})!=len(children):return False
    points=[]
    for c in children:
        t=records.get(c['id']);p=c.get('optical',{})
        if t is None or t.get('observation_epoch')!=c.get('epoch'):return False
        if not(c.get('prior_pulses',0)>=15 and c.get('hits',0)>=5 and p.get('samples',0)>=16
            and p.get('span_s',0)>=3 and p.get('cycles',0)>=2 and p.get('amplitude',0)>=18
            and p.get('unique_core') and c.get('core_in_support')):return False
        b=t.get('body_box',t['box']);point=np.asarray(p.get('point',[]))
        if point.shape!=(2,) or not np.isfinite(point).all():return False
        if not(b[0]<=point[0]<=b[2] and b[1]<=point[1]<=b[3]):return False
        for other in children:
            if other['id']==c['id']:continue
            peer=records.get(other['id'])
            if peer is None:return False
            bb=peer.get('body_box',peer['box'])
            if iou(b,bb)>.25 or (bb[0]<=point[0]<=bb[2] and bb[1]<=point[1]<=bb[3]):return False
        if any(np.linalg.norm(point-q)<5 for q in points):return False
        points.append(point)
    return True


class SourceSeparation:
    def __init__(self):self.samples={};self.parent_samples={};self.groups=[]
    def suppresses(self,box,tracks,epochs):
        self.groups=[g for g in self.groups if all(c['id'] in tracks and epochs.get(c['id'],0)==c['epoch'] for c in g)]
        for group in self.groups:
            bodies=[tracks[c['id']].body_box for c in group]
            if all(potential_fixture(box,b) for b in bodies):return True
        return False
    def observe(self,frame,tracks,epochs,ts,valid):
        self.parent_samples={k:v for k,v in self.parent_samples.items() if valid and k[0] in tracks and epochs.get(k[0],0)==k[1]}
        eligible={t.id:t for t in tracks.values() if valid and t.confidence>=.65 and t.hits>=5 and t.confirmed
            and t.source=='night_flashes' and t.pulses.count>=15 and t.position_observable and t.locator.locked}
        self.samples={k:v for k,v in self.samples.items() if k[0] in eligible and epochs.get(k[0],0)==k[1]}
        proofs={}
        for key,t in eligible.items():
            b=t.body_box;x0,y0,x1,y1=np.rint(b).astype(int)
            if min(x0,y0)<0 or x1>frame.shape[1] or y1>frame.shape[0] or min(x1-x0,y1-y0)<4:continue
            token=(key,epochs.get(key,0));samples=self.samples.setdefault(token,deque(maxlen=32))
            if samples and ts-samples[-1][0]>.5:samples.clear()
            crop=cv.resize(signal_plane(frame[y0:y1,x0:x1]),(24,24)).astype(float)
            samples.append((ts,crop))
            while samples and ts-samples[0][0]>5:samples.popleft()
            proof=temporal_core(samples,b)
            if proof is None:continue
            mask=t.flash.support_mask
            if mask is None:continue
            r=(mask.shape[0]-1)//2;px,py=np.rint(np.asarray(proof['point'])-center(t.box)+r).astype(int)
            if not(0<=py<mask.shape[0] and 0<=px<mask.shape[1] and mask[py,px]):continue
            proofs[key]=dict(id=key,epoch=token[1],prior_pulses=t.pulses.count,hits=t.hits,optical=proof,core_in_support=True)
        offers=[]
        for parent in tracks.values():
            if parent.confirmed or parent.pulses.count>=15:continue
            children=[t for key,t in eligible.items() if potential_fixture(parent.body_box,t.body_box)]
            if len(children)<2:continue
            if any(iou(a.body_box,b.body_box)>.25 or np.linalg.norm(center(a.body_box)-center(b.body_box))<5
                   for i,a in enumerate(children) for b in children[i+1:]):continue
            # Cross-check in one common image region. Separate cropped views
            # of one halo are not sufficient evidence of independent emitters.
            x0,y0,x1,y1=np.rint(parent.body_box).astype(int)
            if min(x0,y0)<0 or x1>frame.shape[1] or y1>frame.shape[0] or min(x1-x0,y1-y0)<4:continue
            token=(parent.id,epochs.get(parent.id,0));samples=self.parent_samples.setdefault(token,deque(maxlen=32))
            if samples and ts-samples[-1][0]>.5:samples.clear()
            samples.append((ts,cv.resize(signal_plane(frame[y0:y1,x0:x1]),(24,24)).astype(float)))
            while samples and ts-samples[0][0]>5:samples.popleft()
            if len(samples)<16 or samples[-1][0]-samples[0][0]<3 or any(t.id not in proofs for t in children):continue
            stack=np.stack([s[1] for s in samples]);lo,hi=np.percentile(stack,[20,90],axis=0);spread=hi-lo
            count,labels,stats,_=cv.connectedComponentsWithStats(np.uint8(spread>=max(25.,float(spread.max())*.5)))
            components=[j for j in range(1,count) if stats[j,cv.CC_STAT_AREA]>=3]
            if len(components)!=len(children):continue
            resolved=[]
            for component in components:
                joint=temporal_core([(stamp,np.where(labels==component,crop,0)) for stamp,crop in samples],parent.body_box)
                if joint is None:break
                point=np.asarray(joint['point'])
                owners=[t for t in children if t.body_box[0]<=point[0]<=t.body_box[2] and t.body_box[1]<=point[1]<=t.body_box[3]]
                if len(owners)!=1:break
                t=owners[0];mask=t.flash.support_mask;r=(mask.shape[0]-1)//2
                px,py=np.rint(point-center(t.box)+r).astype(int)
                if not(0<=py<mask.shape[0] and 0<=px<mask.shape[1] and mask[py,px]):break
                resolved.append(dict(proofs[t.id],optical=joint,component=int(component)))
            if len(resolved)!=len(children):continue
            resolved.sort(key=lambda c:c['id']);children=sorted(children,key=lambda t:t.id)
            proof=dict(children=resolved,group_core_count=len(components))
            records={t.id:dict(box=t.box,body_box=t.body_box,observation_epoch=epochs.get(t.id,0)) for t in children}
            if separation_verified(proof,records):offers.append((parent,children,proof))
        return offers
