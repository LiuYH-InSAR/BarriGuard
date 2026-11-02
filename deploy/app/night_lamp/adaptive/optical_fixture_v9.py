"""Causal optical proof for a small head within one detector body hypothesis."""
from collections import deque
from dataclasses import dataclass,field
from functools import lru_cache
import cv2 as cv
import numpy as np
from .adaptive_prototype import center,iou
from .nested_fixture_v6 import area,containment
from .adaptive_tracking_v5 import BackgroundAnchor
from .adaptive_flash_signal import signal_plane

def potential_fixture(outer,inner):
    # The same immutable geometry is queried by masking, optical verification
    # and the evidence audit. Bound the pure-value cache across camera threads.
    return _potential_fixture(tuple(outer[:4]),tuple(inner[:4]))

@lru_cache(maxsize=16384)
def _potential_fixture(outer,inner):
    ratio=area(outer)/max(1e-9,area(inner))
    return (1.4<=ratio<=6 and containment(outer,inner)>=.8 and iou(outer,inner)<.5
            and min(inner[2]-inner[0],inner[3]-inner[1])>=4
            and np.linalg.norm(center(outer)-center(inner))<=.4*max(outer[2]-outer[0],outer[3]-outer[1]))
def same_fixture_hypothesis(a,b):return potential_fixture(a,b) or potential_fixture(b,a)


def compact_core_ownership(stack,spread,labels,component,bounds,series):
    """Explain diffuse halo variation without assigning another source to a body."""
    ax,ay,bx,by=bounds;inside=np.zeros(spread.shape,bool);inside[ay:by,ax:bx]=True
    weights=spread*(labels==component);total=float(weights.sum())
    fraction=float(weights[inside].sum()/total) if total else 0.
    outside=(labels>0)&~inside;unrelated=int(np.count_nonzero(outside&(labels!=component)))
    if fraction<.9 or unrelated:return None
    count=int(outside.sum());correlation=None
    if count:
        halo=stack[:,outside].mean(axis=1)
        if np.std(series)<=0 or np.std(halo)<=0:return None
        correlation=float(np.corrcoef(series,halo)[0,1])
        if not np.isfinite(correlation) or correlation<.9:return None
        correlation=float(np.clip(correlation,-1.,1.))
    return dict(ownership_basis='verified_compact_core',ownership_component_fraction=fraction,
        ownership_outside_strong_pixels=count,ownership_unrelated_strong_pixels=unrelated,
        ownership_halo_correlation=correlation)


def compact_ownership_verified(proof,proven_fixtures,emitter_epochs,family):
    """Consumer checks metrics and its own prior provenance, not a detector flag."""
    prior=proof.get('ownership_prior',{});key=prior.get('id');kind=prior.get('kind')
    known=(kind=='optical_fixture' and key in proven_fixtures) or (
        kind=='admitted_emitter' and key in emitter_epochs and emitter_epochs[key]==prior.get('observation_epoch'))
    count=proof.get('ownership_outside_strong_pixels',-1);correlation=proof.get('ownership_halo_correlation')
    halo_ok=(count==0 and correlation is None) or (count>0 and isinstance(correlation,(int,float)) and .9<=correlation<=1.)
    return bool(proof.get('ownership_basis')=='verified_compact_core' and known and key in family
        and proof.get('child_hits',0)>=5 and proof.get('ownership_peak_inside')
        and proof.get('ownership_component_fraction',0)>=.9
        and proof.get('ownership_unrelated_strong_pixels')==0 and count>=0 and halo_ok)

@dataclass
class Evidence:
    anchor:object
    initial_detector_hits:int=0
    samples:object=field(default_factory=lambda:deque(maxlen=32))
    last_probe:float=-1e9
    proof:dict=field(default_factory=dict)

class OpticalResolver:
    def __init__(self):
        self.pairs={}
        # Supplied by the tracker only for fixtures with an earlier successful
        # optical merge in this view. This is not a geometry-only whitelist.
        self.proven_fixture_ids=frozenset()
        self.established_emitter_epochs={}
    def observe(self,frame,tracks,ts,frame_valid):
        gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY);offers=[];active=set()
        for parent in tracks:
            children=[t for t in tracks if t.id!=parent.id and potential_fixture(parent.body_box,t.body_box) and t.confidence>=.65]
            groups=[]
            for t in sorted(children,key=lambda t:(-t.hits,-t.confidence)):
                if not any(np.linalg.norm(center(t.body_box)-center(o.body_box))<=2.5 and iou(t.body_box,o.body_box)>=.5 for o in groups):groups.append(t)
            if len(groups)!=1:continue
            child=groups[0];key=(parent.id,child.id);active.add(key)
            if parent.confidence<.65:continue
            state=self.pairs.setdefault(key,Evidence(BackgroundAnchor(gray,parent.body_box),parent.hits+child.hits))
            outer=parent.body_box;inner=child.body_box
            neighbors=[t.body_box for t in tracks if t.id not in key and not same_fixture_hypothesis(outer,t.body_box)]
            stable=frame_valid and state.anchor.evaluate(gray,outer,neighbors)['ok']
            if not stable:
                state.samples.clear();state.initial_detector_hits=parent.hits+child.hits;continue
            if state.samples and ts-state.samples[-1][0]>.5:
                state.samples.clear();state.initial_detector_hits=parent.hits+child.hits
            a,b,c,d=np.rint(outer).astype(int)
            if a<0 or b<0 or c>frame.shape[1] or d>frame.shape[0] or c<=a or d<=b:continue
            crop=signal_plane(frame[b:d,a:c]).astype(float)
            crop=cv.resize(crop,(32,32))
            state.samples.append((ts,crop,parent.hits+child.hits))
            while state.samples and ts-state.samples[0][0]>5.:state.samples.popleft()
            family_updates=parent.hits+child.hits-state.initial_detector_hits
            prior_fixture=(child.id in self.proven_fixture_ids and child.hits>=5 and parent.hits>=1)
            established=(child.id in self.established_emitter_epochs and child.hits>=5 and parent.hits>=1)
            recent_updates=parent.hits+child.hits-state.samples[0][2]
            # The detector may alternate a body mode and a head mode. A single
            # high-confidence head box can be supported by fresh family hits
            # and repeated pixel evidence; requiring that exact box mode twice
            # stranded otherwise well-observed families indefinitely.
            if len(state.samples)<16 or ts-state.samples[0][0]<3 or ts-state.last_probe<.5 or child.hits<1 or family_updates<3:continue
            if parent.hits<5 and not prior_fixture and not established:continue
            if parent.hits<5 and established and not prior_fixture and recent_updates<3:continue
            state.last_probe=ts;stack=np.stack([sample[1] for sample in state.samples]);low,high=np.percentile(stack,[20,90],axis=0);spread=high-low
            maximum=float(spread.max())
            if maximum<25:continue
            mask=(spread>=max(25,maximum*.5)).astype(np.uint8)
            count,labels,stats,centroids=cv.connectedComponentsWithStats(mask)
            components=[j for j in range(1,count) if stats[j,cv.CC_STAT_AREA]>=3]
            if len(components)!=1:continue
            component=components[0];weights=spread*(labels==component);yy,xx=np.indices(spread.shape)
            px=float((weights*xx).sum()/weights.sum());py=float((weights*yy).sum()/weights.sum())
            peak=np.array([a+(px+.5)*(c-a)/32,b+(py+.5)*(d-b)/32])
            if not(inner[0]-1<=peak[0]<=inner[2]+1 and inner[1]-1<=peak[1]<=inner[3]+1):continue
            # A separately observed neighbor forbids assigning its source to this family.
            competing=[t for t in tracks if t.id not in key and t.confidence>=.65 and t.hits>=2
                       and iou(t.body_box,inner)<.5 and np.linalg.norm(center(t.body_box)-center(inner))>3
                       and containment(outer,t.body_box)>=.8]
            if competing:continue
            ax=max(0,int((inner[0]-a)*32/(c-a)));bx=min(32,int(np.ceil((inner[2]-a)*32/(c-a))))
            ay=max(0,int((inner[1]-b)*32/(d-b)));by=min(32,int(np.ceil((inner[3]-b)*32/(d-b))))
            if bx<=ax or by<=ay:continue
            total=float(spread.sum());inside=float(spread[ay:by,ax:bx].sum()/max(total,1e-6))
            compact=inside<.7
            if compact and not(prior_fixture or established):continue
            residual=stack[:,ay:by,ax:bx]-low[ay:by,ax:bx];flat=residual.reshape(len(stack),-1);k=max(1,flat.shape[1]//4)
            series=np.partition(flat,-k,axis=1)[:,-k:].mean(1);lo,hi=np.percentile(series,[10,90]);amplitude=float(hi-lo)
            if amplitude<18:continue
            phase='unknown';cycles=0
            for v in series:
                if v<=lo+.3*amplitude:cycles+=phase=='on';phase='off'
                elif v>=lo+.65*amplitude and phase=='off':phase='on'
            if cycles<2 or sum(series>=lo+.65*amplitude)<3 or sum(series<=lo+.3*amplitude)<3:continue
            ownership={}
            if compact:
                if not(inner[0]<=peak[0]<=inner[2] and inner[1]<=peak[1]<=inner[3]):continue
                ownership=compact_core_ownership(stack,spread,labels,component,(ax,ay,bx,by),series)
                if ownership is None:continue
                prior=dict(kind='optical_fixture' if prior_fixture else 'admitted_emitter',id=child.id)
                if not prior_fixture:prior['observation_epoch']=self.established_emitter_epochs[child.id]
                ownership.update(ownership_peak_inside=True,ownership_prior=prior)
            proof=dict(proof_kind='unique_temporal_core',samples=len(state.samples),span_s=ts-state.samples[0][0],cycles=int(cycles),
                amplitude=amplitude,unique_child=True,unique_emitter=True,peak_in_child=True,inside_variation_fraction=inside,
                parent_hits=parent.hits,child_hits=child.hits,family_detection_updates=family_updates,
                peak=peak.tolist(),containment=containment(outer,inner),neighbor_veto_clear=True,
                minimum_child_edge=float(min(inner[2]-inner[0],inner[3]-inner[1])))
            proof.update(ownership)
            if parent.hits<5:
                if prior_fixture:proof.update(proof_kind='proven_fixture_body_variant',prior_fixture_id=child.id)
                else:proof.update(proof_kind='established_emitter_body_variant',prior_fixture_id=child.id,
                    prior_observation_epoch=self.established_emitter_epochs[child.id],recent_family_detection_updates=recent_updates)
            state.proof=proof;offers.append((parent,child,proof))
        self.pairs={k:v for k,v in self.pairs.items() if k in active};return offers
