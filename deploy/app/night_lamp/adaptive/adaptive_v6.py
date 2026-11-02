"""Offline v6: reconcile body/head identities and isolate provisional candidates."""
from dataclasses import dataclass
import cv2 as cv
import numpy as np
from .adaptive_v5 import AdaptiveV5,V5Params
from .adaptive_prototype import center,iou,PulseCounter
from .adaptive_tracking_v5 import BackgroundAnchor,EmitterLocator,ContextFlashSignal
from .adaptive_flash_signal import SignalParams
from .adaptive_observation_audit import ObservationAudit
from .nested_fixture_v6 import NestedResolver,containment,area


@dataclass(frozen=True)
class V6Params(V5Params):
    candidate_confidence:float=.65
    nested_probe_interval_s:float=.8


class AdaptiveV6(AdaptiveV5):
    def __init__(self,backend,params=None,vision=None):
        super().__init__(backend,params or V6Params(),vision)
        self.nested=NestedResolver();self.fixture_regions={};self.last_nested_probe=-1e9
        self.current_frame=None;self.current_geometry_ok=False;self.epoch={}

    def stable_body(self,t,ts):
        return bool(t.confidence>=self.p.candidate_confidence and t.hits>=self.p.promotion_hits
                    and ts-t.created>=self.p.promotion_span_s)

    def frame_evidence_hook(self,frame,ts,quality,frame_ok,geometric):
        self.current_geometry_ok=bool(frame_ok and geometric and quality['fresh'] and quality['clear'] and quality['continuous'])

    def influential_tracks(self,owner,ts):
        # A search-only false detection cannot take pixels from a known lamp.
        # Confirmed lamps constrain candidates; stable independent candidates
        # may constrain each other before either obtains a flash identity.
        return [t for t in self.tracks.values() if t.id!=owner.id and
                ((t.confirmed and t.confidence>=self.p.candidate_confidence)
                 or (not owner.confirmed and self.stable_body(t,ts)))]

    def background_tracks(self,owner,ts):
        # Excluding a stable bright region from background photometry is not
        # the same as assigning it ownership of another lamp's signal pixels.
        # Even a non-target streetlight is a nuisance light source and must not
        # become the photometric background of a real warning lamp.
        return [t for t in self.tracks.values() if t.id!=owner.id]

    def background_box(self,track,ts):
        if track.id not in self.fixture_regions:return self.frame_boxes[track.id]
        # Head sampling gets smaller after reconciliation, but the whole lamp
        # body/glow must still be excluded from a neighbor's background model.
        # Inter-check transforms are translations; full-check boxes are already
        # in current-frame coordinates.
        delta=np.asarray(self.frame_boxes[track.id])-track.box
        return (np.asarray(self.envelope(track))+delta).tolist()

    def envelope(self,t):
        relative=self.fixture_regions[t.id];b=np.asarray(t.body_box);w=b[2]-b[0];h=b[3]-b[1]
        return (b+relative*np.array([w,h,w,h])).tolist()

    def associate(self,frame,detections,ts,night):
        # Repeated coarse-body and fine-head detections are one hit per frame
        # once this fixture relation has been proved. Unverified nesting remains
        # separate and is still handled by the evidence audit.
        consumed=set();selected=[]
        for key in list(self.fixture_regions):
            if key not in self.tracks:self.fixture_regions.pop(key);continue
            t=self.tracks[key];outer=self.envelope(t);matches=[]
            for j,b in enumerate(detections):
                if j in consumed:continue
                family=(iou(t.body_box,b)>=.5 or (iou(outer,b)>=.5 and containment(outer,b)>=.9)
                        or (area(b)<=area(t.body_box) and containment(t.body_box,b)>=.9))
                if family and np.linalg.norm(center(b)-center(t.body_box))<=.4*max(outer[2]-outer[0],outer[3]-outer[1]):matches.append(j)
            if matches:
                best=max(matches,key=lambda j:(iou(t.body_box,detections[j]),detections[j][4]))
                selected.append(detections[best]);consumed.update(matches)
        super().associate(frame,selected+[b for j,b in enumerate(detections) if j not in consumed],ts,night)

    def extra_proposals(self,ts):
        if ts-self.last_nested_probe<self.p.nested_probe_interval_s:return []
        pending=[(key,state) for key,state in self.nested.pairs.items() if all(k in self.tracks for k in key)]
        if not pending:return []
        key,state=min(pending,key=lambda p:p[1].last_probe)
        self.last_nested_probe=ts
        return [(5.5,center(self.tracks[key[1]].body_box).tolist(),'nested_verification')]

    def related_fixture_aliases(self,parent,child,proof):return []

    def merge_tracks(self,ts):
        # Called by the base tracker only at a full geometry check, so canonical
        # boxes and the current frame use the same coordinates.
        offers=self.nested.observe(self.current_frame,list(self.tracks.values()),ts,self.current_geometry_ok)
        used=set()
        for parent,child,proof in offers:
            if parent.id in used or child.id in used or parent.id not in self.tracks or child.id not in self.tracks:continue
            family=[parent,child,*self.related_fixture_aliases(parent,child,proof)]
            family=list({t.id:t for t in family}.values())
            if any(t.id in used for t in family):continue
            retained=min(family,key=lambda t:(t.created,t.id));removed=[t for t in family if t is not retained]
            outer=parent.body_box[:];inner=child.body_box[:]
            old=[dict(id=t.id,pulses=t.pulses.count,valid_s=t.pulses.valid_s) for t in family]
            retained.box=inner[:];retained.body_box=inner[:]
            retained.aliases=list(dict.fromkeys([*retained.aliases,*[key for t in removed for key in [t.id,*t.aliases]]]))
            retained.hits=max(t.hits for t in family);retained.confidence=max(t.confidence for t in family)
            retained.last_seen=max(t.last_seen for t in family)
            retained.confirmed=False;retained.source='resolved_body_head';retained.corrected=True
            retained.discarded_valid_s+=max(t.pulses.valid_s for t in family)
            retained.pulses=PulseCounter();retained.observations.clear()
            retained.flash=self.make_flash_signal()
            retained.anchor=BackgroundAnchor(cv.cvtColor(self.current_frame,cv.COLOR_BGR2GRAY),inner)
            retained.locator=EmitterLocator(inner)
            # The new epoch begins after the current frame; old-region signal
            # and time are never assigned to the head just selected.
            retained.flash.last=dict(valid=False,ready=False,reason='calibrating_dark_baseline',signal=0.,raw=None)
            retained.position_observable=False
            retained.quality_details=dict(reasons=['calibrating_dark_baseline'],trust_source='resolved_fixture')
            w=inner[2]-inner[0];h=inner[3]-inner[1]
            self.fixture_regions[retained.id]=(np.asarray(outer)-inner)/[w,h,w,h]
            self.epoch[retained.id]=self.epoch.get(retained.id,0)+1
            for t in removed:self.tracks.pop(t.id);self.frame_boxes.pop(t.id,None)
            self.frame_boxes[retained.id]=inner
            used.update(t.id for t in family)
            self.events.append(dict(ts=ts,event='nested_identity_resolved',retained=retained.id,
                                    aliases=retained.aliases[:],proof=proof,old_evidence=old,
                                    observation_reset=True,after_pulses=0,after_valid_s=0.,epoch=self.epoch[retained.id]))
        super().merge_tracks(ts)

    def step(self,frame,ts,frame_id,night=False):
        self.current_frame=frame
        # Stream/scene quality is checked again in the base method. The resolver
        # additionally requires its own frozen local-background comparison.
        self.current_geometry_ok=True
        out=super().step(frame,ts,frame_id,night)
        resolved={e['retained'] for e in out['events'] if e['event']=='nested_identity_resolved'}
        for t in out['tracks']:
            live=self.tracks[t['id']]
            if night and t['confirmed'] and t['confidence']<self.p.candidate_confidence:
                live.confirmed=False;t['confirmed']=False;t['pending_reason']='need_lamp_type_evidence'
            t['candidate_role']='confirmed_fixture' if t['confirmed'] else 'stable_body' if self.stable_body(live,ts) else 'search_only'
            t['observation_epoch']=self.epoch.get(t['id'],0)
            t['nested_resolved']=t['id'] in self.fixture_regions
            if t['id'] in resolved:t['body_box']=live.body_box[:]
        out['confirmed']=[t for t in out['tracks'] if t['confirmed']]
        return out


class ObservationAuditV6(ObservationAudit):
    def apply(self,result):
        tracks={t['id']:t for t in result['tracks']}
        for event in result.get('events',[]):
            if event.get('event')!='nested_identity_resolved':continue
            p=event.get('proof',{});t=tracks.get(event.get('retained'))
            verified=(t is not None and event.get('observation_reset') and t['pulses']==0 and t['valid_s']==0
                      and p.get('samples',0)>=16 and p.get('span_s',0)>=3 and p.get('single_body_fraction',0)>=.6
                      and p.get('cycles',0)>=2 and p.get('parent_hits',0)>=3 and p.get('child_hits',0)>=2
                      and p.get('unique_child') and p.get('peak_in_child'))
            if verified:
                self.quarantined.difference_update([event['retained'],*event['aliases']])
        return super().apply(result)


class LampPipelineV6:
    def __init__(self,backend,params=None,vision=None):
        self.detector=AdaptiveV6(backend,params,vision);self.audit=ObservationAuditV6()
    def step(self,frame,ts,frame_id,night=False):return self.audit.apply(self.detector.step(frame,ts,frame_id,night))
