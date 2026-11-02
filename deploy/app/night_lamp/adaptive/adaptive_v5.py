"""Offline v5: independent signal cadence, background trust and emitter identities.

This module does not import annotations, report review IDs or production configs.
It deliberately preserves v4 code/results for A/B review.
"""
from collections import deque
from dataclasses import dataclass,field
import time
import cv2 as cv
import numpy as np
from .adaptive_prototype import AdaptiveDetector,Track,Params,center,iou,nms,grid,warp_boxes,PulseCounter
from .adaptive_flash_signal import AdaptiveFlashSignal,SignalParams,signal_plane
from .adaptive_tracking_v5 import BackgroundAnchor,EmitterLocator,held_translation,roi_signal_plane,ContextFlashSignal
from .search_plan import plan_search


@dataclass(frozen=True)
class V5Params(Params):
    search_policy:str='legacy'
    input_imgsz:int=640
    coarse_confidence:float=.35
    detector_period_s:float=.2
    max_geometry_age_s:float=.5
    emitter_update_s:float=.5
    merge_distance_px:float=2.5
    detector_tolerance_s:float=.05
    emission_peak_v:float=150.
    emission_rise:float=12.


@dataclass
class LampTrack(Track):
    anchor:object=None
    locator:object=None
    aliases:list=field(default_factory=list)
    observations:object=field(default_factory=lambda:deque(maxlen=80))
    quality_details:dict=field(default_factory=dict)
    corrected:bool=False
    discarded_valid_s:float=0.
    body_box:object=None


class AdaptiveV5(AdaptiveDetector):
    def __init__(self,backend,params=None,vision=None):
        super().__init__(backend,params or V5Params(),vision)
        self.origin=None;self.next_detector_ts=None;self.geometry_gray=None;self.geometry_ts=None
        self.geometry_valid=False;self.force_geometry=False;self.events=[];self.frame_boxes={}
        self.fixed_reference=None;self.applied_transform=np.eye(3)

    def influential_tracks(self,owner,ts):
        """Override point for candidate admission; v5 behavior is unchanged."""
        return [t for t in self.tracks.values() if t.id!=owner.id]

    def extra_proposals(self,ts):return []

    def background_tracks(self,owner,ts):return self.influential_tracks(owner,ts)

    def background_box(self,track,ts):return self.frame_boxes[track.id]

    def background_exclusion_boxes(self,track,neighbors,ts):return neighbors

    def frame_evidence_hook(self,frame,ts,quality,frame_ok,geometric):pass

    def make_flash_signal(self):
        signal=ContextFlashSignal(SignalParams(max_gap_s=self.p.max_gap_s))
        signal.min_peak_v=self.p.emission_peak_v;signal.min_absolute_rise=self.p.emission_rise
        return signal

    def after_flash_observation(self,track,observation,ts):pass

    def should_recenter(self,track,correction,box,ts):return True

    def independent_signal_neighbor(self,owner,other,own_box,other_box):
        return iou(own_box,other_box)<.5

    def associate(self,frame,detections,ts,night,*,reserved_ids=()):
        pairs=[]
        for key,t in self.tracks.items():
            if key in reserved_ids:continue
            for j,b in enumerate(detections):
                distance=float(np.linalg.norm(center(t.body_box if t.body_box is not None else t.box)-center(b)))
                gate=max(4,min(14,.5*min(t.box[2]-t.box[0],t.box[3]-t.box[1])))
                if distance<=gate:pairs.append((distance,key,j))
        used=set();ids=set()
        for _,key,j in sorted(pairs):
            if key in ids or j in used:continue
            ids.add(key);used.add(j);t=self.tracks[key];b=detections[j]
            t.hits+=1;t.last_seen=ts;t.confidence=max(t.confidence,b[4]);t.fine_hits+=int(b[5]=='fine')
            # Geometry is fixed between background transforms. A noisy single
            # YOLO box cannot drag an established core or its dark template.
        gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        for j,b in enumerate(detections):
            if j in used:continue
            t=LampTrack(self.next_id,b[:4],ts,ts,confidence=b[4],fine_hits=int(b[5]=='fine'))
            t.body_box=b[:4]
            t.flash=self.make_flash_signal()
            t.anchor=BackgroundAnchor(gray,t.box);t.locator=EmitterLocator(t.box)
            self.tracks[t.id]=t;self.next_id+=1

    def discover(self,frame,previous,ts,night):
        proposals=self.proposals_from_image(frame,previous,night,ts);detections=[];coarse=0;rois=[]
        if ts-self.last_coarse>=self.p.coarse_period_s-1e-6:
            rr=grid(frame.shape,320,256);coarse=len(rr);raw=self.backend.predict(frame,rr,self.p.input_imgsz,self.p.coarse_confidence)
            detections.extend([b+['coarse'] for b in nms(raw) if not self.masked(center(b),frame.shape)])
            proposals.extend((3.,center(b).tolist(),'coarse_small') for b in raw if min(b[2]-b[0],b[3]-b[1])<18)
            self.last_coarse=ts
        if ts-self.last_fine>=self.p.fine_period_s-1e-6:
            proposals.extend(self.extra_proposals(ts))
            proposals.extend((4.,center(t.box).tolist(),'history') for t in self.tracks.values()
                             if min(t.box[2]-t.box[0],t.box[3]-t.box[1])<=16 and ts-t.last_seen>self.p.fine_period_s)
            plan=plan_search(frame.shape,proposals,self.roi_tested,self.survey_cursor,ts,
                size=self.p.fine_size,budget=self.p.fine_budget,survey_budget=self.p.survey_budget,
                survey=grid(frame.shape,160,128),exclusions=self.p.overlay_exclusions,policy=self.p.search_policy)
            rois=[list(r) for r in plan.regions];self.survey_cursor=plan.next_cursor
            if rois:
                raw=self.backend.predict(frame,rois,self.p.input_imgsz,self.p.fine_confidence)
                detections.extend([b+['fine'] for b in nms(raw) if not self.masked(center(b),frame.shape)])
                for r in rois:self.roi_tested[tuple(r[:2])]=ts
            self.last_fine=ts
        detections=[b for b in nms(detections) if min(b[2]-b[0],b[3]-b[1])>=2 and max(b[2]-b[0],b[3]-b[1])<=80]
        self.associate(frame,detections,ts,night)
        return detections,coarse,rois

    def merge_tracks(self,ts):
        removed=set()
        for a in sorted(self.tracks.values(),key=lambda t:(not t.confirmed,t.created,t.id)):
            if a.id in removed or not a.locator.locked:continue
            for b in list(self.tracks.values()):
                if b.id==a.id or b.id in removed or not b.locator.locked:continue
                if np.linalg.norm(center(a.box)-center(b.box))>self.p.merge_distance_px:continue
                body_a=a.body_box if a.body_box is not None else a.box
                body_b=b.body_box if b.body_box is not None else b.box
                # Independently detected adjacent fixtures must not be merged
                # merely because a weak emitter got attracted to its neighbor.
                if iou(body_a,body_b)<.5 and np.linalg.norm(center(body_a)-center(body_b))>self.p.merge_distance_px:continue
                aa={round(stamp,5):value for stamp,value in a.observations}
                pairs=[(aa[round(stamp,5)],value) for stamp,value in b.observations if round(stamp,5) in aa]
                if len(pairs)<12:continue
                v=np.asarray(pairs)
                if min(v.std(axis=0))<5 or np.corrcoef(v.T)[0,1]<.9:continue
                # Keep one complete stream; never add counts, seconds or frame hits.
                selected=max([a,b],key=lambda t:(t.confirmed,t.pulses.valid_s,t.hits))
                before=[a.pulses.count,b.pulses.count]
                if selected is b:
                    a.pulses=b.pulses;a.flash=b.flash;a.observations=b.observations
                a.hits=max(a.hits,b.hits);a.confidence=max(a.confidence,b.confidence);a.last_seen=max(a.last_seen,b.last_seen)
                a.aliases.extend([b.id,*b.aliases]);a.confirmed=a.confirmed or b.confirmed
                removed.add(b.id)
                self.events.append(dict(ts=ts,event='identity_merge',retained=a.id,alias=b.id,
                                        before_pulses=before,after_pulses=a.pulses.count,evidence_policy='one_stream_only'))
        for key in removed:self.tracks.pop(key,None);self.frame_boxes.pop(key,None)

    def step(self,frame,ts,frame_id,night=False):
        if not np.isfinite(ts):raise ValueError('Timestamp must be finite')
        if frame_id is None:raise ValueError('Unique frame ID required')
        if self.previous_id==frame_id or (self.previous_ts is not None and ts<=self.previous_ts):
            return dict(self.last_output,duplicate_ignored=True)
        started=time.perf_counter();first_event=len(self.events)
        if self.origin is None:self.origin=ts;self.next_detector_ts=ts
        due=ts+self.p.detector_tolerance_s>=self.next_detector_ts-1e-6
        q=self.quality.observe(frame,ts,frame_id)
        metrics=getattr(self.quality,'last_metrics',None)
        gray=metrics.gray if metrics is not None and metrics.matches(frame,ts,frame_id) else cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        frame_ok=bool(q['fresh'] and q['clear']);H=np.eye(3);motion='quality_unavailable';aligned_previous=None
        full=due or self.force_geometry or self.geometry_ts is None or ts-self.geometry_ts>self.p.max_geometry_age_s
        if frame_ok and full:
            frame_ok=bool(self.vision.quality(frame)['ok']);motion='bootstrap'
            if frame_ok and self.fixed_reference is not None:
                al=self.vision.align(self.fixed_reference,frame)
                if al.valid and al.displacement<=self.p.move_fraction:
                    absolute=np.asarray(al.matrix,float)
                    H=absolute@np.linalg.inv(self.applied_transform);motion='aligned'
                    aligned_previous=cv.warpPerspective(self.previous,H,(frame.shape[1],frame.shape[0]))
                    for t in self.tracks.values():
                        t.box=warp_boxes([t.box],H)[0]
                        if t.body_box is not None:t.body_box=warp_boxes([t.body_box],H)[0]
                    self.applied_transform=absolute
                    self.geometry_valid=True
                elif not al.valid:
                    # Occlusion/poor feature coverage is absence of a mapping,
                    # not evidence of a new view. Freeze discovery and sampling
                    # until the original reference can be verified again.
                    motion='registration_unavailable';frame_ok=False
                    if self.geometry_valid:
                        self.events.append(dict(ts=ts,event='geometry_unavailable',reason=al.details.get('reason','unavailable')))
                    self.geometry_valid=False
                else:
                    motion='moved_or_unavailable';self.tracks.clear();self.proposals.clear();self.roi_tested.clear();self.last_coarse=-1e9
                    self.geometry_valid=False
                    self.fixed_reference=frame.copy();self.applied_transform=np.eye(3)
                    self.events.append(dict(ts=ts,event='geometry_reset',reason=al.details.get('reason','unavailable')))
            elif frame_ok:
                self.geometry_valid=True;self.fixed_reference=frame.copy();self.applied_transform=np.eye(3)
            if frame_ok:
                self.previous=frame.copy();self.geometry_gray=gray;self.geometry_ts=ts
            self.force_geometry=not self.geometry_valid
            H=np.eye(3)  # canonical boxes have just moved to this full-check frame
        elif frame_ok and self.geometry_valid:
            estimate=held_translation(self.geometry_gray,gray,self.p.overlay_exclusions)
            if estimate is not None:H=estimate;motion='held_background_translation'
            else:motion='intercheck_mapping_unavailable';self.force_geometry=True
        detections=[];coarse=0;rois=[]
        if due:
            steps=max(1,int(np.floor((ts+self.p.detector_tolerance_s-self.next_detector_ts)/self.p.detector_period_s+1e-6))+1)
            self.next_detector_ts+=steps*self.p.detector_period_s
            if frame_ok:
                detections,coarse,rois=self.discover(frame,aligned_previous,ts,night)
        self.frame_boxes={t.id:warp_boxes([t.box],H)[0] for t in self.tracks.values()}
        body_boxes={t.id:warp_boxes([t.body_box if t.body_box is not None else t.box],H)[0] for t in self.tracks.values()}
        plane,signal_pixels=roi_signal_plane(frame,list(self.frame_boxes.values())) if self.tracks else (None,0)
        geometric=motion in ('aligned','held_background_translation')
        self.frame_evidence_hook(frame,ts,q,frame_ok,geometric)
        for key,t in list(self.tracks.items()):
            if not t.confirmed and ts-t.last_seen>self.p.ttl_s:
                self.tracks.pop(key);self.frame_boxes.pop(key,None);continue
            box=self.frame_boxes[key];influencers=self.influential_tracks(t,ts)
            background_tracks=self.background_tracks(t,ts)
            neighbors=[self.background_box(other,ts) for other in background_tracks]
            background=t.anchor.evaluate(gray,box,self.background_exclusion_boxes(t,neighbors,ts))
            reasons=[]
            if not q['fresh']:reasons.append('stream_frozen_or_stale')
            if not q['clear'] or not frame_ok:reasons.append('frame_quality_failed')
            if not q['continuous']:reasons.append('source_gap')
            if not geometric:reasons.append(motion)
            if not background['ok']:reasons.append(background['reason'])
            if background.get('expanded_context') and ts-t.last_seen>self.p.coarse_period_s+self.p.max_gap_s:
                reasons.append('expanded_context_needs_recent_body_detection')
            supported=t.hits>=self.p.promotion_hits and ts-t.created>=self.p.promotion_span_s
            if not supported and ts-t.last_seen>self.p.coarse_period_s*1.5:reasons.append('position_evidence_expired')
            t.position_observable=not reasons
            t.quality_details=dict(reasons=reasons,background=background,yolo_age_s=ts-t.last_seen,
                                   trust_source='background_and_geometry' if supported else 'recent_yolo_and_background')
            # Locator samples use verified geometry and local background, not an
            # annotation seed or a candidate's existing pulse count.
            own_body=body_boxes[key]
            protected=[]
            for other in background_tracks:
                if other.hits<self.p.promotion_hits:continue
                other_body=body_boxes[other.id]
                if self.independent_signal_neighbor(t,other,own_body,other_body):protected.append(center(other_body))
            correction=t.locator.observe(plane,box,ts,t.position_observable and supported,protected) if night else None
            threshold=max(2.5,.12*min(box[2]-box[0],box[3]-box[1]))
            crosses_neighbor=bool(correction is not None and any(np.linalg.norm(center(box)+correction-point)<3.5 for point in protected))
            if correction is not None and np.linalg.norm(correction)>threshold and not t.corrected and full and not crosses_neighbor and self.should_recenter(t,correction,box,ts):
                before=t.box[:];t.box=(np.asarray(t.box)+np.tile(correction,2)).tolist();box=t.box
                self.frame_boxes[key]=box;t.corrected=True
                t.anchor=BackgroundAnchor(gray,box);t.locator=EmitterLocator(box)
                t.discarded_valid_s+=t.pulses.valid_s
                self.events.append(dict(ts=ts,event='emitter_recenter',id=t.id,before_box=before,after_box=box,
                                        discarded_pulses=t.pulses.count,discarded_valid_s=t.pulses.valid_s))
                t.flash=self.make_flash_signal();t.pulses=PulseCounter();t.observations.clear()
                t.confirmed=False;t.source='yolo'  # old off-center evidence is not a new-core confirmation
            t.flash.context=background.get('photometry')
            t.flash.neighbor_centers=[center(body_boxes[other.id]) for other in influencers
                                     if self.independent_signal_neighbor(t,other,own_body,body_boxes[other.id])]
            radius=t.flash.radius if t.flash.radius is not None else int(max(2,min(18,min(box[2]-box[0],box[3]-box[1])/2)))
            pixels=cv.getRectSubPix(frame,(2*radius+1,2*radius+1),tuple(map(float,center(box))))
            t.flash.raw_peak_pixels=pixels.max(axis=2)
            t.flash.peak_v=float(pixels.max())
            observation=t.flash.observe(plane,box,ts,t.position_observable)
            self.after_flash_observation(t,observation,ts)
            if observation['valid'] and observation['raw'] is not None:t.observations.append((ts,observation['raw']))
            t.pulses.sample(ts,observation['signal'],observation['valid'],self.flash_cfg)
            if supported and ((night and t.pulses.count>=self.p.night_pulses) or (not night and t.confidence>=self.p.day_confidence)):
                t.confirmed=True;t.source='night_flashes' if night else 'day_body'
        if full and frame_ok:self.merge_tracks(ts)
        records=[]
        for t in self.tracks.values():
            o=t.flash.last
            if not t.position_observable:reason=t.quality_details['reasons'][0]
            elif not night:reason='identity_confirmed' if t.confirmed else 'need_independent_yolo_hits'
            elif t.confirmed:reason='identity_confirmed'
            elif not o.get('ready'):reason=o.get('reason','calibrating_dark_baseline')
            elif t.hits<self.p.promotion_hits:reason='need_independent_yolo_hits'
            else:reason='need_15_complete_pulses'
            complete=t.pulses.valid_s>=self.cfg.effective_s
            window='complete_normal' if t.pulses.count>=self.p.night_pulses else 'complete_zero' if complete and t.pulses.count==0 else 'complete_below_threshold' if complete else 'incomplete'
            records.append(dict(id=t.id,box=self.frame_boxes[t.id],confidence=t.confidence,confirmed=t.confirmed,hits=t.hits,
                                pulses=t.pulses.count,valid_s=t.pulses.valid_s,source=t.source,last_seen=t.last_seen,
                                visible_recent=ts-t.last_seen<=.8,position_observable=t.position_observable,
                                flash_signal=o,pending_reason=reason,quality_details=t.quality_details,
                                aliases=t.aliases,emitter_locked=t.locator.locked,corrected=t.corrected,
                                body_box=body_boxes[t.id],
                                discarded_valid_s=t.discarded_valid_s,window_status=window,health_status='offline_window_only'))
        result=dict(ts=ts,motion=motion,quality=q,detections=[b[:5] for b in detections],tracks=records,
                    confirmed=[t for t in records if t['confirmed']],fine_rois=rois,coarse_tiles=coarse,fine_tiles=len(rois),
                    detector_tick=due,geometry_full_check=full,signal_pixels=signal_pixels,events=self.events[first_event:],elapsed_ms=(time.perf_counter()-started)*1000,health_decision=None)
        self.previous_ts=ts;self.previous_id=frame_id;self.last_output=result
        return result
