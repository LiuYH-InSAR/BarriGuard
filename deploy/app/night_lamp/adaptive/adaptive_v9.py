"""Small-head optical ownership; preserve v8 outputs and all business thresholds."""
from .adaptive_v8 import AdaptiveV8
from .adaptive_v6 import V6Params,ObservationAuditV6
from .optical_fixture_v9 import OpticalResolver,same_fixture_hypothesis,compact_ownership_verified
from .adaptive_prototype import center,iou
from .nested_fixture_v6 import containment
import numpy as np
import cv2 as cv
from collections import deque
from .adaptive_tracking_v5 import BackgroundAnchor
from .adaptive_flash_signal import SignalParams
from .exposure_recovery import RecoverableFlashSignal
from .adaptive_tracking_v5 import EmitterLocator
from .adaptive_prototype import PulseCounter
from .source_separation import SourceSeparation,separation_verified
from .single_source import SingleSourceResolver,overlapping_fixture,ownership_verified

class AdaptiveV9(AdaptiveV8):
    def __init__(self,backend,params=None,vision=None):
        super().__init__(backend,params,vision);self.nested=OpticalResolver();self.established_emitters={}
        self.current_night=False;self.audit_quarantined=frozenset();self.held_recenters=set()
        self.separation=SourceSeparation()
        self.single_source=SingleSourceResolver()
    def associate(self,frame,detections,ts,night):
        detections=[b for b in detections if not self.separation.suppresses(b,self.tracks,self.epoch)]
        return super().associate(frame,detections,ts,night)
    def make_flash_signal(self):
        signal=RecoverableFlashSignal(SignalParams(max_gap_s=self.p.max_gap_s))
        signal.min_peak_v=self.p.emission_peak_v;signal.min_absolute_rise=self.p.emission_rise
        return signal
    def step(self,frame,ts,frame_id,night=False):
        self.current_night=night
        return super().step(frame,ts,frame_id,night)
    def frame_evidence_hook(self,frame,ts,quality,frame_ok,geometric):
        super().frame_evidence_hook(frame,ts,quality,frame_ok,geometric)
        if not self.current_geometry_ok:self.single_source.invalidate()
        self.held_recenters={key for key in self.held_recenters if key[0] in self.tracks and key[1]==self.epoch.get(key[0],0)}
        for t in self.tracks.values():
            t.flash.recovery_allowed=bool(self.current_night and self.current_geometry_ok and t.id not in self.audit_quarantined
                and t.hits>=5 and t.confidence>=self.p.candidate_confidence and t.locator.locked
                and 0<=ts-t.last_seen<=self.p.coarse_period_s+self.p.max_gap_s)
            t.flash.detector_hits=t.hits
    def should_recenter(self,t,correction,box,ts):
        # A freshly flashing, already established sampler does not need to move
        # to a peak it already contains. An excluded point inside this sampler
        # is a rejected correction, never a reason to move into masked pixels.
        mask=t.flash.support_mask;last=t.pulses.last_flash
        trusted=(self.current_night and t.confirmed and t.pulses.count>=self.p.night_pulses and last is not None
            and 0<=ts-last<=self.cfg.max_width_s and t.position_observable and t.flash.amplitude is not None)
        if not trusted or mask is None:return True
        radius=(mask.shape[0]-1)//2;px,py=np.rint(np.asarray(correction)+radius).astype(int)
        excluded=0<=px<mask.shape[1] and 0<=py<mask.shape[0] and not mask[py,px]
        supported=1<=px<mask.shape[1]-1 and 1<=py<mask.shape[0]-1 and mask[py-1:py+2,px-1:px+2].all()
        if not excluded and not supported:return True
        key=(t.id,self.epoch.get(t.id,0))
        if key not in self.held_recenters:
            self.held_recenters.add(key)
            self.events.append(dict(event='emitter_region_retained',ts=ts,id=t.id,observation_reset=False,
                reason='excluded_core_cannot_replace_recently_flashing_region' if excluded else 'recent_flashes_in_existing_support',
                candidate_offset=np.asarray(correction).tolist(),pulses=t.pulses.count))
        return False
    def after_flash_observation(self,t,observation,ts):
        super().after_flash_observation(t,observation,ts)
        change=observation.get('exposure_reference_changed')
        if change is None:return
        before=dict(pulses=t.pulses.count,valid_s=t.pulses.valid_s)
        t.discarded_valid_s+=t.pulses.valid_s;t.pulses=PulseCounter();t.observations.clear();t.confirmed=False
        t.source='signal_reference_recalibration';t.locator=EmitterLocator(t.box)
        self.epoch[t.id]=self.epoch.get(t.id,0)+1
        self.events.append(dict(event='signal_reference_changed',ts=ts,id=t.id,observation_reset=True,
            before=before,after_pulses=0,after_valid_s=0.,reference=change,epoch=self.epoch[t.id]))
    def independent_signal_neighbor(self,owner,other,own_box,other_box):
        # Geometry is only a hypothesis: do not steal source pixels while the
        # audit quarantines the unresolved relation. Only temporal proof merges it.
        if same_fixture_hypothesis(own_box,other_box):return False
        return super().independent_signal_neighbor(owner,other,own_box,other_box)
    def related_fixture_aliases(self,parent,child,proof):
        if proof.get('proof_kind') not in {'unique_temporal_core','proven_fixture_body_variant','established_emitter_body_variant'}:return []
        point=np.array(proof['peak']);inner=child.body_box;found=[]
        for t in self.tracks.values():
            if t.id in [parent.id,child.id] or t.confidence<.65:continue
            b=t.body_box
            if not(b[0]-1<=point[0]<=b[2]+1 and b[1]-1<=point[1]<=b[3]+1):continue
            if containment(inner,b)<.8 or np.linalg.norm(center(inner)-center(b))>max(3.5,.4*max(inner[2]-inner[0],inner[3]-inner[1])):continue
            if t.locator is not None and t.locator.locked and np.linalg.norm(center(t.box)-point)>3:continue
            found.append(t)
        proof['related_same_core_ids']=[t.id for t in found]
        return found
    def background_exclusion_boxes(self,track,neighbors,ts):
        if track.id in self.fixture_regions:return [self.background_box(track,ts),*neighbors]
        return neighbors
    def merge_tracks(self,ts):
        offers=self.separation.observe(self.current_frame,self.tracks,self.epoch,ts,self.current_geometry_ok)
        used=set()
        for parent,children,proof in offers:
            if parent.id not in self.tracks or any(t.id in used for t in children):continue
            self.tracks.pop(parent.id);self.frame_boxes.pop(parent.id,None)
            for t,c in zip(children,proof['children']):
                before=dict(pulses=t.pulses.count,valid_s=t.pulses.valid_s)
                t.discarded_valid_s+=t.pulses.valid_s;t.pulses=PulseCounter();t.observations.clear();t.confirmed=False
                t.source='independent_source_reobservation';self.epoch[t.id]=self.epoch.get(t.id,0)+1;c['epoch']=self.epoch[t.id]
                self.events.append(dict(event='independent_source_reset',ts=ts,id=t.id,observation_reset=True,
                    before=before,after_pulses=0,after_valid_s=0.,epoch=self.epoch[t.id]))
            self.separation.groups.append(proof['children'])
            self.events.append(dict(event='independent_sources_resolved',ts=ts,discarded_group=parent.id,proof=proof,observation_reset=True))
            used.update(t.id for t in children)
        self.nested.proven_fixture_ids=frozenset(self.fixture_regions)
        invalid={e['id'] for e in self.events if e.get('ts')==ts and e.get('event')=='emitter_recenter'}
        self.nested.established_emitter_epochs={key:epoch for key,epoch in self.established_emitters.items()
            if key in self.tracks and key not in invalid and self.epoch.get(key,0)==epoch
            and self.tracks[key].confirmed and self.tracks[key].source=='night_flashes'}
        trusted={}
        for key,t in self.tracks.items():
            epoch=self.epoch.get(key,0)
            if (self.established_emitters.get(key)==epoch and t.confirmed and t.pulses.count>=15
                and t.source=='night_flashes' and t.position_observable and t.flash.last.get('ready')
                and t.locator.locked):
                trusted[key]=dict(box=t.box[:],flash=t.flash,pulses=t.pulses,anchor=t.anchor,locator=t.locator,
                    observations=deque(t.observations,maxlen=80),epoch=epoch,discarded_valid_s=t.discarded_valid_s)
        first=len(self.events);super().merge_tracks(ts)
        gray=None
        for event in self.events[first:]:
            if event.get('event')!='nested_identity_resolved':continue
            t=self.tracks.get(event['retained'])
            if t is None:continue
            prior=trusted.get(t.id);proof=event.get('proof',{})
            if (prior is not None
                and proof.get('proof_kind') in {'unique_temporal_core','proven_fixture_body_variant','established_emitter_body_variant'}
                and prior['box']==t.box):
                # The optically verified alias changed, not the sampled pixels.
                # Preserve this one sampler; never sum streams or admit its
                # quarantined interval into the independent business ledger.
                for name in ['flash','pulses','anchor','locator','observations','discarded_valid_s']:setattr(t,name,prior[name])
                t.confirmed=True;t.source='night_flashes';self.epoch[t.id]=prior['epoch']
                event.update(observation_reset=False,evidence_policy='unchanged_trusted_sampler',
                    prior_sampler_id=t.id,
                    before_sampler_box=prior['box'],after_sampler_box=t.box[:],before_observation_epoch=prior['epoch'],
                    epoch=prior['epoch'],after_pulses=t.pulses.count,after_valid_s=t.pulses.valid_s,
                    background_context_radius=t.anchor.radius)
                continue
            # Selecting a small source does not shrink the background evidence
            # to a featureless patch. Anchor it at the selected source, but keep
            # the full fixture's surrounding context and exclude its whole body.
            if gray is None:gray=cv.cvtColor(self.current_frame,cv.COLOR_BGR2GRAY)
            outer=self.envelope(t);anchor=BackgroundAnchor(gray,t.box)
            anchor.radius=int(max(24,min(64,max(outer[2]-outer[0],outer[3]-outer[1])*2)))
            anchor.reference=anchor.patch(gray,t.box);t.anchor=anchor
            event['background_context_radius']=anchor.radius

        offers=self.single_source.observe(self.current_frame,self.tracks,self.epoch,self.audit_quarantined,ts,self.current_geometry_ok)
        used=set()
        for family,proof in offers:
            if any(t.id not in self.tracks or t.id in used for t in family):continue
            retained=min(family,key=lambda t:(self.established_emitters.get(t.id)!=self.epoch.get(t.id,0),t.created,t.id))
            removed=[t for t in family if t is not retained]
            prior=trusted.get(retained.id)
            preserve=bool(prior is not None and prior['box']==retained.box and prior['epoch']==self.epoch.get(retained.id,0))
            before_epoch=self.epoch.get(retained.id,0);before_box=retained.box[:]
            retained.aliases=list(dict.fromkeys([*retained.aliases,*[k for t in removed for k in [t.id,*t.aliases]]]))
            if removed:
                boxes=np.asarray([t.body_box for t in family]);outer=[boxes[:,0].min(),boxes[:,1].min(),boxes[:,2].max(),boxes[:,3].max()]
                b=np.asarray(retained.body_box);w=b[2]-b[0];h=b[3]-b[1]
                self.fixture_regions[retained.id]=(np.asarray(outer)-b)/[w,h,w,h]
            if not preserve:
                retained.discarded_valid_s+=retained.pulses.valid_s
                retained.pulses=PulseCounter();retained.observations.clear();retained.confirmed=False
                retained.flash=self.make_flash_signal();retained.locator=EmitterLocator(retained.box)
                retained.flash.last=dict(valid=False,ready=False,reason='single_source_reobservation',signal=0.,raw=None)
                retained.position_observable=False;retained.source='single_source_reobservation'
                retained.quality_details=dict(reasons=['single_source_reobservation'],trust_source='optical_single_source')
                self.epoch[retained.id]=before_epoch+1
            for t in removed:self.tracks.pop(t.id);self.frame_boxes.pop(t.id,None)
            used.update(t.id for t in family)
            self.events.append(dict(event='same_source_resolved',ts=ts,retained=retained.id,aliases=retained.aliases[:],proof=proof,
                observation_reset=not preserve,evidence_policy='unchanged_trusted_sampler' if preserve else 'new_sampler_epoch',
                before_sampler_box=before_box,after_sampler_box=retained.box[:],before_observation_epoch=before_epoch,
                epoch=self.epoch.get(retained.id,0),after_pulses=retained.pulses.count,after_valid_s=retained.pulses.valid_s))

class ObservationAuditV9(ObservationAuditV6):
    def __init__(self):
        super().__init__();self.proven_fixtures=set();self.established_emitters={};self.previous_records={}
    def conflicts(self,a,b):
        # Coarse search hypotheses below the lamp-type threshold cannot enter
        # a quarantine whose optical resolver refuses to examine them.
        return min(a['confidence'],b['confidence'])>=.65 and (super().conflicts(a,b)
            or overlapping_fixture(a.get('body_box',a['box']),b.get('body_box',b['box'])))
    def apply(self,result):
        tracks=result['tracks']
        for i,a in enumerate(tracks):
            for b in tracks[i+1:]:
                if min(a['confidence'],b['confidence'])>=.65 and (same_fixture_hypothesis(a.get('body_box',a['box']),b.get('body_box',b['box']))
                    or overlapping_fixture(a.get('body_box',a['box']),b.get('body_box',b['box']))):
                    self.quarantined.update([a['id'],b['id']])
        live={t['id']:t for t in tracks}
        for event in result.get('events',[]):
            if event.get('event')=='same_source_resolved':
                t=live.get(event.get('retained'));proof=event.get('proof',{})
                before=event.get('before_observation_epoch');family={event.get('retained'),*event.get('aliases',[])}
                preserved=(t is not None and event.get('evidence_policy')=='unchanged_trusted_sampler'
                    and not event.get('observation_reset') and self.established_emitters.get(t['id'])==before
                    and t.get('observation_epoch')==before and t['confirmed'] and t['pulses']>=15
                    and event.get('before_sampler_box')==event.get('after_sampler_box')==t['box'])
                reset=(t is not None and isinstance(before,int) and event.get('observation_reset')
                    and t.get('observation_epoch')==before+1 and t['pulses']==0 and t['valid_s']==0)
                verified=bool(t is not None and t['id'] in {m['id'] for m in proof.get('members',[])}
                    and all(m['id']==t['id'] or m['id'] not in live for m in proof.get('members',[]))
                    and (preserved or reset) and ownership_verified(proof,live,self.previous_records))
                event['consumer_verified']=verified
                if verified:
                    self.quarantined.difference_update(family)
                    if len(proof['members'])==2:self.proven_fixtures.add(t['id'])
                    for key in family:self.established_emitters.pop(key,None)
            if event.get('event')=='independent_sources_resolved':
                p=event.get('proof',{});records={t['id']:t for t in tracks}
                children=p.get('children',[])
                if (event.get('observation_reset') and event.get('discarded_group') not in records
                    and separation_verified(p,records)
                    and all(records[c['id']]['pulses']==0 and records[c['id']]['valid_s']==0 for c in children)):
                    self.quarantined.difference_update(c['id'] for c in children)
            if event.get('event')=='geometry_reset':self.established_emitters.clear()
            if event.get('event') in {'emitter_recenter','signal_support_changed','signal_reference_changed'}:
                self.established_emitters.pop(event['id'],None)
            if event.get('event')!='nested_identity_resolved':continue
            p=event.get('proof',{});t=live.get(event.get('retained'))
            family={event.get('retained'),*event.get('aliases',[])}
            original=(p.get('proof_kind')=='unique_temporal_core' and p.get('parent_hits',0)>=5 and p.get('child_hits',0)>=1)
            repeat=(p.get('proof_kind')=='proven_fixture_body_variant' and p.get('prior_fixture_id') in self.proven_fixtures
                and p.get('prior_fixture_id') in family and p.get('parent_hits',0)>=1 and p.get('child_hits',0)>=5)
            established=(p.get('proof_kind')=='established_emitter_body_variant'
                and p.get('prior_fixture_id') in self.established_emitters and p.get('prior_fixture_id') in family
                and self.established_emitters[p['prior_fixture_id']]==p.get('prior_observation_epoch')
                and p.get('parent_hits',0)>=1 and p.get('child_hits',0)>=5 and p.get('recent_family_detection_updates',0)>=3)
            ownership=(p.get('inside_variation_fraction',0)>=.7 or compact_ownership_verified(
                p,self.proven_fixtures,self.established_emitters,family))
            retained_sampler=(t is not None and event.get('evidence_policy')=='unchanged_trusted_sampler'
                and not event.get('observation_reset') and event.get('prior_sampler_id')==t['id']
                and self.established_emitters.get(t['id'])==event.get('before_observation_epoch')==t.get('observation_epoch')
                and isinstance(event.get('before_observation_epoch'),int)
                and event.get('before_sampler_box')==event.get('after_sampler_box')==t['box']
                and t['confirmed'] and t['pulses']>=15 and t.get('flash_signal',{}).get('ready'))
            reset_sampler=(t is not None and event.get('observation_reset') and t['pulses']==0 and t['valid_s']==0)
            verified=(t is not None and (original or repeat or established) and (reset_sampler or retained_sampler)
                and p.get('samples',0)>=16 and p.get('span_s',0)>=3
                and p.get('cycles',0)>=2 and p.get('amplitude',0)>=18 and ownership
                and p.get('family_detection_updates',0)>=3
                and p.get('minimum_child_edge',0)>=4
                and all(p.get(k) for k in ['unique_child','unique_emitter','peak_in_child','neighbor_veto_clear']))
            if verified:
                self.quarantined.difference_update(family)
                self.proven_fixtures.difference_update(family)
                self.proven_fixtures.add(event['retained'])
            for key in family:self.established_emitters.pop(key,None)
        self.proven_fixtures.intersection_update(live)
        out=super().apply(result)
        self.previous_records={t['id']:dict(id=t['id'],box=t['box'][:],body_box=t.get('body_box',t['box'])[:],
            observation_epoch=t.get('observation_epoch',0)) for t in result['tracks']}
        live={t['id']:t for t in out['tracks']}
        self.established_emitters={key:epoch for key,epoch in self.established_emitters.items() if key in live
            and live[key].get('observation_epoch',0)==epoch and live[key].get('raw_confirmed',False)}
        quality=out.get('quality',{})
        if all(quality.get(k,False) for k in ['fresh','clear','continuous']) and out.get('motion') in {'aligned','held_background_translation'}:
            for t in out['tracks']:
                signal=t.get('flash_signal',{})
                if (t.get('evidence_admitted') and t.get('confirmed') and t.get('position_observable')
                    and t.get('source')=='night_flashes' and t.get('emitter_locked') and t['confidence']>=.65 and t.get('hits',0)>=5
                    and t['pulses']>=15 and signal.get('valid') and signal.get('ready') and signal.get('emission_supported')
                    and t.get('quality_details',{}).get('background',{}).get('ok')):
                    self.established_emitters[t['id']]=t.get('observation_epoch',0)
        return out

class LampPipelineV9:
    def __init__(self,backend,params=None,vision=None):
        self.detector=AdaptiveV9(backend,params or V6Params(),vision);self.audit=ObservationAuditV9()
    def step(self,frame,ts,frame_id,night=False):
        self.detector.established_emitters=dict(self.audit.established_emitters)
        self.detector.audit_quarantined=frozenset(self.audit.quarantined)
        return self.audit.apply(self.detector.step(frame,ts,frame_id,night))
