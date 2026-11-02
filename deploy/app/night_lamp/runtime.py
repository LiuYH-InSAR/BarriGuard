"""v9 vision -> current-window evidence -> production state/archive.

The vision tracker owns calibration and optical identities. LampMonitor alone
owns business windows. No lifetime tracker counts/durations enter its counters.
Shared camera references are read-only here; cross-view ID migration is deferred.
"""
from dataclasses import replace, asdict
from .adaptive import LampPipelineV9, V6Params
from .adaptive.adaptive_v8 import ContentEvidenceQuality
from .adaptive.adaptive_prototype import center, iou
from .controller import LampController
from .tracked import PulseCounter
import numpy as np


class AdaptiveLampController(LampController):
    def __init__(self, offset, backend, root, params=None, *, detector_spec=None):
        super().__init__(offset, backend, root, params, managed=True)
        self.vision_params=V6Params(max_gap_s=self.monitor.params.max_sample_gap_s,
                                   night_pulses=self.monitor.params.normal_pulses)
        if detector_spec is not None:
            self.vision_params=replace(self.vision_params,
                input_imgsz=detector_spec.imgsz, coarse_confidence=detector_spec.confidence,
                fine_confidence=max(.65,detector_spec.confidence),
                detector_period_s=max(.2,detector_spec.detect_interval_s))
        self.pipeline=LampPipelineV9(backend,self.vision_params)
        self.signal_params=replace(self.monitor.params,on_delta=.65,off_delta=.30)
        self.quality=ContentEvidenceQuality(max_gap_s=self.monitor.params.max_sample_gap_s)
        self.bindings={};self.pending={};self.epochs={};self._ledger_bucket=None
        self.search_counter=PulseCounter()
        self.last_result=None;self._mode=None;self._frame=None;self._source_ts=None
        self._geometry_blocked_at=None
        self.visibility_mode=getattr(detector_spec,'visibility','background')
        self._view=self.monitor.reference
        self._last_checkpoint=None
        saved=self.archive.load_current()
        self.prior_policy_history=(saved or {}).get('prior_policy_history',{})
        if saved and (saved.get('adaptive',{}).get('version')!='v9c'
                      or saved.get('adaptive',{}).get('evidence_policy')!='pulse_first_window_v4'
                      or saved.get('observation_policy')!=asdict(self.monitor.params)):
            # Old policy decisions remain reviewable, but cannot establish six
            # validated v9 zero-flash windows after an upgrade/config change.
            self.prior_policy_history=dict(saved_at=saved['saved_at'],monitor=saved['monitor'])
            for lamp in self.monitor.lamps.values():lamp.history=[]
            self.monitor.results=[];self._last_saved_bucket=None
        # A restored coordinate/history is a prior, never visibility evidence.
        for lamp in self.monitor.lamps.values():lamp.observable=False

    def _new_pipeline(self):
        self.pipeline=LampPipelineV9(self.detector,self.vision_params)
        self.bindings.clear();self.pending.clear();self.epochs.clear()
        self.last_result=None

    def pause(self, source_ts):
        for counter in [self.search_counter,*self.monitor.observations.values(),*self.pending.values()]:
            counter.sample(source_ts,0.,False,self.signal_params)
            counter.previous_valid=False;counter.phase='unknown';counter.rise=None
        detector=self.pipeline.detector
        for track in detector.tracks.values():
            track.pulses.sample(source_ts,0.,False,detector.flash_cfg)
            track.pulses.previous_valid=False;track.pulses.phase='unknown';track.pulses.rise=None
            track.flash.previous_valid=False
        detector.nested.pairs.clear()

    def reset_reference(self,result):
        monitor=self.monitor
        if result.camera_id!=monitor.camera_id or result.status!='committed' or not result.new_reference:
            raise ValueError('Only committed shared transitions from this camera may reset lamps')
        if monitor.reference==result.new_reference:return
        if monitor.reference!=result.old_reference and monitor.lamps:
            raise ValueError('Lamp target reference cannot skip a scene transition')
        if monitor.bucket is not None:monitor._finish(result.checked_at)
        # Preserve old IDs and history as unknown; do not predict/associate them
        # into the new view while cross-view migration is deferred.
        for lamp in monitor.lamps.values():lamp.observable=False
        monitor.reference=result.new_reference
        monitor._applied.add((result.task_id,result.version))
        monitor.bucket=None;monitor.observations={};monitor.accepted={};monitor.candidates={}
        self._view=result.new_reference;self._geometry_blocked_at=None;self._new_pipeline()
        self.checkpoint(result.checked_at)

    def _position_ok(self,wall_ts):
        latest=self.offset.latest
        return bool(latest and latest.status=='aligned' and latest.matrix is not None
            and 0<=wall_ts-latest.checked_at<=self.monitor.params.cadence_s+300
            and self.offset.pending is None and self.monitor.reference==self.offset.references.view)

    def on_frame(self,frame,wall_ts,source_ts,frame_id,is_night,*,discovery_allowed=True):
        if not np.isfinite(source_ts) or not np.isfinite(wall_ts):raise ValueError('Nonfinite frame timestamp')
        self.tick(wall_ts,is_night)
        if frame_id is None or frame_id==self._frame or (self._source_ts is not None and source_ts<=self._source_ts):return
        self._frame,self._source_ts=frame_id,source_ts
        mode='night' if is_night else 'day'
        if self._mode is not None and mode!=self._mode:
            # Day/night photometry cannot share a frozen dark template.
            self.pause(source_ts);self._new_pipeline()
            for lamp in self.monitor.lamps.values():lamp.observable=False
        self._mode=mode
        if not self.monitor.lamps and self._view!=self.offset.references.view:
            self.monitor.reference=self._view=self.offset.references.view
            self._new_pipeline()
        q=self.quality.observe(frame,source_ts,frame_id)
        allowed=bool(self._position_ok(wall_ts) and q['fresh'] and q['clear']
                     and (is_night and self.monitor.bucket is not None or discovery_allowed and not is_night))
        if allowed and self._geometry_blocked_at is not None:
            if self.offset.latest.checked_at>self._geometry_blocked_at:
                self._new_pipeline();self._geometry_blocked_at=None
            else:allowed=False
        if not allowed:
            self.pause(source_ts)
            return
        # Share stateless pixel work, while keeping each quality stream's own
        # continuity/freeze history (the inner stream may start later).
        quality=self.pipeline.detector.quality
        quality.prepared_metrics=getattr(self.quality,'last_metrics',None)
        try:result=self.pipeline.step(frame,source_ts,frame_id,night=is_night)
        finally:quality.prepared_metrics=None
        self.consume(result,wall_ts,source_ts,is_night)
        if self._last_checkpoint is None or wall_ts-self._last_checkpoint>=60:
            self.checkpoint(wall_ts);self._last_checkpoint=wall_ts

    def _match_known(self,record,used):
        box=record.get('body_box',record['box']);offers=[]
        for key,lamp in self.monitor.lamps.items():
            if key in used or lamp.reference!=self.monitor.reference:continue
            old=lamp.body_box or lamp.box
            distance=float(np.linalg.norm(center(box)-center(old)))
            if iou(old,box)>=.5 and distance<=max(4.,.35*min(old[2]-old[0],old[3]-old[1])):
                offers.append((distance,key))
        # Nearby/overlapping alternatives must not silently exchange IDs.
        if not offers:return None
        offers.sort()
        if len(offers)>1 and offers[1][0]-offers[0][0]<3.:return None
        return offers[0][1]

    def consume(self,result,wall_ts,source_ts,is_night):
        """Admit per-frame signals, not the tracker's accumulated pulse/time fields."""
        monitor=self.monitor
        self.tick(wall_ts,is_night)
        if monitor.bucket!=self._ledger_bucket:
            self.pending={};self.search_counter=PulseCounter();self._ledger_bucket=monitor.bucket
        if result.get('duplicate_ignored'):return
        self.last_result=result
        # A local geometry reset invalidates every within-view binding. The
        # shared owner still decides whether a new scene/reference is committed.
        if any(e['event']=='geometry_reset' for e in result.get('events',[])):
            for lamp in monitor.lamps.values():lamp.observable=False
            monitor.observations={key:PulseCounter() for key in monitor.observations}
            monitor.accepted.clear()
            self.search_counter=PulseCounter()
            self.pending.clear();self.bindings.clear();self.epochs.clear()
            self._geometry_blocked_at=wall_ts
            self.pause(source_ts)
            return
        for event in result.get('events',[]):
            if event['event'] in ('independent_source_reset','emitter_recenter'):
                # A newly selected/independently separated source is not just
                # a photometric recalibration of the accepted physical light.
                monitor.accepted.pop(self.bindings.get(event['id']),None)
            if event['event'] not in ('nested_identity_resolved','identity_merge','same_source_resolved'):continue
            if event['event']=='same_source_resolved' and not event.get('consumer_verified'):continue
            retained=event['retained']
            aliases=event.get('aliases',[event.get('alias')])
            preserved=(event.get('evidence_policy')=='unchanged_trusted_sampler' and not event.get('observation_reset')
                and retained in self.bindings and all(self.bindings.get(a,self.bindings[retained])==self.bindings[retained] for a in aliases))
            if preserved:
                for local in aliases:self.pending.pop(local,None)
                continue
            ids=list(dict.fromkeys(self.bindings[k] for k in [retained,*aliases] if k in self.bindings))
            if ids:
                key=min(ids,key=lambda k:monitor.lamps[k].created_at)
                self.bindings[retained]=key
                for alias_key in ids:
                    monitor.accepted.pop(alias_key,None)
                    # No previous ambiguous-region evidence may support failure.
                    monitor.lamps[alias_key].history=[]
                    if alias_key!=key:
                        monitor.lamps[key].aliases.extend([alias_key,*monitor.lamps[alias_key].aliases])
                        monitor.lamps.pop(alias_key);monitor.observations.pop(alias_key,None)
                for local,persistent in list(self.bindings.items()):
                    if persistent in ids:self.bindings[local]=key
                monitor.observations[key]=PulseCounter()
            for local in [retained,*aliases]:self.pending.pop(local,None)
        resets={e.get('id',e.get('retained')) for e in result.get('events',[])
                if e['event'] in ('nested_identity_resolved','identity_merge','same_source_resolved','emitter_recenter','signal_support_changed','signal_reference_changed','independent_source_reset')
                and (e['event']!='same_source_resolved' or e.get('consumer_verified'))
                and e.get('evidence_policy')!='unchanged_trusted_sampler'}
        seen=set();live={r['id'] for r in result['tracks']}
        for local in list(self.pending):
            if local not in live:self.pending.pop(local)
        for local in list(self.bindings):
            if local not in live:self.bindings.pop(local)
        used=set(self.bindings.values())
        for record in result['tracks']:
            local=record['id'];admitted=record.get('evidence_admitted',False)
            epoch=(record.get('observation_epoch',0),record.get('discarded_valid_s',0.))
            reset=local in resets or (local in self.epochs and self.epochs[local]!=epoch)
            self.epochs[local]=epoch
            counter=self.pending.setdefault(local,PulseCounter())
            key=self.bindings.get(local)
            if key is None and admitted and record.get('hits',0)>=self.vision_params.promotion_hits:
                key=self._match_known(record,used)
                if key is not None:
                    self.bindings[local]=key;used.add(key)
                    # A persistent identity may survive a lost local tracker,
                    # but this new sampler/calibration starts fresh evidence.
                    if monitor.bucket is not None and key not in monitor.accepted:monitor.observations[key]=PulseCounter()
            if key is None and admitted and record.get('confirmed'):
                key=monitor.add_known(record['box'],wall_ts,'night_flashes' if is_night else 'day_detection')
                self.bindings[local]=key;used.add(key)
                if monitor.bucket is not None:
                    # Only this bucket's actually sampled candidate evidence;
                    # a late discovery never receives earlier elapsed time.
                    monitor.observations[key]=counter
            if key is not None:
                lamp=monitor.lamps[key];lamp.box=list(record['box']);lamp.body_box=list(record.get('body_box',record['box']))
                lamp.observable=bool(admitted and record['position_observable'])
                if record.get('visible_recent'):lamp.last_seen=wall_ts
                seen.add(key)
                counter=monitor.observations.setdefault(key,PulseCounter()) if monitor.bucket is not None else counter
                if monitor.bucket is not None and key in monitor.accepted:
                    self.pending[local]=counter
                    continue
            if reset or (not admitted and key is None):
                counter=PulseCounter()
                self.pending[local]=counter
                if key is not None and monitor.bucket is not None:monitor.observations[key]=counter
            elif not admitted:
                # Unknown/ambiguous time pauses a known sampler. It neither
                # accrues no-flash evidence nor erases earlier admitted time.
                counter.sample(source_ts,0.,False,self.signal_params)
            observation=record.get('flash_signal',{})
            usable=bool(admitted and record['position_observable'] and observation.get('valid')
                        and result['quality']['fresh'] and result['quality']['continuous'] and result['quality']['clear'])
            if self.visibility_mode=='detected':usable=usable and bool(record.get('visible_recent'))
            if is_night and monitor.bucket is not None:
                counter.sample(source_ts,observation.get('signal',0.),usable and not reset,self.signal_params)
                if key is not None and counter.last_flash==source_ts:monitor.lamps[key].last_flash=wall_ts
                if key is not None and usable and not reset:monitor.accept_normal(key,wall_ts)
            self.pending[local]=counter
        for key,lamp in monitor.lamps.items():
            if key not in seen:
                lamp.observable=False
                if key in monitor.observations:monitor.observations[key].sample(source_ts,0.,False,self.signal_params)
        if is_night and monitor.bucket is not None:
            q=result['quality']
            self.search_counter.sample(source_ts,0.,bool(q['fresh'] and q['continuous'] and q['clear']
                and result.get('motion','aligned') in ('aligned','held_background_translation')),self.signal_params)
            keys=[k for k,lamp in monitor.lamps.items() if lamp.reference==monitor.reference]
            complete=all(k in monitor.accepted and not self.report_identity_ambiguous(k) for k in keys)
            waiting=any(r['id'] not in self.bindings and r.get('evidence_admitted')
                and r['position_observable'] and r.get('hits',0)>=self.vision_params.promotion_hits
                and r.get('confidence',0)>=self.vision_params.candidate_confidence
                and self.pending[r['id']].valid_s<monitor.params.effective_s for r in result['tracks'])
            waiting=waiting or any(r['id'] not in self.bindings and not r.get('evidence_admitted')
                and r.get('raw_confirmed') and r.get('raw_pulses',0)>=monitor.params.normal_pulses
                for r in result['tracks'])
            if complete and not waiting and self.search_counter.valid_s>=monitor.params.effective_s:
                monitor._finish(wall_ts);monitor.bucket=None
                self.tick(wall_ts,True)  # Publish completion; do not reopen the finished bucket.

    def _snapshot(self,stamp):
        state=super()._snapshot(stamp)
        state['observation_policy']=asdict(self.monitor.params)
        state['prior_policy_history']=self.prior_policy_history
        state['adaptive']=dict(version='v9c',evidence_policy='pulse_first_window_v4',cross_view_migration=False,
            bindings={str(k):v for k,v in self.bindings.items()},
            tracks=(self.last_result or {}).get('tracks',[]),
            observation_audit=(self.last_result or {}).get('observation_audit',{}),
            partial_windows_restored=False)
        state['active_window']=dict(bucket=self.monitor.bucket,
            accepted=self.monitor.accepted,
            observations={k:asdict(v) for k,v in self.monitor.observations.items()},
            search_evidence=asdict(self.search_counter),
            restore_policy='diagnostics_only; restart starts fresh evidence')
        state['geometry']=dict(reference=self.monitor.reference,
            lamps={k:dict(box=v.box,body_box=v.body_box,center=center(v.box).tolist(),
                          reference=v.reference,aliases=v.aliases) for k,v in self.monitor.lamps.items()})
        return state

    def report_identity_ambiguous(self,key):
        if self.last_result is None:
            return not self.monitor.lamps[key].observable
        records=[r for r in self.last_result.get('tracks',[]) if self.bindings.get(r['id'])==key]
        return not records or any(not r.get('evidence_admitted',False) for r in records)
