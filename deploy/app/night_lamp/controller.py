"""Explicit integration entry point for a shared OffsetModule and injected detector.

No RTSP, HTTP callback or model download occurs here. The caller provides source
frames, schedule state and reviewed visibility. Only the current lamp observation scheme is supported; visibility thresholds
remain subject to validation with real video.
"""
from datetime import datetime, timedelta, timezone
import cv2 as cv
from .tracked import LampMonitor
from .quality import StreamQuality, ObservationQuality
from .archive import LampArchive

HK=timezone(timedelta(hours=8))


class LampController:
    def __init__(self, offset, detector, root, params=None, visibility=None, *, managed=False):
        self.managed=managed
        self.offset,self.detector=offset,detector
        self.visibility=visibility
        self.archive=LampArchive(root,offset.camera_id)
        self.monitor=LampMonitor(offset.camera_id,offset.references.view,params)
        self.quality=StreamQuality(max_gap_s=self.monitor.params.max_sample_gap_s)
        self._last_precheck=None
        self._last_saved_bucket=None
        self._detector_generation=None
        saved=self.archive.load_current()
        if saved:
            if not managed:self.offset.restore(saved['offset'],saved['saved_at'])
            self.monitor=LampMonitor.restore(saved['monitor'],params)
            if self.monitor.reference!=self.offset.references.view and not managed:
                raise ValueError('Lamp and background reference checkpoint disagree')
            self._last_saved_bucket=(self.monitor.results[-1]['bucket'] if self.monitor.results else None)

    def tick(self,wall_ts,is_night):
        self.monitor.tick(wall_ts,is_night)
        if self.monitor.results and self.monitor.results[-1]['bucket']!=self._last_saved_bucket:
            self.checkpoint(wall_ts)
            self._last_saved_bucket=self.monitor.results[-1]['bucket']

    def on_frame(self,frame,wall_ts,source_ts,frame_id,is_night, *, discovery_allowed=True):
        self.tick(wall_ts,is_night)
        p=self.monitor.params
        next_bucket=(int(wall_ts//p.cadence_s)+1)*p.cadence_s
        due=(not is_night or 0<next_bucket-wall_ts<=300)
        if not self.managed and due and (self._last_precheck is None or wall_ts-self._last_precheck>=self.offset.params.timing.period_s):
            self.offset.inspect(frame,wall_ts,frame_id)
            self._last_precheck=wall_ts
        # A startup has no old position map. Establish a reference even if it is
        # already nighttime; first quality sample does not prove alignment.
        if not self.managed and self.offset.references.anchor is None:
            self.offset.inspect(frame,wall_ts,frame_id)
            self.monitor.reference=self.offset.references.view
        if not self.monitor.lamps and not self.monitor.candidates:
            self.monitor.reference=self.offset.references.view
        q=self.quality.observe(frame,source_ts,frame_id)
        latest=self.offset.latest
        position_ok=bool(latest and latest.status=='aligned' and latest.matrix is not None
                         and wall_ts-latest.checked_at<=p.cadence_s+300
                         and self.offset.pending is None and self.monitor.reference==self.offset.references.view)
        boxes=self.detector.detect(frame) if q['fresh'] and q['clear'] and (is_night or discovery_allowed) else []
        self.latest_boxes=boxes
        generation=getattr(self.detector,'generation',None)
        independent=generation is None or generation!=self._detector_generation
        self._detector_generation=generation
        if hasattr(self.visibility,'update'):
            self.visibility.update(frame,boxes,self.monitor.lamps)
        usable={}
        for key,lamp in self.monitor.lamps.items():
            visible=self.visibility(frame,lamp) if self.visibility else None
            visible=visible if discovery_allowed or is_night else None
            usable[key]=ObservationQuality(q['fresh'],q['continuous'],q['clear'],position_ok,visible).usable
        self.monitor.on_frame(frame,wall_ts,source_ts,frame_id,boxes,usable=usable,
            is_night=is_night,discovery_confirmed=independent,discovery_usable=bool(q['fresh'] and q['clear'] and position_ok and (is_night or discovery_allowed)))

    def apply_offset(self,result,associations):
        changed=self.monitor.migrate(result,associations)
        if not self.managed:self.offset.acknowledge('night_lamp',result.task_id,result.version)
        self.checkpoint(result.checked_at)
        return changed

    def _snapshot(self,stamp):
        return dict(schema=1,saved_at=stamp,offset=self.offset.dump(),monitor=self.monitor.dump())

    def _images(self):
        images={}
        for index,entry in enumerate(self.offset.references.appearances):
            image=self.offset.store.read_image(entry['image'])
            if image is None:raise ValueError('Missing reference during snapshot')
            ok,encoded=cv.imencode('.png',image)
            if not ok:raise OSError('Reference encoding failed')
            images[f'reference_{index}.png']=encoded.tobytes()
        return images

    def checkpoint(self,stamp):
        self.archive.save_current(self._snapshot(stamp),self._images())

    def report_identity_ambiguous(self,key):
        return False

    def report_lamp(self,key,lamp,stamp,cutoff):
        observation=lamp.history[-1] if lamp.history and lamp.history[-1]['bucket']>=cutoff else None
        ambiguous=self.report_identity_ambiguous(key)
        same_view=lamp.reference==self.monitor.reference
        pending=self.offset.pending is not None
        # Report the latest window's health sequence, including the final
        # window before night freeze. Report-send time (07:00) is not a new
        # observation bucket. Real later incomplete windows still stay gray.
        evaluated_at=observation['bucket']+self.monitor.params.cadence_s if observation else stamp
        status=self.monitor.status(key,evaluated_at,require_observable=False) if observation else 'unobservable'
        basis='latest_window'
        spatial=getattr(self,'spatial',None)
        physical=spatial.report_status(lamp,stamp) if spatial is not None else None
        if physical=='missing' and same_view and not pending:
            status='missing';basis='confirmed_day_body_absence'
        elif ambiguous:
            status='unobservable';basis='identity_unverified'
        elif not same_view or pending:
            status='unobservable';basis='reference_unverified'
        elif status=='normal' and not(observation and observation['status']=='normal'
              and observation['pulses']>=self.monitor.params.normal_pulses
              and observation.get('reference')==lamp.reference):
            status='unobservable';basis='normal_evidence_incomplete'
        elif status=='normal' and not lamp.observable:
            # A headlight/occlusion in the last image cannot undo this window's
            # already completed, admitted health evidence. Visibility stays
            # unknown and sampling stays paused; no old window is borrowed.
            basis='completed_valid_window_despite_current_visibility'
        if status=='not_flashing':
            end=observation['bucket']+self.monitor.params.cadence_s
            rows={r['bucket']:r for r in lamp.history}
            expected=[end-self.monitor.params.cadence_s*i for i in range(1,self.monitor.params.failure_windows+1)]
            if not all(b in rows and rows[b].get('reference')==lamp.reference
                and rows[b].get('valid_s',0)>=self.monitor.params.effective_s
                and rows[b].get('pulses')==0 and rows[b].get('status')=='no_flash' for b in expected):
                status='unobservable';basis='failure_evidence_incomplete'
        return dict(lamp_id=key,status=status,last_flash=lamp.last_flash,last_seen=lamp.last_seen,
            box=lamp.box,body_box=lamp.body_box,aliases=lamp.aliases,observation=observation,
            current_observability='observable' if lamp.observable and not ambiguous and same_view and not pending else 'unobservable',
            identity_ambiguous=ambiguous,health_basis=basis,
            position_current=bool(same_view and not pending),
            health_accepted_at=observation.get('accepted_at') if status=='normal' and observation else None,
            physical_presence=dict(status=lamp.presence.get('status','unknown'),
                confirmed_at=lamp.presence.get('confirmed_at'),last_checked_at=lamp.presence.get('last_checked_at'),
                last_present_at=lamp.presence.get('last_present_at'),evidence=lamp.presence.get('last_evidence'),
                valid_windows=lamp.presence.get('history',[])),
            migration=lamp.spatial.get('migration'),
            reference_position=lamp.spatial.get('anchor_body_box'),
            health_window_ended_at=observation['ended'] if status=='normal' and observation else None)

    def finalize(self,stamp):
        self.monitor.tick(stamp,False)
        self.checkpoint(stamp)
        local=datetime.fromtimestamp(stamp,HK)
        cutoff=(local-timedelta(hours=12)).replace(hour=12,minute=0,second=0,microsecond=0).timestamp()
        report=dict(self._snapshot(stamp),night_key=(datetime.fromtimestamp(stamp,HK)-timedelta(hours=12)).date().isoformat(),
            lamps=[self.report_lamp(k,v,stamp,cutoff) for k,v in self.monitor.lamps.items()])
        path=self.archive.archive(report,stamp,self._images())
        self.archive.sweep(stamp)
        return report,path
