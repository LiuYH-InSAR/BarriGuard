"""Production bridge for reference coordinates, ID migration and body presence."""
from copy import deepcopy
import cv2 as cv
import numpy as np
from camera_motion.geometry import warp_boxes,iou
from .landmarks import matrix,inside,midpoint,capture,compare,migrate_profiles
from .presence import PresenceLedger
from .relocation import RelocationSession,committed_matches,correspondences
from .tracked import PulseCounter


class LampSpatialRuntime:
    def __init__(self,controller,saved=None):
        self.controller=controller;self.monitor=controller.monitor
        self.presence=PresenceLedger(self.monitor);self.session=None
        self.migrations=(saved or {}).get('adaptive',{}).get('migrations',[])[-64:]
        self._pipeline=None;self._shared_to_pipeline=None;self._last_check=None;self._last_map=None
        for lamp in self.monitor.lamps.values():
            # Completed absence windows remain historical evidence, not a
            # partially sampled day interval resumed after a process restart.
            lamp.presence.update(previous_valid=False,bucket=None,valid_s=0.,checks=0)

    def pause(self):self.presence.pause()

    def unique_site(self,key):
        lamp=self.monitor.lamps[key];box=lamp.spatial.get('anchor_body_box')
        if box is None:return False
        for other_key,other in self.monitor.lamps.items():
            if other_key==key or other.reference!=lamp.reference:continue
            other_box=other.spatial.get('anchor_body_box')
            if other_box is None:continue
            if iou(box,other_box)>.5:return False
        return True

    def _mapping(self,frame,frame_id,establish=False,result=None):
        c=self.controller;detector=c.pipeline.detector
        if self._pipeline is not detector:
            self._pipeline=detector;self._shared_to_pipeline=None
        if not detector.geometry_valid:return None
        applied=matrix(detector.applied_transform)
        if self._shared_to_pipeline is None:
            if not establish:return None
            latest=c.offset.latest
            if latest and latest.frame_id==str(frame_id) and latest.status=='aligned' and latest.matrix is not None:
                shared=matrix(latest.matrix)
            else:
                alignment=c.offset.references.align(frame)
                if alignment is None or not alignment.valid:return None
                shared=matrix(alignment.matrix)
            self._shared_to_pipeline=np.linalg.inv(applied)@shared
        transform=applied@self._shared_to_pipeline
        if (result or c.last_result or {}).get('motion')=='held_background_translation':
            for key,box in detector.frame_boxes.items():
                if key in detector.tracks:
                    shift=midpoint(box)-midpoint(detector.tracks[key].box)
                    translation=np.eye(3);translation[:2,2]=shift;transform=translation@transform;break
        return transform

    def on_frame(self,frame,stamp,source_ts,frame_id,night):
        c=self.controller;out=c.last_result or {};q=out.get('quality',{})
        if (not c._position_ok(stamp) or out.get('motion') not in ('aligned','held_background_translation')
            or not all(q.get(k) for k in ['fresh','continuous','clear'])):
            self.pause();return
        bound={c.bindings[r['id']]:r for r in out.get('tracks',[]) if r['id'] in c.bindings}
        needs_anchor=any(not l.spatial.get('anchor_body_box') for k,l in self.monitor.lamps.items() if k in bound)
        due=self._last_check is None or stamp-self._last_check>=self.monitor.params.presence_period_s
        transform=self._mapping(frame,frame_id,establish=due or needs_anchor)
        if transform is None:self.pause();return
        self._last_map=transform
        for key,lamp in self.monitor.lamps.items():
            record=bound.get(key)
            if (lamp.reference==self.monitor.reference and lamp.spatial.get('reference')==lamp.reference
                and lamp.spatial.get('anchor_body_box') and not(record and record.get('visible_recent'))):
                lamp.box=warp_boxes([lamp.spatial.get('anchor_box',lamp.spatial['anchor_body_box'])],transform)[0]
                lamp.body_box=warp_boxes([lamp.spatial['anchor_body_box']],transform)[0]
        if not night:
            for lamp in self.monitor.lamps.values():
                profile=lamp.spatial.get('profiles',{}).get('day')
                if lamp.presence.get('previous_valid') and profile and lamp.reference==self.monitor.reference:
                    check=compare(frame,profile,transform)
                    if check['verdict']=='present':self.presence.positive(lamp,stamp,'body_template')
                    elif check['verdict']!='absent':self.presence.pause(lamp)
        if not due and not needs_anchor:return
        self._last_check=stamp;inverse=np.linalg.inv(transform);light='night' if night else 'day'
        for key,lamp in self.monitor.lamps.items():
            if lamp.reference!=self.monitor.reference:continue
            record=bound.get(key)
            positive=bool(record and record.get('evidence_admitted') and record.get('position_observable')
                and record.get('confirmed') and record.get('visible_recent') and record.get('hits',0)>=3)
            if positive:
                body=record.get('body_box',record['box'])
                lamp.spatial.update(reference=lamp.reference,anchor_box=warp_boxes([record['box']],inverse)[0],
                    anchor_body_box=warp_boxes([body],inverse)[0],verified_at=stamp,position_verified=True)
                profiles=lamp.spatial.setdefault('profiles',{})
                if light not in profiles:
                    profile=capture(frame,body,inverse,lamp.reference,light,stamp)
                    if profile is not None:profiles[light]=profile
                if not night:self.presence.positive(lamp,stamp)
                continue
            if night:continue  # No night non-emission can prove a dropped body.
            profile=lamp.spatial.get('profiles',{}).get('day')
            if (not profile or not lamp.spatial.get('position_verified')
                or lamp.spatial.get('reference')!=lamp.reference or not self.unique_site(key)):
                self.presence.observe(lamp,stamp,source_ts,'unknown',dict(reason='no_verified_day_body_reference'));continue
            expected=warp_boxes([lamp.spatial['anchor_body_box']],transform)[0]
            if not inside(expected,frame.shape,2) or min(np.asarray(expected[2:])-expected[:2])<self.monitor.params.presence_min_body_px:
                self.presence.observe(lamp,stamp,source_ts,'unknown',dict(reason='body_too_small_or_outside'));continue
            # Two independent detector crop scales, with a low threshold: a
            # possible body vetoes absence even when regular discovery missed it.
            center=midpoint(expected);h,w=frame.shape[:2];regions=[]
            for size in (160,320):
                side=min(size,h,w);x=int(np.clip(center[0]-side/2,0,w-side));y=int(np.clip(center[1]-side/2,0,h-side))
                regions.append([x,y,x+side,y+side])
            detected=c.detector.predict(frame,regions,c.vision_params.input_imgsz,.15)
            veto=any(iou(expected,b)>.15 or np.linalg.norm(midpoint(expected)-midpoint(b))<max(4.,.6*min(np.asarray(expected[2:])-expected[:2])) for b in detected)
            evidence=compare(frame,profile,transform)
            if veto:
                # A weak detector hit is not sufficient to rewrite identity or
                # a frozen template, but it must block a missing declaration.
                evidence.update(verdict='unknown',reason='targeted_detector_body_candidate')
                if lamp.presence.get('status')=='missing':lamp.presence['status']='unknown'
            self.presence.observe(lamp,stamp,source_ts,evidence['verdict'],evidence)

    def observe_candidate(self,result,reference_frame,frame,stamp,source_ts,frame_id,night=False):
        if reference_frame is None:return
        if self.session is None or self.session.task!=result.task_id:
            self.session=RelocationSession(self.controller,result,reference_frame,night)
        self.session.observe(result,frame,stamp,source_ts,frame_id)

    def reassociation_offers(self,result,frame,frame_id,night):
        c=self.controller
        active={r['id'] for r in result['tracks'] if r.get('evidence_admitted')
            and r.get('position_observable') and r.get('visible_recent')}
        used={key for local,key in c.bindings.items() if local in active}
        lamps={k:l for k,l in self.monitor.lamps.items() if k not in used and l.reference==self.monitor.reference
            and l.spatial.get('migration') and l.spatial.get('reference')==l.reference}
        if not lamps:return {}
        q=result.get('quality',{})
        if not all(q.get(k) for k in ['fresh','continuous','clear']):return {}
        transform=self._mapping(frame,frame_id,establish=True,result=result)
        if transform is None:return {}
        candidates={r['id']:r for r in result['tracks'] if r['id'] not in c.bindings and r.get('confirmed')
            and r.get('evidence_admitted') and r.get('position_observable') and r.get('visible_recent')}
        matches=correspondences(lamps,candidates,transform,frame.shape,self.monitor.params.migration_error_px,
            frame,np.eye(3),'night' if night else 'day')
        # A lost optical sampler must not reserve a persistent site forever.
        # Release it only after unique independent current-view correspondence;
        # the new sampler will start fresh business evidence in consume().
        for key in matches:
            for local,persistent in list(c.bindings.items()):
                if persistent==key:
                    c.bindings.pop(local)
            self.monitor.accepted.pop(key,None)
            self.monitor.observations.pop(key,None)
        return {local:key for key,local in matches.items()}

    def near_migrated_site(self,record):
        box=record.get('body_box',record['box'])
        return any(l.reference==self.monitor.reference and l.spatial.get('migration')
            and (iou(box,l.body_box or l.box)>.15 or np.linalg.norm(midpoint(box)-midpoint(l.body_box or l.box))<6.)
            for k,l in self.monitor.lamps.items())

    def feedback_ready(self,result,stamp):
        eligible=any(l.reference==result.old_reference and l.spatial.get('reference')==result.old_reference
            for l in self.monitor.lamps.values())
        if not eligible:return True
        if self.session is None or self.session.task!=result.task_id:return False
        if self.session.last_stamp is not None and stamp-self.session.last_stamp>2.:return False
        return self.session.ready(result)

    def feedback(self,result,stamp):
        if (self.session is None or self.session.task!=result.task_id or self.session.last_stamp is None
            or not 0<=stamp-self.session.last_stamp<=2.):return None
        return self.session.feedback(result)

    def apply(self,result):
        monitor=self.monitor;c=self.controller
        if result.camera_id!=monitor.camera_id or result.status!='committed' or not result.new_reference:
            raise ValueError('Only committed shared transitions from this camera may migrate lamps')
        token=(result.task_id,result.version)
        if token in monitor._applied or monitor.reference==result.new_reference:
            # A previous in-memory application may have failed its disk write.
            # Do not let the shared owner acknowledge it before durable save.
            c.checkpoint(result.checked_at)
            return
        if monitor.reference!=result.old_reference and monitor.lamps:raise ValueError('Lamp reference cannot skip a transition')
        transform=matrix(result.matrix) if result.matrix is not None else None
        matches=committed_matches(result,monitor.lamps,monitor.params) if transform is not None else {}
        support=np.float32([midpoint(monitor.lamps[k].spatial['anchor_body_box']) for k in matches])
        hull=cv.convexHull(support) if len(support)>=4 else None
        supported=bool(hull is not None and cv.contourArea(hull)>=100
            and result.details.get('group_errors',{}).get('night_lamp',{}).get('max_px',float('inf'))<=monitor.params.migration_error_px)
        if monitor.bucket is not None:monitor._finish(result.checked_at)
        migrated=[];unknown=[]
        for key,lamp in monitor.lamps.items():
            lamp.observable=False
            spatial=lamp.spatial
            if (lamp.reference!=result.old_reference or transform is None
                or spatial.get('reference')!=result.old_reference or not spatial.get('anchor_body_box')):
                unknown.append(key);self.presence.pause(lamp);continue
            old=dict(box=deepcopy(spatial.get('anchor_box',lamp.box)),body=deepcopy(spatial['anchor_body_box']))
            predicted=warp_boxes([old['box'],old['body']],transform)
            proof=matches.get(key)
            point=tuple(map(float,midpoint(old['body'])))
            prediction_verified=bool(supported and cv.pointPolygonTest(hull,point,True)>=-monitor.params.migration_error_px
                and inside(predicted[1],result.shape,2))
            lamp.box=list(proof['box']) if proof else predicted[0]
            lamp.body_box=list(proof['body_box']) if proof else predicted[1]
            lamp.reference=result.new_reference
            spatial.update(reference=lamp.reference,anchor_box=lamp.box[:],anchor_body_box=lamp.body_box[:],
                position_verified=bool(proof or prediction_verified),migration=dict(task_id=result.task_id,version=result.version,
                    old_reference=result.old_reference,new_reference=result.new_reference,
                    old_position=old,predicted=predicted,independent_match=proof,at=result.checked_at))
            migrate_profiles(spatial.get('profiles',{}),transform,result.new_reference)
            self.presence.migrate(lamp);migrated.append(key)
        session=self.session
        monitor.reference=result.new_reference;monitor._applied.add(token)
        monitor.bucket=None;monitor.observations={};monitor.accepted={};monitor.candidates={}
        c._view=result.new_reference;c._geometry_blocked_at=None;c._new_pipeline()
        c.search_counter=PulseCounter();c._ledger_bucket=None
        if session is not None and session.task==result.task_id and matches and not session.invalid:
            # The candidate pipeline is already in NEW-reference coordinates.
            # Reuse its optical identities, but never its counts as business
            # evidence: all observation counters above are fresh.
            c.pipeline=session.pipeline
            c.bindings={proof['local_id']:key for key,proof in matches.items()
                if 'local_id' in proof and proof['local_id'] in session.pipeline.detector.tracks
                and session.candidates.get(proof['candidate_id'],{}).get('local_id')==proof['local_id']}
            c._mode='night' if session.night else 'day'
        self._shared_to_pipeline=None;self._last_check=None;self.session=None
        self.migrations.append(dict(task_id=result.task_id,version=result.version,at=result.checked_at,
            old_reference=result.old_reference,new_reference=result.new_reference,matrix=result.matrix,
            migrated_ids=migrated,independently_matched_ids=list(matches),unmapped_ids=unknown))
        self.migrations=self.migrations[-64:];c.checkpoint(result.checked_at)

    def report_status(self,lamp,stamp):
        if (not lamp.spatial.get('position_verified') or not self.controller._position_ok(stamp)
            or not self.unique_site(lamp.lamp_id)):return None
        status=self.presence.status(lamp,stamp)
        if status=='missing':
            # An unresolved but freshly detected flashing source at this site
            # vetoes physical absence while identity is being reconciled.
            box=lamp.body_box or lamp.box
            for r in (self.controller.last_result or {}).get('tracks',[]):
                if (r.get('raw_confirmed') and r.get('visible_recent')
                    and iou(box,r.get('body_box',r['box']))>.2):return None
        return status
