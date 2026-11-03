"""Camera-scoped owner of references, checks, migration dispatch and checkpoints."""
from dataclasses import replace
from pathlib import Path
import logging
import json
import numpy as np
from .params import MotionParams
from .service import OffsetModule
from .state import CameraState

LOG=logging.getLogger(__name__)
PROJECT=Path(__file__).resolve().parents[3]


class CameraOffsetRuntime:
    def __init__(self,camera_id,spec,clock,offset=None):
        self.spec,self.clock=spec,clock
        params=MotionParams()
        params=replace(params,timing=replace(params.timing,period_s=spec.period_s,fast_s=spec.fast_s,
            rebuild_periods=spec.stable_s/spec.period_s),reference=replace(params.reference,refresh_s=spec.refresh_s),
            storage=replace(params.storage,root=Path(spec.root) if spec.root else PROJECT/'output/camera_motion',
                            retention_s=spec.reference_ttl_s))
        self.offset=offset or OffsetModule(camera_id,params)
        self.consumers={}
        self._next_check=None
        self._last_frame=None
        self._last_save=None
        self._candidate_frame=None
        self._candidate_frame_id=None
        self._stable_since=None
        self._stable_previous=None
        self._failed_consumers=set()
        # Only this owner restores common state. Consumers restore business state only.
        checkpoint=self.offset.store.root/'current.json'
        saved=json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.is_file() else None
        if saved and 'offset' in saved:
            try:
                self.offset.restore(saved['offset'],clock.wall().timestamp())

            except (OSError,ValueError,KeyError,TypeError):
                LOG.exception('Shared offset checkpoint rejected for %s',camera_id)
                raise  # Do not silently pair old business coordinates with a new view.

    def attach(self,name,consumer):
        self.consumers[name]=consumer
        # Reapply only missing committed transitions, in order. Consumer methods are idempotent.
        for result in self.offset.commits:
            self._apply(name,consumer,result)

    def _apply(self,name,consumer,result):
        key=(name,result.task_id)
        if key in self.offset._acks:return
        reference=getattr(consumer,'offset_reference',None)
        if isinstance(reference,str) and reference not in (result.old_reference,result.new_reference):
            # A fresh consumer already on a later view must not replay older transforms.
            return
        try:
            consumer.apply_shared_offset(result)
            self.offset.acknowledge(name,result.task_id,result.version)
            self._failed_consumers.discard(name)
        except Exception:
            self._failed_consumers.add(name)
            LOG.exception('Offset consumer %s failed to apply %s',name,result.task_id)

    def process(self,frame,stamp,mono,frame_id,light_hint=None,source_ts=None):
        if frame_id is None or frame_id==self._last_frame:return self.offset.latest
        self._last_frame=frame_id
        source_ts=stamp if source_ts is None else source_ts
        if self.offset.pending is not None:
            self._observe_candidates(frame,stamp,source_ts,frame_id,light_hint)
        for result in self.offset.commits:
            for name,consumer in self.consumers.items():self._apply(name,consumer,result)
        if self._next_check is not None and mono<self._next_check:return self.offset.latest
        offset=self.offset
        if offset.pending is not None:
            if offset.pending.details.get('accepted'):
                result=offset.commit()
            else:
                waiting=[]
                for name,consumer in self.consumers.items():
                    ready=getattr(consumer,'offset_feedback_ready',None)
                    if callable(ready) and ready(offset.pending,stamp) is False:waiting.append(name)
                # Waiting is not an optimization iteration. Continue feeding
                # independent frames until ready, with a bounded final fallback.
                if waiting and stamp+self.spec.fast_s<offset.pending.details['deadline']:
                    self._next_check=mono+self.spec.fast_s
                    return offset.pending
                for consumer in self.consumers.values():
                    fn=getattr(consumer,'offset_feedback',None)
                    if fn:
                        try:
                            feedback=fn(offset.pending,self._candidate_frame)
                            if feedback is not None:offset.submit(feedback)
                        except Exception:LOG.exception('Rejected migration feedback')
                refined=offset.refine(stamp,max_iterations=self.spec.max_iterations,error_px=self.spec.error_px)
                if refined.details.get('accepted'):result=offset.commit()
                elif refined.status=='failed':result=offset.commit_unmapped('mapping_unavailable')
                else:
                    self._next_check=mono+self.spec.fast_s
                    return refined
            self._failed_consumers.clear()
            # Persist scene transition first; unacknowledged consumers can replay it after restart.
            self.save(stamp)
            for name,consumer in self.consumers.items():self._apply(name,consumer,result)
            self.save(stamp)
            self._stable_since=self._stable_previous=None
            self._next_check=mono+self.spec.fast_s
            return result
        expired=bool(offset.references.appearances) and all(
            stamp-e['created']>=self.spec.reference_ttl_s for e in offset.references.appearances)
        if expired:
            from .contracts import MotionResult
            offset.state.state=CameraState.RECOVER
            offset._version+=1
            result=MotionResult(offset.camera_id,'',str(frame_id),stamp,offset.references.view,
                offset.references.view,offset._version,'unavailable',None,tuple(frame.shape[:2]),0.,
                {'reason':'reference_expired'})
            offset.latest=result
        else:
            result=offset.inspect(frame,stamp,frame_id,reference_light=light_hint)
        state=offset.state.state
        if result.status=='returned':
            quality=offset.vision.quality(frame)
            if result.matrix is not None and quality['ok']:
                offset.references.add_appearance(frame,stamp,quality,np.asarray(result.matrix))
        if state in (CameraState.REBUILD,CameraState.RECOVER):
            quality=offset.vision.quality(frame)
            stable=False
            if quality['ok']:
                if self._stable_previous is None:
                    self._stable_since=mono
                else:
                    recent=offset.vision.align(self._stable_previous,frame)
                    if not recent.valid or recent.displacement>offset.state.tolerance:
                        self._stable_since=mono
                self._stable_previous=frame.copy()
                stable=self._stable_since is not None and mono-self._stable_since>=self.spec.stable_s
            else:
                self._stable_since=self._stable_previous=None
            if stable:
                offset.begin_migration(frame,stamp,frame_id,self.spec.feedback_timeout_s,source_ts=source_ts)
                self._candidate_frame=frame.copy()
                self._candidate_frame_id=frame_id
                self._observe_candidates(frame,stamp,source_ts,frame_id,light_hint)
        else:
            self._stable_since=self._stable_previous=None
        # A newly created/restored reference has not proved alignment yet.
        # Confirm it on the fast cadence instead of spending a normal period
        # with every business consumer unable to observe.
        self._next_check=mono+(self.spec.period_s if state==CameraState.NORMAL and result.status=='aligned' else self.spec.fast_s)
        if self._last_save is None or stamp-self._last_save>=self.spec.period_s:self.save(stamp)
        return offset.pending or result

    def _observe_candidates(self,frame,stamp,source_ts,frame_id,light_hint):
        for name,consumer in self.consumers.items():
            observe=getattr(consumer,'observe_offset_candidate',None)
            if not callable(observe):continue
            try:
                observe(self.offset.pending,self._candidate_frame,frame,stamp,source_ts,
                        frame_id,night=light_hint=='night')
            except Exception:
                LOG.exception('Candidate observation failed for %s; no migration evidence accepted',name)

    def request_check(self,mono):
        """A consumer may request a check; only this owner interprets motion."""
        due=mono+self.spec.fast_s
        self._next_check=due if self._next_check is None else min(self._next_check,due)

    def alignment(self):
        result=self.offset.latest
        valid=(self.offset.pending is None and result is not None
               and result.status=='aligned' and result.matrix is not None and not self._failed_consumers)
        matrix=np.asarray(result.matrix) if valid else None
        return result,matrix

    def save(self,stamp=None):
        stamp=self.clock.wall().timestamp() if stamp is None else stamp
        if self.offset.references.anchor is not None:
            self.offset.store.snapshot(stamp,dict(offset=self.offset.dump()))
            self._last_save=stamp
            protected={r['record'] for r in self.offset.references.appearances}|{r['image'] for r in self.offset.references.appearances}
            if self.offset._candidate is not None:
                protected|={r['record'] for r in self.offset._candidate.appearances}|{r['image'] for r in self.offset._candidate.appearances}
            for group in ('references','migrations'):
                folder=self.offset.store.root/group
                if not folder.is_dir():continue
                for path in folder.iterdir():
                    if path.is_symlink() or not path.is_file() or path.relative_to(self.offset.store.root).as_posix() in {str(v).replace(chr(92),'/') for v in protected}:continue
                    try:created=float(path.stem.split('_')[0])
                    except ValueError:continue
                    if stamp-created>=7*24*3600:path.unlink()

    def day_reference(self):
        entries=[r for r in self.offset.references.appearances if r['light']=='day']
        if not entries:return None
        entry=entries[-1]
        return self.offset.store.read_image(entry['image'])
