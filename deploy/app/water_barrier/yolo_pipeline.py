"""YOLO 水马缺失后端：实现现有 Algorithm 协议，原灯光功能不参与本模块。"""
import hashlib
import copy
import math
import time
from dataclasses import asdict

import cv2 as cv
import numpy as np
from loguru import logger

from server.algo import AlgoResult, Annotation, Event
from .yolo_camera import CameraState
from .yolo_debug import Debugger
from .yolo_detector import Detector
from .yolo_geometry import match_boxes, missing_groups, union_box, warp_boxes
from .yolo_history import Episodes, History
from .yolo_params import PARAMS
from .yolo_policy import Accumulation, Daylight
from .yolo_reference import References
from .yolo_store import Store
from .yolo_vision import Vision, visible_box


class YoloWaterAlgorithm:
    name = 'water_gap'
    cadence = 'per_frame'

    def __init__(self, frame_shape, calib=None, camera_id='', *, params=PARAMS,
                 detector=None, vision=None, wall_clock=time.time):
        self._p = params.validate()
        self._camera = camera_id
        self._shape = frame_shape
        weight_path = params.inference.weights
        self._model_id = f'{weight_path.parent.parent.name}/{weight_path.name}'
        self._model_sha = hashlib.sha256(weight_path.read_bytes()).hexdigest() if weight_path.is_file() else None
        self._clock = wall_clock
        self._store = Store(params,camera_id)
        self._debugger = Debugger(params,self._store)
        self._vision = vision or Vision(params)
        self._detector = detector or Detector(params)
        self._refs = References(params,self._store,self._vision)
        self._motion = CameraState(params)
        self._episodes = Episodes(params)
        self._histories = {}
        self._accumulation = Accumulation(params)
        self._daylight = Daylight(params)
        self._is_daylight = None
        self._previous_missing = set()
        self._old_refs = None
        self._matrix = None
        self._alignment_ok = False
        self._alignment_details = {}
        self._annots = []
        self._status = 'reference_pending'
        self._next_align = self._next_detect = self._next_fast = None
        self._last_step = None
        self._last_detect = None
        self._last_observation_stamp = None
        self._last_fingerprint = None
        self._evidence_frame = None
        self._suspect_saved = {}
        self._last_checkpoint = None
        self._last_statistics = {}
        self._last_decisions = {}
        self._presence_patches = {}
        self._load_state(self._clock())
        logger.info('[{}] YOLO water initialized; parameters={}',camera_id,asdict(params))

    def _load_state(self, stamp):
        data = self._store.load(stamp)
        if not data:
            return
        episodes = data.get('episodes',{})
        self._episodes = Episodes(self._p,episodes.get('records'),episodes.get('guard_until',0.0))
        if data.get('model_sha256') != self._model_sha:
            # 首次切到V6时不继承旧模型可能错误建立的正常排列，仍保留通知冷却。
            self._episodes.guard_new_view(stamp)
            return
        current_policy = data.get('policy_version') == 3
        self._accumulation = Accumulation(self._p,data.get('accumulation',{}) if current_policy else {})
        self._presence_patches = data.get('presence_patches',{}) if current_policy else {}
        try:
            if self._refs.restore(data['references'],stamp):
                for identity,history in (data.get('histories',{}) if current_policy else {}).items():
                    self._histories[identity] = History(self._p,history['entries'],history['started'])
                self._is_daylight = data.get('daylight')
                self._daylight.restore(data.get('lighting',{}))
                if stamp-data['saved_at'] > self._p.timing.period_s*self._p.timing.max_gap_periods:
                    self._histories = {}
                    self._accumulation.pause()
                    self._daylight.pause()
                self._status = 'alignment_pending'
            else:
                self._refs = References(self._p,self._store,self._vision)
                self._accumulation = Accumulation(self._p)
            if data.get('motion_state') in (CameraState.DETECT,CameraState.RECOVER) and self._refs.anchor is not None:
                self._motion.state = data['motion_state']
                self._accumulation.pause()
                self._status = self._motion.state
            elif data.get('motion_state') == CameraState.REBUILD:
                old = data.get('previous_references') or data['references']
                self._old_refs = References(self._p,self._store,self._vision)
                self._old_refs.restore(old,stamp)
                self._refs = References(self._p,self._store,self._vision)
                self._motion.state = CameraState.REBUILD
                self._histories = {}
                self._accumulation = Accumulation(self._p)
                self._status = 'rebuilding'
        except (KeyError,ValueError,TypeError):
            logger.exception('[{}] invalid reference checkpoint; rebuilding',self._camera)
            self._refs = References(self._p,self._store,self._vision)
            self._histories = {}
            self._accumulation = Accumulation(self._p)

    def reset(self):
        # 日间生命周期reset不能抹掉持久报警记录或正常排列。
        self._histories = {}
        self._accumulation.pause()
        self._alignment_ok = False
        self._annots = []
        self._next_align = self._next_detect = self._next_fast = None
        self._status = 'alignment_pending'
        self._previous_missing = set()
        self._evidence_frame = None

    def step(self, frame, now):
        stamp = self._clock()
        self._store.cleanup(stamp)
        if self._next_align is None:
            self._next_align = now+self._p.timing.period_s*self._p.timing.alignment_phase
            self._next_detect = now+self._p.timing.period_s
            self._next_fast = now
        if self._last_step is not None and (now < self._last_step or
                now-self._last_step > self._p.timing.period_s*self._p.timing.max_gap_periods):
            self._histories = {}
            self._alignment_ok = False
            self._next_align = now
        self._last_step = now
        events = []
        if self._motion.state != CameraState.NORMAL:
            if now >= self._next_fast:
                self._next_fast = now+self._p.timing.fast_s
                self._fast(frame,now,stamp)
        elif now >= self._next_align:
            self._next_align = self._advance_due(self._next_align,now)
            self._align(frame,now,stamp)
        elif now >= self._next_detect:
            self._next_detect = self._advance_due(self._next_detect,now)
            events = self._detect(frame,now,stamp)
        if self._last_checkpoint is None or stamp-self._last_checkpoint >= self._p.storage.checkpoint_s or events:
            self._checkpoint(stamp)
        return AlgoResult(events=events,annots=list(self._annots),debug={
            'frame_status':self._status,'water_yolo':self.status()})

    def status(self):
        return dict(state=self._motion.state,status=self._status,view=self._refs.view,
                    model=self._model_id,model_sha256=self._model_sha,
                    baseline_ready=bool(self._refs.boxes),alignment_valid=self._alignment_ok,
                    jitter_tolerance=self._motion.tolerance,statistics=self._last_statistics,
                    daylight=self._is_daylight,initial_gap_indices=self._refs.inferred_indices,
                    accumulation={i:self._accumulation.result(i,self._is_daylight)
                                  for i in self._accumulation.records},episodes=self._episodes.dump(),
                    lighting=self._daylight.dump(),observations=self._last_decisions,
                    baseline_observations=len(self._refs._candidates),
                    baseline_required=self._p.reference.min_baseline_observations,
                    alignment_details=self._alignment_details,observation_stamp=self._last_observation_stamp)

    def _advance_due(self, due, now):
        period = self._p.timing.period_s
        return due+(math.floor((now-due)/period)+1)*period

    def _quality(self, frame, stamp):
        quality = dict(self._vision.quality(frame))
        # 夜间亮路面不能挤掉白天外观版本；查询候选不推进昼夜确认次数。
        quality['reference_light'] = self._daylight.candidate_for(stamp,quality) or 'transition'
        return quality

    def _align(self, frame, now, stamp):
        quality = self._quality(frame,stamp)
        self._debugger.record('reference_quality',stamp,quality)
        if self._refs.anchor is None:
            if quality['ok']:
                self._refs.start_view(frame,stamp,quality)
                self._matrix = np.eye(3)
                self._alignment_ok = True
                self._status = 'baseline_pending'
            return
        if self._refs.expired(stamp):
            self._status = 'reference_expired'
            self._alignment_ok = False
            self._annots = []
            self._histories = {}
            if quality['ok']:
                # 到期后只用历史资料尝试迁移，不再直接用于缺失判断。
                self._old_refs = self._refs
                self._refs = References(self._p,self._store,self._vision)
                self._finish_reference(frame,now,stamp,quality)
            return
        alignment = self._refs.align(frame)
        self._alignment_details = alignment.details if alignment else {'reason':'no_reference'}
        self._debugger.alignment(stamp,self._alignment_details,self._vision.last_matches)
        action = self._motion.check(alignment)
        self._alignment_ok = bool(alignment and alignment.valid and quality['ok'] and action=='aligned')
        if action in ('suspected_move','alignment_recovery'):
            self._accumulation.pause()
            self._annots = []
            self._status = self._motion.state
            self._next_fast = now+self._p.timing.fast_s
            logger.info('[{}] camera state normal -> {} {}',self._camera,self._status,self._alignment_details)
            self._debugger.record('state_transition',stamp,dict(state=self._status,alignment=self._alignment_details))
            self._checkpoint(stamp)
            return
        if not self._alignment_ok:
            self._status = 'alignment_unavailable'
            self._annots = []
            return
        self._matrix = alignment.matrix
        if self._status not in ('inference_unavailable','low_quality_or_frozen'):
            self._status = 'ok' if self._refs.boxes else 'baseline_pending'
        self._vision.quality(frame,learn=True)
        due = self._refs.refresh_due(quality,stamp)
        unresolved = (len(self._previous_missing)/max(1,len(self._refs.boxes))
                      > self._p.reference.max_unresolved_fraction)
        if due and not unresolved:
            self._refs.add_appearance(frame,stamp,quality,self._matrix)
            self._debugger.record('reference_updated',stamp,dict(view=self._refs.view,quality=quality),frame)

    def _fast(self, frame, now, stamp):
        self._alignment_ok = False
        self._annots = []
        self._last_decisions = {}
        if self._motion.state in (CameraState.DETECT,CameraState.RECOVER):
            alignment = self._refs.align(frame)
            quality = self._quality(frame,stamp)
            action = self._motion.check(alignment if quality['ok'] else None)
            details = alignment.details if alignment else {'reason':'no_reference'}
            self._alignment_details = dict(action=action,**details)
            self._debugger.alignment(stamp,dict(action=action,**details),self._vision.last_matches)
            if action == 'confirmed_move':
                self._store.record('movement',stamp,dict(view=self._refs.view,alignment=details),frame)
                if self._refs.anchor is not None:
                    self._store.record('movement',stamp,dict(view=self._refs.view,role='before'),self._refs.anchor)
                logger.info('[{}] camera moved; old view {} disabled; {}',self._camera,self._refs.view,details)
                self._old_refs = self._refs
                self._refs = References(self._p,self._store,self._vision)
                self._histories = {}
                self._accumulation = Accumulation(self._p)
                self._presence_patches = {}
                self._previous_missing = set()
                self._motion.begin_rebuild(frame,now)
                self._status = 'rebuilding'
                self._checkpoint(stamp)
            elif action == 'returned':
                if quality['ok'] and details.get('method') == 'background_phase':
                    self._refs.add_appearance(frame,stamp,quality,alignment.matrix)
                    self._store.record('alignment_recovery',stamp,
                        dict(view=self._refs.view,alignment=details,quality=quality),frame)
                    logger.info('[{}] stationary background recovered; reference updated without resetting counts',self._camera)
                self._status = 'alignment_pending'
                self._next_align = now
                self._next_detect = now+self._p.timing.period_s*self._p.timing.alignment_phase
                self._checkpoint(stamp)
            else:
                self._status = self._motion.state
            return
        quality = self._quality(frame,stamp)
        stable,reason = self._motion.stable(frame,now,self._vision,quality)
        self._status = 'rebuilding'
        self._debugger.record('rebuild_stability',stamp,dict(reason=reason,
                    stable_since=self._motion.stable_since,quality=quality,
                    required_s=self._p.timing.period_s*self._p.timing.rebuild_periods))
        if reason != 'stable':
            logger.info('[{}] rebuild stability window: {}',self._camera,reason)
        if stable and quality['ok']:
            self._finish_reference(frame,now,stamp,quality)

    def _finish_reference(self, frame, now, stamp, quality):
        self._refs.start_view(frame,stamp,quality)
        inherited = False
        if self._old_refs is not None and self._old_refs.boxes:
            # 旧参照只在这里作一次历史关联，绝不重新启用为在线外观参照。
            link = self._old_refs.align(frame)
            if link and link.valid:
                boxes = warp_boxes(self._old_refs.boxes,link.matrix)
                if all(visible_box(b,frame.shape,self._p.motion.min_visible_ratio) for b in boxes):
                    self._refs.boxes = boxes
                    self._refs.rows = self._old_refs.rows
                    self._refs.identities = self._old_refs.identities
                    self._refs.inferred_indices = list(self._old_refs.inferred_indices)
                    self._refs.gap_scan_complete = self._old_refs.gap_scan_complete
                    self._refs.baseline_created = stamp
                    self._store.record('baselines',stamp,self._refs.baseline_dump())
                    inherited = True
        if not inherited:
            self._episodes.guard_new_view(stamp)
        self._old_refs = None
        self._histories = {}
        self._accumulation = Accumulation(self._p)
        self._presence_patches = {}
        self._previous_missing = set()
        self._evidence_frame = None
        self._matrix = np.eye(3)
        self._alignment_ok = True
        self._motion.state = CameraState.NORMAL
        self._motion.jitter.clear()
        self._next_align = now+self._p.timing.period_s
        self._next_detect = now+self._p.timing.period_s*self._p.timing.alignment_phase
        self._status = 'ok' if inherited else 'baseline_pending'
        self._store.record('reconstruction',stamp,dict(view=self._refs.view,inherited=inherited,quality=quality),frame)
        logger.info('[{}] rebuild complete, view={}, inherited={}',self._camera,self._refs.view,inherited)
        self._checkpoint(stamp)

    def _detect(self, frame, now, stamp):
        if not self._alignment_ok or self._matrix is None or self._refs.expired(stamp):
            self._unknown(stamp)
            return []
        quality = self._quality(frame,stamp)
        fingerprint = hashlib.sha256(frame.tobytes()).digest()
        if not quality['ok'] or fingerprint == self._last_fingerprint:
            self._status = 'low_quality_or_frozen'
            self._unknown(stamp)
            return []
        self._last_fingerprint = fingerprint
        if self._last_detect is not None and now-self._last_detect > self._p.timing.period_s*self._p.timing.max_gap_periods:
            self._histories = {}
        self._last_detect = now
        try:
            detected = self._detector.detect(frame)
        except Exception:
            self._status = 'inference_unavailable'
            self._unknown(stamp)
            logger.exception('[{}] YOLO inference failed; missing judgment skipped',self._camera)
            self._debugger.record('inference_failed',stamp,dict(status=self._status))
            return []
        self._last_observation_stamp = stamp
        self._daylight.observe(stamp,quality)
        daylight = True if self._daylight.mode == 'day' else False if self._daylight.mode == 'night' else None
        if daylight != self._is_daylight:
            # 夜间未检出不进入白天缺失窗口；黎明重新收集15分钟证据。
            self._histories = {}
            self._is_daylight = daylight
        if not self._refs.boxes:
            # 夜间漏检不能固化为不完整的新排列；待白天再收敛建立。
            ready = self._daylight.candidate == 'day' and self._refs.collect(detected,self._matrix,stamp)
            self._annots = self._detection_annots(detected)
            self._last_decisions = {}
            self._status = 'ok' if ready else 'baseline_pending'
            self._debugger.record('baseline_build',stamp,dict(ready=ready,detected=detected,
                         baseline=self._refs.baseline_dump()),frame)
            return []
        if daylight:
            self._refs.scan_initial_gaps(stamp)
        expected = warp_boxes(self._refs.boxes,self._matrix)
        diag = np.hypot(*frame.shape[:2])
        error_ratio = self._alignment_details.get('error_px',0)/max(1,diag)
        matches,missing = match_boxes(expected,detected,self._p.matching,error_ratio)
        decisions,statistics = {},{}
        gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY).astype(np.float32)
        previous = self._evidence_frame
        self._evidence_frame = gray_frame
        delta = None
        if previous is not None and previous.shape == gray_frame.shape:
            change = gray_frame-previous
            delta = np.abs(change-np.median(change))
        for index,identity in enumerate(self._refs.identities):
            hold = False
            accumulated_value = 'unknown'
            if not visible_box(expected[index],frame.shape,self._p.motion.min_visible_ratio):
                value,detail = 'unknown',dict(reason='outside_view',eligible_for_accumulation=False)
            elif index in matches:
                confidence = float(detected[matches[index]][4])
                reliable = confidence >= self._p.policy.recovery_confidence
                value = 'present'
                detail = dict(reason='present_reliable' if reliable else 'present_weak',
                    confidence=confidence,detection=matches[index],eligible_for_accumulation=False)
                accumulated_value = 'present' if reliable else 'weak_present'
                if reliable:
                    patch = self._appearance_patch(gray_frame,expected[index])
                    if patch is not None:
                        self._presence_patches[identity] = dict(created=stamp,pixels=patch.tolist())
                    self._refs.verify_slot(index,stamp)
            else:
                value,detail,hold = self._classify_absence(frame,gray_frame,delta,expected,index,
                    matches,identity,daylight,stamp)
                if detail['eligible_for_accumulation']:
                    accumulated_value = 'suspect'
            self._histories.setdefault(identity,History(self._p)).observe(stamp,value)
            result = self._histories[identity].result(stamp)
            accumulated = self._accumulation.observe(identity,stamp,accumulated_value,daylight,hold)
            confirmed = daylight is True and result['alarm']
            timeout = daylight is True and accumulated_value == 'suspect' and accumulated['timeout']
            result.update(alarm=bool(confirmed or timeout),recovered=accumulated['recovered'],
                          error_reason='confirmed_missing' if confirmed else 'suspect_timeout' if timeout else None,
                          accumulation=accumulated,daylight=daylight)
            statistics[identity] = result
            decisions[index] = dict(value=value,box=expected[index],**detail)
        self._last_decisions = {self._refs.identities[i]:v for i,v in decisions.items()}
        self._previous_missing = set(missing)
        before_episodes = copy.deepcopy(self._episodes.dump())
        notifications = set(self._episodes.decide(self._refs.identities,statistics,stamp))
        self._last_statistics = statistics
        self._status = 'ok'
        self._annots = self._detection_annots(detected,matches)
        events = []
        for group in missing_groups(missing,self._refs.rows):
            ids = [self._refs.identities[i] for i in group]
            box = list(map(int,union_box([expected[i] for i in group])))
            active = daylight and any(self._episodes.records.get(i,{}).get('active') for i in ids)
            level = 'alarm' if active else 'suspected'
            causes = sorted({decisions[i]['reason'] for i in group})
            short_ids = ','.join(i.rsplit(':',1)[-1] for i in ids)
            if len(ids) > 1:
                short_ids = ids[0].rsplit(':',1)[-1]+'+'+str(len(ids)-1)
            abbreviations = dict(small_target_unresolved='small',background_unresolved='unresolved',
                lighting_pending='light_pending',suspected_occlusion='occlusion',
                suspected_occlusion_held='occlusion_held',initial_arrangement_gap='initial_gap',
                night_missing_candidate='night_candidate')
            label = ('error ' if active else 'suspect ') + short_ids + ' ' + '/'.join(abbreviations.get(c,c) for c in causes)
            self._annots.append(Annotation(kind='box',box=tuple(box),label=label,level=level))
            payload = dict(box=box,target='|'.join(ids),barrier_ids=ids,view=self._refs.view,
                           statistics={i:statistics[i] for i in ids},
                           reasons={str(i):decisions[i] for i in group},backend='yolo')
            payload.update(model=self._model_id,model_sha256=self._model_sha)
            reasons = {i:self._episodes.records.get(i,{}).get('error_reason') for i in ids}
            payload.update(severity='error' if active else 'suspect',error_reasons=reasons,
                           error_reason=('confirmed_missing' if 'confirmed_missing' in reasons.values()
                                         else 'suspect_timeout') if active else None,
                           daylight=daylight)
            should_notify = any(i in notifications for i in ids)
            if should_notify:
                image = self._render(frame)
                try:
                    path = self._store.record('alarms',stamp,payload,image)
                except OSError:
                    self._episodes = Episodes(self._p,**before_episodes)
                    raise
                ok,jpeg = cv.imencode('.jpg',image,[cv.IMWRITE_JPEG_QUALITY,self._p.storage.jpeg_quality])
                events.append(Event(self._camera,self.name,stamp,'alarm',payload=payload,
                              evidence_path=str(self._store.root/path),image_jpeg=jpeg.tobytes() if ok else None))
                logger.info('[{}] missing alarm {}',self._camera,payload['target'])
            elif not active:
                key = '|'.join(ids)
                last = self._suspect_saved.get(key)
                if last is None or stamp-last >= self._p.evidence.suspect_save_s:
                    self._store.record('suspect',stamp,payload,self._render(frame))
                    self._suspect_saved[key] = stamp
        self._debugger.detection(stamp,frame,getattr(self._detector,'last',{}),expected,matches,decisions,statistics)
        self._debugger.record('alarm_decision',stamp,dict(notifications=list(notifications),episodes=self._episodes.dump()))
        self._store.record('observations',stamp,dict(view=self._refs.view,decisions=decisions,
                           statistics=statistics,status=self._status))
        return events

    def _detection_annots(self, detected, matches=None):
        identities = {j:self._refs.identities[i].rsplit(':',1)[-1]
                      for i,j in (matches or {}).items()}
        return [Annotation(kind='box',box=tuple(map(int,b[:4])),
            label=f'{identities.get(i,"D"+str(i))} {b[4]:.2f}',level='info')
            for i,b in enumerate(detected)]

    def _appearance_patch(self, gray_frame, box):
        a,b,c,d = map(int,box)
        patch = gray_frame[max(0,b):min(gray_frame.shape[0],d),max(0,a):min(gray_frame.shape[1],c)]
        if not patch.size:
            return None
        side = self._p.evidence.appearance_patch_px
        return cv.resize(patch,(side,side),interpolation=cv.INTER_AREA).astype(np.float32)

    def _classify_absence(self, frame, gray_frame, delta, expected, index, matches, identity, daylight, stamp):
        cfg = self._p.evidence
        box = expected[index]
        detail = dict(eligible_for_accumulation=False,inferred_slot=index in self._refs.inferred_indices)
        if daylight is None:
            return 'unknown',dict(detail,reason='lighting_pending'),False
        if min(box[2]-box[0],box[3]-box[1]) < cfg.patch_px:
            return 'unknown',dict(detail,reason='small_target_unresolved'),False
        if detail['inferred_slot']:
            return 'unknown',dict(detail,reason='initial_arrangement_gap',eligible_for_accumulation=True),False
        held = self._accumulation.records.get(identity,{}).get('occlusion_held',False)
        if held:
            return 'unknown',dict(detail,reason='suspected_occlusion_held',occlusion_held=True,
                                  eligible_for_accumulation=True),False
        road,evidence = self._vision.road_evidence(frame,expected,index,matches,
                                                  self._refs.rows,self._previous_missing)
        detail.update(evidence=evidence)
        patch = self._appearance_patch(gray_frame,box)
        template = self._presence_patches.get(identity)
        changed = False
        if template and patch is not None and 0 <= stamp-template['created'] <= cfg.appearance_max_age_s:
            old = np.asarray(template['pixels'],dtype=np.float32)
            if float(old.std()) >= cfg.texture_floor:
                difference = float(np.mean(np.abs((old-old.mean())/old.std()
                    -(patch-patch.mean())/max(float(patch.std()),cfg.texture_floor))))
                detail['appearance_difference'] = difference
                changed = difference >= cfg.appearance_difference
        moving = False
        if delta is not None:
            a,b,c,d = map(int,box)
            local = delta[max(0,b):min(delta.shape[0],d),max(0,a):min(delta.shape[1],c)]
            fraction = float(np.mean(local > cfg.motion_luma_delta)) if local.size else 0.0
            detail['changed_fraction'] = fraction
            moving = fraction > cfg.history_motion_ratio
        cause = evidence['reason']
        occlusion = (cause == 'occluded' or (cause == 'simultaneous_disappearance' and daylight is True)
                     or (not road and changed and evidence.get('neighbor_ratio',0) >= cfg.neighbor_ratio))
        if occlusion:
            return 'unknown',dict(detail,reason='suspected_occlusion',occlusion_held=True,
                                  eligible_for_accumulation=True),True
        if road and not moving:
            return ('missing' if daylight is True else 'unknown'),dict(detail,
                reason='road_exposed' if daylight is True else 'night_missing_candidate',
                eligible_for_accumulation=True),False
        return 'unknown',dict(detail,reason='region_changing' if moving else 'background_unresolved'),False

    def _unknown(self, stamp):
        self._annots = []
        self._accumulation.pause()
        self._daylight.pause()
        self._last_decisions = {}
        for identity,history in self._histories.items():
            history.observe(stamp,'unknown')
        self._last_statistics = {i:h.result(stamp) for i,h in self._histories.items()}

    def _render(self, frame):
        from server.render import draw_annots
        return draw_annots(frame,self._annots)

    def _checkpoint(self, stamp):
        # 只滚动保存当前状态；原始图片、历史记录的生成时间不刷新。
        self._store.snapshot(stamp,dict(references=self._refs.dump(),
             episodes=self._episodes.dump(),histories={i:h.dump() for i,h in self._histories.items()},
             motion_state=self._motion.state,previous_references=self._old_refs.dump() if self._old_refs else None,
             policy_version=3,accumulation=self._accumulation.dump(),
             lighting=self._daylight.dump(),presence_patches=self._presence_patches,
             daylight=self._is_daylight,daylight_streak=self._daylight.streak,
             model_sha256=self._model_sha))
        self._last_checkpoint = stamp
