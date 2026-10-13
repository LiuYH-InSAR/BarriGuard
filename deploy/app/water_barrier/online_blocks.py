"""Online adapter: confirmed fixed-block gaps use the existing alarm transport.

Reuses the tested detector only; replay, plot and matrix exports are never called.
Hypothesis: recent missing support plus recovery latching limits transient alarms.
"""
from collections import deque
from datetime import datetime, timedelta, timezone
import importlib
import json
from pathlib import Path
import sys
import uuid

import cv2 as cv
import numpy as np

from server.algo import AlgoResult, Annotation, Event

EVIDENCE_FRAMES = 5
JPEG_QUALITY = 90


def _core():
    # Keep a single implementation; deployment must include deploy/tools sources.
    tools = str(Path(__file__).resolve().parents[2]/'tools')
    if tools not in sys.path:
        sys.path.append(tools)
    return importlib.import_module('water_manual')


def load_config(spec):
    core = _core()
    parameters = json.loads(Path(spec.manual_parameters).read_text(encoding='utf-8-sig'))
    parameters.setdefault('pixel_fallback',True)
    parameters['recent_missing'] = True
    parameters['hold_s'] = spec.alarm_hold_s
    parameters['max_observation_gap_s'] = max(parameters.get('max_observation_gap_s',1.),2/spec.detect_hz)
    core.ManualSpec(**parameters).validate()
    reference = cv.imread(spec.manual_reference)
    if reference is None:
        raise ValueError('Block reference image missing')
    data, _, _ = core.load_labels(spec.manual_blocks)
    if reference.shape[:2] != (data['imageHeight'],data['imageWidth']):
        raise ValueError('Block reference and labels dimensions differ')
    return dict(backend='blocks',labels=spec.manual_blocks,reference=reference,
                parameters=parameters,detect_hz=spec.detect_hz,
                alarm_interval=spec.alarm_min_interval_s)


class BlockAlgorithm:
    name = 'water_gap'
    cadence = 'per_frame'

    def __init__(self, frame_shape, calib, camera_id=''):
        if tuple(frame_shape[:2]) != tuple(calib['reference'].shape[:2]):
            raise ValueError('Online frame size differs from block reference')
        self._config, self._camera = calib,camera_id
        self.reset()

    def reset(self):
        core = _core()
        self._detector = core.ManualDetector(self._config['labels'],self._config['reference'],
                                             core.ManualSpec(**self._config['parameters']))
        self._records = deque(maxlen=EVIDENCE_FRAMES)
        self._active = {}
        self._session = uuid.uuid4().hex
        self._last = None
        self._gray = None
        self._latest = AlgoResult([],[],dict(frame_status='WARMUP'))

    def _annots(self, results):
        annots = [Annotation('polygon',box=tuple(map(tuple,np.rint(poly).astype(int))),
                             label=f'row{row}',level='info')
                  for row,poly in self._detector.rows.items()]
        for block, result in zip(self._detector.blocks,results):
            if result['status'] not in ('*','error'):
                continue
            annots.append(Annotation('polygon',box=tuple(map(tuple,np.rint(block['poly']).astype(int))),
                          label=block['label'],level='alarm' if result['status']=='error' else 'suspected'))
        return annots

    def _picture(self, label):
        candidates = [r for r in self._records if r['blocks'][label]['evidence']==1]
        chosen = min(candidates,key=lambda r:(r['motion'],-r['blocks'][label]['valid']))
        from server import render
        results = [dict(b,status='error') if b['label']==label else b for b in chosen['blocks'].values()]
        image = render.draw_annots(chosen['frame'],self._annots(results))
        ok, encoded = cv.imencode('.jpg',image,[cv.IMWRITE_JPEG_QUALITY,JPEG_QUALITY])
        return (encoded.tobytes() if ok else None),chosen['time']

    def step(self, frame, now):
        if self._last is not None and now-self._last < 1/self._config['detect_hz']-1e-6:
            return AlgoResult([],self._latest.annots,self._latest.debug)
        results = self._detector.step(frame,now)
        dt = 0 if self._last is None else max(0,now-self._last)
        if dt > self._detector.spec.max_observation_gap_s:
            dt = 0
            for active in self._active.values():
                active['recovery'] = 0.
                active['last_vote'] = None
        self._last = now
        compact = {b['label']:{k:b[k] for k in ('label','row','number','status','evidence','valid','accum','reason')}
                   for b in results}
        gray = cv.resize(cv.cvtColor(frame,cv.COLOR_BGR2GRAY),(320,180))
        motion = 1. if self._gray is None else float(np.mean(np.abs(gray.astype(float)-self._gray.astype(float))>25))
        self._gray = gray
        self._records.append(dict(frame=frame.copy(),blocks=compact,time=now,motion=motion))
        events = []
        for block, result in zip(self._detector.blocks,results):
            label = block['label']
            active = self._active.get(label)
            if active is not None:
                active['recovery'] = active['recovery']+dt if result['evidence']==0 and active['last_vote']==0 else 0.
                active['last_vote'] = result['evidence']
                if active['recovery'] >= self._detector.spec.reset_s:
                    del self._active[label]
                    active = None
            if result['status']!='error' or result['evidence']!=1:
                continue
            if active is None:
                active = dict(id=f'{self._session}_{label}_{now:.6f}',last_push=None,
                              recovery=0.,last_vote=1)
                self._active[label] = active
            if active['last_push'] is not None and now-active['last_push'] < self._config['alarm_interval']:
                continue
            jpeg, evidence_time = self._picture(label)
            x,y,w,h = cv.boundingRect(np.rint(block['poly']).astype(np.int32))
            payload = dict(block_id=label,row_id=block['row'],block_number=block['number'],
                           episode_id=active['id'],box=[x,y,x+w,y+h],accum_s=result['accum'],
                           confirmed_at_monotonic_s=now,evidence_at_monotonic_s=evidence_time,
                           evidence_age_s=now-evidence_time,reason=result['reason'],backend='blocks')
            payload['confidence'] = min(1.,result['accum']/self._detector.spec.hold_s)
            confirmed = datetime.now(timezone.utc)
            payload['confirmed_at'] = confirmed.isoformat()
            payload['evidence_at'] = (confirmed-timedelta(seconds=now-evidence_time)).isoformat()
            events.append(Event(self._camera,self.name,now,'alarm',payload,image_jpeg=jpeg))
            active['last_push'] = now
        frame_status = 'UNDETERMINED' if all(b['status']=='?' for b in results) else 'OK'
        self._latest = AlgoResult(events,self._annots(results),dict(frame_status=frame_status,
                                  backend='blocks',blocks=list(compact.values())))
        return self._latest
