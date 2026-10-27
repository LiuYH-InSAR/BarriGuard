"""一个总开关控制全部调试输出；业务证据独立保存。"""
import cv2 as cv
import numpy as np


class Debugger:
    def __init__(self, params, store):
        self._p = params
        self._store = store

    def record(self, kind, stamp, data, image=None):
        if not self._p.debug.enabled:
            return
        folder = self._store.root/'debug'
        if folder.is_dir():
            records = sorted(folder.glob('*.json'),key=lambda p:p.name)
            for path in records[:max(0,len(records)-self._p.debug.max_records+1)]:
                path.with_suffix('.jpg').unlink(missing_ok=True)
                path.unlink(missing_ok=True)
        self._store.record('debug',stamp,dict(kind=kind,**data),image)

    def alignment(self, stamp, details, matches):
        if not self._p.debug.enabled:
            return
        image = None
        if matches is not None:
            a,b,src,dst = matches
            image = np.hstack([a,b])
            for x,y in zip(src,dst):
                cv.line(image,tuple(x.astype(int)),tuple((y+[a.shape[1],0]).astype(int)),
                        (0,255,0),self._p.debug.line_width)
        self.record('alignment',stamp,details,image)

    def detection(self, stamp, frame, detector, expected, matches, decisions, stats):
        if not self._p.debug.enabled:
            return
        image = frame.copy()
        cfg = self._p.debug
        for x0,y0,x1,y1 in detector.get('regions',[]):
            cv.rectangle(image,(x0,y0),(x1,y1),(255,200,0),cfg.line_width)
        for box in detector.get('raw',[]):
            cv.rectangle(image,tuple(map(int,box[:2])),tuple(map(int,box[2:4])),(150,150,150),cfg.line_width)
        for index,box in enumerate(expected):
            color = (0,255,0) if index in matches else (0,160,255)
            cv.rectangle(image,tuple(map(int,box[:2])),tuple(map(int,box[2:4])),color,cfg.line_width)
            cv.putText(image,str(index),tuple(map(int,box[:2])),cv.FONT_HERSHEY_SIMPLEX,
                       cfg.font_scale,color,cfg.line_width)
        self.record('detection',stamp,dict(detector=detector,expected=expected,matches=matches,
                    decisions=decisions,statistics=stats),image)
