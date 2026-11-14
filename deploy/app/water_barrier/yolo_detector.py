"""仅本层加载模型。测试可以注入 detector，不触发下载或训练。"""
import os
import threading
import time
from pathlib import Path

from .yolo_geometry import merge_boxes, tile_regions

_MODELS = {}
_LOCK = threading.Lock()


class Detector:
    def __init__(self, params):
        self._p = params
        self.last = {}
        self._model = None

    def _load(self):
        path = Path(self._p.inference.weights).resolve()
        if not path.is_file():
            raise FileNotFoundError(f'YOLO weights do not exist: {path}')
        # 配置缓存也必须在新项目目录内，不能写原项目或用户配置目录。
        config_dir = self._p.storage.root / 'model_settings'
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault('YOLO_CONFIG_DIR', str(config_dir))
        with _LOCK:
            if path not in _MODELS:
                from ultralytics import YOLO
                _MODELS[path] = YOLO(str(path), task='detect')
            self._model = _MODELS[path]

    def detect(self, frame):
        started = time.perf_counter()
        if self._model is None:
            self._load()
        cfg = self._p.inference
        regions = tile_regions(frame.shape, cfg)
        raw = []
        for x0,y0,x1,y1 in regions:
            crop = frame[y0:y1,x0:x1]
            with _LOCK:
                result = self._model.predict(crop, imgsz=cfg.imgsz, conf=cfg.confidence,
                    rect=cfg.rect,
                    iou=cfg.nms_iou, classes=[cfg.class_id], device=cfg.device,
                    max_det=cfg.max_detections, verbose=False, save=False)[0]
                values = result.boxes.data.detach().cpu().numpy()
            margin = cfg.border_fraction * min(crop.shape[:2])
            for a,b,c,d,conf,_cls in values:
                if min(c-a,d-b) < cfg.min_box_px:
                    continue
                # 内部裁剪边缘上的残框交给相邻图块，避免残框成为基准。
                if ((x0 and a < margin) or (y0 and b < margin)
                        or (x1 < frame.shape[1] and c > crop.shape[1]-margin)
                        or (y1 < frame.shape[0] and d > crop.shape[0]-margin)):
                    continue
                raw.append([float(a+x0),float(b+y0),float(c+x0),float(d+y0),float(conf)])
        merged = merge_boxes(raw, cfg.merge_iou)
        self.last = dict(regions=regions, raw=raw, merged=merged,
                         seconds=time.perf_counter()-started)
        return merged

    def detect_whole(self, frame):
        """供离线尺度对比使用，不改变实时后端的分块策略。"""
        if self._model is None:
            self._load()
        cfg = self._p.inference
        with _LOCK:
            result = self._model.predict(frame,imgsz=cfg.imgsz,conf=cfg.confidence,
                rect=cfg.rect,
                iou=cfg.nms_iou,classes=[cfg.class_id],device=cfg.device,
                max_det=cfg.max_detections,verbose=False,save=False)[0]
            return result.boxes.data.detach().cpu().numpy().tolist()
