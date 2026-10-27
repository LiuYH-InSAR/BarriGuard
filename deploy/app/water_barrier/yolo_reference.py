"""有版本、质量门槛和TTL的外观参照，以及正常水马排列的收敛建立。"""
import numpy as np
from dataclasses import replace

from .yolo_geometry import build_rows, match_boxes, warp_boxes, infer_initial_gaps


class References:
    def __init__(self, params, store, vision):
        self._p = params
        self._store = store
        self._vision = vision
        self.view = ''
        self.anchor = None
        self.appearances = []
        self.boxes = []
        self.rows = []
        self.identities = []
        self.baseline_created = None
        self.last_refresh = None
        self._candidates = []
        self._light = None
        self._light_count = 0
        self.inferred_indices = []
        self.gap_scan_complete = False

    def expired(self, stamp):
        ttl = self._p.storage.retention_s
        self.appearances = [r for r in self.appearances if stamp-r['created'] < ttl]
        return not self.appearances or (self.baseline_created is not None and stamp-self.baseline_created >= ttl)

    def start_view(self, frame, stamp, quality):
        self.view = f'{stamp:.6f}'
        self.anchor = frame.copy()
        self.appearances = []
        self.boxes,self.rows,self.identities = [],[],[]
        self.baseline_created = None
        self._candidates = []
        self.inferred_indices = []
        self.gap_scan_complete = False
        self.add_appearance(frame,stamp,quality,np.eye(3))

    def add_appearance(self, frame, stamp, quality, anchor_to_frame):
        path = self._store.record('references',stamp,dict(view=self.view,quality=quality,
                        anchor_to_frame=anchor_to_frame,created=stamp),frame)
        data = self._store.read_record(path)
        category_light = quality.get('reference_light',quality['light'])
        entry = dict(record=path,image=data['image'],created=stamp,light=category_light,
                     matrix=np.asarray(anchor_to_frame).tolist())
        self.appearances.append(entry)
        category = [r for r in self.appearances if r['light']==category_light]
        discarded = category[:-self._p.reference.per_light]
        self.appearances = [r for r in self.appearances if r not in discarded]
        self.last_refresh = stamp
        # 只续存已确认位置，不根据漏检删除水马；视角坐标保持不变。
        if self.boxes:
            self.baseline_created = stamp
            self._store.record('baselines',stamp,self.baseline_dump())

    def refresh_due(self, quality, stamp):
        light = quality.get('reference_light',quality['light'])
        if light == self._light:
            self._light_count += 1
        else:
            self._light,self._light_count = light,1
        has_light = any(r['light']==light for r in self.appearances)
        return ((self.last_refresh is None or stamp-self.last_refresh >= self._p.reference.refresh_s)
                or (not has_light and self._light_count >= self._p.reference.light_confirm))

    def align(self, frame):
        best = None
        best_matches = None
        for entry in reversed(self.appearances):
            image = self._store.read_image(entry['image'])
            if image is None:
                continue
            matrix = np.asarray(entry['matrix'])
            exclusion = warp_boxes(self.boxes,matrix)
            alignment = self._vision.align(image,frame,exclusion)
            if not alignment.valid:
                if best is None:
                    best = alignment
                continue
            alignment.matrix = alignment.matrix @ matrix
            # 位移相对整个视角基准，不能被不断刷新的外观参照归零。
            h,w = frame.shape[:2]
            import cv2 as cv
            points = np.float32([[0,0],[w,0],[0,h],[w,h]])
            shifted = cv.perspectiveTransform(points[None],alignment.matrix)[0]
            alignment.displacement = float(np.max(np.linalg.norm(shifted-points,axis=1))/np.hypot(h,w))
            if best is None or not best.valid or alignment.details['inliers'] > best.details['inliers']:
                best = alignment
                best_matches = self._vision.last_matches
        self._vision.last_matches = best_matches
        return best

    def collect(self, detections, anchor_to_frame, stamp):
        boxes = warp_boxes(detections,np.linalg.inv(anchor_to_frame))
        cfg = self._p.reference
        if len(boxes) < cfg.min_barriers:
            self._candidates = []
            return False
        if self._candidates:
            first = self._candidates[0]
            matches,_ = match_boxes(first,boxes,self._p.matching)
            change = abs(len(first)-len(boxes))/max(len(first),len(boxes))
            if change > cfg.baseline_count_change or len(matches)/len(first) < cfg.baseline_presence:
                self._candidates = []
        self._candidates.append(boxes)
        self._candidates = self._candidates[-cfg.max_baseline_observations:]
        if len(self._candidates) < cfg.min_baseline_observations:
            return False
        seed = self._candidates[0]
        samples = [[b] for b in seed]
        for observation in self._candidates[1:]:
            matches,_ = match_boxes(seed,observation,self._p.matching)
            for a,b in matches.items():
                samples[a].append(observation[b])
        stable = [np.median(v,axis=0).tolist() for v in samples
                  if len(v)/len(self._candidates) >= cfg.baseline_presence]
        if len(stable) < cfg.min_barriers:
            return False
        self.boxes = stable
        self.rows = build_rows(stable,cfg)
        self.identities = [f'{self.view}:B{i}' for i in range(len(stable))]
        self.baseline_created = stamp
        self._store.record('baselines',stamp,self.baseline_dump())
        return True

    def baseline_dump(self):
        return dict(view=self.view,boxes=self.boxes,rows=self.rows,identities=self.identities,
                    baseline_created=self.baseline_created,inferred_indices=self.inferred_indices,
                    gap_scan_complete=self.gap_scan_complete)

    def scan_initial_gaps(self, stamp):
        if self.gap_scan_complete or not self.boxes:
            return
        # 缺口本身可能切断普通邻接排，先扩大候选连接，再由局部方向和间距筛选。
        candidate_rows = build_rows(self.boxes,replace(self._p.reference,
            row_distance_ratio=self._p.reference.row_distance_ratio*self._p.initial_gaps.max_gap_ratio))
        gaps = infer_initial_gaps(self.boxes,candidate_rows,self._p.initial_gaps)
        first = len(self.boxes)
        self.boxes.extend(gaps)
        self.inferred_indices.extend(range(first,len(self.boxes)))
        self.identities.extend(f'{self.view}:G{i}' for i in range(len(gaps)))
        self.rows = build_rows(self.boxes,self._p.reference)
        self.gap_scan_complete = True
        self._store.record('baselines',stamp,self.baseline_dump())

    def verify_slot(self, index, stamp):
        if index in self.inferred_indices:
            self.inferred_indices.remove(index)
            self._store.record('baselines',stamp,self.baseline_dump())

    def dump(self):
        return dict(**self.baseline_dump(),appearances=self.appearances,last_refresh=self.last_refresh)

    def restore(self, data, stamp):
        self.view = data['view']
        self.boxes = data.get('boxes',[])
        self.rows = data.get('rows',[])
        self.identities = data.get('identities',[])
        self.inferred_indices = data.get('inferred_indices',[])
        self.gap_scan_complete = data.get('gap_scan_complete',False)
        self.baseline_created = data.get('baseline_created')
        self.appearances = data.get('appearances',[])
        self.last_refresh = data.get('last_refresh')
        if self.expired(stamp):
            return False
        anchor = next((r for r in self.appearances if np.allclose(r['matrix'],np.eye(3))),None)
        # anchor图到期后仍使用统一坐标变换，不再依赖原始anchor像素。
        self.anchor = self._store.read_image((anchor or self.appearances[0])['image'])
        return self.anchor is not None
