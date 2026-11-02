"""有版本、质量门槛和TTL的外观参照，以及正常水马排列的收敛建立。"""
import numpy as np
from dataclasses import replace

from .yolo_geometry import build_rows, match_boxes, warp_boxes, infer_initial_gaps


from camera_motion.references import ReferenceLibrary


class References:
    def __init__(self, params, store, vision, library=None):
        self._p = params
        self._store = store
        self._vision = vision
        self.shared = library is not None
        self.library = library or ReferenceLibrary(params, store, vision, lambda: self.boxes)
        self.boxes = []
        self.rows = []
        self.identities = []
        self.baseline_created = None
        self._candidates = []
        self.inferred_indices = []
        self.gap_scan_complete = False

    def expired(self, stamp):
        empty = (not any(stamp-r['created']<self.library._p.storage.retention_s
                         for r in self.library.appearances)) if self.shared else self.library.expired(stamp)
        return empty or (self.baseline_created is not None
                         and stamp-self.baseline_created >= self._p.storage.retention_s)

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
        self.library.add_appearance(frame, stamp, quality, anchor_to_frame)
        if self.boxes:
            self.baseline_created = stamp
            self._store.record('baselines', stamp, self.baseline_dump())

    def refresh_due(self, quality, stamp):
        return self.library.refresh_due(quality, stamp)

    def align(self, frame):
        return self.library.align(frame)

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
        if self.shared:
            if (self.library.anchor is not None and self.library.view != data['view']
                    and data['view'] not in getattr(self.library,'migration_sources',())):
                return False
            self.restored_view=data['view']
            if self.library.anchor is None:
                # One-time import from the previous water-private store. Never
                # publish private relative paths into the shared library.
                entries=[]
                for entry in data.get('appearances',[]):
                    if stamp-entry['created']>=self._p.storage.retention_s:continue
                    image=self._store.read_image(entry['image'])
                    if image is None:continue
                    path=self.library._store.record('references',entry['created'],
                        dict(view=data['view'],created=entry['created'],quality={'light':entry['light']},
                             anchor_to_frame=entry['matrix']),image)
                    record=self.library._store.read_record(path)
                    entries.append(dict(entry,record=path,image=record['image']))
                if not entries:return False
                self.library.view=data['view']
                self.library.appearances=entries
                self.library.last_refresh=data.get('last_refresh')
                self.library.anchor=self.library._store.read_image(entries[0]['image'])
            self.boxes=data.get('boxes',[])
            self.rows=data.get('rows',[])
            self.identities=data.get('identities',[])
            self.inferred_indices=data.get('inferred_indices',[])
            self.gap_scan_complete=data.get('gap_scan_complete',False)
            self.baseline_created=data.get('baseline_created')
            return bool(self.boxes) and not self.expired(stamp)
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
        self.anchor = self.library._store.read_image((anchor or self.appearances[0])['image'])
        return self.anchor is not None

    @property
    def view(self):
        return self.library.view

    @view.setter
    def view(self, value):
        self.library.view = value

    @property
    def anchor(self):
        return self.library.anchor

    @anchor.setter
    def anchor(self, value):
        self.library.anchor = value

    @property
    def appearances(self):
        return self.library.appearances

    @appearances.setter
    def appearances(self, value):
        self.library.appearances = value

    @property
    def last_refresh(self):
        return self.library.last_refresh

    @last_refresh.setter
    def last_refresh(self, value):
        self.library.last_refresh = value
