"""有版本、质量门槛和TTL的外观参照，以及正常水马排列的收敛建立。"""
import numpy as np
from dataclasses import replace

from .yolo_geometry import build_rows, iou, match_boxes, warp_boxes, infer_initial_gaps


from camera_motion.references import ReferenceLibrary


def baseline_segment(box, shape):
    """Image-space coverage indicator, independent of target confidence."""
    height,width = shape[:2]
    side = 'left' if box[0] + box[2] < width else 'right'
    depth = 'far' if box[3] < height * .55 else 'near'
    return f'{side}_{depth}'


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
        self._pending = []
        self.rebuilding = False
        self.rebuild_unmapped = False
        self.carried_indices = []
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
        self.reset_pending()
        self.rebuilding = False
        self.rebuild_unmapped = False
        self.carried_indices = []
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
        recent = self._candidates[-cfg.min_baseline_observations:]
        seed = recent[0]
        samples = [[b] for b in seed]
        for observation in recent[1:]:
            matches,_ = match_boxes(seed,observation,self._p.matching)
            for a,b in matches.items():
                samples[a].append(observation[b])
        stable = [np.median(v,axis=0).tolist() for v in samples
                  if len(v) == len(recent)]
        if len(stable) < cfg.min_barriers:
            return False
        self.boxes = stable
        self.rows = build_rows(stable,cfg)
        self.identities = [f'{self.view}:B{i}' for i in range(len(stable))]
        self.baseline_created = stamp
        self._store.record('baselines',stamp,self.baseline_dump())
        return True

    def reset_pending(self):
        self._candidates = []
        self._pending = []

    def _track_observed_slots(self, boxes):
        """Collect independent observations; a missed sample breaks a track."""
        cfg = self._p
        matches, _ = match_boxes([samples[0] for samples in self._pending], boxes,
                                 cfg.matching)
        pending = [self._pending[i] + [boxes[j]] for i, j in matches.items()]
        used = set(matches.values())
        pending.extend([box] for j, box in enumerate(boxes) if j not in used)
        required = cfg.reference.min_baseline_observations
        self._pending = [samples[-required:] for samples in pending]
        return [np.median(samples, axis=0).tolist()
                for samples in self._pending if len(samples) >= required]

    @property
    def pending_observations(self):
        return max((len(samples) for samples in self._pending), default=len(self._candidates))

    def rebuild_baseline(self, detected, anchor_to_frame, stamp):
        """Confirm positions incrementally while retaining unmatched migrated slots."""
        expected = warp_boxes(self.boxes, anchor_to_frame)
        provisional = set(self.carried_indices) | set(self.inferred_indices)
        matches, _ = match_boxes(expected, detected, self._p.matching)
        confirmed = {j for i, j in matches.items() if i not in provisional}
        return self.extend_baseline([box for j, box in enumerate(detected) if j not in confirmed],
                                    anchor_to_frame, stamp)

    def finish_rebuild(self, shape, stamp, confirmed_missing=()):
        """Close a mapped rebuild when each inherited position is accounted for."""
        if not self.rebuilding or set(self.carried_indices).difference(confirmed_missing):
            return
        measured = [box for i, box in enumerate(self.boxes) if i not in self.inferred_indices]
        if len(measured) < self._p.reference.min_barriers:
            return
        if self.rebuild_unmapped and len({baseline_segment(box,shape) for box in measured}) < 4:
            return
        completed = self.baseline_dump()
        completed.pop('rebuild_unmapped', None)
        completed['rebuilding'] = False
        self._store.record('baselines', stamp, completed)
        self.rebuilding = self.rebuild_unmapped = False

    def extend_baseline(self, unmatched, anchor_to_frame, stamp):
        """Register repeatedly visible new slots in the existing reference view."""
        cfg = self._p
        threshold = cfg.reference.slot_overlap_iou
        boxes = warp_boxes(unmatched, np.linalg.inv(anchor_to_frame))
        provisional = sorted(set(self.carried_indices) | set(self.inferred_indices))
        measured = [box for i, box in enumerate(self.boxes) if i not in provisional]
        stable = self._track_observed_slots([box for box in boxes
                  if not any(iou(box, old) >= threshold for old in measured)])
        matches, _ = match_boxes([self.boxes[i] for i in provisional],
                                 stable, cfg.matching)
        confirmed = []
        for index, observed in matches.items():
            slot = provisional[index]
            self.boxes[slot] = stable[observed]
            confirmed.append(slot)
        self.carried_indices = [i for i in self.carried_indices if i not in confirmed]
        self.inferred_indices = [i for i in self.inferred_indices if i not in confirmed]
        used = set(matches.values())
        added = self._append_slots(box for j, box in enumerate(stable) if j not in used)
        if added or confirmed:
            self._save_layout(stamp)
        return added or bool(confirmed)

    def _append_slots(self, boxes):
        cfg = self._p
        identities = set(self.identities)
        next_id = len(self.boxes)
        added = False
        for box in boxes:
            if (len(self.boxes) >= cfg.initial_gaps.max_slots
                    or any(iou(box, old) >= cfg.reference.slot_overlap_iou for old in self.boxes)):
                continue
            while f'{self.view}:B{next_id}' in identities:
                next_id += 1
            self.identities.append(f'{self.view}:B{next_id}')
            next_id += 1
            self.boxes.append(box)
            added = True
        return added

    def _save_layout(self, stamp):
        self.rows = build_rows(self.boxes, self._p.reference)
        self.baseline_created = stamp
        self.gap_scan_complete = False
        self._store.record('baselines', stamp, self.baseline_dump())

    def baseline_dump(self):
        data = dict(view=self.view,boxes=self.boxes,rows=self.rows,identities=self.identities,
                    baseline_created=self.baseline_created,inferred_indices=self.inferred_indices,
                    gap_scan_complete=self.gap_scan_complete)
        if self.rebuilding or self.carried_indices:
            data.update(rebuilding=self.rebuilding, carried_indices=self.carried_indices)
        if self.rebuilding:
            data['rebuild_unmapped'] = self.rebuild_unmapped
        return data

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
        self.identities.extend(f'{self.view}:G{i}' for i in range(first, len(self.boxes)))
        self.rows = build_rows(self.boxes,self._p.reference)
        self.gap_scan_complete = True
        self._store.record('baselines',stamp,self.baseline_dump())

    def dump(self):
        return dict(**self.baseline_dump(),appearances=self.appearances,last_refresh=self.last_refresh)

    def restore(self, data, stamp):
        self.rebuilding = data.get('rebuilding', False)
        self.carried_indices = data.get('carried_indices', [])
        self.rebuild_unmapped = data.get('rebuild_unmapped', self.rebuilding and not data.get('boxes'))
        if self.rebuilding and 'rebuild_unmapped' not in data and not self.carried_indices:
            inferred = set(data.get('inferred_indices', []))
            self.carried_indices = [i for i in range(len(data.get('boxes', []))) if i not in inferred]
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
            return (bool(self.boxes) or self.rebuilding) and not self.expired(stamp)
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
