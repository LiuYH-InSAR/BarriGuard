"""Shared background library; target identities and absence policies stay in consumers."""
import numpy as np
from .geometry import warp_boxes


class ReferenceLibrary:
    def __init__(self, params, store, vision, exclusions=lambda: ()):
        self._p, self._store, self._vision = params, store, vision
        self.exclusions = exclusions
        self.view = ''
        self.anchor = None
        self.appearances = []
        self.last_refresh = None
        self._light = None
        self._light_count = 0

    def expired(self, stamp):
        self.appearances = [r for r in self.appearances
                        if stamp-r['created'] < self._p.storage.retention_s]
        return not self.appearances

    def start_view(self, frame, stamp, quality):
        self.view = f'{stamp:.6f}'
        self.anchor = frame.copy()
        self.appearances = []
        self.add_appearance(frame, stamp, quality, np.eye(3))

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
            exclusion = warp_boxes(self.exclusions(),matrix)
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

    def migration_hypothesis(self,frame):
        """A weaker background proposal, never a normal/aligned decision.

        Spatially sparse ORB support can propose a new-view transform only
        when an independent business module subsequently corroborates it.
        """
        feature=getattr(self._vision,'_feature_align',None)
        if not callable(feature):return None
        cfg=self._p.motion;best=None;matches=None
        for entry in self.appearances:
            image=self._store.read_image(entry['image'])
            if image is None:continue
            prior=np.asarray(entry['matrix'],float)
            result=feature(image,frame,warp_boxes(self.exclusions(),prior))
            d=result.details
            if (result.matrix is None or d.get('method')!='orb' or d.get('inliers',0)<cfg.min_inliers
                or d.get('inlier_ratio',0)<cfg.min_inlier_ratio or d.get('error_px',float('inf'))>cfg.max_error_px
                or min(d.get('cells',0),d.get('destination_cells',0))<cfg.stationary_min_cells
                or abs(d.get('scaling',0)-1)>cfg.max_scale_error):continue
            result.matrix=np.asarray(result.matrix)@prior
            if best is None or d['inliers']>best.details['inliers']:
                best=result;matches=self._vision.last_matches
        self._vision.last_matches=matches
        return best

