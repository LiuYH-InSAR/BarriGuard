"""Water-specific road evidence; shared registration lives in camera_motion."""
import cv2 as cv
import numpy as np
from camera_motion.vision import Alignment, Vision as BackgroundVision, gray


class Vision(BackgroundVision):
    def road_evidence(self, frame, expected, index, matches, rows, previous_missing, *, historical_road=None):
        """从目标下方、排除全部预期水马的多个背景块提取局部路面模型。

        这是背景相似度证据，不是路面语义识别。多只同步消失先保留 suspect。
        """
        cfg = self._p.evidence
        box = expected[index]
        height,width = frame.shape[:2]
        x0,y0,x1,y1 = [int(round(v)) for v in box]
        x0,x1 = max(0,x0),min(width,x1)
        y0,y1 = max(0,y0),min(height,y1)
        short_side = min(x1-x0, y1-y0)
        if short_side < cfg.min_patch_px * 2:
            return False,dict(reason='small_or_outside')
        row = next((r for r in rows if index in r),[index])
        others = [i for i in row if i != index]
        visible_count = sum(i in matches for i in others)
        visible = visible_count/max(1,len(others))
        absent = [i for i in row if i not in matches]
        newly_missing = [i for i in absent if i not in previous_missing
                         and min(expected[i][2]-expected[i][0],expected[i][3]-expected[i][1]) >= cfg.min_patch_px * 2]
        if len(newly_missing) >= cfg.simultaneous_min:
            return False,dict(reason='simultaneous_disappearance',neighbor_ratio=visible)
        # Several consecutive missing barriers can make the row ratio low.
        # The first shared disappearance is still excluded above.
        weak_neighbors = not others or (visible < cfg.neighbor_ratio and visible_count < cfg.min_visible_neighbors)
        if weak_neighbors and (not others or index not in previous_missing or historical_road is None):
            return False,dict(reason='neighbors_unreliable',neighbor_ratio=visible)
        patch = max(cfg.min_patch_px, min(cfg.patch_px, short_side // 3))
        profile, count = self._road_profile(frame, expected, (x0,y0,x1,y1), patch)
        if profile is None:
            return False,dict(reason='insufficient_road_samples',road_patches=count)
        target = np.asarray(profile['color'])
        texture_limit = profile['texture_limit']
        if weak_neighbors:
            if (np.linalg.norm(target - historical_road['color']) > cfg.color_distance
                    or texture_limit > historical_road['texture_limit'] * cfg.max_texture_ratio):
                return False, dict(reason='historical_road_changed', neighbor_ratio=visible)
            texture_limit = min(texture_limit, historical_road['texture_limit'])
        fraction = self._road_fraction(frame[y0:y1,x0:x1], patch, target, texture_limit)
        return fraction >= cfg.min_road_fraction,dict(reason='road_exposed' if fraction >= cfg.min_road_fraction
            else 'background_not_confirmed',road_fraction=fraction,road_patches=count,neighbor_ratio=visible,
            road_color=target.tolist(),texture_limit=texture_limit,historical_road_used=weak_neighbors)

    def road_profile(self, frame, expected, index):
        """Keep the adjacent road sample only when the occupied site looks different."""
        height,width = frame.shape[:2]
        x0,y0,x1,y1 = [int(round(v)) for v in expected[index]]
        x0,y0,x1,y1 = max(0,x0),max(0,y0),min(width,x1),min(height,y1)
        cfg = self._p.evidence
        short_side = min(x1-x0,y1-y0)
        if short_side < cfg.min_patch_px * 2:
            return None
        patch = max(cfg.min_patch_px, min(cfg.patch_px, short_side // 3))
        profile, _ = self._road_profile(frame, expected, (x0,y0,x1,y1), patch)
        if (profile is not None and self._road_fraction(frame[y0:y1,x0:x1], patch,
                profile['color'], profile['texture_limit']) < cfg.min_road_fraction):
            return profile
        return None

    def _road_fraction(self, crop, patch, color, texture_limit):
        lab = cv.cvtColor(crop,cv.COLOR_BGR2LAB)
        gray_image = cv.cvtColor(crop,cv.COLOR_BGR2GRAY)
        votes = []
        for yy in range(0,crop.shape[0]-patch+1,patch):
            for xx in range(0,crop.shape[1]-patch+1,patch):
                sample = lab[yy:yy+patch,xx:xx+patch]
                distance = np.linalg.norm(np.median(sample.reshape(-1,3),axis=0)-color)
                texture = gray_image[yy:yy+patch,xx:xx+patch].std()
                votes.append(distance <= self._p.evidence.color_distance and texture <= texture_limit)
        return float(np.mean(votes)) if votes else 0.0

    def _road_profile(self, frame, expected, box, patch):
        cfg = self._p.evidence
        x0,y0,x1,y1 = box
        band_end = min(frame.shape[0],y1+int((y1-y0)*cfg.band_height_ratio))
        if band_end-y1 < patch:
            return None, 0
        band = frame[y1:band_end,x0:x1]
        lab = cv.cvtColor(band,cv.COLOR_BGR2LAB).astype(np.float32)
        gray_image = cv.cvtColor(band,cv.COLOR_BGR2GRAY)
        forbidden = np.zeros(band.shape[:2],np.uint8)
        for a,b,c,d in expected:
            if c >= x0 and a < x1 and d >= y1 and b < band_end:
                cv.rectangle(forbidden,(max(0,int(a))-x0,max(0,int(b))-y1),
                             (min(frame.shape[1]-1,int(c))-x0,min(frame.shape[0]-1,int(d))-y1),1,-1)
        road, textures = [], []
        for yy in range(0,band.shape[0]-patch+1,patch):
            for xx in range(0,band.shape[1]-patch+1,patch):
                if forbidden[yy:yy+patch,xx:xx+patch].any():
                    continue
                sample = lab[yy:yy+patch,xx:xx+patch]
                color = np.median(sample.reshape(-1,3),axis=0)
                if np.linalg.norm(color[1:]-128) <= cfg.max_chroma:
                    road.append(color)
                    textures.append(float(gray_image[yy:yy+patch,xx:xx+patch].std()))
        if len(road) < cfg.min_road_patches:
            return None, len(road)
        texture_limit = max(cfg.texture_floor,float(np.median(textures)))*cfg.max_texture_ratio
        return dict(color=np.median(road,axis=0).tolist(),texture_limit=texture_limit), len(road)


def visible_box(box, shape, ratio):
    h,w = shape[:2]
    x0,y0,x1,y1 = box
    total = max(0,x1-x0)*max(0,y1-y0)
    inside = max(0,min(w,x1)-max(0,x0))*max(0,min(h,y1)-max(0,y0))
    return total > 0 and inside/total >= ratio
