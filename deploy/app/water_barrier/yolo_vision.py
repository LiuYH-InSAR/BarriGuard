"""Water-specific road evidence; shared registration lives in camera_motion."""
import cv2 as cv
import numpy as np
from camera_motion.vision import Alignment, Vision as BackgroundVision, gray


class Vision(BackgroundVision):
    def road_evidence(self, frame, expected, index, matches, rows, previous_missing):
        """从目标下方、排除全部预期水马的多个背景块提取局部路面模型。

        这是背景相似度证据，不是路面语义识别。多只同步消失先保留 suspect。
        """
        cfg = self._p.evidence
        box = expected[index]
        height,width = frame.shape[:2]
        x0,y0,x1,y1 = [int(round(v)) for v in box]
        x0,x1 = max(0,x0),min(width,x1)
        y0,y1 = max(0,y0),min(height,y1)
        if x1-x0 < cfg.patch_px or y1-y0 < cfg.patch_px:
            return False,dict(reason='small_or_outside')
        row = next((r for r in rows if index in r),[index])
        others = [i for i in row if i != index]
        visible = sum(i in matches for i in others)/max(1,len(others))
        absent = [i for i in row if i not in matches]
        newly_missing = [i for i in absent if i not in previous_missing
                         and min(expected[i][2]-expected[i][0],expected[i][3]-expected[i][1]) >= cfg.patch_px]
        if len(newly_missing) >= cfg.simultaneous_min:
            return False,dict(reason='simultaneous_disappearance',neighbor_ratio=visible)
        if not others or visible < cfg.neighbor_ratio:
            return False,dict(reason='neighbors_unreliable',neighbor_ratio=visible)
        lab = cv.cvtColor(frame,cv.COLOR_BGR2LAB).astype(np.float32)
        gray_image = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        forbidden = np.zeros((height,width),np.uint8)
        for a,b,c,d in expected:
            cv.rectangle(forbidden,(max(0,int(a)),max(0,int(b))),
                         (min(width-1,int(c)),min(height-1,int(d))),1,-1)
        band_end = min(height,y1+int((y1-y0)*cfg.band_height_ratio))
        road = []
        textures = []
        patch = cfg.patch_px
        for yy in range(y1,band_end-patch+1,patch):
            for xx in range(x0,x1-patch+1,patch):
                if forbidden[yy:yy+patch,xx:xx+patch].any():
                    continue
                sample = lab[yy:yy+patch,xx:xx+patch]
                color = np.median(sample.reshape(-1,3),axis=0)
                if np.linalg.norm(color[1:]-128) > cfg.max_chroma:
                    continue
                road.append(color)
                textures.append(float(gray_image[yy:yy+patch,xx:xx+patch].std()))
        if len(road) < cfg.min_road_patches:
            return False,dict(reason='insufficient_road_samples',road_patches=len(road))
        target = np.median(road,axis=0)
        texture_limit = max(cfg.texture_floor,float(np.median(textures)))*cfg.max_texture_ratio
        votes = []
        for yy in range(y0,y1-patch+1,patch):
            for xx in range(x0,x1-patch+1,patch):
                sample = lab[yy:yy+patch,xx:xx+patch]
                color = np.median(sample.reshape(-1,3),axis=0)
                distance = float(np.linalg.norm(color-target))
                texture = float(gray_image[yy:yy+patch,xx:xx+patch].std())
                votes.append(distance <= cfg.color_distance and texture <= texture_limit)
        fraction = float(np.mean(votes)) if votes else 0.0
        return fraction >= cfg.min_road_fraction,dict(reason='road_exposed' if fraction >= cfg.min_road_fraction
            else 'background_not_confirmed',road_fraction=fraction,road_patches=len(road),neighbor_ratio=visible,
            road_color=target.tolist(),texture_limit=texture_limit)


def visible_box(box, shape, ratio):
    h,w = shape[:2]
    x0,y0,x1,y1 = box
    total = max(0,x1-x0)*max(0,y1-y0)
    inside = max(0,min(w,x1)-max(0,x0))*max(0,min(h,y1)-max(0,y0))
    return total > 0 and inside/total >= ratio
