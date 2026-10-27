"""固定背景对齐、图像质量与保守路面证据。没有语义遮挡模型。"""
from collections import defaultdict, deque
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from .yolo_geometry import warp_boxes


def gray(frame, width):
    scale = min(1.0, width / frame.shape[1])
    image = cv.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY), scale


@dataclass
class Alignment:
    valid: bool
    matrix: object
    displacement: float
    details: dict


class Vision:
    def __init__(self, params):
        self._p = params
        self._orb = cv.ORB_create(nfeatures=params.motion.features)
        self._quality = defaultdict(lambda: deque(maxlen=params.reference.quality_samples))
        self.last_matches = None

    def quality(self, frame, learn=False):
        image, _ = gray(frame, self._p.motion.width)
        cfg = self._p.reference
        luma = float(np.median(image))
        upper_height = max(1,round(image.shape[0]*cfg.lighting_upper_fraction))
        mask = self._background_mask(image.shape,1)[:upper_height] > 0
        upper = image[:upper_height][mask]
        upper_luma = float(np.median(upper)) if upper.size else luma
        light = 'night' if luma < cfg.night_luma else 'day' if luma >= cfg.day_luma else 'transition'
        blur = float(cv.Laplacian(image, cv.CV_32F).var())
        contrast = float(np.std(image))
        clipped = float(np.mean((image <= cfg.dark_pixel) | (image >= cfg.bright_pixel)))
        points = self._orb.detect(image, None)
        history = self._quality[light]
        adaptive = np.median(history) * cfg.adaptive_quality_ratio if history else cfg.blur_floor
        threshold = max(cfg.blur_floor, float(adaptive))
        ok = (blur >= threshold and contrast >= cfg.contrast_floor
              and clipped <= cfg.clipped_max and len(points) >= cfg.min_features)
        if ok and learn:
            history.append(blur)
        return dict(ok=bool(ok), light=light, luma=luma, upper_luma=upper_luma,blur=blur, blur_threshold=threshold,
                    contrast=contrast, clipped=clipped, features=len(points))

    def align(self, reference, frame, exclude=()):
        result = self._feature_align(reference,frame,exclude)
        if result.valid or reference.shape != frame.shape:
            return result
        recovery = self._background_align(reference,frame,exclude)
        if recovery.valid:
            self.last_matches = None
            return recovery
        result.details['background_recovery'] = recovery.details
        return result

    def _background_mask(self, shape, scale, exclude=()):
        mask = np.full(shape,255,np.uint8)
        cfg = self._p.motion
        for x0,y0,x1,y1 in exclude:
            w,h = x1-x0,y1-y0
            cv.rectangle(mask,(int((x0-w*cfg.feature_margin)*scale),int((y0-h*cfg.feature_margin)*scale)),
                (int((x1+w*cfg.feature_margin)*scale),int((y1+h*cfg.feature_margin)*scale)),0,-1)
        for a,b,c,d in cfg.overlay_exclusions:
            cv.rectangle(mask,(int(a*shape[1]),int(b*shape[0])),
                         (int(c*shape[1]),int(d*shape[0])),0,-1)
        return mask

    def _feature_align(self, reference, frame, exclude=()):
        cfg = self._p.motion
        self.last_matches = None
        if reference.shape != frame.shape:
            return Alignment(False, None, 0.0, {'reason':'resolution_changed'})
        a, scale = gray(reference, cfg.width)
        b, _ = gray(frame, cfg.width)
        mask = self._background_mask(a.shape,scale,exclude)
        # 匹配描述子前增强局部对比，减少日出日落的整体亮度影响。
        aa,bb = cv.equalizeHist(a),cv.equalizeHist(b)
        ka,da = self._orb.detectAndCompute(aa, mask)
        kb,db = self._orb.detectAndCompute(bb,self._background_mask(b.shape,scale))
        if da is None or db is None:
            return Alignment(False,None,0.0,{'reason':'no_features'})
        pairs = cv.BFMatcher(cv.NORM_HAMMING).knnMatch(da,db,k=2)
        good = [pair[0] for pair in pairs if len(pair)==2 and pair[0].distance < cfg.ratio_test*pair[1].distance]
        if len(good) < cfg.min_matches:
            return Alignment(False,None,0.0,{'reason':'few_matches','matches':len(good)})
        src = np.float32([ka[m.queryIdx].pt for m in good])
        dst = np.float32([kb[m.trainIdx].pt for m in good])
        affine, inliers = cv.estimateAffinePartial2D(src,dst,method=cv.RANSAC,
                                                   ransacReprojThreshold=cfg.ransac_px)
        if affine is None or inliers is None:
            return Alignment(False,None,0.0,{'reason':'ransac_failed'})
        keep = inliers.ravel().astype(bool)
        transformed = cv.transform(src[None],affine)[0]
        error = float(np.median(np.linalg.norm(transformed[keep]-dst[keep],axis=1)))
        cells = {(min(cfg.grid-1,int(x/a.shape[1]*cfg.grid)),min(cfg.grid-1,int(y/a.shape[0]*cfg.grid)))
                 for x,y in src[keep]}
        dst_cells = {(min(cfg.grid-1,int(x/a.shape[1]*cfg.grid)),min(cfg.grid-1,int(y/a.shape[0]*cfg.grid)))
                     for x,y in dst[keep]}
        scaling = float(np.hypot(affine[0,0],affine[1,0]))
        matrix = np.eye(3)
        matrix[:2] = affine
        matrix[:2,2] /= scale
        corners = np.float32([[0,0],[frame.shape[1],0],[0,frame.shape[0]],[frame.shape[1],frame.shape[0]]])
        shifted = cv.perspectiveTransform(corners[None],matrix)[0]
        displacement = float(np.max(np.linalg.norm(shifted-corners,axis=1))/np.hypot(*frame.shape[:2]))
        stationary = (displacement <= cfg.jitter_floor
                      and min(len(cells),len(dst_cells)) >= cfg.stationary_min_cells
                      and int(keep.sum()) >= cfg.stationary_min_inliers
                      and float(keep.mean()) >= cfg.stationary_min_inlier_ratio)
        valid = (int(keep.sum()) >= cfg.min_inliers and float(keep.mean()) >= cfg.min_inlier_ratio
                 and (min(len(cells),len(dst_cells)) >= cfg.min_cells or stationary)
                 and error <= cfg.max_error_px and abs(scaling-1) <= cfg.max_scale_error)
        self.last_matches = (reference,frame,src[keep]/scale,dst[keep]/scale)
        details = dict(reason='ok' if valid else 'quality_failed',method='orb',matches=len(good),
                       inliers=int(keep.sum()),inlier_ratio=float(keep.mean()),
                       cells=len(cells),destination_cells=len(dst_cells),error_px=error,
                       scaling=scaling,displacement=displacement,stationary_fallback=bool(stationary
                           and min(len(cells),len(dst_cells)) < cfg.min_cells))
        return Alignment(bool(valid),matrix,displacement,details)

    def _background_align(self, reference, frame, exclude=()):
        """独立背景块的梯度相位一致性，只恢复近静止视角，不能确认移动。"""
        cfg = self._p.alignment_recovery
        a,scale = gray(reference,self._p.motion.width)
        b,_ = gray(frame,self._p.motion.width)
        mask = self._background_mask(a.shape,scale,exclude)
        def edges(image):
            image = cv.GaussianBlur(image,(cfg.blur_kernel,cfg.blur_kernel),0)
            return cv.magnitude(cv.Sobel(image,cv.CV_32F,1,0,ksize=cfg.gradient_kernel),
                                cv.Sobel(image,cv.CV_32F,0,1,ksize=cfg.gradient_kernel))
        aa,bb = edges(a),edges(b)
        patches = []
        for row in range(cfg.rows):
            y0,y1 = row*a.shape[0]//cfg.rows,(row+1)*a.shape[0]//cfg.rows
            for col in range(cfg.columns):
                x0,x1 = col*a.shape[1]//cfg.columns,(col+1)*a.shape[1]//cfg.columns
                if np.mean(mask[y0:y1,x0:x1] == 0) > cfg.max_excluded_fraction:
                    continue
                src,dst = aa[y0:y1,x0:x1].copy(),bb[y0:y1,x0:x1].copy()
                if min(float(src.std()),float(dst.std())) < cfg.min_texture or min(src.shape) <= 1:
                    continue
                shift,response = cv.phaseCorrelate(src,dst,cv.createHanningWindow((x1-x0,y1-y0),cv.CV_32F))
                if (np.isfinite(response) and np.isfinite(shift).all() and response >= cfg.min_response
                        and np.linalg.norm(shift) <= self._p.motion.jitter_floor*np.hypot(*a.shape)):
                    patches.append(dict(row=row,col=col,center=[(x0+x1)/2,(y0+y1)/2],
                                        shift=list(shift),response=float(response)))
        support = []
        if patches:
            shifts = np.asarray([r['shift'] for r in patches])
            groups = [np.flatnonzero(np.linalg.norm(shifts-v,axis=1) <= cfg.consistency_px) for v in shifts]
            support = [patches[i] for i in max(groups,key=len)]
        centers = np.array([p['center'] for p in support])
        valid = (len(support) >= cfg.min_support
            and len({p['col'] for p in support}) >= cfg.min_columns
            and len({p['row'] for p in support}) >= cfg.min_rows
            and np.ptp(centers[:,0]) >= a.shape[1]*cfg.min_span_x
            and np.ptp(centers[:,1]) >= a.shape[0]*cfg.min_span_y)
        matrix = np.eye(3)
        if support:
            matrix[:2,2] = np.median([p['shift'] for p in support],axis=0)/scale
        displacement = float(np.linalg.norm(matrix[:2,2])/np.hypot(*frame.shape[:2]))
        return Alignment(bool(valid),matrix,displacement,dict(method='background_phase',
            reason='stationary_background' if valid else 'insufficient_background_consensus',
            inliers=len(support),patches=patches,support=support,displacement=displacement,
            stationary_fallback=True))

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
