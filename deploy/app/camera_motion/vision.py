"""Independent background registration and image quality."""
from collections import defaultdict, deque
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from .geometry import warp_boxes


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

