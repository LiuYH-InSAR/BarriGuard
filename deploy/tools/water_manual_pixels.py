"""Optional dense poly presence recovery; never produces a missing vote.

Hypothesis: dense connected colour support survives narrow-window geometry.
Tiny/unexposed polys and substantial reference colour loss remain unknown.
"""
import cv2 as cv
import numpy as np


class PixelFallback:
    def __init__(self, poly, reference, spec, expected):
        self.spec,self.expected = spec,expected
        pts=np.rint(poly).astype(np.int32)
        self.x,self.y,self.w,self.h=cv.boundingRect(pts)
        self.w=min(self.w,reference.shape[1]-self.x)
        self.h=min(self.h,reference.shape[0]-self.y)
        self.mask=np.zeros((self.h,self.w),np.uint8)
        cv.fillPoly(self.mask,[pts-[self.x,self.y]],1)
        self.area=int(self.mask.sum())
        self.reference_ratio=self.measure(reference,1.)['colour_fraction']

    def measure(self, frame, factor):
        spec=self.spec
        image=frame[self.y:self.y+self.h,self.x:self.x+self.w]
        hsv=cv.cvtColor(image,cv.COLOR_BGR2HSV).astype(float)
        h,s,v=hsv.transpose(2,0,1)
        valid=(self.mask>0)&(v>=spec.dark_v)&~((v>=250)&(s<10))
        corrected=np.minimum(v/max(factor,.01),255) if spec.light_colour else v
        red=((h<=12)|(h>=165))&(s>=70)&(corrected>=50)
        white=((s<=70)&(corrected>=150))|((s>70)&(s<=120)&(corrected>=80)&(corrected<150)&(h>=85))
        if spec.gray_white:
            spread=image.max(axis=2).astype(float)-image.min(axis=2)
            if spec.light_colour:
                spread/=max(factor,.01)
            white|=(corrected>=spec.gray_v_min)&(s<=spec.gray_s_max)&(spread<=spec.gray_rgb_spread)
        candidate=valid&(red if self.expected==0 else white&~red)
        _,_,stats,_=cv.connectedComponentsWithStats(candidate.astype(np.uint8),8)
        largest=int(stats[1:,cv.CC_STAT_AREA].max()) if len(stats)>1 else 0
        support=int(valid.sum())
        return dict(poly_pixels=self.area,valid_pixels=support,valid_fraction=support/max(self.area,1),
                    colour_pixels=int(candidate.sum()),colour_fraction=float(candidate.sum()/max(support,1)),
                    largest_component=largest)

    def assess(self, frame, factor):
        spec=self.spec
        info=self.measure(frame,factor)
        required=max(spec.pixel_component_min,int(np.ceil(info['valid_pixels']*spec.pixel_component_fraction)))
        conflict=self.reference_ratio-info['colour_fraction']>=spec.colour_loss
        info.update(expected='red' if self.expected==0 else 'white',required_component=required,
                    reference_colour_fraction=self.reference_ratio,reference_colour_loss=bool(conflict))
        info['recovered']=bool(info['valid_pixels']>=spec.pixel_min_pixels
            and info['valid_fraction']>=spec.block_valid
            and info['colour_fraction']>=spec.colour_min
            and info['largest_component']>=required and not conflict)
        return info
