"""Causal, per-track flash signal for the offline prototype (8-bit pixel units).

The dark template and thresholds use only past usable samples. Calibration does
not retrospectively produce pulses. A threshold pair is frozen once learned;
noise is never stretched to unit amplitude. This is not an occlusion guarantee.
"""
from collections import deque
from dataclasses import dataclass
import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class SignalParams:
    calibration_s: float = 3.
    calibration_samples: int = 12
    min_amplitude: float = 18.
    min_snr: float = 4.
    min_state_samples: int = 3
    off_fraction: float = .30
    on_fraction: float = .65
    exposure_limit: float = 12.
    max_gap_s: float = .5


def signal_plane(frame):
    # Grayscale sees emission changes in saturated yellow bodies; the minimum
    # BGR channel also sees the white lamp core. V alone can be clipped in both
    # bright and dark phases. Blur reduces codec noise and subpixel sensitivity.
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    white=np.min(frame,axis=2)
    return cv.GaussianBlur(cv.addWeighted(gray,.75,white,.25,0),(3,3),0)


class AdaptiveFlashSignal:
    def __init__(self,params=None):
        self.p=params or SignalParams()
        self.samples=deque(maxlen=60);self.template=None;self.background=None;self.radius=None
        self.previous_ts=None;self.previous_valid=False;self.calibration_valid_s=0.
        self.history=deque(maxlen=60);self.low=None;self.amplitude=None;self.noise=0.
        self.last={}

    def _patch(self,plane,box):
        h,w=plane.shape
        x,y=(np.asarray(box[:2])+np.asarray(box[2:4]))/2
        if self.radius is None:
            self.radius=int(max(2,min(18,min(box[2]-box[0],box[3]-box[1])/2)))
        r=self.radius;s=3*r
        if x-s<0 or y-s<0 or x+s>=w or y+s>=h:return None
        patch=cv.getRectSubPix(plane,(2*s+1,2*s+1),(float(x),float(y))).astype(np.float32)
        mask=np.ones(patch.shape,bool);mask[s-r-1:s+r+2,s-r-1:s+r+2]=False
        ring=patch[mask];bg=np.percentile(ring,[20,50,80])
        core=patch[s-r:s+r+1,s-r:s+r+1]-bg[1]
        return core,bg

    @staticmethod
    def _contrast(core,template):
        residual=(core-template).ravel()
        # Static bright objects disappear under the per-pixel dark template.
        # The upper quarter retains a small emitter inside a larger lamp box.
        k=max(1,int(np.ceil(residual.size*.25)))
        return float(np.mean(np.partition(residual,-k)[-k:]))

    def observe(self,plane,box,ts,valid):
        sample=self._patch(plane,box) if valid else None
        reason='unusable_position_or_frame'
        if sample is not None:
            core,bg=sample
            # Conservatively pause on large local-background exposure changes.
            # A brightness change shared by the lamp and background is not a
            # lamp pulse. This heuristic cannot prove visibility or RTSP health.
            if self.background is None:self.background=bg.copy()
            if np.max(np.abs(bg-self.background))>self.p.exposure_limit:
                valid=False;reason='background_exposure_change'
        else:valid=False
        gap=None if self.previous_ts is None else ts-self.previous_ts
        connected=valid and self.previous_valid and gap is not None and 0<gap<=self.p.max_gap_s
        self.previous_ts=ts;self.previous_valid=bool(valid)
        result=dict(valid=False,signal=0.,raw=None,ready=False,reason=reason,
                    amplitude=self.amplitude,noise=self.noise,calibration_s=self.calibration_valid_s)
        if not valid:self.last=result;return result
        if self.template is None:
            if connected:self.calibration_valid_s+=gap
            self.samples.append(core.copy())
            if len(self.samples)<self.p.calibration_samples or self.calibration_valid_s<self.p.calibration_s:
                result.update(reason='calibrating_dark_baseline',calibration_s=self.calibration_valid_s)
                self.last=result;return result
            stack=np.stack(list(self.samples));self.template=np.percentile(stack,20,axis=0)
            self.history.extend(self._contrast(c,self.template) for c in stack)
            self.samples.clear()
        raw=self._contrast(core,self.template)
        self.history.append(raw)
        if self.amplitude is None:
            values=np.asarray(self.history);low,high=np.percentile(values,[10,90]);span=high-low
            quiet=values[values<=np.percentile(values,60)]
            self.noise=max(1.,1.4826*float(np.median(np.abs(quiet-np.median(quiet)))))
            enough=(span>=self.p.min_amplitude and span>=self.p.min_snr*self.noise
                    and np.count_nonzero(values>=low+self.p.on_fraction*span)>=self.p.min_state_samples
                    and np.count_nonzero(values<=low+self.p.off_fraction*span)>=self.p.min_state_samples)
            if enough:self.low=float(low);self.amplitude=float(span)
        ready=self.amplitude is not None
        result.update(valid=True,signal=(raw-self.low)/self.amplitude if ready else 0.,raw=raw,
                      ready=ready,reason='tracking_flash' if ready else 'no_reliable_flash_amplitude',
                      amplitude=self.amplitude,noise=self.noise,calibration_s=self.calibration_valid_s)
        self.last=result
        return result
