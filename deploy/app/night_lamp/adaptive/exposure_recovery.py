"""Causal recovery of a stale photometric baseline; no geometry/health decisions."""
from collections import deque
import numpy as np
from .adaptive_v7 import SupportEpochSignal


class ExposureEvidence:
    def __init__(self,params):self.p=params;self.samples=deque(maxlen=60)
    def clear(self):self.samples.clear()
    def observe(self,ts,photometry,allowed,hits):
        if not allowed or photometry is None:
            self.clear();return None
        vector=np.asarray(photometry,float)
        if vector.shape!=(3,) or not np.isfinite(vector).all():self.clear();return None
        if self.samples and not 0<ts-self.samples[-1][0]<=self.p.max_gap_s:self.clear()
        tolerance=self.p.exposure_limit*.25
        if self.samples:
            values=np.asarray([v for _,v,_ in self.samples]+[vector])
            if float(np.ptp(values,axis=0).max())>tolerance:self.clear()
        self.samples.append((ts,vector.copy(),hits))
        while self.samples and ts-self.samples[0][0]>max(5.,self.p.calibration_s+self.p.max_gap_s):self.samples.popleft()
        span=ts-self.samples[0][0];updates=hits-self.samples[0][2]
        if len(self.samples)<self.p.calibration_samples or span<self.p.calibration_s or updates<1:return None
        return dict(samples=len(self.samples),span_s=span,detector_updates=updates,
            maximum_photometry_range=float(np.ptp(np.asarray([v for _,v,_ in self.samples]),axis=0).max()),photometry=vector.tolist())


class RecoverableFlashSignal(SupportEpochSignal):
    recovery_allowed=False
    detector_hits=0
    def __init__(self,params=None):
        super().__init__(params);self.exposure_evidence=ExposureEvidence(self.p)
    def observe(self,plane,box,ts,valid):
        result=super().observe(plane,box,ts,valid)
        if result.get('reason')!='background_exposure_change':
            self.exposure_evidence.clear();return result
        proof=self.exposure_evidence.observe(ts,self.context,valid and self.recovery_allowed,self.detector_hits)
        if proof is None:return result
        previous=dict(background=None if self.background is None else self.background.tolist(),amplitude=self.amplitude,
            calibration_s=self.calibration_valid_s,support_epoch=self.support_epoch)
        radius=self.radius;mask=None if self.support_mask is None else self.support_mask.copy();epoch=self.support_epoch
        # Same pixel region and monotonic exclusions. No old template, pulse
        # threshold or absolute-emission calibration survives the new reference.
        RecoverableFlashSignal.__init__(self,self.p)
        self.radius=radius;self.support_mask=mask;self.support_epoch=epoch+1
        self.previous_ts=ts;self.previous_valid=False
        result=dict(valid=False,ready=False,signal=0.,raw=None,reason='calibrating_exposure_baseline',amplitude=None,
            noise=0.,calibration_s=0.,support_epoch=self.support_epoch,support_changed=None,
            support_pixels=None if mask is None else int(mask.sum()),emission_supported=False,
            exposure_reference_changed=dict(before=previous,proof=proof))
        self.last=result;return result
