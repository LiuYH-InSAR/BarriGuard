"""Offline fix: thresholds and observation evidence belong to one pixel support.

When a neighboring emitter is admitted, a shrinking mask changes the sampled
signal. Old calibration thresholds must not be applied to this new signal, and
old counts/time must not be silently carried into the new observation epoch.
No annotation coordinates, camera IDs, or reviewed target types are inputs.
"""
import numpy as np
from .adaptive_v6 import AdaptiveV6, V6Params, ObservationAuditV6
from .adaptive_tracking_v5 import ContextFlashSignal, midpoint,nearby_centers
from .adaptive_flash_signal import SignalParams
from .adaptive_prototype import PulseCounter


class SupportEpochSignal(ContextFlashSignal):
    def __init__(self,params=None):
        super().__init__(params)
        self.support_epoch=0

    def desired_support(self,box):
        radius=self.radius
        if radius is None:
            radius=int(max(2,min(18,min(box[2]-box[0],box[3]-box[1])/2)))
        yy,xx=np.indices((2*radius+1,2*radius+1))
        distance=np.hypot(xx-radius,yy-radius);keep=np.ones(distance.shape,bool)
        for point in nearby_centers(box,radius,self.neighbor_centers):
            delta=np.asarray(point)-midpoint(box)
            neighbor=np.hypot(xx-radius-delta[0],yy-radius-delta[1])
            keep &= (distance<=neighbor)&(neighbor>3.)
        return keep

    def observe(self,plane,box,ts,valid):
        change=None
        if valid and self.context is not None and self.support_mask is not None:
            keep=self.support_mask & self.desired_support(box)
            removed=int(self.support_mask.sum()-keep.sum())
            if removed:
                change=dict(old_pixels=int(self.support_mask.sum()),new_pixels=int(keep.sum()),
                            removed_pixels=removed,old_amplitude=self.amplitude,
                            old_calibration_s=self.calibration_valid_s)
                # Reset every quantity calibrated on the old spatial support.
                # Preserve current-frame inputs (context/neighbor/raw pixels)
                # and keep exclusions monotonic throughout the track lifetime.
                radius=self.radius
                ContextFlashSignal.__init__(self,self.p)
                # Geometry jitter may cross an integer radius boundary. The
                # new calibration keeps the existing patch grid so the saved
                # exclusion mask and current raw-peak pixels have equal shape.
                self.radius=radius
                self.support_mask=keep
                self.support_epoch+=1
        result=super().observe(plane,box,ts,valid)
        result.update(support_epoch=self.support_epoch,support_changed=change,
                      support_pixels=None if self.support_mask is None else int(self.support_mask.sum()))
        self.last=result
        return result


class AdaptiveV7(AdaptiveV6):
    def influential_tracks(self,owner,ts):
        # Separate high-confidence optical hypotheses before calibrating a new
        # candidate, rather than waiting for their third hit to change its
        # signal definition. An established lamp still cannot lose pixels to
        # an unconfirmed hypothesis. Neither branch uses reviewed positions.
        if owner.confirmed:return super().influential_tracks(owner,ts)
        return [t for t in self.tracks.values() if t.id!=owner.id
                and t.confidence>=self.p.candidate_confidence]

    def make_flash_signal(self):
        signal=SupportEpochSignal(SignalParams(max_gap_s=self.p.max_gap_s))
        signal.min_peak_v=self.p.emission_peak_v;signal.min_absolute_rise=self.p.emission_rise
        return signal

    def after_flash_observation(self,track,observation,ts):
        change=observation.get('support_changed')
        if change is None:return
        previous=dict(pulses=track.pulses.count,valid_s=track.pulses.valid_s)
        track.discarded_valid_s+=track.pulses.valid_s
        track.pulses=PulseCounter();track.observations.clear();track.confirmed=False
        track.source='signal_support_recalibration'
        self.epoch[track.id]=self.epoch.get(track.id,0)+1
        self.events.append(dict(event='signal_support_changed',ts=ts,id=track.id,
            observation_reset=True,before=previous,after_pulses=0,after_valid_s=0.,
            support=change,epoch=self.epoch[track.id]))


class LampPipelineV7:
    def __init__(self,backend,params=None,vision=None):
        self.detector=AdaptiveV7(backend,params or V6Params(),vision)
        self.audit=ObservationAuditV6()

    def step(self,frame,ts,frame_id,night=False):
        return self.audit.apply(self.detector.step(frame,ts,frame_id,night))
