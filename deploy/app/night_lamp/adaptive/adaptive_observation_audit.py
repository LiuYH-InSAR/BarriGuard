"""Read-only evidence admission after detection/tracking, independent of weights.

Nested lamp-body/head identities are not independent lamps. Until resolved, an
observation window must not interpret their mutually masked pixels as no-flash
evidence. Quarantine is deliberately sticky for this single observation window.
Raw diagnostics are preserved; no counts, timing or detector state are modified.
"""
from copy import deepcopy


def nested_conflict(a,b):
    a=a.get('body_box',a['box']);b=b.get('body_box',b['box'])
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    if min(aa,bb)<=0:return False
    return overlap/min(aa,bb)>=.9 and max(aa,bb)/min(aa,bb)>=1.4 and overlap/(aa+bb-overlap)<.5


class ObservationAudit:
    def __init__(self):self.quarantined=set()

    def conflicts(self,a,b):return nested_conflict(a,b)

    def apply(self,result):
        out=deepcopy(result);new=set();tracks=out['tracks']
        for i,a in enumerate(tracks):
            for b in tracks[i+1:]:
                if self.conflicts(a,b):new.update([a['id'],b['id']])
        self.quarantined.update(new)
        for t in tracks:
            if any(alias in self.quarantined for alias in t.get('aliases',[])):self.quarantined.add(t['id'])
            t.update(raw_pulses=t['pulses'],raw_valid_s=t['valid_s'],raw_confirmed=t['confirmed'],
                     raw_window_status=t.get('window_status'),evidence_admitted=t['id'] not in self.quarantined)
            if t['id'] not in self.quarantined:continue
            t.update(pulses=0,valid_s=0.,confirmed=False,position_observable=False,
                     pending_reason='nested_identity_ambiguous',window_status='incomplete',health_status='unobservable')
            t.setdefault('quality_details',{}).setdefault('reasons',[]).append('nested_identity_ambiguous')
            t.setdefault('flash_signal',{}).update(valid=False,reason='nested_identity_ambiguous')
        out['confirmed']=[t for t in tracks if t['confirmed']]
        out['observation_audit']=dict(quarantined_ids=sorted(self.quarantined),policy='nested_identity_window_quarantine')
        return out


class LampPipelineV5:
    """Consumer-facing prototype: tracker proposals followed by evidence audit."""
    def __init__(self,backend,params=None,vision=None):
        from .adaptive_v5 import AdaptiveV5
        self.detector=AdaptiveV5(backend,params,vision);self.audit=ObservationAudit()

    def step(self,frame,ts,frame_id,night=False):
        return self.audit.apply(self.detector.step(frame,ts,frame_id,night))
