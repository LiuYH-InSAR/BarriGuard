"""Temporal brightness disturbance masks; never compress excluded time."""
import numpy as np
from .periodicity import clean_train

PARAMETERS = dict(enabled=True, window_s=5.0, reference_s=30.0,
                  enter_delta=10.0, exit_delta=10.0,
                  enter_hold_s=.48, recovery_hold_s=.8, padding_s=1.0)


def describe_mask(mask,times):
    mask=np.asarray(mask,bool);t=np.asarray(times)
    edges=np.diff(np.r_[False,mask,False].astype(int))
    intervals=[[float(t[a]),float(t[b-1])] for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]
    return {'intervals':intervals,'masked_samples':int(mask.sum()),'valid_samples':int((~mask).sum())}


def detect_matrix(values,times,parameters=None):
    """Vectorized spatial blocks; each pixel still uses only its own time series."""
    p={**PARAMETERS,**(parameters or {})};x=np.asarray(values,np.float32);t=np.asarray(times,float)
    if not p['enabled']:return np.zeros(x.shape,bool)
    left=np.searchsorted(t,t-p['window_s']/2,'left');right=np.searchsorted(t,t+p['window_s']/2,'right')
    sums=np.concatenate((np.zeros((len(x),1)),np.cumsum(x,axis=1,dtype=np.float64)),axis=1)
    mean=(sums[:,right]-sums[:,left])/(right-left)
    reference=np.median(mean[:,t<=t[0]+p['reference_s']],axis=1)
    suspect=np.full(len(x),-1,int);start=suspect.copy();recovery=suspect.copy()
    edges=np.zeros((len(x),len(t)+1),np.int16)
    for i in range(len(t)):
        dev=np.abs(mean[:,i]-reference);idle=start<0
        suspect[idle&(dev<=p['enter_delta'])]=-1
        new=idle&(dev>p['enter_delta'])&(suspect<0);suspect[new]=i
        entered=idle&(suspect>=0)&((t[i]-t[np.maximum(suspect,0)])>=p['enter_hold_s'])
        start[entered]=suspect[entered];recovery[entered]=-1
        active=start>=0;recovery[active&(dev>p['exit_delta'])]=-1
        new=active&(dev<=p['exit_delta'])&(recovery<0);recovery[new]=i
        ended=active&(recovery>=0)&((t[i]-t[np.maximum(recovery,0)])>=p['recovery_hold_s'])
        ids=np.flatnonzero(ended)
        if len(ids):
            lo=np.searchsorted(t,t[start[ids]]-p['padding_s'],'left');hi=np.searchsorted(t,t[recovery[ids]]+p['padding_s'],'right')
            edges[ids,lo]+=1;edges[ids,hi]-=1
            start[ids]=suspect[ids]=recovery[ids]=-1
    ids=np.flatnonzero(start>=0)
    if len(ids):
        lo=np.searchsorted(t,t[start[ids]]-p['padding_s'],'left');edges[ids,lo]+=1;edges[ids,len(t)]-=1
    return np.cumsum(edges[:,:-1],axis=1)>0


def detect(gray, times, parameters=None):
    p = {**PARAMETERS, **(parameters or {})}
    x = np.asarray(gray,float); t=np.asarray(times,float)
    finite=np.isfinite(x)&np.isfinite(t)
    if not finite.any():
        return np.zeros(len(x),bool), np.full(len(x),np.nan), {'parameters':p,'intervals':[]}
    # Centered 2-second windows; sum/count ignore unavailable observations.
    sums=np.r_[0,np.cumsum(np.where(finite,x,0))]
    counts=np.r_[0,np.cumsum(finite)]
    left=np.searchsorted(t,t-p['window_s']/2,'left')
    right=np.searchsorted(t,t+p['window_s']/2,'right')
    mean=np.divide(sums[right]-sums[left],counts[right]-counts[left],out=np.full(len(x),np.nan),where=(counts[right]-counts[left])>0)
    reference=float(np.nanmedian(mean[finite & (t<=t[finite][0]+p['reference_s'])]))
    mask=np.zeros(len(x),bool); intervals=[]
    if p['enabled']:
        start=None; suspect=None; recovery=None
        for i in range(len(x)):
            if not finite[i]:
                continue
            deviation=abs(mean[i]-reference)
            if start is None:
                if deviation>p['enter_delta']:
                    suspect=i if suspect is None else suspect
                    if t[i]-t[suspect]>=p['enter_hold_s']:
                        start=suspect; recovery=None
                else:
                    suspect=None
            elif deviation<=p['exit_delta']:
                recovery=i if recovery is None else recovery
                if t[i]-t[recovery]>=p['recovery_hold_s']:
                    intervals.append([max(float(t[0]),float(t[start]-p['padding_s'])), min(float(t[-1]),float(t[recovery]+p['padding_s']))])
                    start=suspect=recovery=None
            else:
                recovery=None
        if start is not None:
            intervals.append([max(float(t[0]),float(t[start]-p['padding_s'])),float(t[-1])])
    merged=[]
    for lo,hi in intervals:
        if merged and lo<=merged[-1][1]:merged[-1][1]=max(hi,merged[-1][1])
        else:merged.append([lo,hi])
    for lo,hi in merged:mask|=(t>=lo)&(t<=hi)
    return mask,mean,dict(parameters=p,reference_mean=reference,masked_samples=int(mask.sum()),valid_samples=int((finite&~mask).sum()),intervals=merged,
        reference_scope='median of centered window means in first reference_s; must inspect if initial period disturbed')


def periodic_features(states, valid, lag_lo, lag_hi, period):
    """Lag pairs retain original sample indices; exclude pairs touching masks."""
    states=np.asarray(states,bool); valid=np.asarray(valid,bool)
    starts=valid & states & np.r_[True, valid[:-1]&~states[:-1]]
    onsets=int(starts.sum())
    if valid.all():
        from .periodicity import ac_limited
        peak,lag,curve=ac_limited(states,lag_lo,lag_hi)
    else:
        x=states.astype(float)-(states[valid].mean() if valid.any() else 0)
        energy=float(x[valid]@x[valid]);curve=[]
        for lag in range(lag_lo,min(lag_hi,len(x)-1)+1):
            pairs=valid[:-lag]&valid[lag:]
            curve.append(float(x[:-lag][pairs]@x[lag:][pairs])/energy if energy else 0.)
        curve=np.asarray(curve);j=int(curve.argmax()) if len(curve) else -1
        peak=float(curve[j]) if j>=0 else 0.;lag=lag_lo+j if j>=0 else -1
    periodic,peaks=clean_train(curve,lag_lo,period.peak_floor,period.min_peaks,period.gap_tol)
    return onsets,peak,lag,curve,periodic,peaks
