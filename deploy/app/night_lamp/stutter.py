"""High-mask fallback; a trigger is not proof of video freezing."""
import numpy as np
from .disturbance import detect

SETTINGS=dict(enabled=True,shared_candidate=True,mask_ratio=.8,window_s=20.,enter_delta=20.,duty_floor=.2,
              min_samples=40,onset_min=5,peak_floor=.1,min_peaks=3,gap_tol=5)


def duration_ratio(mask,times):
    t=np.asarray(times,float);mask=np.asarray(mask,bool)
    if len(t)<2 or t[-1]<=t[0]:return 0.
    # Each observation represents time until the next observation.
    return float(np.diff(t)@mask[:-1]/(t[-1]-t[0]))


def shared_branch(branch,count,total,valid,radius,duty_floor,max_area):
    """One candidate grouping pass, then cover the existing lookup radius."""
    import cv2 as cv
    duty=np.divide(count,total,out=np.zeros_like(count),where=total>0)
    _,labels,stats,_=cv.connectedComponentsWithStats(((duty>=duty_floor)&valid).astype(np.uint8),8)
    touched=np.unique(labels[branch&valid]);selected=np.zeros_like(branch)
    for label in touched:
        if label==0 or stats[label,cv.CC_STAT_AREA]>=max_area:continue
        selected|=labels==label
    if radius:
        yy,xx=np.ogrid[-radius:radius+1,-radius:radius+1]
        selected=cv.dilate(selected.astype(np.uint8),((yy*yy+xx*xx)<=radius*radius).astype(np.uint8))>0
    return branch|(selected&valid)


def detect_path(signal,times,parameters,settings=None,force=None):
    s=dict(settings or {'enabled':False})
    mask,mean,info=detect(signal,times,parameters)
    ratio=duration_ratio(mask,times)
    triggered=bool(s.get('enabled') and (ratio>s['mask_ratio'] if force is None else force))
    if triggered:
        selected={**parameters,'window_s':s['window_s']}
        if 'enter_delta' in s:selected['enter_delta']=s['enter_delta']
        mask,mean,info=detect(signal,times,selected)
    info.update(suspected_stutter=triggered,normal_mask_duration_ratio=ratio,
                mask_duration_ratio=duration_ratio(mask,times),fallback_settings=s)
    return mask,mean,info


def select_parameters(parameters,triggered):
    p=dict(parameters);s=p.get('stutter',{'enabled':False})
    if triggered and s.get('enabled'):
        p.update({k:s[k] for k in ('min_samples','onset_min','peak_floor','min_peaks','gap_tol','duty_floor')})
    return p


def mask_report(info,mask,times,folder=None,row=None,col=None):
    """Separate the trigger pixel's full-run ratio from the plotted mask union."""
    final=duration_ratio(mask,times)
    info['final_mask_duration_ratio']=final
    trigger=None
    if folder is not None:
        from pathlib import Path
        import json
        folder=Path(folder);path=folder/'normal_mask_duration_ratio.npy'
        if path.exists():
            meta=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
            y,x=row-meta['origin_row'],col-meta['origin_col']
            saved=np.load(path,mmap_mode='r')
            if 0<=y<saved.shape[0] and 0<=x<saved.shape[1]:trigger=float(saved[y,x])
    info['trigger_pixel_full_run_mask_ratio']=trigger
    lines=[f'Mask duration ratio: final plotted mask (including core union)={final:.2%}',
           f'Original 5s signal mask ratio={info["normal_mask_duration_ratio"]:.2%}; selected {info["parameters"]["window_s"]:g}s signal mask ratio={info["mask_duration_ratio"]:.2%}']
    if trigger is not None:
        threshold=info.get('fallback_settings',{}).get('mask_ratio')
        info['trigger_mask_threshold']=threshold
        verdict='PASS' if threshold is not None and trigger>threshold else 'FAIL' if threshold is not None else 'UNKNOWN'
        lines.append(f'Branch trigger: full-run pixel original 5s mask={trigger:.2%}; >{threshold:.0%}: {verdict}' if threshold is not None else f'Branch trigger: full-run pixel original 5s mask={trigger:.2%}; threshold UNKNOWN')
    return lines
