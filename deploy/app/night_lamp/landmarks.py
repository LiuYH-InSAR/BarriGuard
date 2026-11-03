"""Frozen body/background evidence in explicitly versioned reference pixels."""
import base64
from functools import lru_cache
import cv2 as cv
import numpy as np

SIZE=64


def matrix(value):
    m=np.asarray(value,float)
    if m.shape!=(3,3) or not np.isfinite(m).all() or abs(np.linalg.det(m))<1e-8:
        raise ValueError('Invalid reference transform')
    return m


def inside(box,shape,margin=0.):
    h,w=shape[:2];a=np.asarray(box,float)
    return bool(a.shape==(4,) and np.isfinite(a).all() and
                margin<=a[0]<a[2]<=w-margin and margin<=a[1]<a[3]<=h-margin)


def midpoint(box):return (np.asarray(box[:2])+box[2:4])/2


def correlation(a,b):
    a=np.asarray(a,float).ravel();b=np.asarray(b,float).ravel()
    a=a-a.mean();b=b-b.mean();den=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(a@b/den) if den>1e-6 else 0.


@lru_cache(maxsize=128)
def decode(encoded):
    data=np.frombuffer(base64.b64decode(encoded,validate=True),np.uint8)
    image=cv.imdecode(data,cv.IMREAD_GRAYSCALE)
    if image is None or image.shape!=(SIZE,SIZE):raise ValueError('Invalid body reference PNG')
    return image


def capture(frame,box,frame_to_reference,reference,light,stamp):
    if not inside(box,frame.shape):return None
    x,y=midpoint(box);w,h=np.asarray(box[2:4])-box[:2]
    side=max(w,h)*3.+8.
    # Unit image pixels -> original source-frame pixels -> shared reference.
    unit=np.array([[side/(SIZE-1),0,x-side/2],[0,side/(SIZE-1),y-side/2],[0,0,1.]])
    corners=cv.perspectiveTransform(np.float32([[[0,0],[63,0],[63,63],[0,63]]]),unit)[0]
    if not inside([*corners.min(0),*corners.max(0)],frame.shape,1):return None
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    patch=cv.warpPerspective(gray,np.linalg.inv(unit),(SIZE,SIZE))
    inverse=np.linalg.inv(unit)
    body=cv.perspectiveTransform(np.float32([[box[:2],box[2:4]]]),inverse)[0]
    bounds=[int(np.floor(body[0,0])),int(np.floor(body[0,1])),int(np.ceil(body[1,0])),int(np.ceil(body[1,1]))]
    ok,encoded=cv.imencode('.png',patch)
    if not ok:raise OSError('Body reference encoding failed')
    return dict(reference=reference,light=light,captured_at=stamp,body=bounds,
                unit_to_reference=(matrix(frame_to_reference)@unit).tolist(),
                png=base64.b64encode(encoded).decode('ascii'))


def project(frame,profile,reference_to_frame):
    transform=matrix(reference_to_frame)@matrix(profile['unit_to_reference'])
    corners=cv.perspectiveTransform(np.float32([[[0,0],[63,0],[63,63],[0,63]]]),transform)[0]
    if not inside([*corners.min(0),*corners.max(0)],frame.shape,1):return None
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    return cv.warpPerspective(gray,np.linalg.inv(transform),(SIZE,SIZE))


def compare(frame,profile,reference_to_frame):
    now=project(frame,profile,reference_to_frame)
    if now is None:return dict(verdict='unknown',reason='body_context_outside_frame')
    ref=decode(profile['png']);a,b,c,d=profile['body']
    if not(1<=a<c<63 and 1<=b<d<63):return dict(verdict='unknown',reason='invalid_body_mask')
    mask=np.zeros((SIZE,SIZE),np.uint8);mask[max(0,b-2):min(SIZE,d+2),max(0,a-2):min(SIZE,c+2)]=255
    ring=mask==0;body=np.zeros_like(ring);body[b:d,a:c]=True
    rv,nv=ref[ring].astype(float),now[ring].astype(float)
    rlo,rhi=np.percentile(rv,[10,90]);nlo,nhi=np.percentile(nv,[10,90])
    if rhi-rlo<8 or nhi-nlo<5:return dict(verdict='unknown',reason='body_context_low_texture')
    gain=(rhi-rlo)/(nhi-nlo)
    if not .5<=gain<=2.:return dict(verdict='unknown',reason='body_context_photometry')
    normalized=np.clip((now.astype(float)-np.median(nv))*gain+np.median(rv),0,255)
    corr=correlation(ref[ring],normalized[ring]);error=float(np.median(np.abs(ref[ring]-normalized[ring])))
    evidence=dict(context_correlation=corr,context_error=error,
                  body_correlation=correlation(ref[body],normalized[body]))
    if corr<.85 or error>12:return dict(verdict='unknown',reason='body_context_changed_or_occluded',**evidence)
    actual=float(np.mean(np.abs(ref[body]-normalized[body])))
    if actual<10 or (evidence['body_correlation']>=.80 and actual<22):
        return dict(verdict='present',reason='frozen_body_template_present',**evidence)
    empty=cv.inpaint(ref,mask,5,cv.INPAINT_TELEA).astype(float)
    distinctive=np.abs(ref.astype(float)-empty)>18
    distinctive &= body
    if distinctive.sum()<max(12,.20*body.sum()):
        return dict(verdict='unknown',reason='body_not_distinguishable_from_background',**evidence)
    residual=np.abs(normalized-empty)
    empty_error=float(np.mean(residual[body]))
    removed=float(np.mean(residual[distinctive]<12))
    evidence.update(empty_background_error=empty_error,removed_body_fraction=removed)
    # A missing box alone is never absence. Require an independently frozen
    # body to disappear into the continuation of an unchanged textured context.
    if empty_error<=8 and removed>=.80:
        return dict(verdict='absent',reason='body_removed_background_exposed',**evidence)
    return dict(verdict='unknown',reason='body_changed_or_locally_occluded',**evidence)


def migrate_profiles(profiles,old_to_new,new_reference):
    for profile in profiles.values():
        profile['unit_to_reference']=(matrix(old_to_new)@matrix(profile['unit_to_reference'])).tolist()
        profile['reference']=new_reference
