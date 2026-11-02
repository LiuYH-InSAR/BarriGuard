"""Local background and temporal emitter geometry; no camera-specific coordinates."""
from collections import deque
import cv2 as cv
import numpy as np
from .adaptive_flash_signal import signal_plane,AdaptiveFlashSignal


def midpoint(box):return (np.asarray(box[:2])+box[2:4])/2


def nearby_centers(box,radius,points):
    """Discard points provably unable to affect the square sampler's mask.

    Every pixel is at most sqrt(2)*radius from the center. A neighbor farther
    than 3*radius+4 is farther than every self-distance and the 3px exclusion
    disk, with a margin. Actual mask comparisons and thresholds stay unchanged.
    """
    if len(points)==0:return ()
    values=np.asarray(points,dtype=float)
    delta=values-midpoint(box)
    keep=(np.square(delta).sum(axis=1)<=(3*radius+4)**2)|(~np.isfinite(delta).all(axis=1))
    return values[keep]


def roi_signal_plane(frame,boxes):
    """Compute identical color features only where lamp samplers can read."""
    h,w=frame.shape[:2];regions=[]
    for box in boxes:
        c=midpoint(box);r=int(max(2,min(18,min(box[2]-box[0],box[3]-box[1])/2)))
        s=max(3*r,min(28,max(box[2]-box[0],box[3]-box[1])),8)+3
        region=[max(0,int(c[0]-s)),max(0,int(c[1]-s)),min(w,int(c[0]+s)+1),min(h,int(c[1]+s)+1)]
        changed=True
        while changed:
            changed=False
            for old in regions[:]:
                if min(region[2],old[2])>max(region[0],old[0]) and min(region[3],old[3])>max(region[1],old[1]):
                    region=[min(region[0],old[0]),min(region[1],old[1]),max(region[2],old[2]),max(region[3],old[3])]
                    regions.remove(old);changed=True
        regions.append(region)
    result=np.zeros((h,w),np.uint8)
    for x0,y0,x1,y1 in regions:result[y0:y1,x0:x1]=signal_plane(frame[y0:y1,x0:x1])
    return result,sum((x1-x0)*(y1-y0) for x0,y0,x1,y1 in regions)


class BackgroundAnchor:
    """Vote over textured background tiles, excluding the lamp and neighbors.

    The original anchor is frozen. Flash/glare in one tile does not invalidate
    every tile; broad disagreement still pauses. This is an occlusion heuristic.
    """
    def __init__(self,gray,box):
        self.radius=int(max(24,min(64,max(box[2]-box[0],box[3]-box[1])*2)))
        self.reference=self.patch(gray,box)
        # Frozen alternatives are created now, never from a later obstructed
        # frame. Only a low-texture result can request the wider comparison.
        self.wider=[]
        for radius in (64,96):
            if radius<=self.radius:continue
            other=object.__new__(BackgroundAnchor);other.radius=radius;other.wider=[]
            other.reference=other.patch(gray,box)
            if other.reference is not None:self.wider.append(other)

    def patch(self,gray,box):
        x,y=midpoint(box);s=self.radius
        if x-s<0 or y-s<0 or x+s>=gray.shape[1] or y+s>=gray.shape[0]:return None
        p=cv.getRectSubPix(gray,(2*s+1,2*s+1),(float(x),float(y)))
        return cv.resize(p,(64,64)).astype(np.float32)

    def evaluate(self,gray,box,neighbors=()):
        now=self.patch(gray,box)
        if now is None or self.reference is None:return dict(ok=False,reason='patch_outside_frame',tiles=0)
        mask=np.ones((64,64),bool);scale=64/(2*self.radius+1);c=midpoint(box)
        boxes=np.asarray([b[:4] for b in [box,*neighbors]],dtype=float)
        d=(boxes[:,:2]+boxes[:,2:4])/2-c
        half=np.maximum((boxes[:,2:4]-boxes[:,:2])*.8,6.)
        starts=np.floor((d-half)*scale+32).astype(int)
        ends=np.ceil((d+half)*scale+32).astype(int)
        visible=(ends[:,0]>=0)&(ends[:,1]>=0)&(starts[:,0]<64)&(starts[:,1]<64)
        for a,z in zip(starts[visible],ends[visible]):
            mask[max(0,a[1]):min(64,z[1]+1),max(0,a[0]):min(64,z[0]+1)]=False
        # Vectorized tile correlations, with a fixed reference and shared mask.
        def tiles(a):return a.reshape(4,16,4,16).transpose(0,2,1,3).reshape(16,256)
        weights=tiles(mask).astype(float);n=weights.sum(1);den=np.maximum(n,1)
        ref=tiles(self.reference);cur=tiles(now)
        ar=(ref-(ref*weights).sum(1)[:,None]/den[:,None])*weights
        ac=(cur-(cur*weights).sum(1)[:,None]/den[:,None])*weights
        vr=(ar*ar).sum(1);vc=(ac*ac).sum(1)
        textured=(n>=50)&(np.sqrt(vr/den)>=3.)
        corr=(ar*ac).sum(1)/np.sqrt(np.maximum(vr*vc,1e-8))
        count=int(textured.sum())
        if count<3:
            residual=np.abs(now-self.reference)[mask]
            if residual.size<150 or np.median(residual)>18 or np.mean(residual>35)>.25:
                return dict(ok=False,reason='background_changed_or_occluded',tiles=count)
            for other in self.wider:
                if other.radius<=self.radius:continue
                result=other.evaluate(gray,box,neighbors)
                if result['ok']:
                    result.update(context_radius=other.radius,expanded_context=True,local_tiles=count)
                    return result
                # Textured disagreement is occlusion/change, not an invitation
                # to keep expanding until something far away happens to agree.
                if result.get('reason')=='background_changed_or_occluded':return result
            return dict(ok=False,reason='background_low_texture',tiles=count)
        good=textured&(corr>=.6)&(vc>=vr*.15)
        fraction=float(good.sum()/count);median=float(np.median(corr[textured]))
        ok=bool(good.sum()>=3 and fraction>=.6)
        return dict(ok=ok,reason='background_stable' if ok else 'background_changed_or_occluded',
                    tiles=count,stable_tiles=int(good.sum()),stable_fraction=fraction,median_correlation=median,
                    photometry=np.percentile(now[mask],[20,50,80]).tolist())


class ContextFlashSignal(AdaptiveFlashSignal):
    """Use the masked, wider context instead of a glow-contaminated inner ring."""
    context=None
    peak_v=None
    min_peak_v=150.
    min_absolute_rise=12.
    neighbor_centers=()
    raw_peak_pixels=None

    def __init__(self,params=None):
        super().__init__(params);self.absolute_samples=deque(maxlen=60);self.absolute_template=None;self.absolute_core=None;self.support_mask=None

    def _patch(self,plane,box):
        sample=super()._patch(plane,box)
        if sample is None or self.context is None:return sample
        core,inner=sample;outer=np.asarray(self.context,float)
        self.absolute_core=core+inner[1]
        relative=self.absolute_core-outer[1]
        yy,xx=np.indices(core.shape);r=(core.shape[0]-1)/2;c=midpoint(box)
        distance=np.hypot(xx-r,yy-r);keep=np.ones(core.shape,bool)
        for point in nearby_centers(box,r,self.neighbor_centers):
            d=np.asarray(point)-c;neighbor=np.hypot(xx-r-d[0],yy-r-d[1])
            keep &= (distance<=neighbor)&(neighbor>3.)
        # Exclusions only grow during this calibration epoch. Disappearing
        # candidate boxes must not expose new pixels and fabricate a pulse.
        self.support_mask=keep if self.support_mask is None else self.support_mask&keep
        mask=self.support_mask
        relative[~mask]=self.template[~mask] if self.template is not None else 0.
        self.absolute_core[~mask]=self.absolute_template[~mask] if self.absolute_template is not None else 0.
        if self.raw_peak_pixels is not None and self.raw_peak_pixels.shape==mask.shape:
            self.peak_v=float(self.raw_peak_pixels[mask].max()) if mask.any() else None
        return relative,outer

    def observe(self,plane,box,ts,valid):
        self.absolute_core=None
        result=super().observe(plane,box,ts,valid)
        if self.support_mask is not None and self.support_mask.mean()<.4:
            result.update(valid=False,reason='emitter_region_ambiguous');self.previous_valid=False
        usable=valid and self.absolute_core is not None and (result['valid'] or result['reason']=='calibrating_dark_baseline')
        if usable and self.absolute_template is None:
            self.absolute_samples.append(self.absolute_core.copy())
            if self.template is not None:
                self.absolute_template=np.percentile(np.stack(list(self.absolute_samples)),20,axis=0)
                self.absolute_samples.clear()
        rise=self._contrast(self.absolute_core,self.absolute_template) if usable and self.absolute_template is not None else None
        supported=bool(rise is not None and rise>=self.min_absolute_rise and self.peak_v is not None and self.peak_v>=self.min_peak_v)
        result.update(absolute_rise=rise,absolute_peak_v=self.peak_v,emission_supported=supported)
        if result['valid'] and result['signal']>=self.p.on_fraction and not supported:
            result.update(signal=0.,reason='contrast_change_without_emission')
        self.last=result
        return result


class EmitterLocator:
    """Find a recurring emitting core, requiring multiple complete flashes.

    Only offers a bounded correction after stable observations. Distinct peaks
    remain distinct even if adjacent lamps flash synchronously.
    """
    def __init__(self,box):
        self.radius=int(max(8,min(28,max(box[2]-box[0],box[3]-box[1]))))
        self.samples=deque(maxlen=64);self.offers=deque(maxlen=3);self.last_update=-1e9
        self.locked=False;self.peak=None;self.details={}

    def observe(self,plane,box,ts,valid,exclude_centers=()):
        c=midpoint(box);r=self.radius
        if not valid:return None
        if min(c)<r or c[0]+r>=plane.shape[1] or c[1]+r>=plane.shape[0]:return None
        patch=cv.getRectSubPix(plane,(2*r+1,2*r+1),tuple(map(float,c))).astype(float)
        patch-=np.median(np.r_[patch[0],patch[-1],patch[:,0],patch[:,-1]])
        self.samples.append((ts,patch))
        while self.samples and ts-self.samples[0][0]>5.:self.samples.popleft()
        if len(self.samples)<12 or ts-self.samples[0][0]<2.5 or ts-self.last_update<.5:return None
        self.last_update=ts;stack=np.stack([p for _,p in self.samples])
        lo,hi=np.percentile(stack,[20,90],axis=0);spread=hi-lo
        limit=max(4.,min(12.,max(box[2]-box[0],box[3]-box[1])*.85))
        yy,xx=np.indices(spread.shape);allowed=np.hypot(xx-r,yy-r)<=limit
        for point in nearby_centers(box,r,exclude_centers):
            relative=np.asarray(point)-c+r
            allowed &= np.hypot(xx-relative[0],yy-relative[1])>3.
        peaks=(spread>=cv.dilate(spread.astype(np.float32),np.ones((5,5),np.uint8))-1e-5)&allowed&(spread>=25)
        count,_,_,centroids=cv.connectedComponentsWithStats(peaks.astype(np.uint8))
        points=centroids[1:]
        if not len(points):return None
        # Select the closest recurring peak, never just the brightest point in a search crop.
        x,y=np.rint(min(points,key=lambda p:np.hypot(p[0]-r,p[1]-r))).astype(int)
        amplitude=float(spread[y,x]);series=stack[:,y,x];low=lo[y,x]
        phase='unknown';cycles=0;prev=None
        for (stamp,_),value in zip(self.samples,series):
            if prev is not None and stamp-prev>.5:phase='unknown'
            prev=stamp
            if value<=low+.3*amplitude:
                if phase=='on':cycles+=1
                phase='off'
            elif value>=low+.65*amplitude and phase=='off':phase='on'
        if cycles<2 or np.count_nonzero(series>=low+.65*amplitude)<3:return None
        # Weight a compact peak neighborhood, keeping glow edges from pulling the center.
        near=(np.hypot(xx-x,yy-y)<=2)&(spread>=amplitude*.8)
        weights=spread*near;offset=np.array([(weights*xx).sum()/weights.sum()-r,(weights*yy).sum()/weights.sum()-r])
        self.offers.append((ts,offset));self.details=dict(amplitude=amplitude,cycles=cycles,peak_count=len(points))
        if len(self.offers)<3 or self.offers[-1][0]-self.offers[0][0]<.9:return None
        offsets=np.array([o for _,o in self.offers]);estimate=np.median(offsets,axis=0)
        if np.max(np.linalg.norm(offsets-estimate,axis=1))>1.5:return None
        self.locked=True;self.peak=c+estimate
        return estimate


def held_translation(reference,gray,exclusions=()):
    """Small translation consensus between full ORB checks; fail closed."""
    h,w=gray.shape;shape=(480,round(h*480/w))
    def prep(a):
        image=cv.resize(a,shape).astype(np.float32)
        image=cv.GaussianBlur(image,(3,3),0)
        return cv.magnitude(cv.Sobel(image,cv.CV_32F,1,0),cv.Sobel(image,cv.CV_32F,0,1))
    a,b=prep(reference),prep(gray);mask=np.ones(a.shape,bool)
    for x0,y0,x1,y1 in exclusions:
        mask[int(y0*a.shape[0]):int(y1*a.shape[0])+1,int(x0*480):int(x1*480)+1]=False
    support=[]
    for row in range(2):
        for col in range(3):
            y0,y1=row*a.shape[0]//2,(row+1)*a.shape[0]//2;x0,x1=col*160,(col+1)*160
            valid=mask[y0:y1,x0:x1]
            if valid.mean()<.55:continue
            src=a[y0:y1,x0:x1]*valid;dst=b[y0:y1,x0:x1]*valid
            if min(src.std(),dst.std())<3:continue
            # Odd optimal DFT dimensions can produce a half-pixel center bias.
            # Pad both inputs to even optimal sizes before phase correlation.
            ph=cv.getOptimalDFTSize(src.shape[0]);pw=cv.getOptimalDFTSize(src.shape[1])
            if ph%2:ph=cv.getOptimalDFTSize(ph+1)
            if pw%2:pw=cv.getOptimalDFTSize(pw+1)
            src=cv.copyMakeBorder(src,0,ph-src.shape[0],0,pw-src.shape[1],cv.BORDER_CONSTANT)
            dst=cv.copyMakeBorder(dst,0,ph-dst.shape[0],0,pw-dst.shape[1],cv.BORDER_CONSTANT)
            shift,response=cv.phaseCorrelate(src,dst,cv.createHanningWindow((pw,ph),cv.CV_32F))
            if np.isfinite(shift).all() and np.isfinite(response) and response>=.2 and np.linalg.norm(shift)<=2:
                support.append((np.array(shift),row,col))
    if not support:return None
    groups=[[p for p in support if np.linalg.norm(p[0]-v[0])<=.6] for v in support];group=max(groups,key=len)
    if len(group)<3 or len({v[1] for v in group})<2 or len({v[2] for v in group})<2:return None
    offset=np.median([v[0] for v in group],axis=0)*w/480
    if np.linalg.norm(offset)<.25:offset[:]=0
    H=np.eye(3);H[:2,2]=offset
    return H
