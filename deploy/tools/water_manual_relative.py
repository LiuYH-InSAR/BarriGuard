"""Optional same-frame red calibration and independent missing evidence.

Hypothesis: hue plus relative RGB/S support identifies desaturated red bodies;
colour ambiguity must abstain rather than become a missing vote.
"""
import numpy as np


class RelativeRed:
    def __init__(self, frame, hsv, detector, values, states):
        self.frame,self.hsv,self.detector=frame,hsv,detector
        self.values,self.states=values,states

    def profile(self, k):
        d,spec=self.detector,self.detector.spec
        peers=sorted((i for i,b in enumerate(d.blocks) if i!=k and b['row']==d.blocks[k]['row'] and b['number']%2),
                     key=lambda i:abs(d.blocks[i]['number']-d.blocks[k]['number']))[:6]
        groups=[]
        for i in peers:
            if not d.blocks[i]['number']%2 or self.states[d.blocks[i]['label']] in ('*','error'):
                continue
            f=self.values[i]
            selected=d.point_ok(f)&(d.types[i]==0)&(f[:,:,0]>=spec.colour_min)
            coords=set()
            for a,b in np.argwhere(selected):
                yy,xx=d.geometry[i]['pixels'][a][b]
                coords.update(zip(yy.tolist(),xx.tolist()))
            if not coords:
                continue
            yy,xx=np.array(list(coords)).T
            pixels=self.frame[yy,xx].astype(float)
            hsv=self.hsv[yy,xx].astype(float)
            advantage=(pixels[:,2]-np.maximum(pixels[:,0],pixels[:,1]))/np.maximum(hsv[:,2],1)
            hue=(hsv[:,0]<=spec.relative_hue)|(hsv[:,0]>=180-spec.relative_hue)
            good=hue&(hsv[:,1]>=spec.relative_s_floor)&(hsv[:,2]>=spec.dark_v)&(advantage>=spec.relative_rgb_floor)
            if good.sum()>=spec.relative_peer_pixels:
                groups.append((hsv[good,1],advantage[good]))
        reliable=len(groups)>=spec.relative_min_peers
        sat=float(np.median(np.concatenate([a for a,_ in groups]))) if groups else 0.
        rgb=float(np.median(np.concatenate([a for _,a in groups]))) if groups else 0.
        return dict(usable=reliable,peer_blocks=len(groups),
                    s_min=max(spec.relative_s_floor,sat*spec.relative_scale),
                    rgb_min=max(spec.relative_rgb_floor,rgb*spec.relative_scale))

    def sample(self, k, profile, relaxed=False):
        spec=self.detector.spec
        result=np.zeros(self.values[k].shape[:2],np.float32)
        s_min=spec.relative_s_floor if relaxed else profile['s_min']
        rgb_min=spec.relative_rgb_floor if relaxed else profile['rgb_min']
        for i,station in enumerate(self.detector.geometry[k]['pixels']):
            for j,(yy,xx) in enumerate(station):
                if not len(xx):
                    continue
                b,g,r=self.frame[yy,xx].astype(float).T
                h,s,v=self.hsv[yy,xx].astype(float).T
                hue=(h<=spec.relative_hue)|(h>=180-spec.relative_hue)
                advantage=(r-np.maximum(b,g))/np.maximum(v,1)
                result[i,j]=np.mean(hue&(s>=s_min)&(advantage>=rgb_min)&(v>=spec.dark_v))
        return result


def missing_support(valid, relaxed_red, profile, spec):
    # Isolated red specks must not veto a gap; reuse the point/station majority.
    counts=valid.sum(axis=1)
    usable=counts>=spec.min_points
    ambiguous=(relaxed_red>=spec.colour_min)&valid
    missing=usable&((counts-ambiguous.sum(axis=1))/np.maximum(counts,1)>=spec.station_fraction)
    ratio=float(missing[usable].mean()) if usable.any() else 0.
    coverage=float(usable.mean()) if len(usable) else 0.
    sufficient=(profile['usable'] and coverage>=spec.block_valid
                and usable.sum()>=spec.min_stations)
    return dict(calibration_usable=bool(profile['usable']),coverage=coverage,
                valid_stations=int(usable.sum()),missing_station_fraction=ratio,
                ambiguous_stations=int(ambiguous.any(axis=1).sum()),
                allowed=bool(sufficient and ratio>spec.block_fraction))
