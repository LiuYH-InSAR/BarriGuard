"""Optional weak RGB evidence: abstention only, not proof of barrier presence."""
import numpy as np


def weak_region(frame, geometry, features, valid, expected, regions, spec, minimum):
    colour = np.zeros((*features.shape[:2],2),np.float32)
    for i,station in enumerate(geometry['pixels']):
        for j,(yy,xx) in enumerate(station):
            if not len(xx):
                continue
            b,g,r = frame[yy,xx].astype(np.float32).T
            high,low = np.maximum.reduce([b,g,r]),np.minimum.reduce([b,g,r])
            if expected == 0:
                mask = (r >= spec.weak_red_ratio*g) & (r >= spec.weak_red_ratio*b)
                mask &= r-np.maximum(g,b) >= spec.weak_red_delta
            else:
                mask = (high-low)/np.maximum(high,1) <= spec.weak_white_spread
                mask &= features[i,j,4] <= spec.weak_white_s
            mask &= high >= spec.dark_v
            colour[i,j,expected] = mask.mean()
    # Reuse geometry/similarity/support rules; never count skipped pixels.
    ratios = colour * (spec.region_colour_min/spec.weak_colour_min)
    info,_ = regions.assess(features,ratios,valid,expected,minimum)
    info['point_threshold'] = spec.weak_colour_min
    info['candidate_stations'] = int((colour[:,:,expected]>=spec.weak_colour_min).any(axis=1).sum())
    return info
