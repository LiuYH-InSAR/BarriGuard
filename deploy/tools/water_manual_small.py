"""Optional unknown-only presence recovery using independent pixel support.

Hypothesis: small sections need pixel support, not three vertical windows.
No missing/alarm vote is produced by this fallback.
"""
from dataclasses import replace
import numpy as np
from water_manual_regions import ColourRegions


class SmallFallback:
    def __init__(self, geometry, spec):
        self.spec = spec
        self.regions = ColourRegions(geometry,replace(spec,min_points=1))

    def assess(self, features, colour, valid, expected):
        mask = valid & self.regions.independent
        station_pixels = []
        for i,points in enumerate(mask):
            support = set().union(*(self.regions.pixels[i*5+j]
                                   for j in np.flatnonzero(points)))
            station_pixels.append(len(support))
        supported = np.asarray(station_pixels)>=self.spec.fallback_station_pixels
        mask &= supported[:,None]
        info,_ = self.regions.assess(features,colour,mask,expected,2)
        info['station_pixels'] = station_pixels
        info['minimum_station_pixels'] = self.spec.fallback_station_pixels
        info['recovered'] = bool(info['observable'] and info['present'])
        return info
