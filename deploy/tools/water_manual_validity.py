"""Optional small-block observation gates; never classify insufficient pixels normal."""
import math

import numpy as np


class ValidityRules:
    def __init__(self, spec):
        self.spec = spec

    def point_mask(self, features):
        spec = self.spec
        coverage = features[:,:,5]
        threshold = min(spec.point_valid,spec.small_point_valid)
        enough = (coverage >= threshold) & (coverage*spec.patch_size**2 >= spec.small_point_pixels)
        clipped = (features[:,:,3] >= 250) & (features[:,:,4] < 10)
        return enough & (features[:,:,3] >= spec.dark_v) & ~clipped

    def station_minimum(self, n):
        return min(self.spec.min_stations,max(self.spec.small_min_stations,math.ceil(n*self.spec.block_valid)))

    def reference_info(self, features, types, pixels):
        eligible = (self.point_mask(features)&(types >= 0)).sum(axis=1) >= self.spec.min_points
        count = int(eligible.sum())
        coverage = float(eligible.mean()) if len(eligible) else 0.
        return dict(stations=len(eligible),valid_stations=count,coverage=coverage,pixels=pixels,
            minimum_stations=self.station_minimum(len(eligible)),
            supported=coverage>=self.spec.block_valid and count>=self.station_minimum(len(eligible)) and pixels>=self.spec.min_pixels)
