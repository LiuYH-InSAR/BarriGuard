"""Optional sampled colour continuity; absence is independent of the reference.

Hypothesis: a barrier body occupies a connected set of similar red/white patches.
Connectivity follows image coordinates, never merely equal point indices.
"""
import numpy as np


class ColourRegions:
    def __init__(self, geometry, spec):
        self.spec = spec
        centers = geometry['centers']
        self.shape = centers.shape[:2]
        self.edges = []
        self.pixels = [set(zip(yy.tolist(), xx.tolist()))
                       for station in geometry['pixels'] for yy, xx in station]
        self.independent = np.ones(self.shape, bool)
        for i, station in enumerate(centers):
            # Collapsed vertical windows cannot provide five independent points.
            kept = []
            for j, xy in enumerate(station):
                if any(np.linalg.norm(xy-other) < spec.patch_size for other in kept):
                    self.independent[i,j] = False
                else:
                    kept.append(xy)
            for j in range(4):
                self.edges.append((i*5+j, i*5+j+1))
            if i == 0:
                continue
            spacing = min(np.linalg.norm(np.diff(station,axis=0),axis=1).max(),
                          np.linalg.norm(np.diff(centers[i-1],axis=0),axis=1).max())
            reach = max(spec.patch_size, spacing/2+spec.step_px)
            distances = np.linalg.norm(station[:,None]-centers[i-1][None,:],axis=2)
            for j in range(5):
                other = int(distances[j].argmin())
                if distances[j,other] <= reach:
                    self.edges.append((i*5+j,(i-1)*5+other))
            if spec.region_bridge and i >= 2:
                # Skip at most one station; missing pixels never add support.
                distances = np.linalg.norm(station[:,None]-centers[i-2][None,:],axis=2)
                for j in range(5):
                    other = int(distances[j].argmin())
                    if distances[j,other] <= reach+spec.step_px:
                        self.edges.append((i*5+j,(i-2)*5+other))

    def assess(self, features, colour, valid, expected, minimum_stations):
        spec = self.spec
        valid = valid & self.independent
        stations = valid.sum(axis=1) >= spec.min_points
        coverage = float(stations.mean()) if len(stations) else 0.
        support = set().union(*(self.pixels[i] for i in np.flatnonzero(valid.ravel())))
        observable = (coverage >= spec.block_valid and stations.sum() >= minimum_stations
                      and len(support) >= spec.min_pixels)
        maps = np.full((2,*self.shape),-1,np.int32)
        components = []
        flat = features.reshape(-1,6)
        for c in (0,1):
            candidate = (valid & (colour[:,:,c] >= spec.region_colour_min)
                         & (colour[:,:,c] > colour[:,:,1-c])).ravel()
            parent = np.arange(len(candidate))
            def root(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a
            for a,b in self.edges:
                if not candidate[a] or not candidate[b]:
                    continue
                similar_v = abs(flat[a,3]-flat[b,3]) <= max(spec.region_v_delta,
                    spec.region_v_fraction*max(flat[a,3],flat[b,3]))
                if similar_v and abs(flat[a,4]-flat[b,4]) <= spec.region_s_delta:
                    parent[root(a)] = root(b)
            groups = {}
            for a in np.flatnonzero(candidate):
                groups.setdefault(root(a),[]).append(int(a))
            summary = []
            for cid,nodes in enumerate(groups.values()):
                maps[c].ravel()[nodes] = cid
                pixels = set().union(*(self.pixels[a] for a in nodes))
                span = len(set(a//5 for a in nodes))
                summary.append(dict(id=cid,points=len(nodes),stations=span,pixels=len(pixels),
                                    station_fraction=span/max(int(stations.sum()),1)))
            components.append(summary)
        required_span = max(2,int(np.ceil(stations.sum()*spec.region_station_fraction)))
        present = any(c['stations'] >= required_span and c['pixels'] >= spec.region_min_pixels
                      for c in components[expected])
        return dict(expected='red' if expected==0 else 'white', observable=bool(observable),
                    present=bool(present),coverage=coverage,pixels=len(support),
                    required_stations=required_span,components=components), maps
