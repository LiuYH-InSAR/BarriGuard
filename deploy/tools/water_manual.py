"""Optional fixed-block matrix detector. No dependency from production service.

Hypothesis: persistent local colour loss relative to a normal labelled image,
after peer brightness compensation, identifies changed fixed barrier positions.
Thresholds are provisional and exposed in ManualSpec; UNKNOWN never votes normal.
"""
import base64
from dataclasses import asdict, dataclass
import html
import json
from pathlib import Path
import re

import cv2 as cv
import numpy as np

from water_barrier import signal as S

FEATURES = ['red', 'white', 'foreign', 'V', 'S', 'valid']
UNKNOWN, NORMAL, ERROR = -1, 0, 1


@dataclass
class ManualSpec:
    step_px: float = S.STEP
    patch_size: int = 5
    point_valid: float = .8
    dark_v: float = 25
    exposure_fraction: float = .8
    min_points: int = 3
    station_fraction: float = .6
    block_fraction: float = .7
    block_valid: float = .7
    min_stations: int = 5
    min_pixels: int = 100
    colour_min: float = .2
    colour_loss: float = .25
    brightness_residual: float = .3
    saturation_change: float = 35
    min_peers: int = 3
    max_light_change: float = .4
    hold_s: float = 10
    reset_s: float = 5
    max_observation_gap_s: float = 1
    gray_white: bool = False
    light_colour: bool = False
    gray_v_min: float = 80
    gray_s_max: float = 70
    gray_rgb_spread: float = 40
    relative_red: bool = False
    missing_support: bool = False
    relative_hue: float = 20
    relative_s_floor: float = 15
    relative_rgb_floor: float = .03
    relative_scale: float = .2
    relative_min_peers: int = 2
    relative_peer_pixels: int = 25
    pixel_fallback: bool = False
    pixel_min_pixels: int = 30
    pixel_component_min: int = 12
    pixel_component_fraction: float = .15
    small_fallback: bool = False
    fallback_max_stations: int = 12
    fallback_station_pixels: int = 25
    weak_red: bool = False
    weak_white: bool = False
    recent_missing: bool = False
    weak_red_ratio: float = 1.15
    weak_red_delta: float = 10
    weak_white_spread: float = .35
    weak_white_s: float = 90
    weak_colour_min: float = .2
    recent_missing_fraction: float = .95
    region_support: bool = False
    region_bridge: bool = False
    region_weak_stations: int = 2
    colour_regions: bool = False
    region_colour_min: float = .4
    region_v_delta: float = 25
    region_v_fraction: float = .25
    region_s_delta: float = 35
    region_station_fraction: float = .2
    region_min_pixels: int = 50
    validity_optimize: bool = False
    small_point_valid: float = .6
    small_point_pixels: int = 15
    small_min_stations: int = 2

    def validate(self):
        for key in ('point_valid','exposure_fraction','station_fraction','block_fraction','block_valid','colour_min','colour_loss','small_point_valid','region_colour_min','region_station_fraction','region_v_fraction','weak_white_spread','weak_colour_min','recent_missing_fraction','pixel_component_fraction','relative_rgb_floor','relative_scale'):
            if not 0 < getattr(self,key) <= 1:
                raise ValueError(f'{key} must be in (0,1]')
        for key,value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        if self.patch_size != 5 or self.step_px <= 0 or not 1 <= self.min_points <= 5 or self.min_peers < 1 or self.min_stations < 1 or self.hold_s <= 0:
            raise ValueError('Invalid patch size, minimum counts or duration')
        if not 0 < self.relative_hue <= 30 or self.relative_s_floor < 10 or self.relative_min_peers < 2 or self.relative_peer_pixels < 15:
            raise ValueError('Invalid same-frame red calibration limits')
        if self.pixel_min_pixels < 30 or self.pixel_component_min < 12:
            raise ValueError('Dense poly fallback needs >=30 valid pixels and >=12 connected pixels')
        if self.fallback_max_stations < 2 or self.fallback_station_pixels < 15:
            raise ValueError('Small fallback needs >=2 stations and >=15 station pixels')
        if self.region_weak_stations < 2 or int(self.region_weak_stations) != self.region_weak_stations:
            raise ValueError('region_weak_stations must be an integer >=2')
        if not 1<=self.small_point_pixels<=25 or self.small_min_stations<2:
            raise ValueError('Small-block validity requires 1..25 pixels and at least 2 stations')


def load_labels(path):
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    rows, blocks = {}, []
    w, h = data['imageWidth'], data['imageHeight']
    for shape in data['shapes']:
        name = shape['label']
        match = re.fullmatch(r'row_?(\d+)(?:_?block_?(\d+))?', name)
        if not match or shape['shape_type'] != 'polygon':
            raise ValueError(f'Expected rowX or rowXblockYY polygon: {name}')
        poly = np.asarray(shape['points'], np.float32)
        if len(poly) < 3 or not np.isfinite(poly).all() or np.any(poly < 0) or np.any(poly > [w, h]):
            raise ValueError(f'Invalid polygon: {name}')
        row = int(match[1])
        if match[2] is None:
            if row in rows:
                raise ValueError(f'Duplicate row: {name}')
            rows[row] = poly
        else:
            blocks.append(dict(label=name, row=row, number=int(match[2]), poly=poly))
    keys = [(b['row'], b['number']) for b in blocks]
    if not blocks or len(set(keys)) != len(keys):
        raise ValueError('Missing blocks or duplicate block numbers')
    if any(b['row'] not in rows for b in blocks):
        raise ValueError('Each block needs a matching row polygon')
    return data, rows, sorted(blocks, key=lambda b: (b['row'], b['number']))


class ManualDetector:
    def __init__(self, label_path, reference, spec=None):
        self.spec = spec or ManualSpec()
        self.spec.validate()
        self.validity = None
        if self.spec.validity_optimize:
            from water_manual_validity import ValidityRules
            self.validity = ValidityRules(self.spec)
        self.recent = None
        if self.spec.recent_missing:
            from water_manual_recent import RecentMissing
            self.recent = RecentMissing(self.spec)
        self.data, rows, self.blocks = load_labels(label_path)
        self.rows = rows
        h, w = reference.shape[:2]
        if (w, h) != (self.data['imageWidth'], self.data['imageHeight']):
            raise ValueError('Reference size does not match labels')
        self.shape = reference.shape
        self.geometry = []
        self.state = {}
        for block in self.blocks:
            poly = block['poly']
            row = rows[block['row']]
            center = row.mean(axis=0)
            _, _, vt = np.linalg.svd(row-center, full_matrices=False)
            direction = vt[0]
            if direction[0] < 0:
                direction = -direction
            row_u = (row-center) @ direction
            block_u = (poly-center) @ direction
            stations = np.arange(row_u.min()+self.spec.step_px/2, row_u.max(), self.spec.step_px)
            stations = stations[(stations >= block_u.min()) & (stations <= block_u.max())]
            mask = np.zeros((h, w), np.uint8)
            cv.fillPoly(mask, [np.rint(poly).astype(np.int32)], 1)
            ys, xs = np.where(mask)
            ids = np.rint((((np.c_[xs, ys]-center) @ direction)-row_u.min()-self.spec.step_px/2)/self.spec.step_px).astype(int)
            centers, pixels = [], []
            for u in stations:
                sid = int(round((u-row_u.min()-self.spec.step_px/2)/self.spec.step_px))
                take = ids == sid
                if not take.any():
                    continue
                top, bottom = ys[take].min(), ys[take].max()
                cy = top+(bottom-top)*np.array([.1, .3, .5, .7, .9])
                cx = np.full(5, xs[take].mean()) if abs(direction[0]) < 1e-6 else center[0]+(u-(cy-center[1])*direction[1])/direction[0]
                point_centers = np.rint(np.c_[cx, cy]).astype(int)
                patches = []
                radius = self.spec.patch_size//2
                for x, y in point_centers:
                    xx, yy = np.meshgrid(np.arange(x-radius, x+radius+1), np.arange(y-radius, y+radius+1))
                    inside = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
                    xx, yy = xx[inside], yy[inside]
                    good = mask[yy, xx] > 0
                    patches.append((yy[good], xx[good]))
                centers.append(point_centers)
                pixels.append(patches)
            unique = set((int(y), int(x)) for station in pixels for yy, xx in station for y, x in zip(yy, xx))
            self.geometry.append(dict(centers=np.asarray(centers).reshape(-1, 5, 2), pixels=pixels, area=len(unique)))
            self.state[block['label']] = dict(status='?', accum=0., intact=0., last=None)
        self.regions = None
        if self.spec.colour_regions:
            from water_manual_regions import ColourRegions
            self.regions = [ColourRegions(g,self.spec) for g in self.geometry]
        self.small = {}
        if self.spec.small_fallback:
            from water_manual_small import SmallFallback
            self.small = {k:SmallFallback(g,self.spec) for k,g in enumerate(self.geometry)
                          if len(g['centers'])<=self.spec.fallback_max_stations}
        self.pixel = {}
        if self.spec.pixel_fallback:
            from water_manual_pixels import PixelFallback
            for k,(block,g) in enumerate(zip(self.blocks,self.geometry)):
                height = max(float(np.ptp(block['poly'][:,1])),1.)
                if len(g['centers'])<=self.spec.fallback_max_stations or g['area']/height<=self.spec.patch_size:
                    self.pixel[k]=PixelFallback(block['poly'],reference,self.spec,0 if block['number']%2 else 1)
        self.reference = self.sample(reference)
        self.types = []
        for ref in self.reference:
            types = np.full(ref.shape[:2], UNKNOWN, np.int8)
            valid = self.point_ok(ref)
            types[valid & (ref[:, :, 0] >= self.spec.colour_min) & (ref[:, :, 0] > ref[:, :, 1])] = 0
            types[valid & (ref[:, :, 1] >= self.spec.colour_min) & (ref[:, :, 1] > ref[:, :, 0])] = 1
            self.types.append(types)
        self.reference_validity = []
        if self.validity is not None:
            self.reference_validity = [self.validity.reference_info(f,t,g['area']) for f,t,g in zip(self.reference,self.types,self.geometry)]

    def sample(self, frame):
        if frame.shape != self.shape:
            raise ValueError('Frame size differs from labels/reference')
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        red, white = S.red_m(frame), S.white_m(frame)
        if self.spec.gray_white:
            spread = frame.max(axis=2).astype(float)-frame.min(axis=2)
            white |= (hsv[:,:,2] >= self.spec.gray_v_min) & (hsv[:,:,1] <= self.spec.gray_s_max) & (spread <= self.spec.gray_rgb_spread)
        foreign = (hsv[:, :, 1] > 100) & (hsv[:, :, 0] > 12) & (hsv[:, :, 0] < 165)
        values = []
        for geometry in self.geometry:
            f = np.zeros((len(geometry['pixels']), 5, 6), np.float32)
            for i, station in enumerate(geometry['pixels']):
                for j, (yy, xx) in enumerate(station):
                    if not len(xx):
                        continue
                    f[i, j] = [red[yy, xx].mean(), (white[yy, xx] & ~red[yy, xx]).mean(), foreign[yy, xx].mean(),
                               np.median(hsv[yy, xx, 2]), np.median(hsv[yy, xx, 1]), len(xx)/self.spec.patch_size**2]
            values.append(f)
        return values

    def corrected_colour(self, frame, k, factor, colour, hsv):
        """Classify original patch pixels at peer-normalized V; retain raw features."""
        spec = self.spec
        result = np.zeros((len(self.geometry[k]['pixels']),5),np.float32)
        for i,station in enumerate(self.geometry[k]['pixels']):
            for j,(yy,xx) in enumerate(station):
                if not len(xx):
                    continue
                h,s,v = hsv[yy,xx].astype(float).T
                v = np.minimum(v/max(factor,.01),255)
                red = ((h<=12)|(h>=165)) & (s>=70) & (v>=50)
                white = ((s<=70)&(v>=150)) | ((s>70)&(s<=120)&(v>=80)&(v<150)&(h>=85))
                if spec.gray_white:
                    spread = (frame[yy,xx].max(axis=1).astype(float)-frame[yy,xx].min(axis=1))/max(factor,.01)
                    white |= (v>=spec.gray_v_min)&(s<=spec.gray_s_max)&(spread<=spec.gray_rgb_spread)
                result[i,j] = np.mean(red if colour==0 else white & ~red)
        return result

    def point_ok(self, f):
        if self.validity is not None:
            return self.validity.point_mask(f)
        clipped = (f[:, :, 3] >= 250) & (f[:, :, 4] < 10)
        return (f[:, :, 5] >= self.spec.point_valid) & (f[:, :, 3] >= self.spec.dark_v) & ~clipped

    def step(self, frame, now):
        spec = self.spec
        values = self.sample(frame)
        frame_ok, _ = S.frame_valid(frame)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        exposed = np.mean((hsv[:, :, 2] < spec.dark_v) | ((hsv[:, :, 2] >= 250) & (hsv[:, :, 1] < 10))) < spec.exposure_fraction
        outputs = []
        previous_states = {key:value['status'] for key,value in self.state.items()}
        relative = None
        if spec.relative_red or spec.missing_support:
            from water_manual_relative import RelativeRed
            relative = RelativeRed(frame,hsv,self,values,previous_states)
        for k, (block, f, ref, types) in enumerate(zip(self.blocks, values, self.reference, self.types)):
            ok = self.point_ok(f) & self.point_ok(ref) & (types >= 0)
            point = np.full(types.shape, UNKNOWN, np.int8)
            evidence_colour = f[:,:,:2].copy()
            light = {}
            for colour in (0, 1):
                peer_v, peer_s, peer_count = [], [], 0
                peers = sorted((i for i,b in enumerate(self.blocks) if i != k and b['row'] == block['row']), key=lambda i: abs(self.blocks[i]['number']-block['number']))[:6]
                for i in peers:
                    if previous_states[self.blocks[i]['label']] in ('*', 'error'):
                        continue
                    pf, pr = values[i], self.reference[i]
                    colour_ok = pf[:,:,colour] >= spec.colour_min
                    if spec.light_colour:
                        colour_ok |= abs(pf[:,:,4]-pr[:,:,4]) <= spec.saturation_change
                    usable = self.point_ok(pf) & self.point_ok(pr) & (self.types[i] == colour) & colour_ok
                    if usable.any():
                        peer_v.append(pf[:, :, 3][usable]/np.maximum(pr[:, :, 3][usable],1))
                        peer_s.append(pf[:, :, 4][usable]-pr[:, :, 4][usable])
                        peer_count += int(usable.sum())
                factor = float(np.median(np.concatenate(peer_v))) if peer_v else 1.
                sat_delta = float(np.median(np.concatenate(peer_s))) if peer_s else 0.
                reliable = peer_count >= spec.min_peers and abs(factor-1) <= spec.max_light_change
                light[str(colour)] = dict(peers=peer_count, v_factor=factor, s_delta=sat_delta, usable=reliable)
                selected = ok & (types == colour) & frame_ok & exposed & reliable
                if spec.light_colour and reliable:
                    evidence_colour[:,:,colour] = self.corrected_colour(frame,k,factor,colour,hsv)
                observed = evidence_colour[:,:,colour]
                loss = ref[:, :, colour]-observed
                residual = abs(f[:, :, 3]/np.maximum(ref[:, :, 3]*factor,1)-1)
                sat_residual = abs(f[:, :, 4]-ref[:, :, 4]-sat_delta)
                abnormal = (loss >= spec.colour_loss) & ((observed < spec.colour_min) | (residual > spec.brightness_residual) | (sat_residual > spec.saturation_change))
                point[selected] = abnormal[selected].astype(np.int8)
            valid_points = (point >= 0).sum(axis=1)
            station = np.full(len(f), UNKNOWN, np.int8)
            usable_stations = valid_points >= spec.min_points
            station[usable_stations] = (((point == ERROR).sum(axis=1)[usable_stations]/valid_points[usable_stations]) >= spec.station_fraction).astype(np.int8)
            coverage = float(usable_stations.mean()) if len(f) else 0.
            ratio = float((station[usable_stations] == ERROR).mean()) if usable_stations.any() else 0.
            minimum_stations = self.validity.station_minimum(len(f)) if self.validity is not None else spec.min_stations
            sufficient = coverage >= spec.block_valid and usable_stations.sum() >= minimum_stations and self.geometry[k]['area'] >= spec.min_pixels
            evidence = ERROR if sufficient and ratio > spec.block_fraction else NORMAL if sufficient else UNKNOWN
            support_colour = evidence_colour.copy() if relative is not None else evidence_colour
            relative_info, relaxed = None,None
            if relative is not None and block['number']%2:
                relative_info = relative.profile(k)
                relaxed = relative.sample(k,relative_info,True)
                if spec.relative_red and relative_info['usable']:
                    adapted = relative.sample(k,relative_info)
                    evidence_colour[:,:,0] = np.maximum(evidence_colour[:,:,0],adapted)
                    evidence_colour[:,:,1] = np.minimum(evidence_colour[:,:,1],1-evidence_colour[:,:,0])
                    relative_info['candidate_points'] = int((adapted>=spec.region_colour_min).sum())
            region_info, region_map = None, None
            if self.regions is not None:
                expected = 0 if block['number'] % 2 else 1
                region_info, region_map = self.regions[k].assess(f,evidence_colour,
                    self.point_ok(f),expected,minimum_stations)
                if (spec.weak_red if expected==0 else spec.weak_white) and not region_info['present']:
                    from water_manual_weak import weak_region
                    weak = weak_region(frame,self.geometry[k],f,self.point_ok(f),expected,
                        self.regions[k],spec,minimum_stations)
                    region_info['weak_colour'] = weak
                    if weak['observable'] and weak['present']:
                        region_info['observable'] = False
                if spec.region_support:
                    from water_manual_support import colour_support
                    support = colour_support(support_colour,self.point_ok(f),ref,
                        self.point_ok(ref),expected,spec)
                    region_info['colour_support'] = support
                    if not region_info['present'] and support['ambiguous']:
                        region_info['observable'] = False
                region_info['observable'] &= bool(frame_ok and exposed and light[str(expected)]['usable'])
                if relative_info is not None:
                    region_info['relative_red'] = relative_info
                    if spec.relative_red and relative_info['usable'] and region_info['observable'] and region_info['present']:
                        if evidence != ERROR:
                            evidence = NORMAL
                if not region_info['observable']:
                    evidence = UNKNOWN
                elif not region_info['present']:
                    evidence = ERROR
            missing_info = None
            if spec.missing_support and evidence == ERROR and block['number']%2:
                from water_manual_relative import missing_support
                missing_info = missing_support(self.point_ok(f),relaxed,relative_info,spec)
                if not missing_info['allowed']:
                    evidence = UNKNOWN
            fallback = None
            if evidence == UNKNOWN and k in self.small:
                expected = 0 if block['number']%2 else 1
                fallback = self.small[k].assess(f,evidence_colour,self.point_ok(f),expected)
                fallback['quality_ok'] = bool(frame_ok and exposed and light[str(expected)]['usable'])
                fallback['conflicting_station'] = bool((station==ERROR).any())
                fallback['recovered'] &= fallback['quality_ok'] and not fallback['conflicting_station']
                if fallback['recovered']:
                    evidence = NORMAL
            pixel_info = None
            if evidence == UNKNOWN and k in self.pixel:
                expected = 0 if block['number']%2 else 1
                pixel_info = self.pixel[k].assess(frame,light[str(expected)]['v_factor'])
                pixel_info['quality_ok'] = bool(frame_ok and exposed and light[str(expected)]['usable'])
                pixel_info['recovered'] &= pixel_info['quality_ok']
                if pixel_info['recovered']:
                    evidence = NORMAL
            state = self.state[block['label']]
            dt = max(0.,now-state['last']) if state['last'] is not None else 0.
            if dt > spec.max_observation_gap_s:
                dt = 0.
                state['accum'] = 0.
            recent_confirmed = None
            if self.recent is not None:
                recent_accum,recent_confirmed = self.recent.update(block['label'],evidence,now)
            if evidence == ERROR:
                if state['status'] == '?':
                    dt = 0.
                state['accum'] += dt
                state['intact'] = 0.
                state['status'] = 'error' if (recent_confirmed if self.recent is not None else state['accum'] >= spec.hold_s) else '*'
            elif evidence == NORMAL:
                state['intact'] += dt if state['status'] != '?' else 0.
                state['accum'] = 0.
                if state['status'] != 'error' or state['intact'] >= spec.reset_s:
                    state['status'] = 'normal'
            else:
                state['status'] = '?'
            if self.recent is not None:
                state['accum'] = recent_accum
            state['last'] = now
            reasons = []
            if not frame_ok or not exposed:
                reasons.append('frame quality/exposure')
            if not any(v['usable'] for v in light.values()):
                reasons.append('peer light reference insufficient/unstable')
            if coverage < spec.block_valid:
                reasons.append('valid station coverage')
            if usable_stations.sum() < minimum_stations:
                reasons.append('too few valid stations')
            if self.geometry[k]['area'] < spec.min_pixels:
                reasons.append('pixel support')
            if self.validity is not None and not self.reference_validity[k]['supported']:
                reasons.append('normal reference not sufficiently observable')
            if region_info is not None:
                if region_info.get('weak_colour',{}).get('present'):
                    reasons.append('weak '+region_info['expected']+' region; cannot confirm absence')
                elif not region_info['observable']:
                    reasons.append('weak/fragmented body colour; cannot confirm absence' if
                        region_info.get('colour_support',{}).get('ambiguous') and not region_info['present']
                        else 'colour region support/light insufficient')
                elif not region_info['present']:
                    reasons.append('expected continuous '+region_info['expected']+' region absent')
            if fallback is not None:
                reasons.append('small block fallback: '+('presence recovered' if fallback['recovered'] else 'insufficient/conflicting evidence'))
            if pixel_info is not None:
                reasons.append('dense poly fallback: '+('presence recovered' if pixel_info['recovered'] else 'insufficient/conflicting evidence'))
            if missing_info is not None and not missing_info['allowed']:
                reasons.append('missing evidence insufficient: red recognition/ambiguity')
            if relative_info is not None and relative_info['usable'] and spec.relative_red and region_info is not None and region_info['present']:
                reasons.append('same-frame relative red presence')
            reason = '; '.join(reasons) or 'OK'
            outputs.append(dict(label=block['label'], row=block['row'], number=block['number'], n=len(f),
                features=f.transpose(2,0,1).reshape(6,-1), anomaly=np.vstack([point.T,station]),
                colour_evidence=evidence_colour.transpose(2,0,1).reshape(2,-1),
                status=state['status'], evidence=int(evidence), valid=coverage, abnormal=ratio,
                accum=state['accum'], pixels=self.geometry[k]['area'], reason=reason, light=light))
            if missing_info is not None:
                outputs[-1]['missing_support'] = missing_info
            if relative_info is not None:
                outputs[-1]['relative_red'] = relative_info
            if pixel_info is not None:
                outputs[-1]['pixel_fallback'] = pixel_info
            if fallback is not None:
                outputs[-1]['small_fallback'] = fallback
            if region_info is not None:
                outputs[-1]['regions'] = region_info
                outputs[-1]['region_map'] = region_map.reshape(2,-1)
            if self.validity is not None:
                outputs[-1]['reference_validity'] = self.reference_validity[k]
        return outputs


def reference_image(path, data, override=None):
    if override:
        image = cv.imread(str(override))
    elif data.get('imageData'):
        image = cv.imdecode(np.frombuffer(base64.b64decode(data['imageData']),np.uint8),cv.IMREAD_COLOR)
    else:
        candidates = [Path(path).parent/data['imagePath'], Path(__file__).resolve().parents[2]/'tmp'/'roi_relabel'/Path(data['imagePath']).name]
        image = next((cv.imread(str(p)) for p in candidates if p.is_file()),None)
    if image is None:
        raise ValueError('Normal reference image missing; supply --block-reference')
    return image


def draw_result(frame, detector, outputs):
    from water_result import POLY_COLOUR, POLY_ALPHA
    image = frame.copy()
    overlay = image.copy()
    polygons = [np.rint(poly).astype(np.int32) for poly in detector.rows.values()]
    for pts in polygons:
        cv.fillPoly(overlay,[pts],POLY_COLOUR)
    image = cv.addWeighted(overlay,POLY_ALPHA,image,1-POLY_ALPHA,0)
    for pts in polygons:
        cv.polylines(image,[pts],True,POLY_COLOUR,1,cv.LINE_AA)
    for block, result in zip(detector.blocks, outputs):
        if result['status'] not in ('*','error'):
            continue
        colour = {'*':(0,180,255), 'error':(0,0,255)}[result['status']]
        pts = np.rint(block['poly']).astype(np.int32)
        cv.polylines(image,[pts],True,colour,2)
        x,y = pts.mean(axis=0).astype(int)
        text = f"{block['label']} {result['status']} {result['abnormal']:.0%}"
        cv.putText(image,text,(max(0,x-30),max(12,y)),cv.FONT_HERSHEY_SIMPLEX,.35,colour,1,cv.LINE_AA)
    return image


def web_page(title, body):
    return '<!doctype html><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>body{font:15px Microsoft YaHei,sans-serif;margin:24px}img{max-width:100%}td,th{border:1px solid #ccc;padding:6px}table{border-collapse:collapse}canvas{display:block} .scroll{overflow-x:auto}pre{white-space:pre-wrap}</style><h1>'+html.escape(title)+'</h1>'+body


def matrix_page(out, index, now, results, enhanced=False, name=None, frame=None, detector=None):
    body = f'<a href="../index.html">总览</a><p>帧{index}，t={now:.3f}s；P1–P5从上到下。悬停查看精确站号、点号和值。</p>'
    image_name = f'samples_{index:08d}.png'
    if name is not None and frame is not None:
        image_name = name+'.png'
        cv.imwrite(str(out/image_name),frame)
    if (out/image_name).is_file():
        body += f'<img src="../{image_name}">'
    arrays = {}
    for result in results:
        label = result['label']
        arrays[label+'_features'] = result['features']
        arrays[label+'_anomaly'] = result['anomaly']
        if 'colour_evidence' in result:
            arrays[label+'_colour_evidence'] = result['colour_evidence']
        if 'region_map' in result:
            arrays[label+'_region_map'] = result['region_map']
            body += '<p>连续色块：'+html.escape(json.dumps(result['regions'],ensure_ascii=False))+'</p>'
            body += '<p>region_map：2×5n，行=红/白，列=S0P1…S0P5、S1P1…；-1无候选，非负值为同色连通区域编号。存在检查独立于原anomaly矩阵。</p>'
        body += f'<h2 id="{label}">{label} · {result["status"]}</h2><p>n={result["n"]}；有效站={result["valid"]:.1%}；异常站={result["abnormal"]:.1%}；累计={result["accum"]:.2f}s</p>'
        if frame is not None and detector is not None:
            k = next(i for i,b in enumerate(detector.blocks) if b['label']==label)
            x,y,w,h = cv.boundingRect(np.rint(detector.blocks[k]['poly']).astype(int))
            crop_name = f'{name or index}_{label}.png'
            cv.imwrite(str(out/crop_name),frame[max(0,y):y+h,max(0,x):x+w])
            body += f'<img src="../{crop_name}"><p>正常参考：<a href="../reference.png">原图</a>；<a href="../reference.npz">6×5n矩阵</a>。features为原始观测，colour_evidence为光照修正后判断用比例。</p>'
            reference = detector.reference[k].transpose(2,0,1).reshape(6,-1)
            arrays[label+'_reference'] = reference
        if enhanced:
            body += '<p>横轴：S=沿row方向排列的站；每站P1–P5从截面上到下。特征：0低（浅色）→1高（深蓝）；V/S为0–255。异常：绿=0正常、红=1异常、灰=-1无法判断；第6行=站判断。</p><div style="width:220px;height:15px;background:linear-gradient(to right,rgb(245,250,255),rgb(35,95,165))"></div><p>低 ← 数值 → 高</p>'
        matrices = [('features',FEATURES),('anomaly',['P1','P2','P3','P4','P5','station'])]
        if enhanced and detector is not None:
            matrices += [('reference',FEATURES),('colour_evidence',['red_corrected','white_corrected'])]
        if 'region_map' in result:
            matrices += [('region_map',['red_region','white_region'])]
        for kind,names in matrices:
            data = (reference if kind=='reference' else result[kind]).tolist()
            cid = label+'_'+kind
            body += f'<h3>{kind}: {len(data)} × {len(data[0])}</h3><div class="scroll"><canvas id="{cid}"></canvas></div>'
            script = '<script>(()=>{const d='+json.dumps(data)+',names='+json.dumps(names)+',feature='+str(kind=='features').lower()+',c=document.getElementById('+json.dumps(cid)+');const cell=5,rh=27,left=95,top=30;c.width=left+d[0].length*cell;c.height=top+6*rh+30;const ctx=c.getContext("2d");ctx.font="12px sans-serif";for(let r=0;r<6;r++){ctx.fillStyle="#222";ctx.fillText(names[r]+(feature?(r==3||r==4?" [0..255]":" [0..1]"):""),0,top+r*rh+17);for(let j=0;j<d[r].length;j++){const v=d[r][j],q=feature?v/(r==3||r==4?255:1):v;ctx.fillStyle=feature?`rgb(${Math.round(255*q)},80,${Math.round(255*(1-q))})`:(v<0?"#aaa":v>0?"#e44":"#6b8");ctx.fillRect(left+j*cell,top+r*rh,cell,rh-1);}}const stride=feature?5:1;for(let j=0;j<d[0].length;j+=stride){ctx.strokeStyle="#ddd";ctx.beginPath();ctx.moveTo(left+j*cell,top);ctx.lineTo(left+j*cell,top+6*rh);ctx.stroke();if(j%(stride*10)==0){ctx.fillStyle="#222";ctx.fillText("S"+(j/stride),left+j*cell,20);}}c.onmousemove=e=>{const x=Math.floor((e.offsetX-left)/cell),r=Math.floor((e.offsetY-top)/rh);c.title=x>=0&&x<d[0].length&&r>=0&&r<6?names[r]+" / S"+Math.floor(x/stride)+(feature?" P"+(x%5+1):"")+" = "+d[r][x]:"";};})();</script>'
            if kind == 'region_map':
                script = script.replace('r<6;r++','r<d.length;r++').replace('const stride=feature?5:1','const stride=5')
                script = script.replace('(feature?" P"+(x%5+1):"")','(" P"+(x%5+1))')
                script = script.replace('(v<0?"#aaa":v>0?"#e44":"#6b8")','(v<0?"#ddd":`hsl(${v*137.5%360},65%,55%)`)')
            if enhanced:
                if kind in ('reference','colour_evidence'):
                    script = script.replace('feature=false','feature=true')
                script = script.replace('r<6;r++','r<d.length;r++')
                script = script.replace('`rgb(${Math.round(255*q)},80,${Math.round(255*(1-q))})`','`rgb(${Math.round(245-210*q)},${Math.round(250-155*q)},${Math.round(255-90*q)})`')
                script = script.replace('c.onmousemove=', 'if(feature){ctx.font="9px sans-serif";for(let j=0;j<Math.min(5,d[0].length);j++)ctx.fillText("P"+(j+1),left+j*cell,top+6*rh+15);}c.onmousemove=')
            body += script
        body += '<details><summary>光照与有效性详情</summary><pre>'+html.escape(json.dumps({k:v for k,v in result.items() if k not in ('features','anomaly','colour_evidence','region_map')},ensure_ascii=False,indent=2))+'</pre></details>'
    filename = name or f'frame_{index:08d}'
    np.savez_compressed(out/'data'/f'{filename}.npz',**arrays)
    (out/'pages'/f'{filename}.html').write_text(web_page('单块矩阵',body),encoding='utf-8')


def replay_manual(args):
    from datetime import datetime
    from time import perf_counter
    started = perf_counter()
    detect_hz = getattr(args,'detect_hz',0.)
    if not np.isfinite(detect_hz) or detect_hz < 0:
        raise ValueError("--detect-hz must be finite and nonnegative")
    summary_skip = getattr(args,'summary_skip',5.)
    if not np.isfinite(summary_skip) or summary_skip < 0:
        raise ValueError('--summary-skip must be finite and nonnegative')
    label = Path(args.manual_blocks)
    data,_,_ = load_labels(label)
    reference = reference_image(label,data,args.block_reference)
    spec = ManualSpec()
    supplied = {}
    if args.manual_parameters:
        supplied = json.loads(Path(args.manual_parameters).read_text(encoding='utf-8'))
        unknown = set(supplied)-set(asdict(spec))
        if unknown:
            raise ValueError(f'Unknown manual parameters: {unknown}')
        spec = ManualSpec(**supplied)
    spec.validate()
    spec.gray_white = args.gray_white or spec.gray_white
    spec.light_colour = args.light_colour or spec.light_colour
    spec.validity_optimize = getattr(args,'validity_optimize',False) or spec.validity_optimize
    spec.colour_regions = getattr(args,'colour_regions',False) or spec.colour_regions
    spec.region_support = getattr(args,'region_support',False) or spec.region_support
    spec.region_bridge = getattr(args,'region_bridge',False) or spec.region_bridge
    for option in ('weak_red','weak_white','recent_missing','small_fallback','relative_red','missing_support'):
        setattr(spec,option,getattr(args,option,False) or getattr(spec,option))
    pixel_override = getattr(args,'pixel_fallback',None)
    spec.pixel_fallback = supplied.get('pixel_fallback',True) if pixel_override is None else pixel_override
    detector = ManualDetector(label,reference,spec)
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[2]/'output'/'water_manual'/f'{args.camera}_{datetime.now():%Y%m%d_%H%M%S}'
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    for name in ('data','pages'):
        (out/name).mkdir()
    cv.imwrite(str(out/'reference.png'),reference)
    np.savez_compressed(out/'reference.npz',**{b['label']:f.transpose(2,0,1).reshape(6,-1) for b,f in zip(detector.blocks,detector.reference)})
    (out/'labels.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    geometry = {b['label']:dict(centers=g['centers'].tolist(),unique_pixels=g['area'],types=t.tolist()) for b,g,t in zip(detector.blocks,detector.geometry,detector.types)}
    for block in detector.blocks:
        geometry[block['label']]['expected_body'] = 'red' if block['number'] % 2 else 'white'
    (out/'geometry.json').write_text(json.dumps(geometry),encoding='utf-8')
    if detector.validity is not None:
        (out/'reference_validity.json').write_text(json.dumps({b['label']:r for b,r in zip(detector.blocks,detector.reference_validity)},indent=2),encoding='utf-8')
    cap = cv.VideoCapture(args.video)
    writer = None
    count, next_save, pages, history = 0,0.,[],[]
    selected, rank, previous = None,(-1,-1),None
    events, captured, scene, scene_rank, previous_gray = [],set(),None,None,None
    summary_time = None
    early_snapshot = None
    summary_fallback = False
    detected, next_detect, next_progress = 0,0.,10.
    try:
        if not cap.isOpened():
            raise ValueError('Cannot decode video')
        fps = args.fps or cap.get(cv.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Invalid FPS')
        if detect_hz:
            if detect_hz > fps:
                raise ValueError('--detect-hz cannot exceed source FPS')
            spec.max_observation_gap_s = max(spec.max_observation_gap_s,1/detect_hz+1/fps+1e-6)
        with (out/'blocks.jsonl').open('w',encoding='utf-8') as log:
            while True:
                ok,frame = cap.read()
                if not ok:
                    break
                now = count/fps
                if args.stop_after is not None and now > args.stop_after:
                    break
                if detect_hz and now+1e-9 < next_detect:
                    if writer is not None:
                        writer.write(draw_result(frame,detector,results))
                    count += 1
                    continue
                if detect_hz:
                    next_detect = (np.floor(now*detect_hz+1e-9)+1)/detect_hz
                results = detector.step(frame,now)
                detected += 1
                compact = [dict((k,v) for k,v in r.items() if k not in ('features','anomaly','colour_evidence','region_map')) for r in results]
                log.write(json.dumps(dict(frame=count,time_s=now,blocks=compact))+'\n')
                history.append((now,[(r['abnormal'],r['valid'],r['accum'],r['status']) for r in results]))
                statuses = tuple(r['status'] for r in results)
                priority = (statuses.count('error'),statuses.count('*'))
                if now < summary_skip:
                    early_snapshot = (frame.copy(),results,compact,dict(frame=count,time_s=now))
                if now >= summary_skip and priority > rank:
                    rank = priority
                    selected = compact
                    summary_time = dict(frame=count,time_s=now)
                    cv.imwrite(str(out/'summary.png'),draw_result(frame,detector,results))
                    cv.imwrite(str(out/'summary_raw.png'),frame)
                if args.dual_scenes:
                    gray = cv.resize(cv.cvtColor(frame,cv.COLOR_BGR2GRAY),(320,180))
                    if previous_gray is not None and gray.std()>10:
                        factor = np.median(gray)/max(np.median(previous_gray),1)
                        motion = float(np.mean(np.abs(gray.astype(float)-previous_gray.astype(float)*factor)>25))
                        candidate = (statuses.count('?'),motion)
                        if scene_rank is None or candidate<scene_rank:
                            scene_rank = candidate
                            scene = dict(frame=count,time_s=now,motion_fraction=motion,selection='fewest unknown blocks, then least inter-frame motion; stationary occluders may remain',blocks=compact)
                            cv.imwrite(str(out/'scene.png'),draw_result(frame,detector,results))
                            cv.imwrite(str(out/'scene_raw.png'),frame)
                    previous_gray = gray
                if args.diagnostics and args.independent_evidence:
                    for result in results:
                        key = (result['label'],result['status'])
                        if result['status'] not in ('*','error') or key in captured:
                            continue
                        name = f'evidence_{result["label"]}_{"error" if result["status"]=="error" else "suspected"}'
                        matrix_page(out,count,now,[result],args.matrix_labels,name,frame,detector)
                        captured.add(key)
                        events.append(dict(label=result['label'],status=result['status'],frame=count,time_s=now,page=name))
                if args.diagnostics and len(pages)<args.max_pages and (now>=next_save or statuses!=previous or any(abs(now-t)<.5/fps for t in args.at_s)):
                    samples = frame.copy()
                    for block,g in zip(detector.blocks,detector.geometry):
                        cv.polylines(samples,[np.rint(block['poly']).astype(int)],True,(255,150,0),1)
                        for station in g['centers']:
                            for j,(x,y) in enumerate(station):
                                cv.rectangle(samples,(int(x)-2,int(y)-2),(int(x)+2,int(y)+2),(0,255,255),1)
                    cv.imwrite(str(out/f'samples_{count:08d}.png'),samples)
                    display = [r for r in results if r['status'] in ('*','error')] if args.abnormal_only else results
                    if display:
                        matrix_page(out,count,now,display,args.matrix_labels,frame=frame,detector=detector)
                        pages.append((count,now))
                        if args.abnormal_details and not args.independent_evidence:
                            events.extend(dict(label=r['label'],status=r['status'],frame=count,time_s=now,page=f'frame_{count:08d}') for r in display if r['status'] in ('*','error'))
                    next_save = now+args.matrix_every
                previous = statuses
                if args.overlay_video:
                    if writer is None:
                        writer = cv.VideoWriter(str(out/'overlay.mp4'),cv.VideoWriter_fourcc(*'mp4v'),fps,(frame.shape[1],frame.shape[0]))
                        if not writer.isOpened():
                            raise ValueError('Cannot create overlay video')
                    writer.write(draw_result(frame,detector,results))
                count += 1
                if now >= next_progress:
                    print(f'Manual blocks: {count} source frames, {detected} detections, t={now:.1f}s; elapsed={perf_counter()-started:.1f}s',flush=True)
                    next_progress = now+10.
        if not count:
            raise ValueError('No decoded frames')
        if selected is None:
            image,results,selected,summary_time = early_snapshot
            cv.imwrite(str(out/'summary.png'),draw_result(image,detector,results))
            cv.imwrite(str(out/'summary_raw.png'),image)
            summary_fallback = True
        import hashlib
        manifest = dict(mode='manual_blocks',complete=True,video=args.video,camera=args.camera,fps=fps,frames=count,parameters=asdict(spec),matrix_every=args.matrix_every,
            matrices=dict(features='6 x 5n',anomaly='6 x n',codes={'normal':0,'abnormal':1,'unknown':-1}),normal_reference='reference.png',diagnostics=args.diagnostics)
        manifest['source_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(S.__file__),label)}
        manifest['decoded_frames'] = count
        manifest['detected_frames'] = detected
        manifest['detect_hz'] = detect_hz
        manifest['elapsed_before_report_s'] = perf_counter()-started
        manifest['snapshots'] = [dict(frame=i,time_s=t) for i,t in pages]
        manifest['options'] = {key:getattr(args,key) for key in ('gray_white','light_colour','abnormal_details','abnormal_only','matrix_labels','independent_evidence','dual_scenes')}
        manifest['options']['validity_optimize'] = spec.validity_optimize
        manifest['options']['colour_regions'] = spec.colour_regions
        manifest['options']['region_support'] = spec.region_support
        manifest['options']['region_bridge'] = spec.region_bridge
        for option in ('weak_red','weak_white','recent_missing','small_fallback','pixel_fallback','relative_red','missing_support'):
            manifest['options'][option] = getattr(spec,option)
        for enabled,filename in [(spec.weak_red or spec.weak_white,'water_manual_weak.py'),(spec.recent_missing,'water_manual_recent.py'),(spec.small_fallback,'water_manual_small.py'),(spec.pixel_fallback,'water_manual_pixels.py'),(spec.relative_red or spec.missing_support,'water_manual_relative.py')]:
            if enabled:
                module = Path(__file__).with_name(filename)
                manifest['source_sha256'][filename] = hashlib.sha256(module.read_bytes()).hexdigest()
        if spec.region_support:
            module = Path(__file__).with_name('water_manual_support.py')
            manifest['source_sha256'][module.name] = hashlib.sha256(module.read_bytes()).hexdigest()
        if spec.colour_regions or spec.small_fallback:
            module = Path(__file__).with_name('water_manual_regions.py')
            manifest['source_sha256'][module.name] = hashlib.sha256(module.read_bytes()).hexdigest()
        manifest['summary_frame'] = summary_time
        manifest['summary_skip_s'] = summary_skip
        manifest['summary_short_video_fallback'] = summary_fallback
        manifest['evidence'] = events
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        (out/'summary.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
        if args.dual_scenes:
            if scene is None:
                scene = dict(**summary_time,selection='fallback: no usable scene candidate',blocks=selected)
                import shutil
                shutil.copyfile(out/'summary.png',out/'scene.png')
            (out/'scene.json').write_text(json.dumps(scene,indent=2),encoding='utf-8')
        body = f'<p>源帧数={count}；检测次数={detected}；抽帧频率={detect_hz if detect_hz else "逐帧"}；报告生成前耗时={manifest["elapsed_before_report_s"]:.2f}s。</p>'
        body += f'<p>总图跳过视频前{summary_skip:g}秒。'+('视频没有达到择帧起点，回退至最后检测帧。' if summary_fallback else '')+'</p>'
        display_results = scene['blocks'] if args.dual_scenes else selected
        display_image = 'scene.png' if args.dual_scenes else 'summary.png'
        display_time = scene if args.dual_scenes else summary_time
        body += f'<p>独立单块模式；仅疑似*=橙框、error=红框；正常及无法判断不画框，无框不代表正常。正常参考需与视频机位一致。</p><p>展示帧：{display_time["time_s"]:.3f}s；'+('采用较少扰动场景图；图和表均为该时刻状态。' if args.dual_scenes else '未启用双场景图，显示异常证据帧。')+f'</p><img src="{display_image}"><h2>Block汇总（展示帧）</h2><table><tr><th>块</th><th>状态</th><th>有效站</th><th>异常站</th><th>累计秒</th><th>像素支持</th><th>原因</th></tr>'
        for r in display_results:
            body += f'<tr><td>{r["label"]}</td><td>{r["status"]}</td><td>{r["valid"]:.1%}</td><td>{r["abnormal"]:.1%}</td><td>{r["accum"]:.2f}</td><td>{r["pixels"]}</td><td>{r["reason"]}</td></tr>'
        body += '</table>'
        if spec.colour_regions:
            body += '<h2>连续色块存在检查</h2><p>按实际坐标连接相似邻点/邻站；奇数块查红色，偶数块查白色。存在不等于确认身份。原异常站比例仍代表参考变化，色块缺失另列。</p><table><tr><th>块</th><th>应有颜色</th><th>可判断</th><th>连续色块存在</th><th>需跨站数</th></tr>'
            for r in display_results:
                q = r['regions']
                body += f'<tr><td>{r["label"]}</td><td>{q["expected"]}</td><td>{q["observable"]}</td><td>{q["present"]}</td><td>{q["required_stations"]}</td></tr>'
            body += '</table>'
        if detector.validity is not None:
            body += '<h2>正常参考可检测性</h2><p>小块优化启用；5×5窗口至少15个有效像素，最低总像素支持保持不变。参考不足不等于缺失。</p><table><tr><th>块</th><th>参考状态</th><th>有效站</th><th>要求站数</th><th>唯一像素</th></tr>'
            for b,r in zip(detector.blocks,detector.reference_validity):
                body += f'<tr><td>{b["label"]}</td><td>{"可检测" if r["supported"] else "参考/分辨率不足"}</td><td>{r["valid_stations"]}/{r["stations"]}</td><td>{r["minimum_stations"]}</td><td>{r["pixels"]}</td></tr>'
            body += '</table>'
        if args.dual_scenes:
            body += f'<details><summary>查看最高异常状态证据帧（可能包含人车扰动）</summary><p>帧{summary_time["frame"]}，t={summary_time["time_s"]:.3f}s；显示该时刻状态。</p><img src="summary.png"></details>'
        if args.abnormal_details:
            body += '<h2>ERROR／疑似专项诊断</h2>'
            for event in events:
                body += f'<p><a href="pages/{event["page"]}.html#{event["label"]}">{event["label"]} {event["status"]} / 帧{event["frame"]} / {event["time_s"]:.3f}s</a></p>'
            if not events:
                body += '<p>没有保存到ERROR／疑似证据。</p>'
        if not args.abnormal_only:
            body += '<h2>单帧矩阵</h2>'+''.join(f'<p><a href="pages/frame_{i:08d}.html">帧{i} / {t:.3f}s</a></p>' for i,t in pages)
        elif not args.abnormal_details:
            body += '<h2>异常Block矩阵</h2>'+''.join(f'<p><a href="pages/frame_{i:08d}.html">帧{i} / {t:.3f}s</a></p>' for i,t in pages)
        body += '<h2>各Block时间曲线</h2><p>蓝=异常站比例；绿=有效站比例；红=累计时间/确认时间（上限1）；纵轴0–1，横轴视频秒。底部状态带：绿正常、橙疑似、红error、灰无法判断。无法判断时异常比例断线。</p>'
        plot_masks = {}
        if getattr(args,'plot_mask',True):
            from water_plot_mask import find_windows, bands
            plot_masks = find_windows(history,detector.blocks)
            (out/'plot_masks.json').write_text(json.dumps(dict(
                display_only=True,max_seconds=5,onset_seconds=1,min_blocks=2,
                blocks=plot_masks),indent=2),encoding='utf-8')
            body += '<p>灰色透明时间带=mask共同短时扰动；同排至少2块在1秒内开始变化且各自在5秒内恢复正常。悬停查看起止秒。仅显示标记，原检测、曲线和累计不变；不代表已识别人车。</p>'
        for k,block in enumerate(detector.blocks):
            duration = max(history[-1][0],.001)
            body += f'<h3>{block["label"]}</h3><svg viewBox="0 0 900 150" style="width:100%;max-width:1000px"><path d="M40 10 V125 H880" fill="none" stroke="black"/><text x="10" y="20">1</text><text x="10" y="125">0</text><text x="40" y="145">0s</text><text x="810" y="145">{duration:.2f}s</text>'
            if plot_masks:
                body += bands(plot_masks[block['label']],duration)
            for component,colour in [(0,'blue'),(1,'green'),(2,'red')]:
                segments, points = [], []
                for t,v in history:
                    if component == 0 and v[k][3] == '?':
                        segments.append(points);points=[]
                        continue
                    points.append(f'{40+840*t/duration:.1f},{125-110*min(v[k][component]/(spec.hold_s if component==2 else 1),1):.1f}')
                segments.append(points)
                for points in segments:
                    body += f'<polyline points="{" ".join(points)}" fill="none" stroke="{colour}" stroke-width="1"/>'
            for j,(t,v) in enumerate(history):
                end = history[j+1][0] if j+1<len(history) else duration
                colour = {'normal':'#6b8','*':'orange','error':'red','?':'#aaa'}[v[k][3]]
                body += f'<rect x="{40+840*t/duration:.1f}" y="126" width="{max(.5,840*(end-t)/duration):.1f}" height="4" fill="{colour}"/>'
            body += '</svg>'
        body += '<details><summary>全部参数</summary><pre>'+html.escape(json.dumps(manifest,indent=2))+'</pre></details>'
        (out/'index.html').write_text(web_page('固定水马Block检查',body),encoding='utf-8')
        manifest['elapsed_total_s'] = perf_counter()-started
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print(f'Done: {count} source frames, {detected} detections, elapsed={manifest["elapsed_total_s"]:.2f}s; index={out.resolve()/"index.html"}',flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
