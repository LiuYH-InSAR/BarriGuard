"""Reconcile jittered duplicates and reobserve orphaned quarantine, using pixels."""
from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
import cv2 as cv
import numpy as np
from .adaptive_prototype import center, iou
from .nested_fixture_v6 import containment
from .optical_fixture_v9 import same_fixture_hypothesis
from .adaptive_flash_signal import signal_plane


def overlapping_fixture(a, b):
    edge = min(max(a[2]-a[0], a[3]-a[1]), max(b[2]-b[0], b[3]-b[1]))
    return bool(iou(a, b) >= .5 and containment(a, b) >= .8
                and np.linalg.norm(center(a)-center(b)) <= .3*edge)


def related(a, b):
    return overlapping_fixture(a, b) or same_fixture_hypothesis(a, b)


@dataclass
class Evidence:
    box: list
    samples: object = field(default_factory=lambda: deque(maxlen=64))
    last_probe: float = -1e9


def optical_proof(samples, box):
    if len(samples) < 16:
        return None
    effective = sum(b[0]-a[0] for a, b in zip(samples, list(samples)[1:])
                    if b[2] == a[2] and 0 < b[0]-a[0] <= .5)
    if effective < 3:
        return None
    stack = np.stack([s[1] for s in samples])
    lo, hi = np.percentile(stack, [20, 90], axis=0)
    spread = hi-lo
    maximum = float(spread.max())
    if maximum < 25:
        return None
    components = []
    for fraction in [.5, .8]:
        count, labels, stats, _ = cv.connectedComponentsWithStats(np.uint8(spread >= max(25., maximum*fraction)))
        strong = [j for j in range(1, count) if stats[j, cv.CC_STAT_AREA] >= 3]
        if len(strong) != 1:
            return None
        components.append(labels == strong[0])
    core = components[0]
    if float(core.mean()) > .55:
        return None  # Whole-patch exposure change is not one localized emitter.
    weights = spread*core
    yy, xx = np.indices(spread.shape)
    total = float(weights.sum())
    point = [box[0]+(float((weights*xx).sum()/total)+.5)*(box[2]-box[0])/32,
             box[1]+(float((weights*yy).sum()/total)+.5)*(box[3]-box[1])/32]
    series = (stack-lo)[:, core].mean(axis=1)
    low, high = np.percentile(series, [10, 90])
    amplitude = float(high-low)
    if amplitude < 18:
        return None
    phase, cycles = 'unknown', 0
    for i, value in enumerate(series):
        if i == 0 or samples[i][2] != samples[i-1][2] or samples[i][0]-samples[i-1][0] > .5:
            phase = 'unknown'
        if value <= low+.3*amplitude:
            cycles += phase == 'on'
            phase = 'off'
        elif value >= low+.65*amplitude and phase == 'off':
            phase = 'on'
    high_samples = int(np.count_nonzero(series >= low+.65*amplitude))
    low_samples = int(np.count_nonzero(series <= low+.3*amplitude))
    if cycles < 2 or min(high_samples, low_samples) < 3:
        return None
    return dict(samples=len(samples), span_s=samples[-1][0]-samples[0][0], effective_s=effective,
                cycles=int(cycles), amplitude=amplitude, point=point, unique_core=True,
                core_fraction=float(core.mean()), high_samples=high_samples, low_samples=low_samples,
                gap_cycles_excluded=True)


def ownership_verified(proof, live, previous):
    members = proof.get('members', [])
    if len(members) not in (1, 2) or len({m['id'] for m in members}) != len(members):
        return False
    if not(proof.get('samples', 0) >= 16 and 3 <= proof.get('effective_s', 0) <= proof.get('span_s', 0)+1e-6
           and proof.get('span_s', 100) <= 12.000001 and proof.get('cycles', 0) >= 2
           and proof.get('amplitude', 0) >= 18 and proof.get('unique_core')
           and 0 < proof.get('core_fraction', 1) <= .55 and proof.get('gap_cycles_excluded')
           and min(proof.get('high_samples', 0), proof.get('low_samples', 0)) >= 3
           and proof.get('emission_samples', 0) >= 3
           and proof.get('detection_updates', 0) >= 3):
        return False
    point = np.asarray(proof.get('point', []))
    if point.shape != (2,) or not np.isfinite(point).all():
        return False
    for member in members:
        prior = previous.get(member['id'])
        b = member['box']
        if (prior is None or prior.get('observation_epoch', 0) != member['epoch']
            or iou(prior.get('body_box', prior['box']), b) < .8
            or member.get('hits', 0) < 3 or member.get('confidence', 0) < .65
            or not(b[0] <= point[0] <= b[2] and b[1] <= point[1] <= b[3])):
            return False
    if len(members) == 2:
        return overlapping_fixture(members[0]['box'], members[1]['box'])
    # An orphan must have both prior physical-lamp evidence and no current
    # unresolved competitor. Disappearance alone can never lift quarantine.
    member = members[0]
    if member.get('prior_pulses', 0) < 15 or member.get('hits', 0) < 5:
        return False
    return not any(t['id'] != member['id'] and t['confidence'] >= .65
                   and related(member['box'], t.get('body_box', t['box'])) for t in live.values())


class SingleSourceResolver:
    def __init__(self):
        self.states = {}
        self.segment = 0

    def invalidate(self):
        # Retain bounded, already valid pixels, but never count a cycle or time
        # across a bad frame. A scene reset constructs a new resolver.
        self.segment += 1

    def observe(self, frame, tracks, epochs, quarantined, ts, valid):
        if not valid:
            self.invalidate()
            return []
        eligible = [t for t in tracks.values() if t.confidence >= .65 and t.hits >= 3
                    and t.position_observable and t.quality_details.get('background', {}).get('ok')]
        families = [(a, b) for a, b in combinations(eligible, 2) if overlapping_fixture(a.body_box, b.body_box)]
        for t in eligible:
            if (t.id in quarantined and t.hits >= 5 and t.confirmed and t.pulses.count >= 15
                and t.locator.locked and t.flash.last.get('ready')
                and not any(o.id != t.id and o.confidence >= .65 and related(t.body_box, o.body_box) for o in tracks.values())):
                families.append((t,))
        active, offers = set(), []
        for family in families[:64]:
            key = tuple((t.id, epochs.get(t.id, 0)) for t in family)
            active.add(key)
            boxes = np.asarray([t.body_box for t in family])
            outer = [float(boxes[:, 0].min()), float(boxes[:, 1].min()), float(boxes[:, 2].max()), float(boxes[:, 3].max())]
            margin = max(3, int(.15*max(outer[2]-outer[0], outer[3]-outer[1])))
            roi = [int(np.floor(outer[0]))-margin, int(np.floor(outer[1]))-margin,
                   int(np.ceil(outer[2]))+margin, int(np.ceil(outer[3]))+margin]
            if min(roi[:2]) < 0 or roi[2] > frame.shape[1] or roi[3] > frame.shape[0]:
                continue
            state = self.states.get(key)
            if state is None or iou(state.box, roi) < .8 or np.linalg.norm(center(state.box)-center(roi)) > 2:
                state = self.states[key] = Evidence(roi)
            x0, y0, x1, y1 = state.box
            crop = cv.resize(signal_plane(frame[y0:y1, x0:x1]), (32, 32)).astype(float)
            state.samples.append((ts, crop, self.segment, sum(t.hits for t in family),
                                  any(t.flash.last.get('emission_supported') for t in family)))
            while state.samples and ts-state.samples[0][0] > 12:
                state.samples.popleft()
            if ts-state.last_probe < .5:
                continue
            state.last_probe = ts
            proof = optical_proof(state.samples, state.box)
            updates = sum(t.hits for t in family)-state.samples[0][3]
            emission_samples = sum(s[4] for s in state.samples)
            if proof is None or updates < 3 or emission_samples < 3:
                continue
            point = proof['point']
            if not all(t.body_box[0] <= point[0] <= t.body_box[2] and t.body_box[1] <= point[1] <= t.body_box[3] for t in family):
                continue
            # A separately detected source near the optical core remains a veto.
            if any(o not in family and o.confidence >= .65 and o.hits >= 3
                   and o.body_box[0] <= point[0] <= o.body_box[2] and o.body_box[1] <= point[1] <= o.body_box[3]
                   for o in tracks.values()):
                continue
            proof.update(kind='overlap_single_source' if len(family) == 2 else 'orphan_single_source',
                detection_updates=updates, emission_samples=emission_samples, members=[dict(id=t.id, epoch=epochs.get(t.id, 0), box=t.body_box[:],
                    hits=t.hits, confidence=t.confidence, prior_pulses=t.pulses.count) for t in family])
            offers.append((family, proof))
        self.states = {key: value for key, value in self.states.items() if key in active}
        return offers
