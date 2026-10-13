"""Optional lower red/white segments for visual review; never feed gap detection."""
from dataclasses import asdict, dataclass

import cv2 as cv
import numpy as np

from water_barrier import signal as S


@dataclass(frozen=True)
class BlockSpec:
    score_min: float = .35
    width_ratio: float = .60
    valid_ratio: float = .90
    white_score_min: float = .05

    def validate(self):
        if not 0 <= self.score_min <= 1:
            raise ValueError('Block score minimum must be between 0 and 1')
        if not 0 <= self.white_score_min <= 1:
            raise ValueError('White block score minimum must be between 0 and 1')
        if not np.isfinite(self.width_ratio) or self.width_ratio <= 0:
            raise ValueError('Block width ratio must be positive')
        if not 0 < self.valid_ratio <= 1:
            raise ValueError('Block valid ratio must be in (0, 1]')


def scan_windows(red, white, valid, widths, spec):
    """Group consecutive stations by the stronger lower colour after filtering."""
    spec.validate()
    red, white = np.asarray(red), np.asarray(white)
    valid, widths = np.asarray(valid, bool), np.asarray(widths)
    if not (len(red) == len(white) == len(valid) == len(widths)):
        raise ValueError('Window arrays must have equal lengths')
    labels = np.zeros(len(red), np.uint8)
    labels[valid & (red >= spec.score_min) & (red > white)] = 1
    labels[valid & (white >= spec.white_score_min) & (white > red)] = 2
    cuts = np.r_[0, np.flatnonzero(np.diff(labels.astype(int)))+1, len(labels)]
    kept = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if a == b or labels[a] == 0:
            continue
        coverage = float(valid[a:b].mean())
        if coverage < spec.valid_ratio:
            continue
        colour = 'red' if labels[a] == 1 else 'white'
        values = red if colour == 'red' else white
        kept.append(dict(start=int(a), end=int(b-1), width=int(b-a), colour=colour,
                         score=float(values[a:b].mean()), valid_ratio=coverage))
    return kept


def estimate_blocks(frame, rows, trim, spec):
    spec.validate()
    blocks, row_data = [], []
    height, width = frame.shape[:2]
    for row in rows:
        sample = {}
        pack = S.sample_row(frame, np.asarray(row['poly'], np.float32), trace=sample)
        valid = S.apply_end_trim(pack, trim)['v']
        def resample(values):
            return cv.resize(np.asarray(values, np.float32).reshape(1, -1), (S.REF, 1)).ravel()
        red = resample(sample.get('red_ratio', np.zeros(pack['M'])))
        white = resample(sample.get('lower_white_ratio', np.zeros(pack['M'])))
        red_sm, white_sm = S.smooth(red, valid), S.smooth(white, valid)
        # Preserve local width as a review reference; colour transitions define boundaries.
        local_height = pack['y1']-pack['y0']+1
        widths = local_height * spec.width_ratio / pack['step'] * S.REF / pack['M']
        windows = scan_windows(red_sm, white_sm, valid, widths, spec)
        for i, candidate in enumerate(windows, 1):
            a, b = candidate['start'], candidate['end']
            blocks.append(dict(row=row['id'], id=i, **candidate,
                width_px=float(candidate['width']/S.REF*pack['M']*pack['step']),
                box=list(S.pixel_box(pack, a, b, width, height))))
        row_data.append(dict(row=row['id'], red=red.tolist(), white=white.tolist(),
            red_smoothed=red_sm.tolist(), white_smoothed=white_sm.tolist(), valid=valid.tolist(),
            window_widths=[float(x) if np.isfinite(x) else None for x in widths],
            native_stations=pack['M'], count=len(windows)))
    return dict(enabled=True, algorithm='lower_colour_runs', parameters=dict(asdict(spec), trim=trim,
        smooth_sigma=S.SMOOTH_SIGMA, ref=S.REF, width_source='poly_height_reference',
        selection='consecutive dominant lower colour'), blocks=blocks, rows=row_data)
