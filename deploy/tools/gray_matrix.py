"""Disk-backed ROI Gray capture for offline coordinate inspection."""
import json
from pathlib import Path

import cv2 as cv
import numpy as np

from night_lamp.session import NightSession

CHUNK_FRAMES = 32


class GrayRecorder:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.stream = None
        self.times = []
        self.offsets = []
        self.mref = None
        self.parameters = None
        self.colour_stream = None
        self.update_records = []

    def append(self, gray, mask, offset, timestamp, bgr=None, background=None, update=None):
        if self.stream is None:
            ys, xs = np.nonzero(~mask)
            if not len(ys):
                return
            self.y0, self.x0 = int(ys.min()), int(xs.min())
            self.y1, self.x1 = int(ys.max()) + 1, int(xs.max()) + 1
            self.valid = ~mask[self.y0:self.y1, self.x0:self.x1]
            self.full_shape = list(gray.shape)
            self.stream = (self.folder / 'frames.dat').open('wb')
            np.save(self.folder / 'valid_mask.npy', self.valid)
            cv.imwrite(str(self.folder / 'reference_gray.jpg'), gray)
            if background is not None:
                np.save(self.folder / 'initial_background.npy', background[self.y0:self.y1, self.x0:self.x1])
        crop = gray[self.y0:self.y1, self.x0:self.x1]
        self.stream.write(np.ascontiguousarray(crop).tobytes())
        self.times.append(float(timestamp))
        self.offsets.append(float(offset))
        self.update_records.append(update)
        if bgr is not None:
            if self.colour_stream is None:
                self.colour_stream = (self.folder / 'bgr_frames.dat').open('wb')
            self.colour_stream.write(np.ascontiguousarray(bgr[self.y0:self.y1, self.x0:self.x1]).tobytes())

    def finish(self):
        if self.stream is None:
            return
        self.stream.close()
        height, width = self.valid.shape
        count = len(self.times)
        source = np.memmap(self.folder / 'frames.dat', dtype=np.uint8,
                           mode='r', shape=(count, height, width))
        matrix = np.lib.format.open_memmap(self.folder / 'gray_raw.npy',
                   mode='w+', dtype=np.uint8, shape=(height, width, count))
        for start in range(0, count, CHUNK_FRAMES):
            matrix[:, :, start:start + CHUNK_FRAMES] = source[start:start + CHUNK_FRAMES].transpose(1, 2, 0)
        matrix.flush()
        del matrix, source
        (self.folder / 'frames.dat').unlink()
        if self.colour_stream is not None:
            self.colour_stream.close()
            source = np.memmap(self.folder / 'bgr_frames.dat', dtype=np.uint8, mode='r', shape=(count,height,width,3))
            matrix = np.lib.format.open_memmap(self.folder / 'bgr_raw.npy', mode='w+', dtype=np.uint8, shape=(height,width,count,3))
            for start in range(0,count,CHUNK_FRAMES):
                matrix[:,:,start:start+CHUNK_FRAMES] = source[start:start+CHUNK_FRAMES].transpose(1,2,0,3)
            matrix.flush()
            del matrix, source
            (self.folder / 'bgr_frames.dat').unlink()
        if self.update_records and all(x is not None for x in self.update_records):
            np.save(self.folder / 'background_updates.npy', np.asarray(self.update_records, np.float64))
        np.save(self.folder / 'time_s.npy', np.asarray(self.times))
        np.save(self.folder / 'exposure_offset.npy', np.asarray(self.offsets))
        metadata = {'axes': ['row', 'column', 'sample'],
                    'mref': self.mref,
                    'parameters': self.parameters,
                    'shape': [height, width, count], 'full_frame_shape': self.full_shape,
                    'origin_row': self.y0, 'origin_col': self.x0,
                    'coordinates': 'original video coordinates, zero-based',
                    'scope': 'expanded ROI bounding rectangle; valid_mask excludes OSD and pixels outside ROI',
                    'sampling': 'algorithm accepted samples, including warmup; actual timestamps in time_s.npy',
                    'corrected': 'gray_raw.astype(float) - exposure_offset; not clipped'}
        metadata['background_update_columns'] = ['reset_after_hit', 'ema_alpha', 'frame_global_median'][:len(self.update_records[0])] if self.update_records and self.update_records[0] is not None else []
        metadata['storage'] = {'gray_bytes':height*width*count,
            'bgr_bytes':height*width*count*3 if self.colour_stream is not None else 0,
            'background_update_bytes':count*len(metadata['background_update_columns'])*8,
            'initial_background_bytes':height*width*4}
        print(f'Gray diagnostics saved: samples={count}; Gray={metadata["storage"]["gray_bytes"]/1024**3:.2f} GiB; BGR={metadata["storage"]["bgr_bytes"]/1024**3:.2f} GiB; folder={self.folder}')
        (self.folder / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


class GrayRecordingSession(NightSession):
    def __init__(self, camera_id, spec, recording_dir):
        super().__init__(camera_id, spec)
        self.recorder = GrayRecorder(recording_dir)
        self._colour_warm_paths = {}
        from night_lamp import evidence as ev
        from night_lamp.tools import nightly_map as nm
        self._captured_on_delta = nm.ON_DELTA
        p = spec.periodicity
        self.recorder.parameters = {
            'on_delta': nm.ON_DELTA, 'duty_floor': nm.CAND_DUTY_LO,
            'baseline_frames': spec.baseline.frames,
            'gap_min': ev.MIN_STATE_GAP, 'weight_min': ev.MIN_STATE_WEIGHT,
            'separation_min': ev.MIN_SEPARATION, 'bic_min': ev.MIN_BIC_GAIN,
            'onset_min': p.onset_min, 'peak_floor': p.peak_floor,
            'min_peaks': p.min_peaks, 'gap_tol': p.gap_tol,
            'lag_lo': max(1, int(round(p.lag_lo_s * 1000 / spec.sampling.interval_ms))),
            'lag_hi': max(2, int(round(p.lag_hi_s * 1000 / spec.sampling.interval_ms)))}
        self.recorder.parameters.update(min_samples=ev.MIN_SAMPLES,
            yellow_support=ev.YELLOW_SUPPORT, yellow_separation=ev.YELLOW_SEPARATION,
            min_colour_gain=ev.MIN_COLOUR_GAIN, fit_iterations=ev.FIT_ITERATIONS,
            variance_floor=ev.VARIANCE_FLOOR, region_radius=ev.REGION_RADIUS,
            ema_half_life_s=self._ema_half_life_s)
        self.recorder.parameters['discovery_constants'] = {k:v for k,v in vars(nm).items() if k.isupper() and isinstance(v,(int,float,str,bool))}
        from dataclasses import asdict
        self.recorder.parameters['session_spec'] = asdict(spec)
        from night_lamp.disturbance import PARAMETERS
        self.recorder.parameters['disturbance'] = dict(PARAMETERS)
        from night_lamp.stutter import SETTINGS
        self.recorder.parameters['stutter'] = dict(SETTINGS)
        from night_lamp import single_pixel as sp
        self.recorder.parameters['single_pixel'] = dict(enabled=sp.ENABLED,
            min_hits=sp.MIN_HITS,max_area=sp.MAX_AREA,max_tracks=sp.MAX_TRACKS,
            min_effective_seconds=sp.MIN_EFFECTIVE_SECONDS)

    def accumulate(self, frame_bgr, ts_mono):
        if not self._warm_done:
            path = self.recorder.folder / f'warm_colour_{len(self._colour_warm_paths):04d}.npy'
            np.save(path, frame_bgr)
            self._colour_warm_paths[float(ts_mono)] = path
        return super().accumulate(frame_bgr, ts_mono)

    def _process(self, gray, med, dt=None, bgr=None, ts=None):
        self.recorder.mref = float(self._detrend)
        from night_lamp.session import _CHANGE_POINT_DELTA, _CHANGE_POINT_RESET_ENABLED, _LOOKUP_R, _DEDUP_R
        reset = self._base_reset or (_CHANGE_POINT_RESET_ENABLED and self._last_med is not None and abs(med-self._last_med)>_CHANGE_POINT_DELTA)
        alpha = self._ema_alpha if dt is None or dt<=0 or self._ema_half_life_s<=0 else 1-.5**(dt/self._ema_half_life_s)
        self.recorder.parameters['change_point_delta'] = _CHANGE_POINT_DELTA
        self.recorder.parameters.update(change_point_reset_enabled=_CHANGE_POINT_RESET_ENABLED,
            lookup_radius_px=_LOOKUP_R, dedup_radius_px=_DEDUP_R)
        colour = bgr
        path = self._colour_warm_paths.pop(float(ts), None)
        if colour is None and path is not None:
            colour = np.load(path)
        self.recorder.append(gray, self._mask, int(round(med - self._detrend)), ts,
            bgr=colour, background=self._base, update=(int(reset), alpha, float(med)))
        if path is not None:
            path.unlink()
        return super()._process(gray, med, dt, bgr, ts)
