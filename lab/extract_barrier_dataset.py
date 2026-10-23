"""Hypothesis: contextual ROI crops preserve barrier appearance for fixed-camera detection.

Extract unlabelled images every N seconds; evaluate after manual annotation.
No training or automatic negative labels are produced.
"""
import argparse
import glob
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


def bounds(points, size, padding):
    width, height = size
    poly = np.asarray(points, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise ValueError('poly must contain at least three [x, y] points')
    if not np.isfinite(poly).all():
        raise ValueError('poly coordinates must be finite')
    if (poly < 0).any() or (poly[:, 0] >= width).any() or (poly[:, 1] >= height).any():
        raise ValueError('poly is outside reference_size')
    if abs(cv2.contourArea(poly.astype(np.float32))) < 1:
        raise ValueError('poly has zero area')
    low, high = poly.min(axis=0), poly.max(axis=0)
    margin = (high - low) * padding
    x0, y0 = np.maximum(0, np.floor(low - margin)).astype(int)
    x1, y1 = np.minimum([width, height], np.ceil(high + margin + 1)).astype(int)
    return int(x0), int(y0), int(x1), int(y1)


def extract(video, camera, split, roi, args, manifest):
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open {video}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f'Invalid FPS: {video}')
        reference = tuple(roi['reference_size'])
        box = bounds(roi['poly'], reference, args.padding)
        token = hashlib.sha256(str(video.resolve()).encode()).hexdigest()[:12]
        dest = args.out / 'images' / split
        dest.mkdir(parents=True, exist_ok=True)
        count, index, sample = 0, 0, 0
        # Decode sequentially to avoid inaccurate seeks on long-GOP recordings.
        while cap.grab():
            target = round(sample * args.interval * fps)
            if index < target:
                index += 1
                continue
            ok, frame = cap.retrieve()
            if not ok:
                raise ValueError(f'Cannot decode frame {index}: {video}')
            height, width = frame.shape[:2]
            if (width, height) != reference:
                raise ValueError(f'{video}: frame size {(width, height)} != reference {reference}')
            x0, y0, x1, y1 = box
            name = f'{camera}_{token}_f{index:09d}.jpg'
            path = dest / name
            if path.exists():
                raise FileExistsError(f'Refusing to overwrite {path}; use a fresh output folder')
            if not cv2.imwrite(str(path), frame[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f'Cannot write {path}')
            manifest.write(json.dumps(dict(image=str(path.relative_to(args.out)),
                video=str(video.resolve()), camera=camera, split=split,
                frame=index, seconds=index / fps, crop_xyxy=box,
                poly=roi['poly'], reference_size=reference), ensure_ascii=False) + '\n')
            count += 1
            sample += 1
            index += 1
        if count == 0:
            raise ValueError(f'No frames decoded: {video}')
        print(f'{camera} {split}: {video.name} -> {count} images', flush=True)
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--padding', type=float, default=0.1,
                        help='Margin per side as fraction of polygon bounding-box size')
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error('--interval must be finite and positive')
    if not math.isfinite(args.padding) or args.padding < 0:
        parser.error('--padding must be finite and nonnegative')
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    jobs, assigned = [], set()
    for camera, roi in config['cameras'].items():
        if not camera or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in camera):
            raise ValueError('Camera IDs must use letters, digits, underscore or hyphen')
        size = roi['reference_size']
        if len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
            raise ValueError('reference_size must be [width, height] positive integers')
        bounds(roi['poly'], size, args.padding)
        for split, patterns in roi['videos'].items():
            if split not in {'train', 'val', 'test'}:
                raise ValueError(f'Unknown split {split}')
            for pattern in patterns:
                absolute = Path(pattern)
                if not absolute.is_absolute():
                    absolute = args.config.resolve().parent / absolute
                files = sorted(Path(p).resolve() for p in glob.glob(str(absolute), recursive=True) if Path(p).is_file())
                if not files:
                    raise ValueError(f'No videos match {pattern}')
                for video in files:
                    key = str(video).casefold()
                    if key in assigned:
                        raise ValueError(f'Video assigned more than once: {video}')
                    assigned.add(key)
                    jobs.append((video, camera, split, roi))
    if not jobs:
        raise ValueError('No videos configured')
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'manifest.jsonl').open('x', encoding='utf-8') as manifest:
        for video, camera, split, roi in jobs:
            extract(video, camera, split, roi, args, manifest)
    print('Extraction complete. Annotate images before training; no labels were generated.')


if __name__ == '__main__':
    main()
