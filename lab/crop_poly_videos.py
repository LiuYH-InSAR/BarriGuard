"""Hypothesis: a padded union of barrier polygons preserves useful detection context.

Read X-AnyLabeling JSON and local videos. Produce unlabelled rectangular crops.
Evaluate crop coverage visually before annotation/training.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

DEFAULT_ROOT = Path('C:/Users/admin/Desktop/projects/BarriGuard_Yolo')
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov'}


def load_roi(path, label, padding):
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    width, height = int(data['imageWidth']), int(data['imageHeight'])
    polygons = []
    for shape in data['shapes']:
        if shape['label'] != label:
            continue
        points = np.asarray(shape['points'], dtype=float)
        if shape['shape_type'] != 'polygon' or points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError(f'{path}: invalid polygon')
        if not np.isfinite(points).all():
            raise ValueError(f'{path}: nonfinite coordinates')
        if (points < 0).any() or (points[:, 0] > width).any() or (points[:, 1] > height).any():
            raise ValueError(f'{path}: polygon outside image')
        if abs(cv2.contourArea(points.astype(np.float32))) <= 0:
            raise ValueError(f'{path}: zero-area polygon')
        polygons.append(points)
    if not polygons:
        raise ValueError(f'{path}: no polygons labelled {label}')
    points = np.concatenate(polygons)
    low, high = points.min(axis=0), points.max(axis=0)
    margin = (high - low) * padding
    x0, y0 = np.maximum(0, np.floor(low - margin)).astype(int)
    x1, y1 = np.minimum([width, height], np.ceil(high + margin + 1)).astype(int)
    box = [int(x0), int(y0), int(x1), int(y1)]
    print(f'{path.stem}: {len(polygons)} polygons; ignored {len(data["shapes"])-len(polygons)}; crop {box}', flush=True)
    return (width, height), box, polygons


def extract(video, camera, roi, args, manifest):
    size, box, polygons = roi
    regions = [(None, box)]
    if args.mode == 'blocks':
        regions = [(f'block_{i:03d}', block_box(poly, size, args.padding))
                   for i, poly in enumerate(polygons, 1)]
    cap = cv2.VideoCapture(str(video))
    count = 0
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open {video}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f'{video}: invalid FPS')
        folder = args.out / 'images' / camera
        folder.mkdir(parents=True, exist_ok=True)
        for block_id, _ in regions:
            if block_id:
                (folder / block_id).mkdir(exist_ok=True)
        frame_index, sample = 0, 0
        # Sequential decode avoids keyframe-seek errors. Time is based on file FPS.
        while cap.grab():
            target = round(sample * args.interval * fps)
            if frame_index < target:
                frame_index += 1
                continue
            ok, frame = cap.retrieve()
            if not ok:
                raise RuntimeError(f'{video}: failed decoding frame {frame_index}')
            if (frame.shape[1], frame.shape[0]) != size:
                raise ValueError(f'{video}: frame resolution differs from polygon reference {size}')
            for block_id, crop_box in regions:
                x0, y0, x1, y1 = crop_box
                dest = folder / block_id if block_id else folder
                suffix = f'_{block_id}' if block_id else ''
                path = dest / f'{video.stem}_f{frame_index:09d}{suffix}.jpg'
                image_key = path.relative_to(args.out).as_posix()
                if args.resume and image_key in args.recorded and path.exists():
                    continue
                if path.exists():
                    raise FileExistsError(path)
                crop = frame[y0:y1, x0:x1]
                if not cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f'Cannot save {path}')
                if image_key in args.recorded:
                    continue
                manifest.write(json.dumps(dict(image=image_key,
                    source=str(video.resolve()), camera=camera, block_id=block_id,
                    frame=frame_index, seconds=frame_index/fps,
                    crop_xyxy=crop_box, reference_size=size)) + '\n')
            manifest.flush()
            # One full-frame preview per clip makes drift and crop coverage inspectable.
            if count == 0:
                preview = frame.copy()
                for poly in polygons:
                    cv2.polylines(preview, [poly.astype(np.int32)], True, (0, 255, 0), 2)
                for block_id, (x0, y0, x1, y1) in regions:
                    cv2.rectangle(preview, (x0, y0), (x1-1, y1-1), (0, 0, 255), 2)
                    if block_id:
                        cv2.putText(preview, block_id, (x0, max(15, y0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                preview_path = args.out / 'previews' / f'{video.stem}.jpg'
                if not cv2.imwrite(str(preview_path), preview):
                    raise OSError(preview_path)
            count += 1
            sample += 1
            frame_index += 1
            if args.limit_per_video and count >= args.limit_per_video:
                break
        if count == 0:
            raise RuntimeError(f'{video}: no frames decoded')
        if not args.limit_per_video and total > 0 and frame_index < total - 1:
            raise RuntimeError(f'{video}: stopped early at {frame_index}/{total}; partial output retained')
        print(f'{video.name}: {count} frames -> {count * len(regions)} crops', flush=True)
        return count * len(regions)
    finally:
        cap.release()


def block_box(poly, size, padding):
    low, high = poly.min(axis=0), poly.max(axis=0)
    margin = (high - low) * padding
    start = np.maximum(0, np.floor(low - margin)).astype(int)
    end = np.minimum(size, np.ceil(high + margin + 1)).astype(int)
    return [int(v) for v in (*start, *end)]


def run_jobs(jobs, args, manifest, errors):
    total, failures = 0, 0
    for video, camera, roi in jobs:
        try:
            total += extract(video, camera, roi, args, manifest)
        except (RuntimeError, ValueError) as exc:
            errors.write(json.dumps(dict(video=str(video), error=str(exc))) + '\n')
            errors.flush()
            failures += 1
            print(f'SKIP {video}: {exc}', flush=True)
    return total, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poly-dir', type=Path, default=DEFAULT_ROOT/'Poly')
    parser.add_argument('--videos', type=Path, default=DEFAULT_ROOT/'tmp/yolo_samples')
    parser.add_argument('--out', type=Path, required=True, help='New output directory')
    parser.add_argument('--resume', action='store_true', help='Reuse matching configuration and already recorded images')
    parser.add_argument('--label', default='BarriGuard')
    parser.add_argument('--mode', choices=['union', 'blocks'], default='union',
                        help='union: one crop per frame; blocks: one crop per polygon')
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--padding', type=float, default=0.1, help='Extra margin per side, fraction of ROI width/height')
    parser.add_argument('--camera', nargs='+', default=['1749', '1750'])
    parser.add_argument('--limit-per-video', type=int, default=0, help='Smoke test: maximum sampled frames per video, 0 means all')
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error('--interval must be finite and positive')
    if not math.isfinite(args.padding) or args.padding < 0:
        parser.error('--padding must be finite and nonnegative')
    if args.limit_per_video < 0 or len(set(args.camera)) != len(args.camera):
        parser.error('Invalid limit or repeated camera')
    jobs, names = [], set()
    for camera in args.camera:
        if not camera.isdigit():
            parser.error('Camera IDs must be numeric')
        roi = load_roi(args.poly_dir/f'{camera}.json', args.label, args.padding)
        videos = sorted(p for p in args.videos.rglob(f'{camera}_*') if p.suffix.lower() in VIDEO_EXTENSIONS)
        if not videos:
            raise ValueError(f'No videos for camera {camera}')
        for video in videos:
            if video.stem.casefold() in names:
                raise ValueError(f'Duplicate video filename: {video.name}')
            names.add(video.stem.casefold())
            jobs.append((video, camera, roi))
    metadata = dict(interval=args.interval, padding=args.padding, label=args.label, mode=args.mode,
                    limit_per_video=args.limit_per_video,
                    cameras={c: dict(reference_size=r[0], crop_xyxy=r[1],
                                     blocks=[dict(block_id=f'block_{i:03d}',
                                                  crop_xyxy=block_box(p, r[0], args.padding))
                                             for i, p in enumerate(r[2], 1)],
                                     polygons=[p.tolist() for p in r[2]]) for _, c, r in jobs})
    config_path = args.out/'crop_config.json'
    args.recorded = set()
    if args.resume:
        old = json.loads(config_path.read_text(encoding='utf-8'))
        # Earlier versions did not store this option; their default was zero.
        old.setdefault('limit_per_video', 0)
        if old != json.loads(json.dumps(metadata)):
            raise ValueError('Resume configuration differs; restore original options/poly or use a new output directory')
        with (args.out/'manifest.jsonl').open(encoding='utf-8') as previous:
            args.recorded = {json.loads(line)['image'] for line in previous if line.strip()}
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out/'previews').mkdir()
        config_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    with (args.out/'manifest.jsonl').open('a', encoding='utf-8') as manifest, \
         (args.out/'errors.jsonl').open('a', encoding='utf-8') as errors:
        total, failures = run_jobs(jobs, args, manifest, errors)
    print(f'Done: {total} image positions processed; {failures} failed videos (errors.jsonl). No training labels generated.', flush=True)


if __name__ == '__main__':
    main()
