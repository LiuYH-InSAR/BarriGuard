"""Hypothesis: fixed-camera polygons provide useful, manually correctable box labels.

prepare creates an X-AnyLabeling review folder; export uses the edited JSONs.
No generated label is treated as human-verified.
"""
import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('C:/Users/admin/Desktop/projects/BarriGuard_Yolo')
SPLITS = ('train', 'val', 'test')


def session(row):
    parts = Path(row['source']).stem.split('_')
    return '_'.join(parts[1:3])[:11]


def assign_splits(rows, ratios, seed):
    counts = Counter(session(r) for r in rows)
    if len(counts) < 3:
        raise ValueError('At least three recording-hour groups required')
    total = sum(counts.values())
    targets = {s: total*r for s, r in zip(SPLITS, ratios)}
    assigned = dict.fromkeys(SPLITS, 0)
    result = {}
    keys = sorted(counts, key=lambda k: (-counts[k], hashlib.sha256(f'{seed}:{k}'.encode()).hexdigest()))
    for key in keys:
        split = max(SPLITS, key=lambda s: targets[s]-assigned[s])
        result[key] = split
        assigned[split] += counts[key]
    if any(v == 0 for v in assigned.values()):
        raise ValueError('Recording-hour groups too imbalanced for all three splits')
    return result


def select_rows(rows, count, preferred):
    if count > len(rows):
        raise ValueError(f'Requested {count} images, only {len(rows)} available in split')
    groups = {}
    for row in rows:
        groups.setdefault((row['camera'], row['block_id']), []).append(row)
    sequences = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda r: (r['source'], r['frame']))
        prior = [r for r in group if Path(r['image']).stem in preferred]
        others = [r for r in group if Path(r['image']).stem not in preferred]
        # Center each equal time interval, rather than always choosing endpoints.
        n = min(len(others), math.ceil(count/max(1, len(groups))))
        indices = [int((i+.5)*len(others)/n) for i in range(n)] if n else []
        chosen = prior + [others[i] for i in indices]
        used = {r['image'] for r in chosen}
        sequences.append(chosen + [r for r in others if r['image'] not in used])
    selected = []
    for i in range(max(map(len, sequences), default=0)):
        for seq in sequences:
            if i < len(seq):
                selected.append(seq[i])
            if len(selected) == count:
                return selected
    return selected


def clip_poly(points, box):
    points = [list(p) for p in points]
    for axis, edge, sign in [(0, box[0], 1), (0, box[2], -1),
                             (1, box[1], 1), (1, box[3], -1)]:
        result = []
        for a, b in zip(points, points[1:] + points[:1]):
            inside_a = (a[axis] - edge)*sign >= 0
            inside_b = (b[axis] - edge)*sign >= 0
            if inside_a:
                result.append(a)
            if inside_a != inside_b:
                fraction = (edge-a[axis])/(b[axis]-a[axis])
                result.append([a[k]+fraction*(b[k]-a[k]) for k in (0, 1)])
        points = result
        if not points:
            break
    return np.asarray(points, dtype=float)


def prepare(args):
    config = json.loads((args.crops/'crop_config.json').read_text(encoding='utf-8'))
    rows = [json.loads(s) for s in (args.crops/'manifest.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    groups = {}
    for row in rows:
        if (args.crops/row['image']).is_file():
            groups.setdefault((row['camera'], row['block_id']), []).append(row)
    surviving = [r for group in groups.values() for r in group]
    preferred = set()
    if args.reuse_review:
        preferred = {p.stem for p in args.reuse_review.glob('*.jpg') if p.with_suffix('.json').is_file()}
    if args.train_count:
        quotas = dict(train=args.train_count, val=args.val_count, test=args.test_count)
        total = sum(quotas.values())
        splits = assign_splits(surviving, [quotas[s]/total for s in SPLITS], args.seed)
        ordered = []
        for split in SPLITS:
            pool = [r for r in surviving if splits[session(r)] == split]
            chosen = select_rows(pool, quotas[split], preferred)
            ordered.extend(dict(r, split=split) for r in chosen)
    else:
        ordered = select_rows(surviving, min(args.limit or len(surviving), len(surviving)), preferred)
    if not ordered:
        raise ValueError('No surviving crops')
    args.out.mkdir(parents=True, exist_ok=False)
    provenance = {}
    reused = 0
    for row in ordered:
        src = args.crops/row['image']
        image = cv2.imread(str(src))
        if image is None:
            raise ValueError(f'Cannot read {src}')
        height, width = image.shape[:2]
        x0, y0, x1, y1 = row['crop_xyxy']
        if (height, width) != (y1-y0, x1-x0):
            raise ValueError(f'Crop dimensions mismatch: {src}')
        shapes = []
        for i, poly in enumerate(config['cameras'][row['camera']]['polygons'], 1):
            clipped = clip_poly(poly, [x0, y0, x1, y1])
            if len(clipped) < 3 or abs(cv2.contourArea(clipped.astype(np.float32))) < 1:
                continue
            low = clipped.min(axis=0) - [x0, y0]
            high = clipped.max(axis=0) - [x0, y0]
            shapes.append(dict(label='BarriGuard', shape_type='rectangle',
                points=[low.tolist(), high.tolist()], group_id=None, flags={},
                description=f'Projected from reference block_{i:03d}; review required', difficult=False))
        dest = args.out/src.name
        if dest.exists():
            raise ValueError(f'Duplicate filename {dest}')
        shutil.copy2(src, dest)
        annotation = dict(version='2.5.4', flags={}, shapes=shapes,
            imagePath=dest.name, imageData=None, imageHeight=height, imageWidth=width)
        previous = args.reuse_review/dest.with_suffix('.json').name if args.reuse_review else None
        if previous and previous.is_file():
            old_image = args.reuse_review/dest.name
            if not old_image.is_file() or hashlib.sha256(old_image.read_bytes()).digest() != hashlib.sha256(src.read_bytes()).digest():
                raise ValueError(f'Cannot reuse annotation with different image: {previous}')
            annotation = json.loads(previous.read_text(encoding='utf-8'))
            if (annotation['imageWidth'], annotation['imageHeight']) != (width, height):
                raise ValueError(f'Reused annotation size mismatch: {previous}')
            annotation['imagePath'] = dest.name
            annotation['imageData'] = None
            reused += 1
        dest.with_suffix('.json').write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding='utf-8')
        provenance[dest.stem] = row
    (args.out/'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    summary = dict(images=len(provenance), reused_annotations=reused,
                   splits=dict(Counter(r.get('split','unassigned') for r in provenance.values())),
                   seed=args.seed, note='Generated boxes require review; reused labels retain your edits')
    (args.out/'preparation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Prepared {summary} in {args.out}', flush=True)


def export(args):
    provenance = json.loads((args.review/'provenance.json').read_text(encoding='utf-8'))
    active = {k:v for k,v in provenance.items() if (args.review/f'{k}.jpg').is_file()}
    if active and all(r.get('split') in SPLITS for r in active.values()):
        splits = {}
        for row in active.values():
            key = session(row)
            if key in splits and splits[key] != row['split']:
                raise ValueError('Recording-hour group assigned to multiple splits')
            splits[key] = row['split']
    else:
        splits = assign_splits(list(active.values()), [.8,.1,.1], args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    counts = dict(train=0, val=0, test=0)
    audit = []
    for stem, row in active.items():
        data = json.loads((args.review/f'{stem}.json').read_text(encoding='utf-8'))
        width, height = data['imageWidth'], data['imageHeight']
        lines = []
        for shape in data['shapes']:
            if shape['label'] != 'BarriGuard' or shape['shape_type'] not in {'rectangle', 'polygon'}:
                raise ValueError(f'Unexpected label/shape in {stem}')
            points = np.asarray(shape['points'], dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                raise ValueError(f'Invalid points in {stem}')
            low, high = points.min(axis=0), points.max(axis=0)
            if (low < 0).any() or (high > [width, height]).any() or (high <= low).any():
                raise ValueError(f'Box outside image or empty in {stem}; correct in editor')
            center = (low+high)/2/[width, height]
            size = (high-low)/[width, height]
            lines.append('0 '+' '.join(f'{v:.8f}' for v in (*center, *size)))
        split = splits[session(row)]
        images, labels = args.out/'images'/split, args.out/'labels'/split
        images.mkdir(parents=True, exist_ok=True); labels.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.review/f'{stem}.jpg', images/f'{stem}.jpg')
        (labels/f'{stem}.txt').write_text('\n'.join(lines)+ ('\n' if lines else ''), encoding='utf-8')
        counts[split] += 1
        audit.append(dict(image=stem, split=split, session=session(row), source=row['source']))
    (args.out/'data.yaml').write_text(f'path: {json.dumps(args.out.resolve().as_posix())}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: BarriGuard\n', encoding='utf-8')
    (args.out/'split_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(f'Exported {counts}; data.yaml: {args.out/"data.yaml"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--crops', type=Path, default=ROOT/'tmp/blocks_30s')
    prep.add_argument('--out', type=Path, required=True)
    prep.add_argument('--limit', type=int, default=200, help='0 = all surviving images')
    prep.add_argument('--train-count', type=int, default=0, help='Select exactly this many train samples, plus val/test quotas; overrides --limit')
    prep.add_argument('--val-count', type=int, default=375)
    prep.add_argument('--test-count', type=int, default=375)
    prep.add_argument('--reuse-review', type=Path, help='Reuse matching edited JSONs when their images are selected')
    prep.add_argument('--seed', type=int, default=42)
    exp = commands.add_parser('export')
    exp.add_argument('--review', type=Path, required=True)
    exp.add_argument('--out', type=Path, required=True)
    exp.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.limit < 0 or args.train_count < 0 or args.val_count <= 0 or args.test_count <= 0:
            parser.error('Invalid sample counts')
        prepare(args)
    else:
        export(args)


if __name__ == '__main__':
    main()
