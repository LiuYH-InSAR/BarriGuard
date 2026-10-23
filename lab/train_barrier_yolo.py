"""Hypothesis: COCO-pretrained YOLO12n can be fine-tuned to detect barrier blocks.

Train uses train/val only. Run --mode test separately after model selection.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path('C:/Users/admin/Desktop/projects/BarriGuard_Yolo')


def check_data(path):
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    root = Path(data.get('path', path.parent))
    if not root.is_absolute():
        root = path.parent/root
    names = data['names']
    if names not in ({0: 'BarriGuard'}, ['BarriGuard']):
        raise ValueError('Expected one class: 0 = BarriGuard')
    for split in ('train', 'val', 'test'):
        images = root/data[split]
        files = sorted(images.glob('*.jpg'))
        if not files:
            raise ValueError(f'No JPEG images in {images}')
        boxes, negatives = 0, 0
        for image in files:
            im = cv2.imread(str(image))
            if im is None:
                raise ValueError(f'Cannot decode {image}')
            label = root/'labels'/split/f'{image.stem}.txt'
            lines = label.read_text(encoding='utf-8-sig').splitlines()
            nonempty = [line for line in lines if line.strip()]
            negatives += not nonempty
            for line in nonempty:
                parts = line.split()
                if len(parts) != 5 or parts[0] != '0':
                    raise ValueError(f'Invalid YOLO label: {label}')
                values = np.asarray([float(v) for v in parts[1:]])
                if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any() or (values[2:] <= 0).any():
                    raise ValueError(f'Invalid normalized coordinates: {label}')
                center, size = values[:2], values[2:]
                if (center-size/2 < -1e-6).any() or (center+size/2 > 1+1e-6).any():
                    raise ValueError(f'Box extends outside image: {label}')
                boxes += 1
        print(f'{split}: {len(files)} images, {boxes} boxes, {negatives} empty-label images', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'tmp/yolo_dataset_3750/data.yaml')
    parser.add_argument('--weights', type=Path, default=ROOT/'models/yolo12n.pt')
    parser.add_argument('--project', type=Path, default=ROOT/'runs')
    parser.add_argument('--name', default='barrier_yolo12n_v1')
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--resume', type=Path, help='Interrupted run weights/last.pt')
    parser.add_argument('--check-only', action='store_true', help='Validate local data and GPU; do not download or train')
    args = parser.parse_args()
    if min(args.epochs, args.batch, args.imgsz) <= 0 or args.workers < 0:
        parser.error('Invalid training settings')
    args.data = args.data.resolve()
    check_data(args.data)

    import torch
    import ultralytics
    from ultralytics import YOLO

    print(f'torch={torch.__version__}; ultralytics={ultralytics.__version__}', flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use the BarriGuard .venv Python interpreter')
    # Execute a small kernel to detect unsupported GPU builds before training.
    probe = torch.ones((16,16), device='cuda')
    (probe @ probe).sum().item()
    print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    if args.check_only:
        print('Checks passed. No weights downloaded; no training started.', flush=True)
        return
    if args.resume:
        if args.mode != 'train' or not args.resume.is_file():
            parser.error('--resume requires train mode and an existing last.pt')
        model = YOLO(str(args.resume))
        model.train(resume=True, device=0, workers=args.workers, batch=args.batch)
        return
    if args.mode == 'test':
        if not args.weights.is_file():
            parser.error('Test requires an existing trained best.pt via --weights')
        YOLO(str(args.weights)).val(data=str(args.data), split='test', device=0,
            imgsz=args.imgsz, batch=args.batch, workers=args.workers,
            project=str(args.project), name=args.name+'_test', plots=True)
        return
    if args.weights.suffix != '.pt':
        parser.error('Transfer learning requires pretrained .pt weights, not a model YAML')
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    model.train(data=str(args.data), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=0, workers=args.workers,
        project=str(args.project), name=args.name, exist_ok=False,
        optimizer='AdamW', lr0=0.001, lrf=0.01, weight_decay=0.0005,
        freeze=0, patience=15, seed=42, deterministic=True,
        mosaic=0.0, mixup=0.0, degrees=0.0, flipud=0.0, fliplr=0.5,
        translate=0.05, scale=0.2, cache=False, amp=True, plots=True)
    print(f'Best weights: {Path(model.trainer.save_dir)/"weights/best.pt"}', flush=True)


if __name__ == '__main__':
    main()
