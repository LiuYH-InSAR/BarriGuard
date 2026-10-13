"""Capture one original RTSP frame per camera for manual polygon labels."""
import argparse
from datetime import datetime
import os
import time
from pathlib import Path

# Set before importing OpenCV; this affects only this standalone process.
os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'rtsp_transport;tcp')
import cv2 as cv
import yaml

DEPLOY = Path(__file__).resolve().parents[1]
TIMEOUT_MS = 10000
WARMUP_SECONDS = 5
MIN_FRAME_STD = 10
MAX_STARTUP_SECONDS = 30


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEPLOY/'config.yaml')
    parser.add_argument('--out', type=Path, default=DEPLOY.parent/'tmp'/'roi_relabel')
    parser.add_argument('--warmup',type=float,default=WARMUP_SECONDS,help='Pre-read seconds before saving (default 5)')
    parser.add_argument('--camera', nargs='+', default=['1749', '1750'])
    args = parser.parse_args()
    if not 0 <= args.warmup < MAX_STARTUP_SECONDS:
        parser.error('--warmup must be in [0,30)')
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    cameras = {str(c['id']): c for c in config['cameras']}
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    failures = []
    for camera in args.camera:
        if camera not in cameras:
            print(f'{camera}: missing camera configuration', flush=True)
            failures.append(camera)
            continue
        print(f'{camera}: opening stream...', flush=True)
        cap = cv.VideoCapture()
        try:
            opened = cap.open(cameras[camera]['rtsp_url'], cv.CAP_FFMPEG,
                [cv.CAP_PROP_OPEN_TIMEOUT_MSEC, TIMEOUT_MS,
                 cv.CAP_PROP_READ_TIMEOUT_MSEC, TIMEOUT_MS])
            ok,frame = False,None
            started = time.monotonic()
            while opened and time.monotonic()-started < MAX_STARTUP_SECONDS:
                ok,frame = cap.read()
                if not ok:
                    break
                elapsed = time.monotonic()-started
                if elapsed>=args.warmup and cv.cvtColor(frame,cv.COLOR_BGR2GRAY).std()>=MIN_FRAME_STD:
                    break
                ok = False
            if not ok:
                print(f'{camera}: failed to read frame', flush=True)
                failures.append(camera)
                continue
            path = args.out/f'{camera}_label_{stamp}.png'
            if not cv.imwrite(str(path), frame):
                raise OSError(f'Cannot save {path}')
            print(f'{camera}: {frame.shape[1]}x{frame.shape[0]} -> {path.resolve()}', flush=True)
        finally:
            cap.release()
    if failures:
        raise SystemExit('Failed cameras: '+', '.join(failures))


if __name__ == '__main__':
    main()
