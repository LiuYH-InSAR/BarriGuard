"""Standalone RTSP recording for local water tests; production never imports it."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS','rtsp_transport;tcp')
import cv2 as cv
import numpy as np
import yaml

DEPLOY = Path(__file__).resolve().parents[1]
WARMUP_SECONDS = 5
MIN_FRAME_STD = 10
MAX_STARTUP_SECONDS = 30


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera',default='1749')
    parser.add_argument('--seconds',type=float,default=60)
    parser.add_argument('--config',type=Path,default=DEPLOY/'config.yaml')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--warmup',type=float,default=WARMUP_SECONDS,help='Pre-read seconds before recording (default 5)')
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be positive')
    if not 0 <= args.warmup < MAX_STARTUP_SECONDS:
        parser.error('--warmup must be in [0,30)')
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    camera = next((c for c in config['cameras'] if str(c['id'])==args.camera),None)
    if camera is None:
        parser.error('Unknown camera')
    out = args.out or DEPLOY.parent/'tmp'/'stream_tests'/f'{args.camera}_{datetime.now():%Y%m%d_%H%M%S}.mp4'
    if out.exists():
        parser.error('Output already exists; choose a new filename')
    out.parent.mkdir(parents=True,exist_ok=True)
    cap,writer = cv.VideoCapture(),None
    times=[]
    started=None
    try:
        if not cap.open(camera['rtsp_url'],cv.CAP_FFMPEG,[cv.CAP_PROP_OPEN_TIMEOUT_MSEC,10000,cv.CAP_PROP_READ_TIMEOUT_MSEC,10000]):
            raise RuntimeError('Cannot open stream')
        fps=cap.get(cv.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps<=0:
            raise RuntimeError('Stream FPS unavailable; cannot write reliable local video')
        connected=time.monotonic()
        while True:
            ok,frame=cap.read()
            now=time.monotonic()
            if not ok:
                raise RuntimeError('Stream interrupted; partial recording retained')
            if started is None:
                if now-connected>=MAX_STARTUP_SECONDS:
                    raise RuntimeError('No usable startup frame within 30 seconds')
                if now-connected<args.warmup or cv.cvtColor(frame,cv.COLOR_BGR2GRAY).std()<MIN_FRAME_STD:
                    continue
                started=now
                size=(frame.shape[1],frame.shape[0])
                writer=cv.VideoWriter(str(out),cv.VideoWriter_fourcc(*'mp4v'),fps,size)
                if not writer.isOpened():
                    raise RuntimeError('Cannot write MP4')
                cv.imwrite(str(out.with_suffix('.png')),frame)
                print(f'Recording {args.seconds:g}s -> {out.resolve()}',flush=True)
            if now-started>=args.seconds:
                break
            if (frame.shape[1],frame.shape[0])!=size:
                raise RuntimeError('Stream dimensions changed')
            writer.write(frame)
            times.append(now-started)
        print(f'Done: {len(times)} frames; FPS={fps:g}; video={out.resolve()}',flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if times:
            out.with_suffix('.json').write_text(json.dumps(dict(camera=args.camera,declared_fps=fps,
                frames=len(times),receive_times_s=times,requested_seconds=args.seconds,
                timing='MP4 uses declared constant FPS; receive timestamps retained for checking dropped frames'),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
