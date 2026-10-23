"""Local diagnostic replay of historical clips or timestamped server recordings."""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy/app"))

from server.config import load_runtime
from server.output import ReportStore
from night_lamp.session import NightSession
from night_lamp.local_recording import _Recorder
from night_intermediates import save_intermediates, write_plot_index

TZ = timezone(timedelta(hours=8))


def historical(session, video, start, origin, seconds, interval):
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps > 0:
            raise RuntimeError(f"Invalid FPS: {video}")
        index, next_sample = 0, 0.0
        while index / fps < seconds:
            time_s = index / fps
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            if time_s + 1e-9 < next_sample:
                continue
            session.accumulate(frame, (start-origin).total_seconds()+time_s)
            next_sample += interval
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("historical", "server"), required=True)
    parser.add_argument("--camera", choices=("1749", "1750"), default="1750")
    parser.add_argument("--periodicity", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "deploy/config_blocks.yaml")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--diagnostics", action="store_true", help="Export matrices, plots and plot index (default off)")
    args = parser.parse_args()
    cfg = load_runtime(args.config)
    spec = next(c for c in cfg.cameras if c.id == args.camera).algorithms["night_lamp"].spec
    if args.periodicity is not None:
        spec.periodicity.enabled = args.periodicity
    spec.overnight.burst_seconds = 300
    interval = spec.sampling.interval_ms / 1000.0
    old = args.group == "historical"
    begin = datetime(2026,9,29,19,tzinfo=TZ) if old else datetime(2026,10,14,20,tzinfo=TZ)
    end = datetime(2026,9,30,6,tzinfo=TZ) if old else datetime(2026,10,15,3,tzinfo=TZ)
    folder = ROOT / ("tmp/hourly10" if old else f"tmp/server_downloads/night_offline/videos/{args.camera}")
    mode = "period_on" if spec.periodicity.enabled else "period_off"
    out = args.out or ROOT / "tmp/lamp_poly_test" / f"{args.camera}_{args.group}_{mode}"
    out.mkdir(parents=True, exist_ok=True)
    store = ReportStore(out, persist_images=True)
    session = NightSession(args.camera, spec)
    recorder = _Recorder(session, spec.overnight, interval)
    processed = 0
    try:
        start = begin
        while start < end:
            channel = "1002490" if args.camera == "1749" else "1002491"
            matches = ([folder / f"{channel}_{start:%Y%m%d_%H%M}.mp4"] if old else
                       sorted(folder.glob(f"{start:%Y%m%d_%H}*.mp4")))
            matches = [p for p in matches if p.exists()]
            if len(matches) > 1:
                raise RuntimeError(f"Multiple recordings for {start}; select one before replay")
            if not matches:
                print(f"Missing: {start.isoformat()}", flush=True)
                if session._lamp_checks:
                    session._lamp_checks.unknown(start.isoformat())
                start += timedelta(hours=1)
                continue
            video = matches[0]
            print(f"Processing {video.name}; periodicity={spec.periodicity.enabled}", flush=True)
            session.begin_burst(start)
            if old:
                historical(session, video, start, begin, 300, interval)
                report = session.snapshot(start+timedelta(minutes=5), force=True)
            else:
                if not video.with_suffix(".jsonl").exists():
                    raise RuntimeError(f"Missing timestamps: {video}")
                recorder.path = video
                report = recorder.snapshot(start+timedelta(minutes=5), force=True)
            if report is not None:
                store.save(report)
                if args.diagnostics:
                    save_intermediates(session, report, out / start.strftime("%Y%m%d_%H%M") / "intermediate")
                    write_plot_index(out)
            processed += 1
            start += timedelta(hours=1)
        if not processed:
            raise RuntimeError("No input videos found")
        session.freeze()
        morning = end.replace(hour=7, minute=0)
        report = session.finalize(None, morning)
        if report is not None:
            store.save(report)
            if report.image_jpeg:
                (out / "final_0700.jpg").write_bytes(report.image_jpeg)
        if args.diagnostics:
            write_plot_index(out)
        print(f"Results: {out}", flush=True)
    finally:
        recorder.close()
        session.release()


if __name__ == "__main__":
    main()
