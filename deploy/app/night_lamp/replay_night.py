"""Recompute one recorded night in an isolated directory; never publish."""
import argparse
from datetime import date, datetime, time, timedelta
import hashlib
import json
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from night_lamp.local_recording import _Recorder
from night_lamp.session import NightSession
from server.config import load_runtime
from server.output import ReportStore


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def night_bounds(day, schedule):
    tz = ZoneInfo(schedule.timezone)
    start = datetime.combine(day - timedelta(days=1), time.fromisoformat(schedule.night_start), tz)
    end_day = day if schedule.night_end <= schedule.night_start else day - timedelta(days=1)
    end = datetime.combine(end_day, time.fromisoformat(schedule.night_end), tz)
    morning = datetime.combine(day, time.fromisoformat(schedule.report_at), tz)
    return start, end, morning


def recordings(folder, start, end, cadence):
    """Use filename wall times; reject ambiguous buckets rather than guess."""
    buckets = {}
    for video in sorted(Path(folder).glob("*.mp4")):
        try:
            wall = datetime.strptime(video.stem[:15], "%Y%m%d_%H%M%S").replace(tzinfo=start.tzinfo)
        except ValueError:
            continue
        if not start <= wall < end:
            continue
        bucket = int(wall.timestamp()) // cadence
        if bucket in buckets:
            raise ValueError(f"Multiple recordings in one bucket: {buckets[bucket][1]} and {video}")
        stamps = video.with_suffix(".jsonl")
        if not stamps.is_file():
            raise ValueError(f"Missing capture timestamps: {stamps}")
        values = [float(line) for line in stamps.read_text(encoding="utf-8").splitlines() if line.strip()]
        import math
        if not values or not all(math.isfinite(t) for t in values) or any(b <= a for a, b in zip(values, values[1:])):
            raise ValueError(f"Empty or invalid capture timestamps: {stamps}")
        buckets[bucket] = (wall, video)
    return [buckets[key] for key in sorted(buckets)]


def recompute(cfg, day, cameras, out=None, input_root=None):
    selected = []
    for camera in cameras:
        cam = next((c for c in cfg.cameras if c.id == camera), None)
        if cam is None or "night_lamp" not in cam.algorithms:
            raise ValueError(f"No night_lamp configuration for camera {camera}")
        spec = cam.algorithms["night_lamp"].spec
        if not (spec.overnight.enabled and spec.overnight.record_local):
            raise ValueError(f"Camera {camera} is not configured for local hourly recordings")
        start, end, morning = night_bounds(day, cfg.schedule)
        root = Path(input_root or spec.overnight.record_root)
        clips = recordings(root / "videos" / camera, start, end,
                           max(spec.overnight.cadence_minutes, 1) * 60)
        if not clips:
            raise ValueError(f"No recordings for {camera}: {start.isoformat()} to {end.isoformat()}")
        selected.append((camera, spec, clips, morning))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    root = Path(input_root or selected[0][1].overnight.record_root)
    out = Path(out) if out else root / "results" / "replays" / str(day) / run_id
    # An existing output is never reused, including a failed or incomplete run.
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"version": 1, "run_id": run_id, "report_date": str(day),
                "complete": False, "cameras": {}, "finals": []}
    manifest_path = out / "run.json"
    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    save_manifest()
    store = ReportStore(out)
    for camera, spec, clips, morning in selected:
        session = NightSession(camera, spec)
        recorder = _Recorder(session, spec.overnight, max(spec.sampling.interval_ms, 1) / 1000.0)
        info = {"videos": [], "hourly_reports": []}
        manifest["cameras"][camera] = info
        try:
            for wall, video in clips:
                print(f"[{camera}] Replaying {video}", flush=True)
                session.begin_burst(wall)
                recorder.path = video
                report = recorder.snapshot(wall + timedelta(seconds=spec.overnight.burst_seconds), force=True)
                info["videos"].append(str(video))
                if report is not None:
                    info["hourly_reports"].append(str(Path(store.save(report)).relative_to(out)))
                save_manifest()
            session.freeze()
            report = session.finalize(None, morning)
            if report is None or not report.image_jpeg or report.status in ("failed", "rejected"):
                raise RuntimeError(f"Camera {camera}: final result unavailable; nothing will be publishable")
            report.metadata["manual_replay"] = {"run_id": run_id, "report_date": str(day),
                                                 "recordings": len(clips)}
            path = Path(store.save(report))
            payload = json.loads(path.read_text(encoding="utf-8"))
            image = path.parent / payload["image"]
            manifest["finals"].append({"camera": camera, "report": str(path.relative_to(out)),
                                       "sha256": digest(path), "image_sha256": digest(image)})
            save_manifest()
        finally:
            recorder.close()
            session.release()
    manifest["complete"] = True
    save_manifest()
    print(f"REPLAY_DIR={out.resolve()}", flush=True)
    print("Review the final JPGs before running night_lamp.publish_night. No results were sent.")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, required=True, help="Morning date, e.g. 2026-10-23")
    parser.add_argument("--camera", nargs="+", choices=("1749", "1750"), default=["1749", "1750"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input-root", type=Path, help="Override night_offline root (contains videos/)")
    parser.add_argument("--out", type=Path, help="New, unused output directory")
    args = parser.parse_args()
    try:
        recompute(load_runtime(args.config), args.date, list(dict.fromkeys(args.camera)), args.out, args.input_root)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Replay failed: {exc}\n")


if __name__ == "__main__":
    main()
