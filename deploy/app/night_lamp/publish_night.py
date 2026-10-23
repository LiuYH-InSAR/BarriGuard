"""Publish only the reviewed final reports from a completed manual replay."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.request

from night_lamp.replay_night import digest
from server.algo import Report
from server.config import load_runtime
from server.reporting import Reporter


def inside(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Result path escapes replay directory: {name}")
    return path


def publish(cfg, folder, cameras=None, resend=False):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or not manifest.get("complete"):
        raise ValueError("Replay is incomplete; refusing to publish")
    cb = cfg.callback
    if not cb.enabled or not cb.url.strip():
        raise ValueError("Configured callback is disabled or empty")
    finals = manifest["finals"]
    if cameras:
        if set(cameras) - {item["camera"] for item in finals}:
            raise ValueError("Selected camera has no final report in this replay")
        finals = [item for item in finals if item["camera"] in cameras]
    if not finals:
        raise ValueError("No final reports to publish")
    # Reuse the exact production payload serializer, with its background sender disabled.
    serializer = Reporter(replace(cfg, callback=replace(cb, enabled=False)))
    prepared = []
    try:
        for item in finals:
            path = inside(folder, item["report"])
            data = json.loads(path.read_text(encoding="utf-8"))
            image = inside(folder, str(path.parent.relative_to(folder) / data["image"]))
            if digest(path) != item["sha256"] or digest(image) != item["image_sha256"]:
                raise ValueError(f"Result changed after replay: {path}")
            if data["camera"] != item["camera"] or data["report_type"] != "night_heatmap" or data["status"] in ("failed", "rejected"):
                raise ValueError(f"Invalid final report: {path}")
            report = Report(camera=data["camera"], algorithm=data["algorithm"],
                            report_type=data["report_type"], created_at=data["created_at"],
                            image_jpeg=image.read_bytes(), metadata=data["metadata"], status=data["status"])
            prepared.append((item["camera"], serializer._body("report", report, None)))
    finally:
        serializer.close()
    # A separate process must not send this run concurrently.
    lock = folder / ".publish.lock"
    lock.open("x", encoding="utf-8").close()
    try:
        for camera, body in prepared:
            receipt = folder / f"published_{camera}.json"
            if receipt.exists() and not resend:
                print(f"[{camera}] Already published; skipped. Use --resend only to send again.")
                continue
            request = urllib.request.Request(cb.url, data=body, headers={"Content-Type": "application/json"})
            for attempt in range(max(int(cb.retries), 0) + 1):
                try:
                    with urllib.request.urlopen(request, timeout=cb.timeout_s) as response:
                        status = response.status
                        if not 200 <= status < 300:
                            raise RuntimeError(f"Callback returned HTTP {status}")
                    break
                except Exception:
                    if attempt >= max(int(cb.retries), 0):
                        raise
                    time.sleep(1)
            receipt.write_text(json.dumps({"camera": camera, "run_id": manifest["run_id"],
                "sent_at": datetime.now(timezone.utc).isoformat(), "url": cb.url,
                "http_status": status}, indent=2), encoding="utf-8")
            print(f"[{camera}] Published final report: HTTP {status}", flush=True)
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--camera", nargs="+", choices=("1749", "1750"))
    parser.add_argument("--resend", action="store_true", help="Explicitly resend an already published result")
    args = parser.parse_args()
    try:
        publish(load_runtime(args.config), args.run_dir, args.camera, args.resend)
    except Exception as exc:
        parser.exit(1, f"Publish failed: {exc}\n")


if __name__ == "__main__":
    main()
