"""Replay each sample independently; isolate memory and save a batch summary."""
import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CAMERAS = {"1002490": "1749", "1002491": "1750"}
NAME = re.compile(r"(100249[01])_(\d{8})_(\d{4})")
COLLECTION = "final_images"
PLOT_COLLECTION = "gray_plots"


def collect_images(out):
    """Copy existing final images with video names so none collide."""
    collection = out / COLLECTION
    collection.mkdir(parents=True, exist_ok=True)
    count = 0
    for image in sorted(out.glob("*/final_night.jpg")):
        shutil.copy2(image, collection / (image.parent.name + ".jpg"))
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO / "tmp/hourly10")
    parser.add_argument("--out", type=Path, default=REPO / "output/night_batch")
    parser.add_argument("--limit", type=int, default=5,
                        help="Number of videos in filename order (default: 5; 0: all)")
    parser.add_argument("--start", type=int, default=1,
                        help="First video in filename order, one-based (default: 1)")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    if args.start < 1 or args.limit < 0:
        parser.error("--start must be >= 1 and --limit must be >= 0")
    out = args.out.resolve()
    if args.collect_only:
        print(f"Collected {collect_images(out)} images: {out / COLLECTION}")
        return
    videos = sorted(args.input.resolve().glob("*.mp4"))
    videos = videos[args.start - 1:]
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        parser.error("No MP4 videos found")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, video in enumerate(videos, 1):
        match = NAME.fullmatch(video.stem)
        if not match:
            raise ValueError(f"Unrecognized camera/date in filename: {video.name}")
        channel, date, time = match.groups()
        start = datetime.strptime(date + time, "%Y%m%d%H%M").isoformat() + "+08:00"
        camera = CAMERAS[channel]
        print(f"[{index}/{len(videos)}] {video.name} camera={camera}", flush=True)
        folder = out / video.stem
        folder.mkdir(exist_ok=True)
        command = [sys.executable, str(REPO / "deploy/tools/run_night_clip.py"),
                   "--video", str(video), "--camera", camera,
                   "--continuous", "--scheme", video.stem, "--start", start,
                   "--out", str(out), "--max-frames", str(args.max_frames)]
        with (folder / "run.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=REPO, stdout=log,
                                    stderr=subprocess.STDOUT)
        row = {"video": video.name, "camera": camera,
               "exit_code": result.returncode, "status": "process_failed",
               "n_flash": "", "n_cand": "", "samples": "",
               "image": "", "log": str(folder / "run.log")}
        metadata = folder / "meta.json"
        if result.returncode == 0 and metadata.exists():
            meta = json.loads(metadata.read_text(encoding="utf-8"))
            final = meta.get("final") or {}
            details = final.get("metadata") or {}
            row.update(status=final.get("status", "unknown"),
                       n_flash=details.get("n_flash", ""),
                       n_cand=details.get("n_cand", ""),
                       samples=meta.get("samples", ""))
            image = folder / "final_night.jpg"
            if image.exists():
                row["image"] = str(image)
                collection = out / COLLECTION
                collection.mkdir(exist_ok=True)
                shutil.copy2(image, collection / (video.stem + ".jpg"))
            plots = out / PLOT_COLLECTION
            duty_maps = out / 'duty_maps'
            duty_maps.mkdir(exist_ok=True)
            for name in ('duty_map.jpg', 'duty_candidate_mask.jpg'):
                if (folder / name).exists():
                    shutil.copy2(folder / name, duty_maps / f'{video.stem}_{name}')
            plots.mkdir(exist_ok=True)
            for plot in sorted((folder / "selected_gray_plots").glob("*.svg")):
                shutil.copy2(plot, plots / f"{video.stem}_{plot.name}")
        rows.append(row)
        with (out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  status={row['status']} n_flash={row['n_flash']}", flush=True)
    print(f"Results: {out}", flush=True)


if __name__ == "__main__":
    main()
