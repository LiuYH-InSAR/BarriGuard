"""Optional local-video replay. Production service never imports this module."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import cv2 as cv
import numpy as np

DEPLOY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY / "app"))



# Local-video block: remove this file to remove the replay entry point.
def replay(args):
    from server import config, registry, render, worker
    from water_result import RunSummary
    path = Path(args.video).resolve()
    if not path.is_file():
        raise ValueError(f"Video not found: {path}")
    cap = cv.VideoCapture(str(path))
    writer = None
    recorder = None
    count = 0
    summary = RunSummary()
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot decode video: {path}")
        fps = args.fps if args.fps is not None else cap.get(cv.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid video FPS; specify --fps")
        cfg = config.load_runtime(args.config)
        cam = next((c for c in cfg.cameras if c.id == args.camera), None)
        if cam is None or "water_gap" not in cam.algorithms:
            raise ValueError(f"Camera has no water_gap configuration: {args.camera}")
        spec = cam.algorithms["water_gap"].spec
        calib = worker._load_calibration("water_gap", spec)
        # Optional block preview: remove this block and its CLI options to detach it.
        block_spec = None
        if args.block_windows:
            from water_blocks import BlockSpec
            block_spec = BlockSpec(args.block_score_min, args.block_width_ratio, args.block_valid_ratio, args.block_white_score_min)
            block_spec.validate()
        out = Path(args.out) if args.out else DEPLOY.parent / "output" / "water_clip" / (
            f"{path.stem}_{args.camera}_{datetime.now():%Y%m%d_%H%M%S}")
        if out.exists() and any(out.iterdir()):
            raise ValueError(f"Output must be empty (existing runs are preserved): {out}")
        out.mkdir(parents=True, exist_ok=True)
        algo = None
        duration = "full video" if args.stop_after is None else f"first {args.stop_after:g}s"
        print(f"Replay: {path}; FPS={fps:g}; {duration}; time=frame_index/FPS; output={out.resolve()}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = count / fps
            if args.stop_after is not None and now > args.stop_after:
                break
            if algo is None:
                if min(frame.shape[:2]) < 32:
                    raise ValueError("Water quality gate requires video dimensions >=32")
                algo = registry.create("water_gap", frame.shape, calib, cam.id)
                # Diagnostic block: no import or trace allocation unless explicitly enabled.
                if args.diagnostics:
                    from water_intermediates import Recorder
                    recorder = Recorder(out, path, fps, frame.shape, calib, spec,
                                        args.page_every, args.at_s, args.max_pages)
                if args.overlay_video:
                    writer = cv.VideoWriter(str(out / "overlay.mp4"),
                                            cv.VideoWriter_fourcc(*"mp4v"), fps,
                                            (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise ValueError("Cannot create overlay.mp4")
                shape = frame.shape
            if frame.shape != shape:
                raise ValueError("Video dimensions changed; calibration requires a fixed frame size")
            trace = {} if recorder is not None else None
            result = algo.step(frame, now, trace=trace)
            summary.add(frame, count, now, algo)
            if recorder is not None:
                recorder.add(count, now, frame, result, trace, algo)
            if writer is not None:
                vis = render.draw_annots(render.overlay(frame, algo.fg), result.annots)
                writer.write(vis)
            count += 1
            if count % max(1, int(fps * 10)) == 0:
                print(f"Processed {count} frames; video t={now:.2f}s", flush=True)
        if not count:
            raise ValueError("Video contains no decodable frames")
        summary.finish(out, path, args.camera, fps, calib['rows'], block_spec,
                       calib['detect']['trim'])
        if recorder is not None:
            recorder.finish()
        else:
            import json
            (out / "run.json").write_text(json.dumps(dict(
                video=str(path), camera=args.camera, fps=fps, frames=count,
                diagnostics=False, time_source="frame_index/FPS"), indent=2), encoding="utf-8")
        print(f"Done: {count} frames -> {out.resolve()}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if recorder is not None:
            recorder.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-video", action="store_true", help="Explicitly enable local replay")
    parser.add_argument("--video", help="Local MP4/AVI; camera calibration must match this view")
    parser.add_argument("--camera", default="1749")
    parser.add_argument("--config", default=str(DEPLOY / "config.yaml"))
    parser.add_argument("--out")
    parser.add_argument("--fps", type=float, help="Override source FPS for replay timestamps")
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--stop-after", type=float, default=60., help="Stop at this video time (default: 60s)")
    duration.add_argument("--full-video", action="store_true", help="Explicitly process the whole video")
    parser.add_argument("--overlay-video", action="store_true", help="Write annotated overlay.mp4")
    parser.add_argument("--diagnostics", action="store_true", help="Save traces and process plots")
    parser.add_argument("--manual-blocks", help="Enable independent fixed-block mode using label JSON")
    parser.add_argument("--block-reference", help="Normal reference image for fixed-block mode")
    parser.add_argument("--manual-parameters", help="JSON overrides for ManualSpec thresholds")
    parser.add_argument("--matrix-every", type=float, default=1., help="Matrix snapshot interval; detection still runs every frame")
    parser.add_argument("--gray-white", action="store_true", help="Manual mode: include gray-white colour evidence")
    parser.add_argument("--light-colour", action="store_true", help="Manual mode: normalize patch V before colour classification")
    parser.add_argument("--abnormal-details", action="store_true", help="Manual mode: ERROR/suspected evidence section")
    parser.add_argument("--abnormal-only", action="store_true", help="Manual mode: replace ordinary matrix snapshots with abnormal blocks")
    parser.add_argument("--matrix-labels", action="store_true", help="Manual mode: enhanced matrix legend and point axes")
    parser.add_argument("--independent-evidence", action="store_true", help="Manual mode: first suspected/alarm evidence outside page quota")
    parser.add_argument("--dual-scenes", action="store_true", help="Manual mode: separate scene and alarm images")
    parser.add_argument("--validity-optimize", action="store_true", help="Manual mode: optional small-block validity rules")
    parser.add_argument("--colour-regions", action="store_true", help="Manual mode: independent neighbouring point/station colour continuity")
    parser.add_argument("--region-support", action="store_true", help="Colour regions: abstain on weak/fragmented body colour evidence")
    parser.add_argument("--region-bridge", action="store_true", help="Colour regions: connect similar patches across at most one station")
    parser.add_argument("--weak-red", action="store_true", help="Colour regions: weak RGB red connected evidence causes abstention")
    parser.add_argument("--weak-white", action="store_true", help="Colour regions: weak neutral connected evidence causes abstention")
    parser.add_argument("--recent-missing", action="store_true", help="Manual mode: confirm missing from recent hold_s window only")
    parser.add_argument("--detect-hz", type=float, default=0., help="Manual replay: detection samples/second; 0 keeps every frame")
    parser.add_argument("--small-fallback", action="store_true", help="Manual mode: recover unknown small blocks from independent pixel/colour support")
    parser.add_argument("--pixel-fallback", action=argparse.BooleanOptionalAction, default=None, help="Manual mode: dense poly recovery enabled by default; --no-pixel-fallback disables it")
    parser.add_argument("--summary-skip", type=float, default=5., help="Manual mode: exclude first seconds from summary selection (default 5; 0 disables)")
    parser.add_argument("--plot-mask", action=argparse.BooleanOptionalAction, default=True, help="Mark common brief disturbances in plots only; --no-plot-mask disables")
    parser.add_argument("--relative-red", action="store_true", help="Manual mode: same-frame HSV/RGB red calibration")
    parser.add_argument("--missing-support", action="store_true", help="Manual mode: require independent red missing evidence")
    # Independent window preview options; disabled unless --block-windows is supplied.
    parser.add_argument("--block-windows", action="store_true", help="Draw estimated barrier windows in blue")
    parser.add_argument("--block-score-min", type=float, default=.35, help="Minimum dominant lower-colour score")
    parser.add_argument("--block-white-score-min", type=float, default=.05, help="Minimum dominant lower-white ratio for block preview")
    parser.add_argument("--block-width-ratio", type=float, default=.60, help="Window width / local poly height")
    parser.add_argument("--block-valid-ratio", type=float, default=.90, help="Minimum valid-station fraction")
    parser.add_argument("--page-every", type=float, default=30., help="Periodic detail page interval (seconds)")
    parser.add_argument("--at-s", type=float, action="append", default=[], help="Additional detail time; repeatable")
    parser.add_argument("--max-pages", type=int, default=20, help="Cap detail snapshots (default: 20); traces remain complete")
    parser.add_argument("--replot", metavar="OUTPUT_DIR", help="Rebuild saved pages; never opens source video")
    args = parser.parse_args()
    if args.full_video:
        args.stop_after = None
    if args.replot:
        if args.local_video or args.video or args.diagnostics or args.overlay_video or args.block_windows or args.manual_blocks:
            parser.error("--replot is separate from replay options")
        from water_intermediates import replot
        replot(Path(args.replot), frame_limit=args.max_pages, stop_after=args.stop_after)
        return
    if not args.local_video or not args.video:
        parser.error("Local replay is disabled: supply --local-video --video PATH")
    values = [args.page_every, *args.at_s]
    if any(not np.isfinite(v) or v < 0 for v in values) or args.page_every == 0:
        parser.error("--page-every must be positive; --at-s must be nonnegative")
    if args.max_pages < 1 or (args.stop_after is not None and (
            not np.isfinite(args.stop_after) or args.stop_after < 0)):
        parser.error("--max-pages must be >=1; --stop-after must be nonnegative")
    try:
        if args.manual_blocks:
            if args.block_windows or not np.isfinite(args.matrix_every) or args.matrix_every <= 0:
                raise ValueError('Manual mode requires positive matrix interval and no --block-windows')
            from water_manual import replay_manual
            replay_manual(args)
            return
        if args.detect_hz != 0:
            raise ValueError("--detect-hz requires --manual-blocks")
        replay(args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
