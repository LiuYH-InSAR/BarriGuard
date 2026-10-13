"""Replay preserves detection; rejected/held candidates remain inspectable."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2 as cv
import numpy as np
import yaml

DEPLOY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY / "app"))
sys.path.insert(0, str(DEPLOY / "tools"))

from server.contracts import WaterGapSpec
from water_barrier.pipeline import WaterGapAlgorithm
from water_barrier import signal as S
from water_barrier import track as T
from water_intermediates import Recorder, replot, chart, table


def calibration():
    spec = WaterGapSpec()
    track = dict(asdict(spec.track), alarm_hold_s=spec.alarm_hold_s,
                 reconfirm_s=spec.reconfirm_s, rer_threshold=spec.confidence)
    return dict(rows=[dict(id="row_0", poly=[[.05, .2], [.95, .2], [.95, .65], [.05, .65]], U=15)],
                road_rois=[[16, 120, 304, 152]], detect=asdict(spec.detect), track=track)


def scene(kind="gap"):
    frame = np.full((160, 320, 3), 100, np.uint8)
    frame[32:75, 16:305] = 240
    frame[75:105, 16:305] = (0, 0, 220)
    if kind == "gap":
        frame[32:105, 140:175] = 100
    if kind == "occluded":
        frame[32:105, 80:220] = (220, 0, 0)
    if kind == "invalid":
        checker = (np.indices((160, 320)).sum(axis=0) % 4 < 2).astype(np.uint8) * 255
        frame = np.repeat(checker[:, :, None], 3, axis=2)
    return frame


class WaterTraceTests(unittest.TestCase):
    def test_block_windows_and_switch(self):
        from water_blocks import BlockSpec, scan_windows
        from water_result import RunSummary
        red, white = np.zeros(24), np.zeros(24)
        red[4:12], white[16:20] = 1, 1
        valid = np.ones(len(red), bool)
        windows = scan_windows(red, white, valid, np.full(len(red), 4), BlockSpec())
        self.assertEqual([(b['start'], b['end'], b['colour']) for b in windows], [(4, 11, 'red'), (16, 19, 'white')])
        valid[16:20] = False
        windows = scan_windows(red, white, valid, np.full(len(red), 4), BlockSpec())
        self.assertEqual(len(windows), 1)
        # Shadow white has weaker mask coverage than red, but remains a white segment.
        windows = scan_windows(np.zeros(24), np.full(24, .08), np.ones(24, bool), np.full(24, 4), BlockSpec())
        self.assertEqual([(b['start'], b['end'], b['colour']) for b in windows], [(0, 23, 'white')])
        summary = RunSummary()
        from types import SimpleNamespace
        summary.add(scene('intact'), 0, 0, SimpleNamespace(_tracks=[]))
        with tempfile.TemporaryDirectory(dir=DEPLOY.parent/'tmp') as temp:
            folder = Path(temp)
            summary.finish(folder, 'test.mp4', '1749', 2, calibration()['rows'], BlockSpec())
            enabled = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
            self.assertTrue(enabled['block_windows']['blocks'])
            self.assertTrue((folder/'summary_raw.png').is_file())
            with patch.dict(sys.modules, {'water_blocks': None}):
                summary.finish(folder, 'test.mp4', '1749', 2, calibration()['rows'])
            disabled = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
            self.assertNotIn('block_windows', disabled)
            self.assertEqual(enabled['tracks'], disabled['tracks'])
            self.assertEqual(enabled['representative'], disabled['representative'])

    def test_lower_red_or_white_score(self):
        poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
        frame = np.full((100, 100, 3), 240, np.uint8)
        frame[60:, :50] = (0, 0, 220)
        trace = {}
        pack = S.sample_row(frame, poly, trace=trace)
        np.testing.assert_allclose(pack['s'], 1)
        self.assertTrue(np.any(trace['lower_white_ratio'] == 1))
        frame[60:, 50:] = 100
        pack = S.sample_row(frame, poly)
        self.assertAlmostEqual(float(pack['s'][-1]), .4, places=6)
        # A red/white HSV boundary pixel still contributes only one vote.
        hsv = np.full((100, 100, 3), (0, 70, 200), np.uint8)
        with patch.object(S, 'red_m', return_value=np.ones((100, 100), bool)), \
             patch.object(S, 'white_m', return_value=np.ones((100, 100), bool)):
            pack = S.sample_row(cv.cvtColor(hsv, cv.COLOR_HSV2BGR), poly)
        np.testing.assert_allclose(pack['s'], 1)

    def test_run_summary_retains_earlier_alarm(self):
        from types import SimpleNamespace
        from water_result import RunSummary
        summary = RunSummary()
        track = dict(state='ALARM', row_id='row_0', box=(16, 32, 60, 100), rer=.8)
        algo = SimpleNamespace(_tracks=[track])
        summary.add(scene(), 0, 0, algo)
        track['state'] = 'NORMAL'
        summary.add(scene(), 1, .5, algo)
        with tempfile.TemporaryDirectory(dir=DEPLOY.parent / 'tmp') as temp:
            summary.finish(temp, 'test.mp4', '1749', 2)
            result = json.loads((Path(temp)/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(result['alarm_frames'], 1)
            self.assertEqual(result['representative']['boxes'][0]['state'], 'ALARM')
            self.assertEqual(result['tracks'][0]['peak_state'], 'ALARM')
            self.assertTrue((Path(temp)/'summary.png').is_file())

    def test_trace_preserves_results(self):
        a = WaterGapAlgorithm(scene().shape, calibration(), "test")
        b = WaterGapAlgorithm(scene().shape, calibration(), "test")
        kinds = ["gap"] * 25 + ["occluded", "invalid"] + ["intact"] * 15
        seen = set()
        for i, kind in enumerate(kinds):
            trace = {}
            plain = a.step(scene(kind), i * .5)
            diagnostic = b.step(scene(kind), i * .5, trace=trace)
            self.assertEqual(plain.events, diagnostic.events)
            self.assertEqual(plain.annots, diagnostic.annots)
            self.assertEqual(plain.debug["diag"], diagnostic.debug["diag"])
            for rid in a._sup["row_ids"]:
                np.testing.assert_array_equal(a._sup["acc"][rid], b._sup["acc"][rid])
                self.assertEqual(a._sup["n_obs"][rid], b._sup["n_obs"][rid])
            seen.update(e.kind for e in diagnostic.events)
            if kind == "occluded":
                self.assertEqual(trace["skip"], {"row_0": "OCCLUDED"})
                self.assertTrue(trace["gaps"])
                self.assertTrue(trace["tracks"][0]["matched"])
        self.assertIn("alarm", seen)

    def test_rejected_candidates_and_multi_core(self):
        sm = np.ones(S.REF, np.float32) * .8
        sm[100:108] = .05
        sm[240:290] = .05
        sm[500:545] = .05
        detail = []
        found = S.detect(sm, np.full(S.REF, .8), np.ones(S.REF, bool), trace=detail)
        self.assertEqual(detail[0]["reason"], "ENVELOPE_FAIL")
        self.assertEqual(len(found), 2)
        self.assertEqual([d["core"][0] for d in detail[1:]], [240, 500])
        # Real frame with two low valleys must annotate each candidate, not only the last.
        frame = scene()
        frame[32:105, 230:265] = 100
        trace = {}
        algo = WaterGapAlgorithm(frame.shape, calibration())
        algo.step(frame, 0, trace=trace)
        candidates = trace["rows"]["row_0"]["candidates"]
        self.assertGreaterEqual(len(candidates), 2)
        self.assertTrue(all("confidence" in c for c in candidates if c.get("core")))

    def test_rer_trace_equal(self):
        frame = scene()
        profile = T.fit_road_profile(frame[120:152, 16:304])
        slot = np.zeros(frame.shape[:2], np.uint8)
        slot[32:105, 140:175] = 255
        fg = np.where(S.red_m(frame) | S.white_m(frame), 255, 0).astype(np.uint8)
        trace = {}
        expected = T.calc_patch_rer(frame, slot, fg, profile)
        actual = T.calc_patch_rer(frame, slot, fg, profile, trace=trace)
        self.assertEqual(expected, actual)
        self.assertGreater(actual, .25)
        self.assertEqual(trace["valid_count"], np.count_nonzero(trace["valid"]))

    def test_saved_replot_without_video(self):
        with tempfile.TemporaryDirectory(dir=DEPLOY.parent / "tmp") as temp:
            out = Path(temp)
            spec = WaterGapSpec()
            algo = WaterGapAlgorithm(scene().shape, calibration())
            recorder = Recorder(out, Path("nonexistent.mp4"), 2, scene().shape,
                                calibration(), spec, 30, [10], 6)
            for i in range(27):
                trace = {}
                frame = scene("occluded" if i == 25 else "gap")
                result = algo.step(frame, i / 2, trace=trace)
                recorder.add(i, i / 2, frame, result, trace, algo)
            recorder.finish()
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["frames"], 27)
            self.assertEqual(manifest["schema"], 2)
            self.assertTrue(manifest["complete"])
            self.assertLessEqual(len(manifest["snapshots"]), 6)
            records = [json.loads(line) for line in (out / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[25]["trace"]["tracks"][0]["quality_skip"], "OCCLUDED")
            self.assertFalse(records[25]["trace"]["tracks"][0]["fresh_hit"])
            self.assertTrue(records[25]["trace"]["tracks"][0]["matched"])
            self.assertTrue(list(out.glob("rer_*.html")))
            with patch("cv2.VideoCapture", side_effect=AssertionError("must not reopen video")):
                replot(out)
            for path in out.glob("pages/*.html"):
                self.assertIn("support", path.read_text(encoding="utf-8"))
            main = (out / "pages/frame_00000000.html").read_text(encoding="utf-8")
            for title in ("红白比例", "颜色分数", "红色识别", "白色识别", "异色识别", "整帧质量", "有效区域", "历史累计参数", "路面判断参数"):
                self.assertIn(title, main)
            self.assertNotIn("<table>", main)
            self.assertIn("开头10秒", (out / "track_0001.html").read_text(encoding="utf-8"))

    def test_gate_colour_and_isolated_point(self):
        text = table([("gate", "FAIL", "bad"), ("next", "SKIP", "unused")])
        self.assertIn("#c62828", text)
        self.assertIn("#777777", text)
        svg = chart("RER", [0., 1., 2.], [("actual", np.array([np.nan, .5, np.nan]))])
        self.assertIn("<circle", svg)
        self.assertNotIn("nan", svg)

    def test_local_replay_requires_switch(self):
        result = subprocess.run([sys.executable, str(DEPLOY / "tools/run_water_clip.py"),
                                 "--video", "nonexistent.mp4"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Local replay is disabled", result.stderr)

    def test_all_anomalies_and_colour_overlap(self):
        from water_plot import masks_from_hsv, series_plot, RED
        from water_intermediates import parameters
        p = parameters(calibration(), WaterGapSpec())
        hsv = np.array([[[90, 110, 100], [90, 20, 170]]], np.uint8)
        masks = masks_from_hsv(hsv, p)
        self.assertTrue(masks["white"][0, 0])
        self.assertTrue(masks["foreign"][0, 0])
        self.assertFalse(masks["foreign"][0, 1])
        with tempfile.TemporaryDirectory(dir=DEPLOY.parent / "tmp") as temp:
            annotations = [(i*3, i*3+1, f"异常编号{i}", RED) for i in range(18)]
            path = series_plot(Path(temp), "test", "异常标记", np.arange(60),
                               [("values", np.ones(60), "#222")], "threshold=.3", annotations=annotations)
            text = (Path(temp) / path).read_text(encoding="utf-8")
            for i in range(18):
                self.assertIn(f"异常编号{i}", text)

    def test_cli_video_and_saved_replot(self):
        with tempfile.TemporaryDirectory(dir=DEPLOY.parent / "tmp") as temp:
            folder = Path(temp)
            video = folder / "test.mp4"
            writer = cv.VideoWriter(str(video), cv.VideoWriter_fourcc(*"mp4v"), 2, (320, 160))
            self.assertTrue(writer.isOpened())
            for _ in range(130):
                writer.write(scene())
            writer.release()
            calib = calibration()
            (folder / "roi.yaml").write_text(yaml.safe_dump({k: calib[k] for k in ("rows", "road_rois")}), encoding="utf-8")
            cfg = yaml.safe_load((DEPLOY / "config.yaml").read_text(encoding="utf-8"))
            cfg["cameras"] = [cfg["cameras"][0]]
            cfg["cameras"][0]["algorithms"]["water_gap"]["calibration"] = str(folder / "roi.yaml")
            (folder / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
            out = folder / "run"
            result = subprocess.run([sys.executable, str(DEPLOY / "tools/run_water_clip.py"),
                                     "--local-video", "--video", str(video), "--config", str(folder / "config.yaml"),
                                     "--out", str(out), "--diagnostics", "--overlay-video", "--max-pages", "4"],
                                    capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["frames"], 121)
            self.assertTrue((out / "overlay.mp4").is_file())
            self.assertTrue((out / "summary.html").is_file())
            self.assertTrue((out / "summary.png").is_file())
            lines = (out / "frames.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(lines[-1])["time_s"], 60.)
            self.assertTrue(list(out.glob("rer_*.html")))
            full = subprocess.run([sys.executable, str(DEPLOY / "tools/run_water_clip.py"),
                                   "--local-video", "--video", str(video), "--config", str(folder / "config.yaml"),
                                   "--out", str(folder / "full"), "--full-video"],
                                  capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(full.returncode, 0, full.stderr)
            self.assertEqual(json.loads((folder / "full/run.json").read_text(encoding="utf-8"))["frames"], 130)
            self.assertTrue((folder/'full/summary.png').is_file())
            block_run = subprocess.run([sys.executable, str(DEPLOY/'tools/run_water_clip.py'),
                '--local-video', '--video', str(video), '--config', str(folder/'config.yaml'),
                '--out', str(folder/'blocks'), '--stop-after', '0', '--block-windows'],
                capture_output=True, text=True, encoding='utf-8', errors='replace')
            self.assertEqual(block_run.returncode, 0, block_run.stderr)
            block_result = json.loads((folder/'blocks/summary.json').read_text(encoding='utf-8'))
            self.assertTrue(block_result['block_windows']['enabled'])
            video.unlink()
            result = subprocess.run([sys.executable, str(DEPLOY / "tools/run_water_clip.py"),
                                     "--replot", str(out)], capture_output=True, text=True,
                                    encoding="utf-8", errors="replace")
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
