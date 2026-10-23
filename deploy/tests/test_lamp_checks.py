import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from night_lamp.lamp_checks import LampChecks, hotspot_region
from server.contracts import LampCheckSpec


class LampCheckTests(unittest.TestCase):
    def test_strength_threshold_suppresses_faint_regions(self):
        duty = np.ones((1,3),np.float32)
        keep = np.ones_like(duty)
        region = hotspot_region(duty, 1, keep, np.array([[1,.35,1]]),
                                np.array([[1,1,.001]]), np.zeros_like(duty,bool),
                                .7,.005,.75)
        np.testing.assert_array_equal(region, [[True,False,False]])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        p = Path(self.folder.name) / "lamps.json"
        p.write_text(json.dumps(dict(imageHeight=20, imageWidth=30,
            shapes=[dict(label="lamp01", shape_type="polygon",
                         points=[[2,2],[8,2],[8,8],[2,8]])])))
        self.checks = LampChecks(LampCheckSpec(enabled=True, polygons=str(p)))
        self.empty = np.zeros((20,30), bool)

    def observe(self, flash=None, coverage=1.0):
        return self.checks.observe(self.empty if flash is None else flash,
                                  self.empty, coverage, True, str(len(self.checks.history)))

    def test_circle_label(self):
        p = Path(self.folder.name) / "circle.json"
        p.write_text(json.dumps(dict(imageHeight=20, imageWidth=30,
            shapes=[dict(label="lamp01", shape_type="circle", points=[[5,5],[8,5]])])))
        checks = LampChecks(LampCheckSpec(enabled=True, polygons=str(p)))
        rows = checks.observe(~self.empty, self.empty, 1.0, True, "hour")
        self.assertEqual(rows[0]["status"], "FLASHING")
        self.assertGreater(rows[0]["heatmap_pixels"], 20)

    def test_strict_area_threshold(self):
        self.checks.spec.heatmap_ratio = 15/49
        heat = self.empty.copy()
        heat[2:5,2:7] = True
        self.assertEqual(self.observe(heat)[0]["status"], "NOT_FLASHING")
        heat[5,2] = True
        self.assertEqual(self.observe(heat)[0]["status"], "FLASHING")

    def test_three_lit_hours_any_order(self):
        self.checks.spec.lit_hours = 3
        self.observe(~self.empty)
        self.observe()
        self.observe(~self.empty)
        self.assertEqual(self.checks.summary()[0]["status"], "NOT_FLASHING")
        self.observe(~self.empty)
        self.observe()
        self.assertEqual(self.checks.summary()[0]["status"], "FLASHING")
        self.assertEqual(self.checks.summary()[0]["lit_hours"], 3)

    def test_diagnostics_do_not_gate(self):
        rows = self.checks.observe(~self.empty, ~self.empty, .1, False, "hour", overflow=True)
        self.assertEqual(rows[0]["status"], "FLASHING")
        self.assertEqual(rows[0]["heatmap_ratio"], 1.0)

    def test_missing_hours_are_not_lit(self):
        self.checks.spec.lit_hours = 3
        self.observe(~self.empty)
        self.checks.unknown("missing")
        self.assertEqual(self.checks.summary()[0]["status"], "NOT_FLASHING")
        self.assertEqual(self.checks.summary()[0]["unavailable_hours"], 1)

    def test_one_hour_is_not_counted_twice(self):
        for stamp in ("2026-09-29T20:05:00", "2026-09-29T20:10:00"):
            self.checks.observe(~self.empty, self.empty, 1, True, stamp)
        self.assertEqual(self.checks.summary()[0]["lit_hours"], 1)

    def test_restore_and_draw(self):
        self.observe()
        value = self.checks.dump()
        self.checks.history = []
        self.checks.restore(value)
        rows = self.checks.summary()
        image = self.checks.draw(np.zeros((20,30,3), np.uint8), rows)
        self.assertTrue(np.any(image[:,:,2] > 0))

    def test_wrong_image_dimensions(self):
        with self.assertRaises(ValueError):
            self.checks.observe(np.zeros((10,10),bool),self.empty,1,True,"hour")

    def test_session_report_and_persistence(self):
        from night_lamp.session import NightSession
        from server.contracts import NightLampSpec, BaselineSpec, OvernightSpec
        from unittest.mock import patch
        p = Path(self.folder.name) / "session.json"
        p.write_text(json.dumps(dict(imageHeight=64, imageWidth=64,
            shapes=[dict(label="lamp01", shape_type="polygon",
                         points=[[27,27],[36,27],[36,36],[27,36]])])))
        spec = NightLampSpec(baseline=BaselineSpec(frames=60, warmup_s=600),
            overnight=OvernightSpec(enabled=True, burst_seconds=160, anchor="wall"),
            lamp_checks=LampCheckSpec(enabled=True, polygons=str(p)))
        with patch("night_lamp.session.nm.osd_mask", return_value=np.zeros((64,64),bool)):
            session = NightSession("test", spec)
            session.begin_burst("hour")
            for i in range(1000):
                frame = np.full((64,64,3),20,np.uint8)
                if i % 6 < 3:
                    frame[28:35,28:35] = [25,110,120]
                session.accumulate(frame,i*.16)
            report = session.snapshot("hour", force=True)
            self.assertEqual(report.metadata["fixed_lamps"][0]["status"], "FLASHING")
            restored = NightSession("test", spec)
            self.assertTrue(restored.load_state(session.dump_state()))
            with patch("night_lamp.session.nm.compose", return_value=np.full((64,64,3), [255,0,0], np.uint8)):
                final = restored.finalize(None,"morning")
            self.assertEqual(final.metadata["fixed_lamps"][0]["status"], "FLASHING")
            import cv2
            image = cv2.imdecode(np.frombuffer(final.image_jpeg, np.uint8), cv2.IMREAD_COLOR)
            self.assertLess(int(image[5,5].max())-int(image[5,5].min()), 3)
            restored.release()
            session.release()
