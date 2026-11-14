"""Rendering must preserve raw frames shared with night sampling."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from unittest.mock import Mock
from night_lamp.tracked import LampMonitor
from server import render
from server.algo import AlgoResult, Annotation
from server.agents.day import DayRunner
from water_barrier.yolo_pipeline import YoloWaterAlgorithm


class RenderIsolation(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((120, 180, 3), dtype=np.uint8)
        self.original = self.frame.copy()
        self.box = Annotation("box", (30, 40, 100, 90), "water", "alarm")

    def assert_clean(self, image):
        self.assertTrue(np.array_equal(self.frame, self.original),
                        "Water rendering contaminated the shared raw frame")
        self.assertFalse(np.shares_memory(image, self.frame))
        self.assertTrue(np.any(image != self.original), "Annotations disappeared")

    def test_annotation_shapes(self):
        annots = [self.box, Annotation("point", (60, 60), "lamp", "info"),
                  Annotation("polygon", ((30, 40), (100, 40), (60, 90)), "roi", "alarm")]
        for annot in annots:
            with self.subTest(kind=annot.kind):
                self.frame[:] = self.original
                self.assert_clean(render.draw_annots(self.frame, [annot]))

    def test_status_only(self):
        self.assert_clean(render.draw_status(self.frame, "alignment_pending"))

    def test_lamp_signal_uses_clean_shared_frame(self):
        monitor=LampMonitor('cam','view')
        key=monitor.add_known([30,40,100,90],0)
        preview=render.draw_status(render.draw_annots(self.frame,[self.box]),'alignment_pending')
        self.assert_clean(preview)
        monitor.on_frame(self.frame,0,0,1,[],usable={key:True})
        self.assertTrue(np.array_equal(self.frame,self.original))

    def test_yolo_evidence(self):
        algo = YoloWaterAlgorithm.__new__(YoloWaterAlgorithm)
        algo._annots = [self.box]
        self.frame[:] = (10, 80, 160)
        self.original = self.frame.copy()
        image = algo._render(self.frame)
        self.assert_clean(image)
        self.assertEqual(tuple(image[110, 170]), (96, 96, 96))
        self.assertEqual(tuple(image[40, 30]), render.LEVEL_COLOR["alarm"])

    def test_day_snapshot(self):
        self.frame[:] = (10, 80, 160)
        self.original = self.frame.copy()
        result = AlgoResult([], [self.box])
        algo = SimpleNamespace(step=lambda frame, now: result, name="water_gap")
        runner = DayRunner(SimpleNamespace(id="1749"), None, algo)
        runner.on_frame(self.frame, 0)
        report = runner.snapshot(datetime.now(timezone.utc))
        self.assertIsNotNone(report.image_jpeg)
        image = cv.imdecode(np.frombuffer(report.image_jpeg, np.uint8), cv.IMREAD_COLOR)
        self.assertEqual(tuple(image[110, 170]), (96, 96, 96))
        self.assertGreater(int(image[40, 55, 2]), int(image[40, 55, 0]) + 40)
        self.assertTrue(np.array_equal(self.frame, self.original))


if __name__ == "__main__":
    unittest.main()
