"""Packaged lamp weight and runtime template must agree on the input size."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'deploy/app'))
from server.config_validate import parse_runtime
from server.contracts import LampDetectorSpec


class V7Configuration(unittest.TestCase):
    def test_production_enables_reports_and_test_template_keeps_them_disabled(self):
        for name,enabled in [('config.yaml',True),('config.test.yaml',False)]:
            with self.subTest(config=name):
                path=ROOT/'deploy'/name;data=yaml.safe_load(path.read_text(encoding='utf-8'))
                self.assertEqual(data['callback']['enabled'],enabled)
                self.assertEqual(data['runtime']['port'],8000)
                self.assertEqual(data['schedule']['report_at'],'07:00')
                cfg=parse_runtime(data,path.parent,source='template activation regression')
                self.assertEqual({c.id for c in cfg.cameras},{'1749','1750'})
                for camera in cfg.cameras:
                    self.assertEqual('night_lamp' in camera.algorithms,enabled)
                    if enabled:self.assertEqual(camera.algorithms['night_lamp'].schedule,'night')

    def test_packaged_weight_and_both_enabled_templates_are_consistent(self):
        profile=json.loads((ROOT/'models/lamp_v7/model.json').read_text())
        weight=ROOT/'models/lamp_v7/best.pt'
        self.assertEqual(hashlib.sha256(weight.read_bytes()).hexdigest(),profile['sha256'])
        self.assertEqual(LampDetectorSpec().imgsz,320)
        for name in ['config.yaml','config.test.yaml']:
            path=ROOT/'deploy'/name;data=yaml.safe_load(path.read_text(encoding='utf-8'))
            baseline=copy.deepcopy(data)
            for camera in data['cameras']:camera['algorithms']['night_lamp']['enabled']=True
            cfg=parse_runtime(data,path.parent,source='test')
            self.assertEqual(len(cfg.cameras),2)
            for camera in cfg.cameras:
                lamp=camera.algorithms['night_lamp'].spec
                self.assertEqual(Path(lamp.detector.weights).resolve(),weight.resolve())
                self.assertEqual(lamp.detector.imgsz,profile['inference_imgsz'])
                self.assertEqual(lamp.observation.effective_s,300)
                self.assertEqual(lamp.observation.normal_pulses,15)
                self.assertEqual(camera.algorithms['water_gap'].spec.backend,'yolo')
            self.assertEqual(yaml.safe_load(path.read_text(encoding='utf-8')),baseline)


if __name__=='__main__':unittest.main()
