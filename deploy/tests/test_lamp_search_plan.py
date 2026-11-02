import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
from local_test_support import OFFLINE
sys.path.insert(0,str(ROOT/'deploy/app'))
sys.path.insert(0,str(OFFLINE))
from night_lamp.adaptive.search_plan import plan_search
from night_lamp.adaptive.adaptive_prototype import grid
from night_lamp.adaptive.adaptive_v5 import AdaptiveV5
try:
    from detection_tape import RecordingDetector,ReplayDetector
except ModuleNotFoundError:
    RecordingDetector=ReplayDetector=None  # Offline tooling is intentionally not shipped.


class Backend:
    def __init__(self):self.requests=[]
    def predict(self,frame,regions,imgsz,confidence):
        self.requests.append((copy.deepcopy(regions),imgsz,confidence))
        return []


class SearchTests(unittest.TestCase):
    def test_discovery_requests_equal_frozen_pre_refactor_code(self):
        path=ROOT/'tmp/lamp_search_split_20261031/baseline/production_adaptive_v5.py'
        if not path.exists():self.skipTest('Local frozen oracle is not shipped')
        spec=importlib.util.spec_from_file_location('night_lamp.adaptive._frozen_v5',path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        a,b=module.AdaptiveV5(Backend()),AdaptiveV5(Backend())
        rng=np.random.default_rng(98);frame=np.zeros((1080,1920,3),np.uint8)
        for i in range(180):
            proposals=[(float(rng.integers(1,6)),rng.integers([0,0],[1920,1080]).tolist(),'image') for _ in range(i%23)]
            for detector in (a,b):detector.proposals_from_image=lambda *args:copy.deepcopy(proposals)
            self.assertEqual(a.discover(frame,None,i*.21,True),b.discover(frame,None,i*.21,True))
            self.assertEqual(a.backend.requests,b.backend.requests)
            self.assertEqual(a.roi_tested,b.roi_tested)
            self.assertEqual(a.survey_cursor,b.survey_cursor)

    def test_bounded_survey_is_independent_and_keeps_budget(self):
        survey=grid((1080,1920),160,128);tested={(0,0):1.};proposals=[(999,[80,80],'history')]*100
        original=copy.deepcopy((tested,proposals));cursor=0;covered=set()
        for i in range(100):
            kw=dict(size=160,budget=12,survey_budget=2,survey=survey,exclusions=(),policy='bounded')
            a=plan_search((1080,1920),proposals,tested,cursor,i,**kw)
            b=plan_search((1080,1920),[],tested,cursor,i,**kw)
            self.assertEqual(a.regions[:2],b.regions[:2]);self.assertLessEqual(len(a.regions),12)
            self.assertEqual(a.sources[:2],('survey','survey'))
            covered.update(a.regions[:2]);cursor=a.next_cursor
        self.assertEqual(len(covered),len(survey));self.assertEqual((tested,proposals),original)

    def test_masked_and_all_survey_budget_terminate(self):
        kw=dict(size=160,budget=2,survey_budget=2,survey=grid((160,320),160,128),policy='bounded')
        a=plan_search((160,320),[],{},0,0,exclusions=((0,0,1,1),),**kw)
        self.assertEqual(a.regions,());self.assertEqual(a.next_cursor,len(kw['survey']))
        b=plan_search((160,320),[(99,[160,80],'history')],{},0,0,exclusions=(),**kw)
        self.assertEqual(b.sources,('survey','survey'));self.assertEqual(len(b.regions),2)

    @unittest.skipUnless(RecordingDetector is not None,'Offline recording tools are not deployed')
    def test_tape_roundtrip_and_reject_different_frame_roi_threshold_model(self):
        frame=np.zeros((12,12,3),np.uint8);regions=[[0,0,8,8]]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tape.gz';writer=RecordingDetector(Backend(),path,'weight')
            writer.predict(frame,regions,640,.65);writer.close()
            reader=ReplayDetector(path,'weight');self.assertEqual(reader.predict(frame,regions,640,.65),[])
            reader.assert_finished();reader.close()
            for change in ['frame','roi','threshold','model']:
                reader=ReplayDetector(path,'other' if change=='model' else 'weight')
                f=frame.copy()
                if change=='frame':f[0,0]=1
                with self.assertRaises(ValueError):reader.predict(f,[[1,0,9,8]] if change=='roi' else regions,640,.35 if change=='threshold' else .65)
                reader.close()
            reader=ReplayDetector(path,'weight','reject_crop_edge_1px_iou0.5')
            with self.assertRaises(ValueError):reader.predict(frame,regions,640,.65)
            reader.close()

    @unittest.skipUnless(RecordingDetector is not None,'Offline crop experiment is not deployed')
    def test_partial_crop_box_is_not_independent_fixture_evidence(self):
        from crop_edge_control import crop_complete
        self.assertFalse(crop_complete([152.650955,104.433289,160.,115.26205],160,160))
        for b in [[0,10,20,30],[10,0,30,20],[10,10,160,30],[10,10,30,160],[1,10,20,30]]:
            self.assertFalse(crop_complete(b,160,160))
        self.assertTrue(crop_complete([5,10,25,30],160,160))

    @unittest.skipUnless(RecordingDetector is not None,'Offline experiment guards are not deployed')
    def test_inference_size_matches_training_unless_explicit_ablation(self):
        from inference_profile import resolve_inference_size
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);(run/'weights').mkdir();weight=run/'weights/best.pt';weight.write_bytes(b'test')
            (run/'args.yaml').write_text('imgsz: 320\n')
            self.assertEqual(resolve_inference_size(weight)['inference_imgsz'],320)
            with self.assertRaises(ValueError):resolve_inference_size(weight,640)
            self.assertTrue(resolve_inference_size(weight,640,True)['intentional_mismatch'])


if __name__=='__main__':unittest.main()
