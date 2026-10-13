import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from night_lamp.stutter import duration_ratio, detect_path, SETTINGS


class StutterTests(unittest.TestCase):
    def test_duration_and_strict_trigger(self):
        t=np.arange(11,dtype=float)
        self.assertEqual(duration_ratio(np.ones(11,bool),t),1.)
        self.assertEqual(duration_ratio(np.zeros(11,bool),t),0.)

    def test_normal_path_unchanged(self):
        from night_lamp.disturbance import detect,PARAMETERS
        t=np.arange(0,60,.16);x=np.full(len(t),100.)
        a=detect(x,t,PARAMETERS)
        b=detect_path(x,t,PARAMETERS,SETTINGS)
        np.testing.assert_array_equal(a[0],b[0])
        self.assertFalse(b[2]['suspected_stutter'])

    def test_high_mask_recomputed_once(self):
        from night_lamp.disturbance import detect,PARAMETERS
        t=np.arange(0,120,.16);x=np.where(t<8,60.,120.)
        p=dict(PARAMETERS,reference_s=5)
        normal=detect(x,t,p)
        self.assertGreater(duration_ratio(normal[0],t),.8)
        actual=detect_path(x,t,p,SETTINGS)
        expected=detect(x,t,dict(p,window_s=20.,enter_delta=20.))
        np.testing.assert_array_equal(actual[0],expected[0])
        self.assertTrue(actual[2]['suspected_stutter'])
        self.assertEqual(actual[2]['parameters']['window_s'],20.)
        self.assertEqual(actual[2]['parameters']['enter_delta'],20.)

    def test_40_samples_and_selected_gates(self):
        from types import SimpleNamespace
        from night_lamp.evidence import analyse_region
        p=SimpleNamespace(onset_min=20,peak_floor=.2,min_peaks=3,gap_tol=1)
        t=np.arange(40)*.16
        x=np.where(np.arange(40)%4==0,160.,100.)
        bgr=np.repeat(x[:,None],3,axis=1)
        normal=analyse_region(bgr,p,2,15,t,force_stutter=False)
        self.assertEqual(normal['reason'],'insufficient_samples')
        result=analyse_region(bgr,p,2,15,t,force_stutter=True)
        self.assertTrue(result['suspected_stutter'])
        self.assertEqual(result['selected_thresholds']['min_samples'],40)
        self.assertEqual(result['selected_thresholds']['onset_min'],5)
        self.assertEqual(result['selected_thresholds']['gap_tol'],5)
        short=analyse_region(bgr[:-1],p,2,15,t[:-1],force_stutter=True)
        self.assertEqual(short['reason'],'insufficient_samples')

    def test_ratio_gate_is_strict(self):
        from types import SimpleNamespace
        from night_lamp.session import NightSession
        session=SimpleNamespace(_series=SimpleNamespace(_pys=[0],_pxs=[0]),
                                _on=np.array([[8.]]),_valid_sample_count=np.array([[40]]))
        result=dict(suspected_stutter=True,flashing=True)
        NightSession._fallback_gate(session,result,0)
        self.assertFalse(result['flashing'])
        session._on[0,0]=9
        result=dict(suspected_stutter=True,flashing=True)
        NightSession._fallback_gate(session,result,0)
        self.assertTrue(result['flashing'])

    def test_background_replay_uses_20s_mask(self):
        import json,tempfile
        from types import SimpleNamespace
        root=Path(__file__).resolve().parents[2]
        sys.path.insert(0,str(root/'deploy/tools'))
        from masked_background import rebuild
        from background_kernels import numpy_replay
        from night_lamp.disturbance import PARAMETERS,detect_matrix
        with tempfile.TemporaryDirectory(dir=root/'tmp') as temp:
            folder=Path(temp);t=np.arange(0,300,.16);n=len(t)
            gray=np.where(t<40,60,120).astype(np.uint8)[None,None,:]
            updates=np.column_stack((np.zeros(n),np.full(n,.001)))
            p=dict(disturbance=dict(PARAMETERS),stutter=dict(SETTINGS),on_delta=40)
            meta=dict(parameters=p,origin_row=0,origin_col=0)
            (folder/'manifest.json').write_text(json.dumps(meta))
            for name,value in dict(gray_raw=gray,time_s=t,exposure_offset=np.zeros(n),
                background_updates=updates,valid_mask=np.ones((1,1),bool),initial_background=np.full((1,1),60,np.float32)).items():
                np.save(folder/f'{name}.npy',value)
            session=SimpleNamespace(recorder=SimpleNamespace(folder=folder,parameters=p),
                _on=np.zeros((1,1),np.float32),_peak=np.zeros((1,1),np.float32),_base=np.full((1,1),60,np.float32))
            rebuild(session)
            self.assertTrue(session._stutter_map[0,0])
            mask=detect_matrix(gray[0].astype(np.float32),t,dict(PARAMETERS,window_s=20.,enter_delta=20.))
            actual=np.unpackbits(session._disturbance_packed[0],axis=1,bitorder='little')[:,:n].astype(bool)
            np.testing.assert_array_equal(actual,mask)
            wanted=numpy_replay(gray[0].astype(np.float32),mask,np.array([60],np.float32),updates,40)
            for actual,expected in zip((session._on[0],session._valid_sample_count[0],session._peak[0],session._base[0]),wanted):
                np.testing.assert_array_equal(actual,expected)
            rebuild(session)
            self.assertTrue(session._stutter_map[0,0])
            del session._disturbance_packed


if __name__=='__main__':unittest.main()
