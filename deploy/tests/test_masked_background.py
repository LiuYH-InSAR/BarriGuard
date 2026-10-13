import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'deploy/tools'));sys.path.insert(0,str(ROOT/'deploy/app'))
from masked_background import rebuild
from night_lamp.disturbance import PARAMETERS
from gray_matrix import GrayRecordingSession
from server.contracts import NightLampSpec,BaselineSpec,OvernightSpec
from background_kernels import numpy_replay, replay
from unittest.mock import patch

class MaskedBackgroundTests(unittest.TestCase):
    def test_batched_mask_and_replay_match_row_reference(self):
        from night_lamp.disturbance import detect_matrix
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
            folder=Path(temp)/'gray_matrix';folder.mkdir()
            t=np.arange(0,70,.16);n=len(t)
            gray=np.full((3,7,n),100,np.uint8)
            gray[0,:, (t>=40)&(t<50)]=160
            gray[1,:, (t>=45)&(t<55)]=40
            valid=np.ones((3,7),bool);valid[2,0]=False
            initial=np.full((3,7),100,np.float32)
            offsets=np.sin(t).astype(np.float32)
            updates=np.column_stack((t==60,np.full(n,.01)))
            parameters={'disturbance':dict(PARAMETERS),'on_delta':40}
            meta={'parameters':parameters,'origin_row':1,'origin_col':2}
            (folder/'manifest.json').write_text(json.dumps(meta))
            for name,value in dict(gray_raw=gray,time_s=t,exposure_offset=offsets,
                    valid_mask=valid,initial_background=initial,background_updates=updates).items():
                np.save(folder/f'{name}.npy',value)
            expected=[np.empty((3,7),dtype=d) for d in (np.float32,np.int32,np.float32,np.float32)]
            masks=[]
            for y in range(3):
                x=gray[y].astype(np.float32)-offsets
                mask=detect_matrix(x,t,PARAMETERS)|~valid[y,:,None]
                masks.append(np.packbits(mask,axis=1,bitorder='little'))
                for target,value in zip(expected,numpy_replay(x,mask,initial[y],updates,40)):target[y]=value
            session=SimpleNamespace(recorder=SimpleNamespace(folder=folder,parameters=parameters),
                _on=np.zeros((5,11),np.float32),_peak=np.zeros((5,11),np.float32),_base=np.zeros((5,11),np.float32))
            with patch('masked_background.BATCH_PIXELS',4):rebuild(session)
            for actual,wanted in zip((session._on,session._valid_sample_count,session._peak,session._base),expected):
                np.testing.assert_array_equal(actual[1:4,2:9],wanted)
            np.testing.assert_array_equal(session._disturbance_packed,np.asarray(masks))
            del session._disturbance_packed

    def test_compiled_replay_matches_numpy_exactly(self):
        rng=np.random.default_rng(56)
        x=rng.uniform(-30,280,(37,750)).astype(np.float32)
        mask=rng.random(x.shape)<.3
        initial=rng.uniform(0,200,37).astype(np.float32)
        updates=np.column_stack((rng.random(750)<.04,rng.uniform(0,.2,750)))
        for delta in (0,30,40):
            for a,b in zip(numpy_replay(x,mask,initial,updates,delta),replay(x,mask,initial,updates,delta)):
                np.testing.assert_array_equal(a,b)

    def test_no_mask_rebuild_preserves_original_computation(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
            spec=NightLampSpec(baseline=BaselineSpec(frames=60),overnight=OvernightSpec(enabled=False))
            session=GrayRecordingSession('test',spec,Path(temp)/'gray_matrix')
            session.recorder.parameters['disturbance']['enabled']=False
            for i in range(100):
                frame=np.full((32,32,3),20,np.uint8)
                frame[12:17,12:17]=80 if i%4==0 else 22
                session.accumulate(frame,i*.16)
            original=session.dump_state()['base'].copy()
            counts=session._on.copy();peak=session._peak.copy();valid=~session._mask
            session.recorder.finish()
            initial=np.load(session.recorder.folder/'initial_background.npy').copy()
            rebuild(session)
            np.testing.assert_array_equal(session._on[valid],counts[valid])
            np.testing.assert_array_equal(session._peak[valid],peak[valid])
            np.testing.assert_array_equal(session._base[valid],original[valid])
            np.testing.assert_array_equal(np.load(session.recorder.folder/'initial_background.npy'),initial)
            self.assertTrue((session._valid_sample_count[valid]==100).all())
            performance=json.loads((session.recorder.folder/'masked_performance.json').read_text())
            self.assertGreater(performance['reused_pixels'],0)
            self.assertEqual(performance['replayed_pixels'],0)
            with patch('masked_background.detect_matrix',side_effect=AssertionError('cache not used')):
                rebuild(session)
            self.assertTrue(json.loads((session.recorder.folder/'masked_performance.json').read_text())['cache_hit'])
            np.testing.assert_array_equal(session._on[valid],counts[valid])
            del session._disturbance_packed
            session.release()

    def test_mask_before_ema_and_both_ratio_counts(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
            folder=Path(temp)/'gray_matrix';folder.mkdir()
            t=np.arange(0,70,.16);n=len(t);x=np.full(n,100,np.uint8)
            x[(t>=40)&(t<50)]=160
            params={'disturbance':PARAMETERS,'baseline_frames':60,'on_delta':40}
            meta={'parameters':params,'origin_row':0,'origin_col':0}
            (folder/'manifest.json').write_text(json.dumps(meta))
            np.save(folder/'gray_raw.npy',x.reshape(1,1,n));np.save(folder/'time_s.npy',t)
            np.save(folder/'exposure_offset.npy',np.zeros(n));np.save(folder/'valid_mask.npy',np.ones((1,1),bool))
            np.save(folder/'background_updates.npy',np.column_stack((np.zeros(n),np.full(n,.1))))
            np.save(folder/'initial_background.npy',np.full((1,1),100,np.float32))
            session=SimpleNamespace(recorder=SimpleNamespace(folder=folder,parameters=params),_on=np.zeros((1,1),np.float32),_peak=np.zeros((1,1),np.float32),_base=np.full((1,1),100,np.float32))
            rebuild(session)
            mask=np.unpackbits(np.load(folder/'disturbance_mask_packed.npy')[0,0],bitorder='little')[:n].astype(bool)
            self.assertTrue(mask[(t>=40)&(t<50)].all())
            self.assertEqual(session._valid_sample_count[0,0],int((~mask).sum()))
            self.assertEqual(session._on[0,0],0)
            self.assertEqual(session._base[0,0],100)
            np.testing.assert_array_equal(np.load(folder/'initial_background.npy'),[[100]])
            previous=session._on.copy()
            with patch('masked_background.detect_matrix',side_effect=AssertionError('cache not used')):
                rebuild(session)
            np.testing.assert_array_equal(session._on,previous)
            meta=json.loads((folder/'manifest.json').read_text())
            meta['parameters']['on_delta']=30
            (folder/'manifest.json').write_text(json.dumps(meta))
            with patch('masked_background.detect_matrix',wraps=__import__('masked_background').detect_matrix) as detect:
                rebuild(session)
                self.assertEqual(detect.call_count,1)
            del session._disturbance_packed

if __name__=='__main__':unittest.main()
