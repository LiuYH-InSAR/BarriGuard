"""Fixed-block geometry, colour loss, unknown gating and matrix export."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2 as cv
import numpy as np

DEPLOY = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(DEPLOY/'app'),str(DEPLOY/'tools')]
from water_manual import ManualDetector, ManualSpec, matrix_page
from water_manual import replay_manual


class ManualTests(unittest.TestCase):
    def test_relative_red_and_missing_gate(self):
        from types import SimpleNamespace
        from water_manual_relative import RelativeRed, missing_support
        spec=ManualSpec()
        frame=np.zeros((5,15,3),np.uint8)
        frame[:,:10]=(120,130,200)
        frame[:,10:]=(180,185,205)
        geometry=[]
        for x in (0,5,10):
            xx,yy=np.meshgrid(np.arange(x,x+5),np.arange(5))
            geometry.append(dict(pixels=[[(yy.ravel(),xx.ravel())]*5]))
        values=[np.zeros((1,5,6)) for _ in range(3)]
        for f in values[:2]: f[:,:,0]=1
        blocks=[dict(row=0,number=n,label=str(n)) for n in (1,3,5)]
        detector=SimpleNamespace(spec=spec,blocks=blocks,geometry=geometry,
            types=[np.zeros((1,5),int)]*3,point_ok=lambda f:np.ones(f.shape[:2],bool))
        rules=RelativeRed(frame,cv.cvtColor(frame,cv.COLOR_BGR2HSV),detector,values,
                          {b['label']:'normal' for b in blocks})
        profile=rules.profile(2)
        self.assertTrue(profile['usable'])
        self.assertTrue((rules.sample(2,profile)==1).all())
        frame[:,10:]=190
        rules.hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        self.assertFalse(rules.sample(2,profile).any())
        rules.states['1']='error'
        self.assertFalse(rules.profile(2)['usable'])
        valid=np.ones((10,5),bool)
        red=np.zeros((10,5)); red[:,:1]=1
        self.assertTrue(missing_support(valid,red,profile,spec)['allowed'])
        red[:,:3]=1
        self.assertFalse(missing_support(valid,red,profile,spec)['allowed'])
        self.assertFalse(missing_support(valid,red*0,dict(usable=False),spec)['allowed'])

    def test_dense_poly_fallback(self):
        from water_manual_pixels import PixelFallback
        spec=ManualSpec(gray_white=True)
        frame=np.full((40,40,3),160,np.uint8)
        poly=np.array([[4,4],[6,4],[6,25],[4,25]])
        white=PixelFallback(poly,frame,spec,1)
        result=white.assess(frame,1.)
        self.assertTrue(result['recovered'])
        self.assertEqual(result['poly_pixels'],66)
        self.assertFalse(white.assess(frame*0,1.)['recovered'])
        tiny=PixelFallback(np.array([[4,4],[5,4],[5,10],[4,10]]),frame,spec,1)
        self.assertFalse(tiny.assess(frame,1.)['recovered'])
        redframe=np.full_like(frame,(0,0,200))
        red=PixelFallback(poly,redframe,spec,0)
        self.assertTrue(red.assess(redframe,1.)['recovered'])
        self.assertFalse(red.assess(frame,1.)['recovered'])
        fragmented=np.full_like(frame,(0,80,0))
        fragmented[4:26:2,4:7]=160
        self.assertFalse(white.assess(fragmented,1.)['recovered'])

    def test_small_fallback_pixel_support(self):
        from water_manual_small import SmallFallback
        centers=np.array([[[5+i*2,5]]*5 for i in range(10)])
        pixels=[]
        for i in range(10):
            xx,yy=np.meshgrid(np.arange(3+i*2,8+i*2),np.arange(3,8))
            pixels.append([(yy.ravel(),xx.ravel())]*5)
        spec=ManualSpec()
        f=np.zeros((10,5,6));f[:,:,3]=120
        colour=np.zeros((10,5,2));colour[:,:,1]=1
        valid=np.ones((10,5),bool)
        rules=SmallFallback(dict(centers=centers,pixels=pixels),spec)
        result=rules.assess(f,colour,valid,1)
        self.assertTrue(result['recovered'])
        self.assertEqual(result['pixels'],115)
        self.assertFalse(rules.assess(f,colour*0,valid,1)['recovered'])
        small=SmallFallback(dict(centers=centers[:8],pixels=pixels[:8]),spec)
        self.assertFalse(small.assess(f[:8],colour[:8],valid[:8],1)['recovered'])

    def test_sampling_keeps_video_time(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=DEPLOY.parent/'tmp') as td:
            folder=Path(td)
            shapes=[dict(label='row0',shape_type='polygon',points=[[0,20],[319,20],[319,120],[0,120]])]
            frame=np.full((160,320,3),100,np.uint8)
            for i,x in enumerate([10,110,210]):
                shapes.append(dict(label=f'row0block{2*i+1:02}',shape_type='polygon',points=[[x,20],[x+80,20],[x+80,120],[x,120]]))
                frame[20:121,x:x+81]=(0,0,220)
            label=folder/'labels.json'
            label.write_text(json.dumps(dict(imageWidth=320,imageHeight=160,shapes=shapes,imagePath='ref.png')))
            cv.imwrite(str(folder/'ref.png'),frame)
            video=folder/'clip.avi'
            writer=cv.VideoWriter(str(video),cv.VideoWriter_fourcc(*'MJPG'),20,(320,160))
            changed=frame.copy();changed[20:121,10:91]=100
            for _ in range(240):writer.write(changed)
            writer.release()
            args=SimpleNamespace(manual_blocks=str(label),block_reference=None,manual_parameters=None,
                gray_white=False,light_colour=False,out=str(folder/'out'),camera='1749',video=str(video),fps=None,
                stop_after=None,diagnostics=False,max_pages=1,matrix_every=5,at_s=[],overlay_video=False,
                abnormal_only=False,abnormal_details=False,matrix_labels=True,independent_evidence=False,
                dual_scenes=False,detect_hz=2,recent_missing=True)
            replay_manual(args)
            out=folder/'out';manifest=json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['decoded_frames'],240)
            self.assertEqual(manifest['detected_frames'],24)
            self.assertTrue(manifest['options']['pixel_fallback'])
            self.assertGreater(manifest['elapsed_total_s'],0)
            self.assertGreaterEqual(manifest['summary_frame']['time_s'],5)
            records=[json.loads(s) for s in (out/'blocks.jsonl').read_text().splitlines()]
            self.assertEqual([r['time_s'] for r in records],list(np.arange(0,12,.5)))
            errors=[r['time_s'] for r in records if r['blocks'][0]['status']=='error']
            self.assertEqual(errors[0],10)
            args.pixel_fallback=False
            args.summary_skip=0
            args.out=str(folder/'disabled')
            replay_manual(args)
            disabled=json.loads((folder/'disabled'/'manifest.json').read_text())
            self.assertFalse(disabled['options']['pixel_fallback'])
            self.assertEqual(disabled['summary_skip_s'],0)

    def test_recent_missing(self):
        from water_manual_recent import RecentMissing
        spec=ManualSpec(hold_s=10)
        recent=RecentMissing(spec)
        for t in range(11):
            seconds,confirmed=recent.update('gap',1,t)
        self.assertTrue(confirmed)
        self.assertEqual(seconds,10)
        for t in range(11,25):
            seconds,confirmed=recent.update('gap',-1,t)
        self.assertFalse(confirmed)
        self.assertEqual(seconds,0)
        for t in range(31):
            _,confirmed=recent.update('noisy',-1 if t%5==0 else 1,t)
            self.assertFalse(confirmed)
        recent.update('gap',0,25)
        self.assertEqual(recent.update('gap',1,26)[0],0)

    def test_weak_rgb_regions(self):
        from water_manual_regions import ColourRegions
        from water_manual_weak import weak_region
        spec=ManualSpec(min_stations=2,min_pixels=10,region_min_pixels=10)
        centers=np.array([[[i*2,j*10] for j in range(5)] for i in range(5)])
        pixels=[[(np.array([j*10]),np.array([i*2])) for j in range(5)] for i in range(5)]
        geometry=dict(centers=centers,pixels=pixels)
        regions=ColourRegions(geometry,spec)
        f=np.zeros((5,5,6));f[:,:,3]=100;f[:,:,4]=40
        valid=np.ones((5,5),bool)
        frame=np.full((50,15,3),(60,65,90),np.uint8)
        self.assertTrue(weak_region(frame,geometry,f,valid,0,regions,spec,2)['present'])
        gray=np.full_like(frame,70)
        self.assertFalse(weak_region(gray,geometry,f,valid,0,regions,spec,2)['present'])
        self.assertTrue(weak_region(gray,geometry,f,valid,1,regions,spec,2)['present'])
        self.assertFalse(weak_region(gray,geometry,f,valid*False,1,regions,spec,2)['observable'])

    def test_optional_region_refinement(self):
        from water_manual_regions import ColourRegions
        from water_manual_support import colour_support
        centers=np.array([[[i*2,j*10] for j in range(5)] for i in range(5)])
        pixels=[[(np.array([j*10]),np.array([i*2])) for j in range(5)] for i in range(5)]
        geometry=dict(centers=centers,pixels=pixels)
        spec=ManualSpec(min_stations=2,min_pixels=10,region_min_pixels=10)
        f=np.zeros((5,5,6));f[:,:,3]=120
        colour=np.zeros((5,5,2));colour[::2,:,0]=1
        valid=np.ones((5,5),bool)
        old,_=ColourRegions(geometry,spec).assess(f,colour,valid,0,2)
        self.assertFalse(old['present'])
        spec.region_bridge=True
        new,_=ColourRegions(geometry,spec).assess(f,colour,valid,0,2)
        self.assertTrue(new['present'])
        self.assertEqual(sum(c['points'] for c in new['components'][0]),15)
        weak=colour*.25
        self.assertTrue(colour_support(weak,valid,f,valid,0,spec)['ambiguous'])
        self.assertFalse(colour_support(weak*0,valid,f,valid,0,spec)['ambiguous'])
        reference=f.copy();reference[:,:,:2]=weak
        self.assertTrue(colour_support(weak*0,valid,reference,valid,0,spec)['ambiguous'])

    def test_optional_validity_rules(self):
        from water_manual_validity import ValidityRules
        spec=ManualSpec(validity_optimize=True)
        rules=ValidityRules(spec)
        f=np.zeros((5,5,6),np.float32);f[:,:,3]=120;f[:,:,5]=.6
        types=np.ones((5,5),np.int8)
        self.assertTrue(rules.point_mask(f).all())
        self.assertTrue(rules.reference_info(f,types,120)['supported'])
        self.assertFalse(rules.reference_info(f[:4],types[:4],77)['supported'])
        f[:,:,5]=.56
        self.assertFalse(rules.point_mask(f).any())

    def test_independent_evidence_and_scenes(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=DEPLOY.parent/'tmp') as td:
            folder=Path(td)
            shapes=[dict(label='row0',shape_type='polygon',points=[[0,20],[319,20],[319,120],[0,120]])]
            frame=np.full((160,320,3),100,np.uint8)
            for i,x in enumerate([10,110,210]):
                shapes.append(dict(label=f'row0block{2*i+1:02}',shape_type='polygon',points=[[x,20],[x+80,20],[x+80,120],[x,120]]))
                frame[20:121,x:x+81]=(0,0,220)
            label=folder/'labels.json'
            label.write_text(json.dumps(dict(imageWidth=320,imageHeight=160,shapes=shapes,imagePath='ref.png')))
            cv.imwrite(str(folder/'ref.png'),frame)
            video=folder/'clip.avi'
            writer=cv.VideoWriter(str(video),cv.VideoWriter_fourcc(*'MJPG'),4,(320,160))
            self.assertTrue(writer.isOpened())
            writer.write(frame)
            changed=frame.copy();changed[20:121,10:91]=100
            for _ in range(3):writer.write(changed)
            writer.release()
            parameters=folder/'parameters.json';parameters.write_text(json.dumps(dict(hold_s=.3)))
            args=SimpleNamespace(manual_blocks=str(label),block_reference=None,manual_parameters=str(parameters),
                gray_white=False,light_colour=False,out=str(folder/'out'),camera='1749',video=str(video),fps=None,
                stop_after=None,diagnostics=True,max_pages=1,matrix_every=5,at_s=[],overlay_video=False,
                abnormal_only=True,abnormal_details=True,matrix_labels=True,independent_evidence=True,dual_scenes=True)
            replay_manual(args)
            out=folder/'out';manifest=json.loads((out/'manifest.json').read_text())
            self.assertTrue(manifest['summary_short_video_fallback'])
            statuses={e['status'] for e in manifest['evidence'] if e['label']=='row0block01'}
            self.assertEqual(statuses,{'*','error'})
            self.assertLessEqual(len(manifest['snapshots']),1)
            self.assertTrue((out/'scene.png').is_file())
            body=(out/'index.html').read_text(encoding='utf-8')
            self.assertIn('ERROR／疑似专项诊断',body)
            self.assertNotIn('<h2>单帧矩阵</h2>',body)
            for event in manifest['evidence']:
                self.assertTrue((out/(event['page']+'.png')).is_file())
                page=(out/'pages'/(event['page']+'.html')).read_text(encoding='utf-8')
                self.assertIn('红',page)
                with np.load(out/'data'/(event['page']+'.npz')) as saved:
                    self.assertIn(event['label']+'_reference',saved.files)

    def test_gray_and_light_switches(self):
        with tempfile.TemporaryDirectory(dir=DEPLOY.parent/'tmp') as td:
            path=Path(td)/'labels.json'
            shapes=[dict(label='row0',shape_type='polygon',points=[[0,20],[319,20],[319,120],[0,120]])]
            for i,x in enumerate([10,110,210]):
                shapes.append(dict(label=f'row0block{2*i+2:02}',shape_type='polygon',points=[[x,20],[x+80,20],[x+80,120],[x,120]]))
            path.write_text(json.dumps(dict(imageWidth=320,imageHeight=160,shapes=shapes)))
            reference=np.full((160,320,3),200,np.uint8)
            dim=np.full_like(reference,130)
            old=ManualDetector(path,reference)
            self.assertEqual(float(old.sample(dim)[0][:,:,1].max()),0)
            gray=ManualDetector(path,reference,ManualSpec(gray_white=True))
            self.assertEqual(float(gray.sample(dim)[0][:,:,1].min()),1)
            corrected=ManualDetector(path,reference,ManualSpec(light_colour=True))
            result=corrected.step(dim,0)[0]
            self.assertEqual(result['status'],'normal')
            self.assertEqual(float(result['features'][1].max()),0)
            self.assertEqual(float(result['colour_evidence'][1].min()),1)

    def test_fixed_block_change_and_unknown(self):
        with tempfile.TemporaryDirectory(dir=DEPLOY.parent/'tmp') as td:
            folder = Path(td)
            shapes = [dict(label='row0',shape_type='polygon',points=[[10,20],[310,20],[310,120],[10,120]])]
            frame = np.full((160,320,3),100,np.uint8)
            for i in range(3):
                x = 10+i*100
                shapes.append(dict(label=f'row0block{2*i+1:02}',shape_type='polygon',points=[[x,20],[x+80,20],[x+80,120],[x,120]]))
                frame[20:121,x:x+81] = (0,0,220)
            path = folder/'labels.json'
            path.write_text(json.dumps(dict(imageWidth=320,imageHeight=160,shapes=shapes)),encoding='utf-8')
            detector = ManualDetector(path,frame,ManualSpec(hold_s=1))
            normal = detector.step(frame,0)
            self.assertEqual(normal[0]['status'],'normal')
            changed = frame.copy()
            changed[20:121,10:91] = 100
            first = detector.step(changed,.1)
            self.assertEqual(first[0]['status'],'*')
            last = detector.step(changed,1.1)
            self.assertEqual(last[0]['status'],'error')
            self.assertEqual(last[1]['status'],'normal')
            self.assertEqual(last[0]['features'].shape,(6,5*last[0]['n']))
            self.assertEqual(last[0]['anomaly'].shape,(6,last[0]['n']))
            before = last[0]['accum']
            dark = detector.step(np.zeros_like(frame),1.2)
            self.assertEqual(dark[0]['status'],'?')
            self.assertEqual(dark[0]['accum'],before)
            self.assertTrue(np.all(dark[0]['anomaly']==-1))
            (folder/'data').mkdir();(folder/'pages').mkdir()
            matrix_page(folder,1,.1,first)
            with np.load(folder/'data/frame_00000001.npz') as arrays:
                self.assertEqual(arrays['row0block01_features'].shape,first[0]['features'].shape)
            page = (folder/'pages/frame_00000001.html').read_text(encoding='utf-8')
            self.assertIn('row0block01',page)
            self.assertIn('station',page)

    def test_global_brightness_and_tiny_block(self):
        with tempfile.TemporaryDirectory(dir=DEPLOY.parent/'tmp') as td:
            path = Path(td)/'labels.json'
            frame = np.full((160,320,3),(0,0,220),np.uint8)
            shapes = [dict(label='row0',shape_type='polygon',points=[[0,20],[319,20],[319,120],[0,120]])]
            for i,(x,width) in enumerate([(20,4),(80,80),(180,80)]):
                shapes.append(dict(label=f'row0block{2*i+1:02}',shape_type='polygon',points=[[x,20],[x+width,20],[x+width,120],[x,120]]))
            path.write_text(json.dumps(dict(imageWidth=320,imageHeight=160,shapes=shapes)))
            detector = ManualDetector(path,frame)
            out = detector.step((frame.astype(float)*.8).astype(np.uint8),0)
            self.assertEqual(out[0]['status'],'?')
            self.assertEqual(out[1]['status'],'normal')
            self.assertAlmostEqual(out[1]['light']['0']['v_factor'],.8,places=2)


if __name__ == '__main__':
    unittest.main()
