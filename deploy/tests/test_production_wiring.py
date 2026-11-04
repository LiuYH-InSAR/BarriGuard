"""Production wiring tests: no RTSP, HTTP, downloads or real callbacks."""
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import copy
import sys
import tempfile
import unittest
import cv2 as cv
import numpy as np
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from server.contracts import CameraSpec,OffsetSpec,LampDetectorSpec,NightLampSpec,WaterGapSpec,AlgorithmBinding,ScheduleState
from server.config_validate import parse_runtime,ConfigError
from server.worker import CameraWorker
from server.algo import Report
from camera_motion.runtime import CameraOffsetRuntime
from camera_motion.service import OffsetModule
from camera_motion.params import MotionParams
from camera_motion.vision import Alignment
from camera_motion.contracts import MotionResult
from night_lamp.production import LampAdapter
from water_barrier.yolo_pipeline import YoloWaterAlgorithm
from water_barrier.yolo_params import PARAMS

ROOT=Path(__file__).resolve().parents[2]

class Clock:
    def __init__(self):self.t=1000000.
    def wall(self):return datetime.fromtimestamp(self.t,timezone.utc)
    def monotonic(self):return self.t-1000000.

class Vision:
    def __init__(self):self.last_matches=None;self.calls=0
    def quality(self,frame,*args,**kwargs):return dict(ok=True,light='day')
    def align(self,a,b,*args):
        self.calls+=1
        moved=int(a[0,0,0])!=int(b[0,0,0])
        matrix=np.eye(3);matrix[0,2]=5 if moved else 0
        points=np.float32([(x,y) for x in (10,30,50,70) for y in (10,30,50,70)])
        self.last_matches=(a,b,points,points+[matrix[0,2],0])
        return Alignment(True,matrix,.1 if moved else 0,dict(inliers=16,inlier_ratio=1.,method='orb'))

class ProductionWiring(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.clock=Clock();self.state=ScheduleState.DAY
        self.scheduler=SimpleNamespace(state=lambda:self.state,is_active=lambda c,n:self.state==ScheduleState.NIGHT if n=='night_lamp' else self.state==ScheduleState.DAY)
        self.frame=np.full((100,100,3),10,np.uint8)
        self.reporter=Mock();self.vision=Vision()
        self.detector=SimpleNamespace(detect=Mock(return_value=[[10,10,25,25],[50,10,65,25]]))
        self.detector.predict=lambda frame,regions,imgsz,confidence:[list(b)+[.9] for b in self.detector.detect(frame)]
    def owner(self,camera_id,spec,clock):
        params=replace(MotionParams(),reference=replace(MotionParams().reference,refresh_s=spec.refresh_s),
            storage=replace(MotionParams().storage,root=self.root/'motion',retention_s=spec.reference_ttl_s))
        return CameraOffsetRuntime(camera_id,spec,clock,OffsetModule(camera_id,params,vision=self.vision))
    def worker(self,water=True,lamp=True):
        bindings={}
        if water:bindings['water_gap']=AlgorithmBinding('day',WaterGapSpec(backend='yolo'))
        if lamp:bindings['night_lamp']=AlgorithmBinding('night',NightLampSpec(
            detector=LampDetectorSpec(root=str(self.root/'lamps'))))
        camera=CameraSpec('cam','camera','rtsp://example.test/cam',algorithms=bindings,
            offset=OffsetSpec(root=str(self.root/'motion'),period_s=1.,stable_s=1.))
        worker=CameraWorker(camera,self.scheduler,self.reporter,clock=self.clock,
            offset_factory=self.owner,lamp_detector_factory=lambda spec:self.detector)
        params=replace(PARAMS,storage=replace(PARAMS.storage,root=self.root/'water'),
            debug=replace(PARAMS.debug,enabled=False))
        def create(name,shape,calib,camera_id,**kwargs):
            return YoloWaterAlgorithm(shape,calib,camera_id,params=params,
                detector=SimpleNamespace(detect=lambda frame:[list(b[:4])+[.9] for b in self.detector.detect(frame)]),**kwargs)
        with patch('server.worker.registry.create',side_effect=create),patch('water_barrier.yolo_pipeline.logger'):
            worker._build(self.frame.shape)
        return worker
    def process(self,worker,advance=1,frame=None):
        self.clock.t+=advance
        worker._process_shared(self.frame if frame is None else frame,self.clock.wall(),self.clock.monotonic(),str(self.clock.t),self.clock.monotonic())
    def test_shared_lamp_only(self):
        self.vision.align=lambda *args:Alignment(True,np.eye(3),0,dict(inliers=16,inlier_ratio=1.))
        w=self.worker(water=False)
        self.assertIsInstance(w._lamp_adapter,LampAdapter)
        self.assertIs(w._lamp_adapter.offset,w._offset_runtime.offset)
        w.start_night();self.assertIs(w._night['night_lamp'],w._lamp_adapter)
        w._lamp_adapter.controller.quality=SimpleNamespace(observe=lambda *a:dict(fresh=True,continuous=True,clear=True))
        w._lamp_adapter.controller.pipeline.detector.vision=self.vision
        self.frame=np.random.default_rng(5).integers(20,150,self.frame.shape,dtype=np.uint8)
        self.frame[0,0]=10
        self.process(w)
        self.frame[-1,-1]=99
        self.process(w)
        self.assertTrue(self.detector.detect.called)
    def test_water_only_has_common_owner(self):
        w=self.worker(lamp=False);self.process(w);self.process(w)
        algo=w._day_runners['water_gap'].algo
        self.assertIs(algo._refs.library,w._offset_runtime.offset.references)
        algo._align(self.frame,1,self.clock.t)
        self.assertTrue(algo._alignment_ok)

    def test_startup_alignment_uses_fast_confirmation_and_consumer_can_request_check(self):
        owner=self.owner('cam',OffsetSpec(period_s=30,fast_s=1),self.clock)
        owner.process(self.frame,100,0,'start')
        self.assertEqual(owner.offset.latest.status,'unavailable')
        owner.process(self.frame,101,1,'confirm')
        self.assertEqual(owner.offset.latest.status,'aligned')
        owner.request_check(2)
        owner.process(self.frame,103,3,'requested')
        self.assertEqual(owner.offset.latest.frame_id,'requested')
    def test_duplicates_do_not_advance_scene(self):
        w=self.worker();self.process(w)
        version=w._offset_runtime.offset._version
        w._process_shared(self.frame,self.clock.wall(),self.clock.monotonic(),str(self.clock.t))
        self.assertEqual(version,w._offset_runtime.offset._version)
    def test_no_private_day_cache(self):
        w=self.worker();self.process(w)
        self.assertFalse(hasattr(w,'_day_ref'))
        image,origin=w._select_day_base()
        self.assertEqual(origin,'shared_day_reference')
        self.assertIsNotNone(image)
    def test_lamp_independent_tick_freeze_finalize_next_night(self):
        w=self.worker();w.start_night();ad=w._lamp_adapter
        self.process(w);self.process(w)
        self.state=ScheduleState.NIGHT
        w._tick_night(self.clock.wall())
        self.assertIsNotNone(ad.controller.monitor.bucket)
        w.freeze_night();self.assertTrue(ad._frozen)
        w.finalize_night(self.clock.wall())
        self.reporter.submit.assert_called_once()
        report=self.reporter.submit.call_args.args[0]
        self.assertEqual(report.report_type,'lamp_status')
        self.assertNotIn('night_lamp',w._night)
        w.start_night();self.assertIs(w._night['night_lamp'],ad);self.assertFalse(ad._frozen)
    def test_live_adapters_restore_business_without_resetting_common(self):
        w=self.worker();self.process(w);self.process(w)
        reference=w._offset_runtime.offset.references
        reference.view='authoritative'
        w._lamp_adapter.controller.monitor.reference='authoritative'
        w.flush_night()
        same=LampAdapter(w._offset_runtime,w.camera.algorithms['night_lamp'].spec,self.clock,self.detector)
        self.assertIs(same.offset.references,reference)
        self.assertEqual(reference.view,'authoritative')
    def test_reference_restore_from_shared_owner(self):
        w=self.worker();self.process(w);self.process(w);w.flush_night()
        owner=self.owner('cam',w.camera.offset,self.clock)
        self.assertEqual(owner.offset.references.view,w._offset_runtime.offset.references.view)
        self.assertTrue(owner.offset.references.appearances)
    def test_automatic_move_commits_once_and_keeps_library(self):
        owner=self.owner('cam',OffsetSpec(period_s=1,stable_s=1),self.clock)
        consumer=Mock();consumer.offset_feedback.return_value=None
        owner.attach('consumer',consumer)
        lib=owner.offset.references
        owner.process(self.frame,100,0,'start')
        owner.process(self.frame,101,1,'aligned')
        changed=np.full_like(self.frame,20)
        for i in range(2,14):owner.process(changed,100+i,i,str(i))
        self.assertEqual(len(owner.offset.commits),1)
        self.assertIs(owner.offset.references,lib)
        consumer.apply_shared_offset.assert_called_once()
        self.assertAlmostEqual(owner.offset.commits[-1].matrix[0][2],5,places=3)
        for i in range(14,17):owner.process(changed,100+i,i,str(i))
        consumer.apply_shared_offset.assert_called_once()
    def test_failed_consumer_retries_without_repeating_successful(self):
        owner=self.owner('cam',OffsetSpec(period_s=1,stable_s=1),self.clock)
        good=Mock();good.offset_feedback.return_value=None
        bad=Mock();bad.offset_feedback.return_value=None
        bad.apply_shared_offset.side_effect=[RuntimeError('temporary'),None]
        owner.attach('good',good);owner.attach('bad',bad)
        owner.process(self.frame,100,0,'start');owner.process(self.frame,101,1,'aligned')
        with self.assertLogs('camera_motion.runtime',level='ERROR'):
            for i in range(2,14):owner.process(np.full_like(self.frame,20),100+i,i,str(i))
        self.assertEqual(good.apply_shared_offset.call_count,1)
        self.assertEqual(bad.apply_shared_offset.call_count,2)
        owner.process(np.full_like(self.frame,20),120,20,'last')
        self.assertIsNotNone(owner.alignment()[1])

    def test_lamp_failure_does_not_invalidate_water_offset(self):
        self.vision.align=lambda *args:Alignment(True,np.eye(3),0,dict(inliers=16,inlier_ratio=1.))
        w=self.worker()
        w._lamp_adapter.controller.quality=SimpleNamespace(observe=lambda *a:dict(fresh=True,continuous=True,clear=True))
        self.detector.detect.side_effect=RuntimeError('inference failed')
        w._lamp_adapter.controller.pipeline.detector.vision=self.vision
        self.frame=np.random.default_rng(5).integers(20,150,self.frame.shape,dtype=np.uint8)
        self.frame[0,0]=10
        with patch('server.worker.logger'):
            self.process(w)
            self.frame[-1,-1]=99
            self.process(w)
        self.assertEqual(w._offset_runtime.offset.latest.status,'aligned')
        self.assertIn('inference failed',w._lamp_adapter.error)
        water=w._day_runners['water_gap'].algo
        water._align(self.frame,2,self.clock.t)
        self.assertTrue(water._alignment_ok)

    def test_actual_water_baseline_with_shared_library(self):
        w=self.worker(lamp=False);algo=w._day_runners['water_gap'].algo
        algo._quality=lambda frame,stamp:dict(ok=True,light='day',luma=180,upper_luma=180)
        algo._daylight.candidate_for=lambda *args:'day'
        for i in range(8):
            frame=np.full_like(self.frame,10);frame[99,99]=i+30
            self.process(w,advance=30,frame=frame)
            algo.step(frame,self.clock.monotonic())
        self.assertTrue(algo._refs.boxes)
        self.assertEqual(algo._target_reference,w._offset_runtime.offset.references.view)
        self.assertIs(algo._refs.library,w._offset_runtime.offset.references)

    def test_water_migration_and_restart_preserve_identity(self):
        w=self.worker(lamp=False);self.process(w);self.process(w)
        algo=w._day_runners['water_gap'].algo
        algo._refs.boxes=[[10,10,25,25]];algo._refs.identities=['B1'];algo._refs.rows=[[0]]
        algo._target_reference=algo._refs.view
        old=algo._refs.view
        r=MotionResult('cam','task','candidate',self.clock.t,old,'new',99,'committed',
                      [[1,0,5],[0,1,0],[0,0,1]],(100,100),1.)
        algo._refs.library.view='new'
        self.assertTrue(algo.apply_shared_offset(r))
        self.assertEqual(algo._refs.boxes,[[15.,10.,30.,25.]])
        self.assertEqual(algo._refs.identities,['B1'])
        self.assertFalse(algo.apply_shared_offset(r))

    def test_unmapped_move_never_keeps_old_water_coordinates(self):
        w=self.worker(lamp=False);self.process(w);self.process(w)
        algo=w._day_runners['water_gap'].algo
        algo._refs.boxes=[[10,10,25,25]];algo._refs.identities=['B1']
        algo._target_reference=algo._refs.view;old=algo._refs.view
        r=MotionResult('cam','task','candidate',self.clock.t,old,'new',99,'committed',None,(100,100),0.)
        algo._refs.library.view='new'
        algo.apply_shared_offset(r)
        self.assertEqual(algo._refs.boxes,[]);self.assertEqual(algo._refs.identities,[])
        self.assertFalse(algo._alignment_ok)

    def test_source_sample_is_atomic_and_does_not_modify_reader_pixels(self):
        from server.source import Reader
        reader=Reader('rtsp://example.test/cam')
        with reader._lock:
            reader._frame=self.frame;reader._seq=5;reader._last_frame_mono=123.
        image,seq,stamp=reader.sample()
        image[:]=99
        self.assertEqual(seq,5);self.assertEqual(stamp,123.)
        self.assertTrue(np.all(reader._frame==10))

    def test_worker_loop_processes_each_decoded_frame_once(self):
        from server.algo import AlgoResult
        w=self.worker();w._built=True
        runner=SimpleNamespace(spec=WaterGapSpec(backend='yolo'),
            on_frame=Mock(return_value=AlgoResult(events=[],annots=[],debug={})))
        w._day_runners={'water_gap':runner};w._process_shared=Mock()
        samples=iter([(self.frame,1,0.),(self.frame,1,0.),(self.frame,1,0.),(self.frame,2,.2)])
        def sample():
            try:return next(samples)
            except StopIteration:w._stop.set();return None,None,None
        w._reader=SimpleNamespace(sample=sample)
        w._loop()
        self.assertEqual(runner.on_frame.call_count,2)
        self.assertEqual(w._process_shared.call_count,2)
        self.assertEqual(w._n,2)

    def test_custom_state_roots_are_protected_from_old_sweeper(self):
        from server.maintenance import Sweeper
        import os,time
        root=self.root/'custom';root.mkdir()
        protected=root/'current.json';protected.write_text('{}')
        other=self.root/'old.json';other.write_text('{}')
        past=time.time()-3*86400
        os.utime(protected,(past,past));os.utime(other,(past,past))
        sweeper=Sweeper([self.root],48);sweeper.protect([root]);sweeper.sweep()
        self.assertTrue(protected.exists());self.assertFalse(other.exists())


    def test_crash_after_common_commit_replays_water_targets_once(self):
        w=self.worker(lamp=False);self.process(w);self.process(w)
        algo=w._day_runners['water_gap'].algo
        algo._refs.boxes=[[10,10,25,25]];algo._refs.identities=['B1'];algo._refs.rows=[[0]]
        algo._refs.baseline_created=self.clock.t;algo._target_reference=algo._refs.view
        algo._checkpoint(self.clock.t)
        owner=w._offset_runtime
        owner.offset.begin_migration(self.frame,self.clock.t+1,'candidate')
        matrix=np.eye(3);matrix[0,2]=5
        owner.offset.pending=replace(owner.offset.pending,matrix=matrix.tolist(),details={'accepted':True})
        result=owner.offset.commit();owner.save(self.clock.t+1)
        # Simulate death before the old water checkpoint sees this commit.
        self.clock.t+=2
        fresh_owner=self.owner('cam',w.camera.offset,self.clock)
        params=algo._p
        with patch('water_barrier.yolo_pipeline.logger'):
            restored=YoloWaterAlgorithm(self.frame.shape,{'backend':'yolo'},'cam',params=params,
                detector=self.detector,offset_module=fresh_owner.offset,wall_clock=lambda:self.clock.t)
        self.assertEqual(restored.offset_reference,result.old_reference)
        fresh_owner.attach('water_gap',restored)
        self.assertEqual(restored._refs.identities,['B1'])
        self.assertEqual(restored._refs.boxes,[[15.,10.,30.,25.]])
        fresh_owner.save()
        again_owner=self.owner('cam',w.camera.offset,self.clock)
        with patch('water_barrier.yolo_pipeline.logger'):
            again=YoloWaterAlgorithm(self.frame.shape,{'backend':'yolo'},'cam',params=params,
                detector=self.detector,offset_module=again_owner.offset,wall_clock=lambda:self.clock.t)
        again_owner.attach('water_gap',again)
        self.assertEqual(again._refs.boxes,[[15.,10.,30.,25.]])

    def test_import_previous_private_water_reference(self):
        params=replace(PARAMS,storage=replace(PARAMS.storage,root=self.root/'water'),debug=replace(PARAMS.debug,enabled=False))
        with patch('water_barrier.yolo_pipeline.logger'):
            old=YoloWaterAlgorithm(self.frame.shape,{'backend':'yolo'},'cam',params=params,
                detector=self.detector,wall_clock=lambda:self.clock.t)
        old._refs.start_view(self.frame,self.clock.t,dict(ok=True,light='day'))
        old._refs.boxes=[[10,10,25,25]];old._refs.identities=['B1'];old._refs.rows=[[0]]
        old._refs.baseline_created=self.clock.t;old._checkpoint(self.clock.t)
        owner=self.owner('cam',OffsetSpec(period_s=1,stable_s=1),self.clock)
        with patch('water_barrier.yolo_pipeline.logger'):
            new=YoloWaterAlgorithm(self.frame.shape,{'backend':'yolo'},'cam',params=params,
                detector=self.detector,offset_module=owner.offset,wall_clock=lambda:self.clock.t)
        self.assertIs(new._refs.library,owner.offset.references)
        self.assertEqual(new._refs.identities,['B1'])
        entry=owner.offset.references.appearances[0]
        self.assertTrue((owner.offset.store.root/entry['image']).is_file())

    def test_new_consumer_does_not_replay_old_transforms(self):
        owner=self.owner('cam',OffsetSpec(period_s=1,stable_s=1),self.clock)
        owner.process(self.frame,100,0,'start');owner.process(self.frame,101,1,'aligned')
        for i in range(2,14):owner.process(np.full_like(self.frame,20),100+i,i,str(i))
        consumer=SimpleNamespace(offset_reference=owner.offset.references.view,apply_shared_offset=Mock())
        owner.attach('new',consumer)
        self.assertEqual(consumer.apply_shared_offset.call_count,1)
        # Consumer sees already-current reference and its own method is responsible for no-op.
        owner.process(np.full_like(self.frame,20),120,20,'after')
        self.assertEqual(consumer.apply_shared_offset.call_count,1)

    def test_night_reference_does_not_replace_day_category(self):
        owner=self.owner('cam',OffsetSpec(period_s=1,stable_s=1,refresh_s=1),self.clock)
        owner.process(self.frame,100,0,'start');owner.process(self.frame,101,1,'day')
        owner.process(self.frame,102,2,'night',light_hint='night')
        categories={r['light'] for r in owner.offset.references.appearances}
        self.assertIn('day',categories);self.assertIn('night',categories)


    def test_restore_never_reuses_old_frame_alignment(self):
        w=self.worker(lamp=False);self.process(w);self.process(w);w.flush_night()
        owner=self.owner('cam',w.camera.offset,self.clock)
        self.assertEqual(owner.offset.latest.status,'unavailable')
        self.assertIsNone(owner.offset.latest.matrix)
        self.assertIsNone(owner.alignment()[1])

    def test_expired_reference_does_not_delete_active_identity(self):
        w=self.worker(lamp=False);self.process(w);self.process(w);w.flush_night()
        view=w._offset_runtime.offset.references.view
        self.clock.t+=4*86400
        owner=self.owner('cam',w.camera.offset,self.clock)
        self.assertEqual(owner.offset.references.view,view)
        owner.process(self.frame,self.clock.t,100,'fresh')
        self.assertEqual(owner.offset.latest.status,'unavailable')
        self.assertEqual(owner.offset.latest.details['reason'],'reference_expired')




    def test_actual_runtime_constructs_isolated_owners_without_starting_streams(self):
        from server.worker import Runtime
        data=yaml.safe_load((ROOT/'deploy/config.yaml').read_text(encoding='utf-8'))
        data['callback']['enabled']=False
        for camera in data['cameras']:camera['algorithms']['night_lamp']['enabled']=False
        data['runtime']['output_dir']=str(self.root/'output')
        for camera in data['cameras']:camera['offset']={'root':str(self.root/'motion')}
        cfg=parse_runtime(data,ROOT/'deploy')
        runtime=Runtime(cfg,clock=self.clock,reporter=self.reporter)
        owners=[w._offset_runtime.offset for w in runtime.workers.values()]
        self.assertEqual(len(owners),2)
        self.assertIsNot(owners[0],owners[1])
        self.assertIsNot(owners[0].references,owners[1].references)
        self.assertTrue(all(not w._reader._thread.is_alive() for w in runtime.workers.values()))
        self.assertEqual(len(runtime.sweeper._protected),2)

    def test_morning_report_never_reuses_previous_night_normal(self):
        from night_lamp.controller import LampController
        from datetime import timedelta
        owner=self.owner('cam',OffsetSpec(),self.clock)
        owner.offset.inspect(self.frame,self.clock.t,'init')
        controller=LampController(owner.offset,self.detector,self.root/'lamps',managed=True)
        key=controller.monitor.add_known([10,10,25,25],self.clock.t)
        previous=self.clock.t-2*86400
        controller.monitor.lamps[key].history=[dict(bucket=previous,ended=previous+300,
            valid_s=300,pulses=15,status='normal',reference=controller.monitor.reference,lamp_id=key)]
        report,path=controller.finalize(self.clock.t)
        self.assertEqual(report['lamps'][0]['status'],'unobservable')


    def test_single_lamp_path_shares_actual_water_library(self):
        w=self.worker();w.start_night()
        water=w._day_runners['water_gap'].algo;lamp=w._night['night_lamp']
        self.assertIsInstance(lamp,LampAdapter)
        self.assertIs(water._offset_module,lamp.offset)
        self.assertIs(water._refs.library,lamp.offset.references)
        self.process(w);self.process(w)
        calls=self.vision.calls
        water._align(self.frame,2,self.clock.t)
        self.assertEqual(self.vision.calls,calls)
        self.assertTrue(water._alignment_ok)

    def test_no_retired_lamp_modules_in_production(self):
        import importlib.util
        for name in ('session','adapter','periodicity','lamp_checks','persist','local_recording','tools',
                     'pulse','evidence','disturbance','stutter','single_pixel','report','publish_night','replay_night'):
            with self.subTest(module=name):
                self.assertIsNone(importlib.util.find_spec('night_lamp.'+name))


    def test_observation_config_reaches_the_only_adapter(self):
        from night_lamp.settings import LampParams
        owner=self.owner('cam',OffsetSpec(),self.clock)
        spec=NightLampSpec(detector=LampDetectorSpec(root=str(self.root/'lamps')),
            observation=replace(LampParams(),normal_pulses=30))
        adapter=LampAdapter(owner,spec,self.clock,self.detector)
        self.assertEqual(adapter.controller.monitor.params.normal_pulses,30)
        self.assertIs(adapter.offset,owner.offset)

    def test_enabled_without_weights_fails_before_starting_reader(self):
        camera=CameraSpec('cam','camera','rtsp://example.test/cam',
            algorithms={'night_lamp':AlgorithmBinding('night',NightLampSpec(
                detector=LampDetectorSpec(weights=str(self.root/'missing.pt'))))})
        with patch('server.worker.Reader') as reader:
            with self.assertRaises(ValueError):
                CameraWorker(camera,self.scheduler,self.reporter,clock=self.clock,offset_factory=self.owner)
            reader.assert_not_called()


class LampConfig(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.data=yaml.safe_load((ROOT/'deploy/config.yaml').read_text(encoding='utf-8'))
        self.data['callback']['enabled']=False
    def parse(self):return parse_runtime(self.data,ROOT/'deploy')
    def enable(self):
        weight=self.root/'local.pt';weight.write_bytes(b'fixture')
        lamp=self.data['cameras'][0]['algorithms']['night_lamp']
        lamp['enabled']=True;lamp['detector']['weights']=str(weight)
        return lamp
    def test_production_template_enables_both_lamp_adapters(self):
        cfg=self.parse()
        self.assertTrue(all(c.offset.enabled for c in cfg.cameras))
        self.assertTrue(all('night_lamp' in c.algorithms for c in cfg.cameras))
    def test_requires_local_weights_when_enabled(self):
        lamp=self.enable();lamp['detector']['weights']=str(self.root/'missing.pt')
        with self.assertRaises(ConfigError):self.parse()
    def test_sole_scheme_has_no_backend_or_poly(self):
        self.enable();spec=self.parse().cameras[0].algorithms['night_lamp'].spec
        self.assertFalse(hasattr(spec,'backend'));self.assertFalse(hasattr(spec,'lamp_checks'))
        self.assertEqual(spec.observation.normal_pulses,15)
    def test_rejects_old_options_even_if_disabled(self):
        for key,value in [('backend','legacy'),('tracked',{}),('lamp_checks',{}),('periodicity',{}),('lamp_roi',{})]:
            data=copy.deepcopy(self.data)
            data['cameras'][0]['algorithms']['night_lamp']['enabled']=False
            data['cameras'][0]['algorithms']['night_lamp'][key]=value
            with self.subTest(key=key):
                with self.assertRaises(ConfigError):parse_runtime(data,ROOT/'deploy')
    def test_cannot_disable_shared_offset(self):
        self.enable();self.data['cameras'][0]['offset']={'enabled':False}
        with self.assertRaises(ConfigError):self.parse()
    def test_invalid_detector_options(self):
        for opts in ({'typo':1},{'confidence':float('nan')},{'detect_interval_s':1.}):
            self.enable();data=copy.deepcopy(self.data)
            data['cameras'][0]['algorithms']['night_lamp']['detector'].update(opts)
            with self.subTest(options=opts):
                with self.assertRaises(ConfigError):parse_runtime(data,ROOT/'deploy')
    def test_invalid_observation_rules(self):
        lamp=self.enable();lamp['observation']['deadline_s']=1800.
        with self.assertRaises(ConfigError):self.parse()
    def test_invalid_offset(self):
        self.data['cameras'][0]['offset']={'stable_s':float('inf')}
        with self.assertRaises(ConfigError):self.parse()

    def test_observation_rejects_nonfinite_and_noninteger_counts(self):
        for key,value in [('on_delta',float('inf')),('normal_pulses',1.5),('candidate_ttl_s',-1.)]:
            self.enable();data=copy.deepcopy(self.data)
            data['cameras'][0]['algorithms']['night_lamp']['observation'][key]=value
            with self.subTest(key=key):
                with self.assertRaises(ConfigError):parse_runtime(data,ROOT/'deploy')

if __name__=='__main__':unittest.main()
