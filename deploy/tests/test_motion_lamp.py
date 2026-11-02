"""Behavioral regression and weight-independent acceptance cases."""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import importlib.util
import json
import sys
import tempfile
import unittest
import cv2 as cv
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from camera_motion.params import MotionParams
from camera_motion.vision import Vision, Alignment
from camera_motion.state import CameraState
from camera_motion.references import ReferenceLibrary
from camera_motion.contracts import MotionResult, Feedback
from camera_motion.service import OffsetModule
from camera_motion.store import Store
from night_lamp.tracked import LampMonitor, LampParams, PulseCounter, Candidate
from night_lamp.archive import LampArchive, RETENTION_S
from night_lamp.quality import ObservationQuality, StreamQuality
from water_barrier.yolo_params import PARAMS
from water_barrier.yolo_reference import References

ROOT=Path(__file__).resolve().parents[2]
BACKUP=ROOT/'tmp/backups/before_lamp_refactor_20261028_115015/snapshot/deploy/app/water_barrier'


def baseline(name):
    spec=importlib.util.spec_from_file_location('water_barrier._baseline_'+name,BACKUP/(name+'.py'))
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod


class MemoryStore:
    def __init__(self):self.entries={};self.images={};self.index=0
    def record(self,group,stamp,data,image=None):
        self.index+=1;key=str(self.index);data=dict(data)
        if image is not None:data['image']=key;self.images[key]=image.copy()
        self.entries[key]=data;return key
    def read_record(self,key):return self.entries[key]
    def read_image(self,key):return self.images[key]


class FixedVision:
    def __init__(self):self.last_matches=None
    def quality(self,frame):return dict(ok=True,light='day')
    def align(self,old,new,*args):
        return Alignment(True,np.eye(3),0.,dict(inliers=40,inlier_ratio=1.))


class ExtractionRegression(unittest.TestCase):
    @unittest.skipUnless(BACKUP.is_dir(), 'Local pre-change snapshot unavailable')
    def test_registration_matches_backup(self):
        old=baseline('yolo_vision').Vision(PARAMS);new=Vision(PARAMS)
        rng=np.random.default_rng(8)
        image=rng.integers(0,256,(480,720,3),dtype=np.uint8)
        shifted=cv.warpAffine(image,np.float32([[1,0,15],[0,1,-7]]),(720,480))
        a,b=old.align(image,shifted),new.align(image,shifted)
        self.assertTrue(a.valid);self.assertEqual(a.details,b.details)
        np.testing.assert_array_equal(a.matrix,b.matrix)
        self.assertEqual(old.quality(image),new.quality(image))

    @unittest.skipUnless(BACKUP.is_dir(), 'Local pre-change snapshot unavailable')
    def test_motion_state_matches_backup(self):
        old=baseline('yolo_camera').CameraState(PARAMS);new=CameraState(PARAMS)
        for displacement in [0.001]*12+[.04]*8+[.001]*8:
            alignment=Alignment(True,np.eye(3),displacement,dict(method='orb'))
            self.assertEqual(old.check(alignment),new.check(alignment))
            self.assertEqual(old.state,new.state)
            self.assertEqual(old.tolerance,new.tolerance)

    @unittest.skipUnless(BACKUP.is_dir(), 'Local pre-change snapshot unavailable')
    def test_reference_lifecycle_matches_backup(self):
        a=baseline('yolo_reference').References(PARAMS,MemoryStore(),FixedVision())
        b=References(PARAMS,MemoryStore(),FixedVision())
        frame=np.zeros((100,200,3),np.uint8);q=dict(ok=True,light='day')
        for refs in (a,b):
            refs.start_view(frame,100,q)
            for t in range(101,107):refs.collect([[10,10,30,30],[40,10,60,30]],np.eye(3),t)
            refs.add_appearance(frame,800,q,np.eye(3))
        self.assertEqual(a.dump(),b.dump())
        self.assertEqual(a.refresh_due(q,1400),b.refresh_due(q,1400))
        self.assertEqual(a.expired(900),b.expired(900))
        np.testing.assert_array_equal(a.align(frame).matrix,b.align(frame).matrix)

    def test_library_shared_without_target_dependency(self):
        lib=ReferenceLibrary(MotionParams(),MemoryStore(),FixedVision())
        refs=References(PARAMS,MemoryStore(),FixedVision(),library=lib)
        refs.start_view(np.zeros((20,20,3),np.uint8),10,dict(light='day',ok=True))
        self.assertIs(refs.appearances,lib.appearances)
        lib.view='shared';self.assertEqual(refs.view,'shared')


class PulseRules(unittest.TestCase):
    def setUp(self):self.cfg=LampParams()
    def test_complete_pulses_not_frames(self):
        c=PulseCounter()
        for i,x in enumerate([0,0,40,40,40,0,0,40,0]):c.sample(i*.1,x,True,self.cfg)
        self.assertEqual(c.count,2)
    def test_starts_bright_is_not_complete(self):
        c=PulseCounter()
        for i,x in enumerate([40,40,0]):c.sample(i*.1,x,True,self.cfg)
        self.assertEqual(c.count,0)
    def test_gap_does_not_join_pulse_or_count_duration(self):
        c=PulseCounter();c.sample(0,0,True,self.cfg);c.sample(.1,40,True,self.cfg)
        c.sample(120,0,True,self.cfg)
        self.assertEqual(c.count,0);self.assertAlmostEqual(c.valid_s,.1)
    def test_invalid_segment_pauses_time(self):
        c=PulseCounter()
        for ts in [0,.1,.2]:c.sample(ts,0,True,self.cfg)
        c.sample(.3,0,False,self.cfg);c.sample(120,0,True,self.cfg);c.sample(120.1,0,True,self.cfg)
        self.assertAlmostEqual(c.valid_s,.3)
    def test_duplicates_do_not_count(self):
        c=PulseCounter();c.sample(0,0,True,self.cfg);c.sample(.1,40,True,self.cfg)
        c.sample(.1,0,True,self.cfg);self.assertEqual(c.count,0)
    def test_long_steady_light_not_flash(self):
        c=PulseCounter()
        for i in range(60):c.sample(i*.1,0 if i in (0,59) else 40,True,self.cfg)
        self.assertEqual(c.count,0)


class WindowRules(unittest.TestCase):
    def monitor(self):
        m=LampMonitor('cam','old');key=m.add_known([10,10,20,20],0);return m,key
    def test_deadline_insufficient_and_no_reopen(self):
        m,key=self.monitor();m.tick(0);m.tick(1500)
        self.assertEqual(m.results[-1]['lamps'][0]['status'],'insufficient_observation')
        m.tick(1501);self.assertIsNone(m.bucket)
        m.tick(1800);self.assertEqual(m.bucket,1800)
    def test_normal_fifteen_and_below(self):
        for count,status in [(15,'normal'),(14,'below_threshold'),(0,'no_flash')]:
            m,key=self.monitor();m.tick(0);m.observations[key]=PulseCounter(valid_s=300,count=count)
            m.tick(1500);self.assertEqual(m.results[-1]['lamps'][0]['status'],status)
    def test_six_windows_and_no_older_substitution(self):
        m,key=self.monitor()
        for i in range(6):
            m.tick(i*1800);m.observations[key]=PulseCounter(valid_s=300);m.tick(i*1800+1500)
        self.assertEqual(m.status(key,10800),'not_flashing')
        m.lamps[key].history[2]['status']='insufficient_observation'
        self.assertNotEqual(m.status(key,10800),'not_flashing')
    def test_recovery(self):
        m,key=self.monitor()
        for i in range(7):
            m.tick(i*1800);m.observations[key]=PulseCounter(valid_s=300,count=15 if i==6 else 0);m.tick(i*1800+1500)
        self.assertEqual(m.status(key,12600),'normal')
    def test_restored_partial_window_not_complete(self):
        m,key=self.monitor();m.tick(0);m.observations[key]=PulseCounter(valid_s=299)
        restored=LampMonitor.restore(json.loads(json.dumps(m.dump())))
        self.assertIsNone(restored.bucket);self.assertEqual(restored.observations,{})
    def test_late_candidate_has_own_clock(self):
        m,key=self.monitor();m.tick(0)
        other=m.add_known([40,10,50,20],120)
        frame=np.zeros((80,80,3),np.uint8)
        m.on_frame(frame,120,120,1,[],usable={key:True,other:True})
        self.assertEqual(m.observations[other].valid_s,0)
    def test_missing_detection_still_observable(self):
        m,key=self.monitor();frame=np.zeros((80,80,3),np.uint8)
        for i in range(3):m.on_frame(frame,i*.1,i*.1,i,[],usable={key:True})
        self.assertAlmostEqual(m.observations[key].valid_s,.2)
    def test_migration_idempotent_and_unmatched_unknown(self):
        m,key=self.monitor();m.candidates['c']=Candidate([15,12,25,22],0,0)
        other=m.add_known([40,10,50,20],0)
        r=MotionResult('cam','t','f',0,'old','new',1,'committed',[[1,0,5],[0,1,2],[0,0,1]],(80,80),1.)
        self.assertTrue(m.migrate(r,{key:'c'}));self.assertFalse(m.migrate(r,{key:'c'}))
        self.assertEqual(m.lamps[key].box,[15,12,25,22]);self.assertFalse(m.lamps[other].observable)
        self.assertEqual(m.status(other,10),'unobservable')


class FeedbackRules(unittest.TestCase):
    def module(self):
        module=OffsetModule('cam',store=MemoryStore(),vision=FixedVision())
        frame=np.zeros((100,100,3),np.uint8)
        module.inspect(frame,1,'initial');module.begin_migration(frame,2,'candidate')
        return module
    def test_reject_stale_or_self_predicted(self):
        m=self.module();r=m.pending
        for version,independent in [(r.version+1,True),(r.version,False)]:
            f=Feedback('cam',r.task_id,'lamp',r.old_reference,r.new_reference,version,r.frame_id,
                       ((0,0),(30,30)),((2,1),(32,31)),independent=independent)
            with self.assertRaises(ValueError):m.submit(f)
    def test_refine_commit_and_ack(self):
        m=self.module();r=m.pending;shared=m.references
        a=tuple((float(x),float(y)) for x in (10,30,50,70) for y in (10,30,50,70))
        b=tuple((x+4,y+3) for x,y in a)
        m.submit(Feedback('cam',r.task_id,'lamp',r.old_reference,r.new_reference,r.version,r.frame_id,a,b))
        updated=m.refine(3);self.assertTrue(updated.details['accepted'])
        np.testing.assert_allclose(np.asarray(updated.matrix)[:2,2],[4,3],atol=1e-4)
        committed=m.commit();self.assertIs(shared,m.references)
        self.assertEqual(shared.view,committed.new_reference)
        self.assertTrue(m.acknowledge('lamp',committed.task_id,committed.version))
        self.assertFalse(m.acknowledge('lamp',committed.task_id,committed.version))
    def test_timeout_cannot_commit(self):
        m=self.module();m.refine(2000)
        with self.assertRaises(RuntimeError):m.commit()
        m.abandon();self.assertIsNone(m.pending)
    def test_clustered_points_rejected(self):
        m=self.module();r=m.pending
        a=tuple((float(i)/100,float(i%2)/100) for i in range(16))
        m.submit(Feedback('cam',r.task_id,'lamp',r.old_reference,r.new_reference,r.version,r.frame_id,a,a))
        self.assertFalse(m.refine(3).details['accepted'])


class ArchiveRules(unittest.TestCase):
    def test_seven_days_current_and_images_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            store=LampArchive(temp,'cam')
            store.save_current({'id':'L1'},{'reference.png':b'image'})
            old=store.archive({'id':'old'},0,{'reference.png':b'old'})
            recent=store.archive({'id':'new'},RETENTION_S)
            self.assertEqual(store.sweep(RETENTION_S),1)
            self.assertFalse(old.exists());self.assertTrue(recent.exists())
            self.assertEqual(store.load_current(),{'id':'L1'})
            self.assertEqual(len(list((store.root/'current').glob('*/reference.png'))),1)
    def test_reject_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):LampArchive(temp,'../escape')
            store=LampArchive(temp,'cam')
            with self.assertRaises(ValueError):store.save_current({}, {'../x.png':b'x'})


class QualityRules(unittest.TestCase):
    def test_unknown_visibility_not_usable(self):
        self.assertFalse(ObservationQuality(True,True,True,True,None).usable)
    def test_duplicate_and_freeze_suspicion(self):
        q=StreamQuality(freeze_s=1.)
        image=np.random.default_rng(3).integers(0,256,(90,160,3),dtype=np.uint8)
        self.assertTrue(q.observe(image,0,1)['fresh'])
        self.assertFalse(q.observe(image,.1,1)['fresh'])
        self.assertTrue(q.observe(image,1.1,2)['freeze_suspected'])


class ControllerIntegration(unittest.TestCase):
    def test_end_to_end_window_and_restart(self):
        from night_lamp.controller import LampController
        with tempfile.TemporaryDirectory() as temp:
            params=MotionParams(storage=replace(MotionParams().storage,root=Path(temp)/'motion'))
            offset=OffsetModule('cam',params,vision=FixedVision())
            detector=SimpleNamespace(detect=lambda frame:[[20,20,30,30]])
            controller=LampController(offset,detector,Path(temp)/'lamps',visibility=lambda frame,lamp:True)
            controller.quality=SimpleNamespace(observe=lambda *args:dict(fresh=True,continuous=True,clear=True))
            frame=np.full((60,60,3),20,np.uint8)
            for i,stamp in enumerate([0.,30.,30.1,30.2,30.3]):
                controller.on_frame(frame,stamp,stamp,i,False)
            self.assertEqual(len(controller.monitor.lamps),1)
            key=next(iter(controller.monitor.lamps))
            controller.on_frame(frame,1750.,1750.,10,True)
            for i in range(3002):
                frame[20:30,20:30]=100 if 10<=i%100<20 else 20
                ts=1800+i*.1
                controller.on_frame(frame,ts,ts,100+i,True)
            controller.tick(3300,True)
            row=controller.monitor.results[-1]['lamps'][0]
            self.assertEqual(row['status'],'normal')
            self.assertEqual(row['pulses'],15)
            self.assertLess(row['valid_s'],300)
            report,path=controller.finalize(3300)
            self.assertTrue(path.is_dir())
            fresh=OffsetModule('cam',params,vision=FixedVision())
            restored=LampController(fresh,detector,Path(temp)/'lamps')
            self.assertIn(key,restored.monitor.lamps)
            self.assertEqual(fresh.references.view,restored.monitor.reference)
            self.assertIsNone(restored.monitor.bucket)

    def test_no_visibility_provider_is_conservative(self):
        from night_lamp.controller import LampController
        with tempfile.TemporaryDirectory() as temp:
            offset=OffsetModule('cam',store=MemoryStore(),vision=FixedVision())
            frame=np.zeros((60,60,3),np.uint8)
            offset.inspect(frame,0,'a');offset.inspect(frame,1,'b')
            controller=LampController(offset,SimpleNamespace(detect=lambda f:[]),temp)
            controller.monitor.reference=offset.references.view
            key=controller.monitor.add_known([20,20,30,30],1)
            controller.quality=SimpleNamespace(observe=lambda *args:dict(fresh=True,continuous=True,clear=True))
            for i in range(20):controller.on_frame(frame,1800+i*.1,1800+i*.1,i,True)
            self.assertEqual(controller.monitor.observations[key].valid_s,0)


if __name__=='__main__':unittest.main()
