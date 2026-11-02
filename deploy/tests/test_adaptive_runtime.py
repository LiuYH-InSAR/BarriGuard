"""Business integration, conservative admission and extracted-v9 parity."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import copy
import json
import sys
import tempfile
import unittest
import cv2 as cv
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'deploy/app'))
from camera_motion.service import OffsetModule
from camera_motion.params import MotionParams
from camera_motion.vision import Alignment
from camera_motion.contracts import MotionResult
from night_lamp.runtime import AdaptiveLampController
from night_lamp.settings import LampParams
from night_lamp.tracked import PulseCounter
from night_lamp.adaptive import LampPipelineV9


class Vision:
    last_matches=None
    def quality(self,*args,**kw):return dict(ok=True,light='night')
    def align(self,*args,**kw):return Alignment(True,np.eye(3),0.,{})


class Backend:
    def predict(self,frame,regions,imgsz,confidence):
        return [[60.,60.,68.,68.,.9]] if any(a<=64<c and b<=64<d for a,b,c,d in regions) else []


def record(local=1,signal=0.,**kw):
    out=dict(id=local,box=[60.,60.,68.,68.],body_box=[60.,60.,68.,68.],confidence=.9,
        hits=6,confirmed=True,position_observable=True,evidence_admitted=True,
        visible_recent=True,observation_epoch=0,discarded_valid_s=0.,aliases=[],
        pulses=999,valid_s=99999.,flash_signal=dict(valid=True,signal=signal))
    out.update(kw)
    return out


def result(rows,events=None,**quality):
    q=dict(fresh=True,continuous=True,clear=True);q.update(quality)
    return dict(tracks=rows,events=events or [],quality=q)


class RuntimeEvidence(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        params=replace(MotionParams(),storage=replace(MotionParams().storage,root=self.root/'motion'))
        self.offset=OffsetModule('cam',params,vision=Vision())
        self.image=np.random.default_rng(9).integers(20,100,(128,192,3),dtype=np.uint8)
        self.offset.inspect(self.image,0,'start');self.offset.inspect(self.image,.2,'aligned')
        self.controller=AdaptiveLampController(self.offset,Backend(),self.root/'lamps')
        self.monitor=self.controller.monitor

    def step(self,t,rows=None,events=None):
        self.controller.consume(result([record(signal=1. if int(round(t*5))%5 in (1,2) else 0.)] if rows is None else rows,events),t,t,True)

    def test_unavailable_registration_pauses_business_time_and_open_pulse(self):
        self.step(0)
        key=next(iter(self.monitor.lamps));counter=self.monitor.observations[key]
        counter.valid_s=120.;counter.count=5;counter.phase='on';counter.rise=0.
        self.step(.2,[record(position_observable=False,flash_signal=dict(valid=False,signal=0.))],
                  events=[dict(event='geometry_unavailable',reason='quality_failed')])
        self.assertIs(self.monitor.observations[key],counter)
        self.assertEqual((counter.valid_s,counter.count),(120.,5))
        self.assertEqual(counter.phase,'unknown')
        self.step(.4,[record(signal=0.)])
        self.assertEqual((counter.valid_s,counter.count),(120.,5))
        self.step(.6,[record(signal=0.)])
        self.assertAlmostEqual(counter.valid_s,120.2)

    def test_counts_complete_cycles_not_detector_hits_and_never_copies_lifetime_evidence(self):
        self.step(0)
        key=next(iter(self.monitor.lamps));obs=self.monitor.observations[key]
        self.assertEqual((obs.valid_s,obs.count),(0,0))
        for i in range(1,1502):self.step(i*.2)
        self.assertLess(obs.valid_s,300);self.assertEqual(obs.count,15)
        self.assertLess(self.monitor.accepted[key]['accepted_at'],30)
        self.assertEqual(self.monitor.status(key,300.2),'normal')
        self.assertIsNone(self.monitor.bucket)

    def test_new_window_does_not_copy_previous_window(self):
        for i in range(1502):self.step(i*.2)
        key=next(iter(self.monitor.lamps))
        self.step(1800)
        self.assertEqual(self.monitor.results[-1]['lamps'][0]['status'],'normal')
        self.assertEqual((self.monitor.observations[key].valid_s,self.monitor.observations[key].count),(0,0))

    def test_completed_window_stops_inference_and_next_scheduled_window_starts(self):
        for i in range(1502):self.step(i*.2)
        self.controller.pipeline.step=Mock(return_value=result([record()]))
        self.controller.on_frame(self.image,301,301,'closed',True)
        self.controller.pipeline.step.assert_not_called()
        frame=self.image.copy();frame[-1,-1]=200
        self.controller.on_frame(frame,1800,1800,'next',True)
        self.controller.pipeline.step.assert_called_once()
        key=next(iter(self.monitor.lamps))
        self.assertEqual(self.monitor.observations[key].valid_s,0)

    def test_late_discovery_has_only_its_own_evidence(self):
        self.step(0,[])
        for i in range(100):self.step(120+i*.2,[record(confirmed=i>=80,signal=1. if i%5 in (1,2) else 0.)])
        key=next(iter(self.monitor.lamps));obs=self.monitor.observations[key]
        self.assertLessEqual(obs.valid_s,19.8)
        self.assertLessEqual(obs.valid_s,self.monitor.accepted[key]['accepted_at']-120+1e-6)
        self.assertLess(obs.count,30)

    def test_independent_source_release_starts_a_new_business_evidence_epoch(self):
        self.step(0);key=next(iter(self.monitor.lamps));counter=self.monitor.observations[key]
        counter.valid_s=150.;counter.count=40
        self.step(.2,[record(observation_epoch=1,signal=1.)],events=[dict(event='independent_source_reset',id=1)])
        counter=self.monitor.observations[key];self.assertEqual((counter.valid_s,counter.count),(0.,0))
        self.step(.4,[record(observation_epoch=1,signal=0.)]);self.assertEqual(counter.count,0)

    def test_known_sampler_quarantine_pauses_and_same_sampler_release_keeps_admitted_time(self):
        self.step(0);key=next(iter(self.monitor.lamps));counter=self.monitor.observations[key]
        counter.valid_s=150.;counter.count=5;counter.phase='on';counter.rise=0.
        self.step(.2,[record(evidence_admitted=False,position_observable=False)])
        self.assertEqual((counter.valid_s,counter.count),(150.,5));self.assertEqual(counter.phase,'unknown')
        event=dict(event='nested_identity_resolved',retained=1,aliases=[2],observation_reset=False,evidence_policy='unchanged_trusted_sampler')
        self.step(1.,[record(signal=0.)],events=[event])
        self.assertIs(self.monitor.observations[key],counter);self.assertEqual((counter.valid_s,counter.count),(150.,5))
        self.step(1.2,[record(signal=0.)]);self.assertAlmostEqual(counter.valid_s,150.2)

    def test_reacquiring_known_id_with_a_new_sampler_does_not_borrow_old_evidence(self):
        for i in range(40):self.step(i*.2)
        key=next(iter(self.monitor.lamps))
        self.step(20,[record(2,confirmed=False,box=[61,60,69,68],body_box=[61,60,69,68])])
        self.assertEqual(self.controller.bindings[2],key)
        self.assertEqual(len(self.monitor.lamps),1)
        self.assertEqual(self.monitor.observations[key].valid_s,0)
        self.assertEqual(self.monitor.observations[key].count,0)

    def test_bad_quality_extends_observation_without_cross_gap_pulse(self):
        for i in range(501):self.step(i*.2,[record(signal=0.)])
        key=next(iter(self.monitor.lamps));before=self.monitor.observations[key].count
        self.controller.pause(101)
        self.step(220,[record(signal=0.)])
        self.assertEqual(self.monitor.observations[key].count,before)
        self.assertAlmostEqual(self.monitor.observations[key].valid_s,100)
        for i in range(1,1001):self.step(220+i*.2,[record(signal=0.)])
        self.assertEqual(self.monitor.observations[key].valid_s,300)

    def test_25_minute_deadline_does_not_postpone_next_bucket(self):
        for i in range(20):self.step(i*.2)
        key=next(iter(self.monitor.lamps))
        self.step(1500)
        self.assertIsNone(self.monitor.bucket)
        self.assertEqual(self.monitor.results[-1]['lamps'][0]['status'],'insufficient_observation')
        self.step(1799)
        self.assertIsNone(self.monitor.bucket)
        self.step(1800)
        self.assertEqual(self.monitor.bucket,1800)
        self.assertEqual(self.monitor.observations[key].valid_s,0)

    def zero_windows(self,broken=None):
        for window in range(6):
            for i in range(1502 if window!=broken else 20):
                t=window*1800+i*.2;self.step(t,[record(signal=0.)])
            self.controller.tick(window*1800+1500,True)

    def test_six_zero_windows_and_recovery_clear_before_next_boundary(self):
        self.zero_windows();key=next(iter(self.monitor.lamps))
        self.assertEqual(self.monitor.status(key,10800),'not_flashing')
        for i in range(1502):self.step(10800+i*.2)
        self.assertEqual(self.monitor.status(key,11100.2),'normal')
        self.controller.tick(11100.2,False)
        self.assertEqual(self.monitor.status(key,11100.2),'normal')

    def test_incomplete_window_breaks_failure_evidence(self):
        self.zero_windows(broken=2);key=next(iter(self.monitor.lamps))
        self.assertNotEqual(self.monitor.status(key,10800),'not_flashing')

    def test_region_epoch_and_quarantine_erase_current_window_evidence(self):
        for i in range(40):self.step(i*.2)
        key=next(iter(self.monitor.lamps))
        self.step(20,[record(observation_epoch=1)],events=[dict(event='signal_support_changed',id=1)])
        self.assertEqual(self.monitor.observations[key].valid_s,0)
        self.step(20.2,[record(observation_epoch=1,evidence_admitted=False)])
        self.assertEqual(self.monitor.observations[key].count,0)
        self.assertFalse(self.monitor.lamps[key].observable)

    def test_private_geometry_loss_discards_evidence_and_waits_for_shared_recheck(self):
        for i in range(100):self.step(i*.2)
        key=next(iter(self.monitor.lamps))
        self.step(20,[],events=[dict(event='geometry_reset')])
        self.assertEqual(self.monitor.observations[key].valid_s,0)
        self.assertEqual(self.controller._geometry_blocked_at,20)
        self.assertFalse(self.monitor.lamps[key].observable)

    def test_photometric_reference_change_keeps_identity_but_restarts_business_evidence(self):
        for i in range(40):self.step(i*.2)
        key=self.controller.bindings[1];old=self.monitor.observations[key]
        self.assertGreater(old.valid_s,0);self.assertGreater(old.count,0)
        geometry=list(self.monitor.lamps[key].box)
        self.step(20,[record(confirmed=False,observation_epoch=1,
            flash_signal=dict(valid=False,signal=0.,reason='calibrating_exposure_baseline'))],
            events=[dict(event='signal_reference_changed',id=1,observation_reset=True)])
        self.assertEqual(self.controller.bindings[1],key)
        self.assertEqual(len(self.monitor.lamps),1)
        self.assertEqual(self.monitor.lamps[key].box,geometry)
        new=self.monitor.observations[key]
        self.assertIsNot(old,new);self.assertIs(self.controller.pending[1],new)
        self.assertEqual((new.valid_s,new.count),(0,0))
        self.step(20.2,[record(confirmed=False,observation_epoch=1,signal=0.)])
        self.assertEqual((new.valid_s,new.count),(0,0))
        self.step(20.4,[record(observation_epoch=1,signal=0.)])
        self.assertAlmostEqual(new.valid_s,.2);self.assertEqual(new.count,0)

    def test_rejected_core_correction_keeps_the_existing_business_sampler(self):
        for i in range(100):self.step(i*.2,[record(signal=0.)])
        key=self.controller.bindings[1];counter=self.monitor.observations[key]
        before=counter.valid_s;pulses=counter.count
        self.step(20,[record(signal=0.)],events=[dict(event='emitter_region_retained',id=1,
            observation_reset=False,reason='excluded_core_cannot_replace_recently_flashing_region')])
        self.assertEqual(self.controller.bindings[1],key)
        self.assertIs(self.monitor.observations[key],counter)
        self.assertAlmostEqual(counter.valid_s,before+.2)
        self.assertEqual(counter.count,pulses)

    def test_alias_merge_resets_one_counter_and_keeps_one_persistent_lamp(self):
        self.step(0,[record(1),record(2,box=[80,60,88,68],body_box=[80,60,88,68])])
        self.assertEqual(len(self.monitor.lamps),2)
        self.step(.2,[record(1,aliases=[2],observation_epoch=1)],events=[
            dict(event='nested_identity_resolved',retained=1,aliases=[2])])
        self.assertEqual(len(self.monitor.lamps),1)
        key=next(iter(self.monitor.lamps))
        self.assertEqual(self.monitor.observations[key].valid_s,0)
        self.assertEqual(len(self.monitor.lamps[key].aliases),1)

    def test_verified_overlapping_duplicate_never_adds_two_business_streams(self):
        self.step(0,[record(1),record(2,box=[61,59,69,67],body_box=[61,59,69,67])])
        keys=[self.controller.bindings[k] for k in [1,2]]
        self.monitor.observations[keys[0]].valid_s=120.;self.monitor.observations[keys[0]].count=20
        self.monitor.observations[keys[1]].valid_s=290.;self.monitor.observations[keys[1]].count=200
        self.step(1.,[record(1,aliases=[2])],events=[dict(event='same_source_resolved',retained=1,aliases=[2],
            consumer_verified=True,observation_reset=False,evidence_policy='unchanged_trusted_sampler')])
        self.assertEqual(len(self.monitor.lamps),1)
        counter=next(iter(self.monitor.observations.values()))
        self.assertEqual((counter.valid_s,counter.count),(0.,0))

    def test_unverified_same_source_event_cannot_merge_or_reset_business_ids(self):
        self.step(0,[record(1),record(2,box=[80,60,88,68],body_box=[80,60,88,68])])
        key=self.controller.bindings[1];self.monitor.observations[key].valid_s=120.
        self.step(1.,[record(1),record(2,box=[80,60,88,68],body_box=[80,60,88,68])],events=[
            dict(event='same_source_resolved',retained=1,aliases=[2],consumer_verified=False,observation_reset=True)])
        self.assertEqual(len(self.monitor.lamps),2);self.assertEqual(self.monitor.observations[key].valid_s,120.)

    def test_orphan_revalidation_restarts_only_its_own_business_evidence(self):
        self.step(0);key=self.controller.bindings[1]
        self.monitor.observations[key].valid_s=120.;self.monitor.observations[key].count=20
        self.step(1.,[record(1,confirmed=False,observation_epoch=1)],events=[dict(event='same_source_resolved',retained=1,
            aliases=[],consumer_verified=True,observation_reset=True,evidence_policy='new_sampler_epoch')])
        self.assertEqual(self.controller.bindings[1],key)
        self.assertEqual((self.monitor.observations[key].valid_s,self.monitor.observations[key].count),(0.,0))

    def test_shared_reference_commit_does_not_claim_deferred_migration(self):
        self.step(0);key=next(iter(self.monitor.lamps));old=self.monitor.reference
        move=MotionResult('cam','move','newframe',1,old,'new',1,'committed',np.eye(3).tolist(),(128,192),0.)
        self.controller.reset_reference(move)
        self.assertEqual(self.monitor.reference,'new')
        self.assertEqual(self.monitor.lamps[key].reference,old)
        self.assertFalse(self.monitor.lamps[key].observable)
        self.controller.reset_reference(move)

    def test_checkpoint_restart_preserves_geometry_history_but_no_partial_evidence(self):
        for i in range(100):self.step(i*.2)
        key=next(iter(self.monitor.lamps));self.controller.checkpoint(20)
        state=self.controller.archive.load_current()
        self.assertEqual(state['adaptive']['version'],'v9c')
        self.assertIn(key,state['geometry']['lamps'])
        self.assertTrue(list((self.controller.archive.root/'current').glob('*/reference_0.png')))
        restored=AdaptiveLampController(self.offset,Backend(),self.root/'lamps')
        self.assertEqual(restored.monitor.lamps[key].box,self.monitor.lamps[key].box)
        self.assertIsNone(restored.monitor.bucket)
        self.assertFalse(restored.monitor.lamps[key].observable)
        restored.consume(result([record()]),21,21,True)
        self.assertEqual(len(restored.monitor.lamps),1)
        self.assertEqual(restored.monitor.observations[key].valid_s,0)

    def test_changed_policy_keeps_old_history_for_review_without_reusing_failure_evidence(self):
        self.zero_windows();key=next(iter(self.monitor.lamps));self.controller.checkpoint(10800)
        restored=AdaptiveLampController(self.offset,Backend(),self.root/'lamps',replace(LampParams(),normal_pulses=30))
        self.assertEqual(restored.monitor.lamps[key].history,[])
        self.assertEqual(len(restored.prior_policy_history['monitor']['lamps'][key]['history']),6)
        self.assertIs(restored.offset,self.offset)

    def test_changed_visual_evidence_policy_keeps_ids_but_retires_failure_history(self):
        self.zero_windows();key=next(iter(self.monitor.lamps))
        saved=self.controller._snapshot(10800);saved['adaptive'].pop('evidence_policy')
        self.controller.archive.save_current(saved,self.controller._images())
        restored=AdaptiveLampController(self.offset,Backend(),self.root/'lamps')
        self.assertIn(key,restored.monitor.lamps);self.assertEqual(restored.monitor.lamps[key].history,[])
        self.assertEqual(len(restored.prior_policy_history['monitor']['lamps'][key]['history']),6)

    def test_duplicate_content_and_shared_position_gate_prevent_inference(self):
        backend=SimpleNamespace(predict=Mock(return_value=[]))
        c=AdaptiveLampController(self.offset,backend,self.root/'other')
        c.pipeline.detector.vision=Vision()
        c.on_frame(self.image,0,0,'a',True)
        calls=backend.predict.call_count
        c.on_frame(self.image,.2,.2,'b',True)
        self.assertEqual(backend.predict.call_count,calls)
        self.offset.latest=replace(self.offset.latest,status='unavailable',matrix=None)
        frame=self.image.copy();frame[-1,-1]=200
        c.on_frame(frame,.4,.4,'c',True)
        self.assertEqual(backend.predict.call_count,calls)


class RegistrationPause(unittest.TestCase):
    def make(self):
        self.backend=Mock(wraps=Backend());self.vision=Vision()
        self.pipeline=LampPipelineV9(self.backend,vision=self.vision)
        self.base=np.random.default_rng(5).integers(20,100,(128,192,3),dtype=np.uint8)
        for i in range(15):self.pipeline.step(self.frame(i),i*.2,i,False)
        self.detector=self.pipeline.detector
        self.track=next(iter(self.detector.tracks.values()))
        self.assertTrue(self.track.confirmed)
        self.track.pulses=PulseCounter(valid_s=120.,count=20,previous_ts=2.8,previous_valid=True,phase='on',rise=2.8)
        return self.track.id

    def frame(self,i):
        out=self.base.copy();out[-1,-1]=i%255;return out

    def test_unknown_mapping_preserves_id_and_reference_without_discovery(self):
        key=self.make();ref=self.detector.fixed_reference.copy()
        self.vision.align=lambda *a:Alignment(False,None,0.,dict(reason='occlusion'))
        self.backend.reset_mock()
        out=self.pipeline.step(self.frame(16),3.,16,True)
        self.assertEqual(out['motion'],'registration_unavailable')
        self.assertIn(key,self.detector.tracks)
        self.backend.predict.assert_not_called()
        self.assertFalse(out['tracks'][0]['position_observable'])
        self.assertEqual((self.track.pulses.valid_s,self.track.pulses.count),(120.,20))
        np.testing.assert_array_equal(self.detector.fixed_reference,ref)
        self.assertFalse(any(e['event']=='geometry_reset' for e in out['events']))

    def test_repeated_unavailable_checks_never_add_time_or_force_new_identity(self):
        key=self.make()
        self.vision.align=lambda *a:Alignment(False,None,0.,dict(reason='occlusion'))
        for i in range(16,216):self.pipeline.step(self.frame(i),i*.2,i,True)
        self.assertIn(key,self.detector.tracks)
        self.assertEqual((self.track.pulses.valid_s,self.track.pulses.count),(120.,20))
        self.vision.align=Vision().align
        out=self.pipeline.step(self.frame(216),43.2,216,True)
        self.assertEqual(out['motion'],'aligned')
        self.assertIn(key,self.detector.tracks)
        self.assertEqual(self.track.pulses.valid_s,120.)

    def test_verified_large_movement_still_discards_old_geometry(self):
        key=self.make();matrix=np.eye(3);matrix[0,2]=20.
        self.vision.align=lambda *a:Alignment(True,matrix,.05,dict(reason='verified_move'))
        out=self.pipeline.step(self.frame(16),3.,16,True)
        self.assertNotIn(key,self.detector.tracks)
        self.assertTrue(any(e['event']=='geometry_reset' for e in out['events']))


class ExtractedVisionParity(unittest.TestCase):
    @unittest.skipUnless((ROOT/'training/lamps/optical_fixture_v9.py').exists(),'offline oracle is intentionally not deployed')
    def test_compact_verified_source_with_diffuse_halo_matches_offline_oracle(self):
        sys.path.insert(0,str(ROOT/'training/lamps'))
        from optical_fixture_v9 import OpticalResolver as Offline
        from night_lamp.adaptive.optical_fixture_v9 import OpticalResolver as Production
        a,b=Offline(),Production()
        for resolver in [a,b]:resolver.established_emitter_epochs={2:7}
        parent=SimpleNamespace(id=1,body_box=[90,90,108,110],hits=1,confidence=.9)
        child=SimpleNamespace(id=2,body_box=[98,98,106,106],hits=5,confidence=.9)
        base=np.random.default_rng(9).integers(20,100,(220,240,3),dtype=np.uint8);proofs=[]
        for i in range(40):
            child.hits+=1;frame=base.copy();frame[90:110,90:108]=30
            if i%5 in [1,2]:frame[90:110,90:108]=45;frame[100:104,100:104]=[0,240,250]
            left=[p for _,_,p in a.observe(frame,[parent,child],i*.2,True)]
            right=[p for _,_,p in b.observe(frame,[parent,child],i*.2,True)]
            self.assertEqual(left,right);proofs.extend(right)
        self.assertTrue(proofs)
        self.assertTrue(all(p['ownership_basis']=='verified_compact_core' and p['inside_variation_fraction']<.7 for p in proofs))

    @unittest.skipUnless((ROOT/'training/lamps/exposure_recovery.py').exists(),'offline oracle is intentionally not deployed')
    def test_photometric_recovery_and_post_reset_calibration_match_offline_oracle(self):
        sys.path.insert(0,str(ROOT/'training/lamps'))
        from exposure_recovery import RecoverableFlashSignal as Offline
        from night_lamp.adaptive.exposure_recovery import RecoverableFlashSignal as Production
        signals=[Offline(),Production()];changed=False
        for signal in signals:
            signal.radius=4;signal.support_mask=np.ones((9,9),bool);signal.support_mask[:,0]=False
            signal.template=np.full((9,9),20.);signal.background=np.array([20.,25.,30.]);signal.amplitude=40.;signal.low=0.
        for i in range(60):
            outputs=[]
            plane=np.full((160,160),100,np.uint8)
            if i%5 in [1,2]:plane[78:82,78:82]=245
            for signal in signals:
                signal.context=[50.,60.,70.];signal.recovery_allowed=True;signal.detector_hits=10+i//10
                outputs.append(signal.observe(plane,[76,76,84,84],i*.2,True))
            self.assertEqual(json.dumps(outputs[0],sort_keys=True),json.dumps(outputs[1],sort_keys=True))
            if outputs[1].get('exposure_reference_changed'):
                changed=True;self.assertFalse(outputs[1]['valid']);self.assertEqual(outputs[1]['calibration_s'],0)
            np.testing.assert_array_equal(signals[0].support_mask,signals[1].support_mask)
        self.assertTrue(changed)

    def variant_parity(self,established=False):
        sys.path.insert(0,str(ROOT/'training/lamps'))
        from optical_fixture_v9 import OpticalResolver as Offline
        from night_lamp.adaptive.optical_fixture_v9 import OpticalResolver as Production
        a,b=Offline(),Production()
        for resolver in [a,b]:
            if established:resolver.established_emitter_epochs={2:7}
            else:resolver.proven_fixture_ids=frozenset([2])
        parent=SimpleNamespace(id=1,body_box=[90,90,108,110],hits=1,confidence=.9)
        child=SimpleNamespace(id=2,body_box=[98,98,106,106],hits=5,confidence=.9)
        base=np.random.default_rng(9).integers(20,100,(220,240,3),dtype=np.uint8);proofs=[]
        for i in range(40):
            child.hits+=1;frame=base.copy();frame[90:110,90:108]=30
            if i%5 in [1,2]:frame[100:104,100:104]=[0,240,250]
            left=[p for _,_,p in a.observe(frame,[parent,child],i*.2,True)]
            right=[p for _,_,p in b.observe(frame,[parent,child],i*.2,True)]
            self.assertEqual(left,right);proofs.extend(right)
        self.assertTrue(proofs)
        kind='established_emitter_body_variant' if established else 'proven_fixture_body_variant'
        self.assertTrue(all(p['proof_kind']==kind and p['parent_hits']==1 for p in proofs))

    @unittest.skipUnless((ROOT/'training/lamps/optical_fixture_v9.py').exists(),'offline oracle is intentionally not deployed')
    def test_proven_fixture_body_variant_matches_oracle_and_keeps_truthful_hits(self):self.variant_parity()

    @unittest.skipUnless((ROOT/'training/lamps/optical_fixture_v9.py').exists(),'offline oracle is intentionally not deployed')
    def test_established_emitter_reverification_matches_offline_oracle(self):self.variant_parity(established=True)

    @unittest.skipUnless((ROOT/'training/lamps/adaptive_v9.py').exists(),'offline oracle is intentionally not deployed')
    def test_extracted_defaults_match_frozen_v9_with_identical_backend_and_frames(self):
        sys.path.insert(0,str(ROOT/'training/lamps'))
        from adaptive_v9 import LampPipelineV9 as Offline
        a=Offline(Backend(),vision=Vision());b=LampPipelineV9(Backend(),vision=Vision())
        base=np.random.default_rng(13).integers(20,100,(128,192,3),dtype=np.uint8)
        for i in range(120):
            frame=base.copy();frame[60:69,60:69]=230 if i%5 in (1,2) else 10
            frame[-1,-1]=i
            left=a.step(frame,i*.2,str(i),True);right=b.step(frame,i*.2,str(i),True)
            left.pop('elapsed_ms');right.pop('elapsed_ms')
            self.assertEqual(json.dumps(left,sort_keys=True),json.dumps(right,sort_keys=True))


if __name__=='__main__':unittest.main()
