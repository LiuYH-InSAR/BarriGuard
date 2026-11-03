"""Independent identity migration and physically evidenced absence; no live I/O."""
from copy import deepcopy
from dataclasses import asdict,replace
from types import SimpleNamespace as NS
from unittest.mock import Mock
from pathlib import Path
import importlib.util
import sys
import unittest
import cv2 as cv
import numpy as np
import test_adaptive_runtime as fixture
from camera_motion.contracts import MotionResult,Feedback
from camera_motion.geometry import warp_boxes
from night_lamp.landmarks import capture,compare,midpoint
from night_lamp.presence import PresenceLedger
from night_lamp.relocation import correspondences,constellation,committed_matches,RelocationSession
from night_lamp.tracked import LampMonitor,PulseCounter
from night_lamp.settings import LampParams
from night_lamp.runtime import AdaptiveLampController


def scene():
    yy,xx=np.indices((128,192));rng=np.random.default_rng(66)
    plane=np.clip(60+xx*.45+15*np.sin(yy*.1)+rng.normal(0,2,xx.shape),0,255).astype(np.uint8)
    empty=cv.cvtColor(plane,cv.COLOR_GRAY2BGR);body=empty.copy()
    body[60:78,60:78]=[15,210,230];body[63:75,66:70]=[20,70,90]
    return body,empty,[60.,60.,78.,78.]


class FrozenBody(unittest.TestCase):
    def setUp(self):
        self.body,self.empty,self.box=scene()
        self.profile=capture(self.body,self.box,np.eye(3),'ref','day',0.)

    def test_body_template_and_exposed_background_are_distinct(self):
        self.assertEqual(compare(self.body,self.profile,np.eye(3))['verdict'],'present')
        self.assertEqual(compare(self.empty,self.profile,np.eye(3))['verdict'],'absent')

    def test_body_only_and_broad_occlusion_are_unknown(self):
        for box in [[60,60,78,78],[40,40,100,100]]:
            image=self.body.copy();a,b,c,d=box;image[b:d,a:c]=0
            self.assertEqual(compare(image,self.profile,np.eye(3))['verdict'],'unknown')

    def test_illumination_change_is_not_body_loss(self):
        image=np.clip(self.body.astype(float)+24,0,255).astype(np.uint8)
        self.assertEqual(compare(image,self.profile,np.eye(3))['verdict'],'present')

    def test_transformed_template_tracks_translation_and_rotation(self):
        m=np.eye(3);m[:2]=cv.getRotationMatrix2D((69,69),4,1.03);m[:2,2]+=[15,2]
        frame=cv.warpPerspective(self.body,m,(192,128))
        self.assertEqual(compare(frame,self.profile,m)['verdict'],'present')
        m[0,2]+=300
        self.assertEqual(compare(frame,self.profile,m)['verdict'],'unknown')

    def test_featureless_context_cannot_prove_absence(self):
        image=np.full_like(self.body,80);image[60:78,60:78]=230
        profile=capture(image,self.box,np.eye(3),'ref','day',0.)
        self.assertEqual(compare(np.full_like(image,80),profile,np.eye(3))['verdict'],'unknown')


class PresenceWindows(unittest.TestCase):
    def setUp(self):
        cfg=replace(LampParams(),effective_s=2.,presence_period_s=.5,missing_min_checks=3)
        self.monitor=LampMonitor('cam','ref',cfg);key=self.monitor.add_known([60,60,78,78],0.)
        self.lamp=self.monitor.lamps[key];self.ledger=PresenceLedger(self.monitor)

    def absent(self,start):
        for i in range(7):self.ledger.observe(self.lamp,start+i*.5,start+i*.5,'absent',dict(reason='body_removed_background_exposed'))

    def test_two_consecutive_day_windows_required(self):
        self.absent(0.);self.assertIsNone(self.ledger.status(self.lamp,4.))
        self.absent(1800.);self.assertEqual(self.ledger.status(self.lamp,1804.),'missing')

    def test_skipped_window_does_not_support_missing(self):
        self.absent(0.);self.absent(3600.)
        self.assertIsNone(self.ledger.status(self.lamp,3604.))

    def test_gap_and_quality_pause_do_not_add_time(self):
        self.ledger.observe(self.lamp,0.,0.,'absent')
        self.ledger.observe(self.lamp,10.,10.,'absent')
        self.assertEqual(self.lamp.presence['valid_s'],0.)
        self.ledger.pause();self.ledger.observe(self.lamp,10.5,10.5,'absent')
        self.assertEqual(self.lamp.presence['valid_s'],0.)

    def test_returning_body_or_trusted_night_flashes_clear_missing(self):
        self.absent(0.);self.absent(1800.)
        self.ledger.positive(self.lamp,1804.,'night_flashes')
        self.assertIsNone(self.ledger.status(self.lamp,1804.))
        self.assertEqual(self.lamp.presence['status'],'present')
        self.assertEqual(self.lamp.presence['events'][-1]['event'],'missing_recovered')

    def test_stale_reference_and_stale_day_evidence_are_unknown(self):
        self.absent(0.);self.absent(1800.)
        self.assertIsNone(self.ledger.status(self.lamp,1804.+86401))
        self.lamp.reference='new';self.assertIsNone(self.ledger.status(self.lamp,1804.))


class Migration(unittest.TestCase):
    def setUp(self):
        f=fixture.RuntimeEvidence();f.setUp();self.addCleanup(f.doCleanups)
        self.f=f;self.c=f.controller;self.m=f.monitor;self.old=self.m.reference
        for i,box in enumerate([[30,35,42,47],[100,35,112,47],[30,85,42,97],[100,85,112,97],[65,60,77,72]]):
            key=self.m.add_known(box,0.,lamp_id='Lamp'+str(i));lamp=self.m.lamps[key]
            lamp.body_box=box[:];lamp.spatial=dict(reference=self.old,anchor_box=box[:],anchor_body_box=box[:],position_verified=True)
        self.transform=np.eye(3);self.transform[:2,2]=[10,5]

    def committed(self,keys=None):
        keys=list(self.m.lamps)[:4] if keys is None else keys
        proof=[];old=[];new=[]
        for i,key in enumerate(keys):
            box=warp_boxes([self.m.lamps[key].box],self.transform)[0]
            old.append(tuple(midpoint(self.m.lamps[key].box)));new.append(tuple(midpoint(box)))
            proof.append(dict(lamp_id=key,candidate_id='new'+str(i),box=box,body_box=box,
                samples=20,span_s=4.,independent_hits=10,cycles=15,night=True,admitted=True,observed_at=6.))
        feedback=Feedback('cam','move','night_lamp',self.old,'new',1,'frame',tuple(old),tuple(new),
            evidence=dict(source='independent_candidate_pipeline',matches=proof))
        return MotionResult('cam','move','frame',1.,self.old,'new',2,'committed',self.transform.tolist(),(128,192),1.,
            dict(deadline=120.,feedback=dict(night_lamp=asdict(feedback)),group_errors=dict(night_lamp=dict(max_px=0.))))

    def test_independent_pairs_keep_old_ids_and_do_not_copy_pulse_evidence(self):
        result=self.committed();self.m.observations['Lamp0']=PulseCounter(valid_s=300,count=15)
        self.m.accepted['Lamp0']=dict(pulses=15,valid_s=300)
        self.c.search_counter=PulseCounter(valid_s=700.);self.c._ledger_bucket=0.
        self.f.offset.references.view='new'
        self.c.reset_reference(result)
        self.assertEqual(set(self.m.lamps),{'Lamp'+str(i) for i in range(5)})
        self.assertEqual(self.m.lamps['Lamp0'].box,[40.,40.,52.,52.])
        self.assertEqual(self.m.lamps['Lamp0'].reference,'new')
        self.assertTrue(self.m.lamps['Lamp0'].spatial['position_verified'])
        self.assertTrue(self.m.lamps['Lamp4'].spatial['position_verified']) # Interpolated inside joint support.
        self.assertFalse(self.m.lamps['Lamp0'].observable)
        self.assertEqual(self.m.accepted,{});self.assertEqual(self.m.observations,{})
        self.assertEqual(self.c.search_counter.valid_s,0.);self.assertIsNone(self.c._ledger_bucket)
        self.c.reset_reference(result);self.assertEqual(len(self.c.spatial.migrations),1)

    def test_unmapped_transition_preserves_old_unknown_archive(self):
        result=replace(self.committed(),matrix=None)
        self.f.offset.references.view='new';self.c.reset_reference(result)
        self.assertEqual(self.m.lamps['Lamp0'].reference,self.old)
        self.assertFalse(self.m.lamps['Lamp0'].observable)

    def test_unmapped_old_pixels_are_counted_unknown_but_not_drawn_in_new_view(self):
        from night_lamp.report_render import render_lamp_report
        self.f.offset.references.view='new';self.c.reset_reference(replace(self.committed(),matrix=None))
        rows=[self.c.report_lamp(k,l,2.,0.) for k,l in self.m.lamps.items()]
        self.assertTrue(all(not r['position_current'] and r['status']=='unobservable' for r in rows))
        canvas,meta=render_lamp_report(self.f.image,rows,'cam','now')
        np.testing.assert_array_equal(canvas[:128],self.f.image)
        self.assertEqual(meta['counts']['unknown'],5)
        self.assertEqual(set(meta['unplotted_ids']),set(self.m.lamps))

    def test_bound_migrated_site_blocks_second_persistent_id(self):
        self.f.offset.references.view='new';self.c.reset_reference(self.committed())
        self.c.bindings={1:'Lamp0'}
        row=fixture.record(2,box=self.m.lamps['Lamp0'].box,body_box=self.m.lamps['Lamp0'].body_box)
        stale=fixture.record(1,box=self.m.lamps['Lamp0'].box,body_box=self.m.lamps['Lamp0'].body_box,
            evidence_admitted=False,visible_recent=False)
        self.c.consume(fixture.result([stale,row]),2.,2.,True)
        self.assertEqual(len(self.m.lamps),5)
        self.assertNotIn(2,self.c.bindings)

    def test_unique_new_sampler_can_release_an_inactive_migrated_binding(self):
        self.f.offset.references.view='new';self.c.reset_reference(self.committed())
        self.c.bindings={1:'Lamp0'}
        self.c.spatial._mapping=Mock(return_value=np.eye(3))
        row=fixture.record(2,box=self.m.lamps['Lamp0'].box,body_box=self.m.lamps['Lamp0'].body_box)
        rows=fixture.result([fixture.record(1,evidence_admitted=False,visible_recent=False),row])
        self.m.observations['Lamp0']=PulseCounter(count=14,valid_s=250)
        offers=self.c.spatial.reassociation_offers(rows,self.f.image,'fresh',True)
        self.assertEqual(offers,{2:'Lamp0'});self.assertNotIn(1,self.c.bindings)
        self.assertNotIn('Lamp0',self.m.observations)
        self.c._migration_offers=offers;self.c.consume(rows,2.,2.,True)
        self.assertEqual(self.c.bindings[2],'Lamp0');self.assertEqual(len(self.m.lamps),5)
        self.assertEqual(self.m.observations['Lamp0'].count,0)

    def test_live_candidate_tracker_reuses_only_verified_id_links_not_business_counts(self):
        result=self.committed();tracker=self.c.pipeline;candidates={}
        for i,p in enumerate(result.details['feedback']['night_lamp']['evidence']['matches'],1):
            p['local_id']=i;tracker.detector.tracks[i]=NS()
            candidates[p['candidate_id']]=dict(local_id=i)
        self.c.spatial.session=NS(task=result.task_id,invalid=False,pipeline=tracker,night=True,candidates=candidates)
        self.f.offset.references.view='new';self.c.reset_reference(result)
        self.assertIs(self.c.pipeline,tracker)
        self.assertEqual(self.c.bindings,{i:'Lamp'+str(i-1) for i in range(1,5)})
        self.assertEqual(self.m.observations,{});self.assertEqual(self.m.accepted,{})

    def test_durable_independent_matches_survive_common_commit_restart(self):
        result=self.committed();self.c.checkpoint(0.)
        recovered=AdaptiveLampController(self.f.offset,fixture.Backend(),self.f.root/'lamps')
        self.f.offset.references.view='new';recovered.reset_reference(result)
        self.assertTrue(recovered.monitor.lamps['Lamp0'].spatial['position_verified'])
        self.assertEqual(recovered.monitor.lamps['Lamp0'].box,[40.,40.,52.,52.])

    def test_failed_disk_publish_retries_before_transition_can_be_acknowledged(self):
        result=self.committed();self.f.offset.references.view='new'
        original=self.c.checkpoint;attempts=[]
        def save(stamp):
            attempts.append(stamp)
            if len(attempts)==1:raise OSError('temporary disk failure')
            return original(stamp)
        self.c.checkpoint=save
        with self.assertRaises(OSError):self.c.reset_reference(result)
        self.c.reset_reference(result)
        self.assertEqual(len(attempts),2);self.assertEqual(len(self.c.spatial.migrations),1)
        recovered=AdaptiveLampController(self.f.offset,fixture.Backend(),self.f.root/'lamps')
        self.assertEqual(recovered.monitor.lamps['Lamp0'].reference,'new')

    def test_ambiguous_or_tampered_pair_cannot_claim_verified_identity(self):
        result=self.committed();data=deepcopy(result.details)
        data['feedback']['night_lamp']['evidence']['matches'][0]['candidate_id']='new1'
        self.assertEqual(committed_matches(replace(result,details=data),self.m.lamps,self.m.params),{})
        self.assertEqual(committed_matches(replace(result,version=99),self.m.lamps,self.m.params),{})
        data=deepcopy(result.details)
        data['feedback']['night_lamp']['old_points']=list(data['feedback']['night_lamp']['old_points'])
        data['feedback']['night_lamp']['old_points'][0]=(float('nan'),1.)
        self.assertEqual(committed_matches(replace(result,details=data),self.m.lamps,self.m.params),{})

    def test_partial_new_constellation_matches_one_to_one(self):
        candidates={}
        for key in ['Lamp3','Lamp0','Lamp2']:
            box=warp_boxes([self.m.lamps[key].box],self.transform)[0]
            candidates['candidate'+key]=dict(body_box=box)
        matches=correspondences(self.m.lamps,candidates,self.transform,(128,192))
        self.assertEqual(matches,{k:'candidate'+k for k in ['Lamp0','Lamp2','Lamp3']})

    def test_equal_alternatives_are_rejected(self):
        lamp=self.m.lamps['Lamp0'];candidates=dict(a=dict(body_box=[39,40,51,52]),b=dict(body_box=[41,40,53,52]))
        self.assertEqual(correspondences({'Lamp0':lamp},candidates,self.transform,(128,192)),{})

    def test_reference_change_reopens_only_a_new_reference_bucket(self):
        self.m.tick(1,True);self.m._finish(2.);self.m.bucket=None
        self.m.tick(3,True);self.assertIsNone(self.m.bucket)
        self.m.reference='new';self.m.tick(4,True);self.assertEqual(self.m.bucket,0)

    def test_candidate_coordinates_are_normalized_before_feedback(self):
        result=replace(self.committed(),status='migration_pending',version=1,details=dict(deadline=120.,source_ts=0.))
        session=RelocationSession(self.c,result,self.f.image,False)
        translation=np.eye(3);translation[0,2]=2
        def step(frame,source,frame_id,night=False):
            boxes=warp_boxes([l.box for l in self.m.lamps.values()],translation@self.transform)
            return fixture.result([fixture.record(i+1,box=b,body_box=b,hits=int(source*5)+5) for i,b in enumerate(boxes)],motion='aligned')
        # fixture.result's quality is explicit; motion belongs to the result root.
        def output(frame,source,frame_id,night=False):
            data=step(frame,source,frame_id,night);data['motion']='aligned';return data
        session.pipeline=NS(step=output,detector=NS(applied_transform=translation))
        for i in range(22):session.observe(result,self.f.image,1+i*.2,i*.2,str(i))
        self.assertTrue(session.ready(result));feedback=session.feedback(result)
        self.assertIsNotNone(feedback);self.assertEqual(len(feedback.old_points),5)
        np.testing.assert_allclose(feedback.observed_points[0],midpoint(self.m.lamps['Lamp0'].box)+[10,5])
        self.assertEqual(self.m.observations,{})


class ConstellationFallback(unittest.TestCase):
    def test_nonuniform_subset_recovers_large_translation_without_background_map(self):
        rng=np.random.default_rng(62);old=rng.uniform([300,150],[1200,650],(20,2))
        angle=.06;rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])*.96
        expected=old@rotation.T+[-250,35];new=np.delete(expected,[2,7,14],axis=0)
        transform=constellation(old,new,(720,1280))
        self.assertIsNotNone(transform)
        np.testing.assert_allclose(old@transform[:2,:2].T+transform[:2,2],expected,atol=.01)

    def test_collinear_and_ambiguous_repeated_grid_do_not_force_ids(self):
        old=np.array([[x,100] for x in range(20,220,20)])
        self.assertIsNone(constellation(old,old+[5,0],(240,320)))
        grid=np.array([[x,y] for x in range(30,231,40) for y in (70,130,190)])
        self.assertIsNone(constellation(grid,grid[3:]+[10,5],(240,320)))


class UnchangedViewParity(unittest.TestCase):
    def test_migration_fixes_preserve_the_completed_original_view_controller(self):
        frozen=Path(__file__).resolve().parents[2]/'tmp/lamp_migration_presence_20261102/frozen_night_runtime/deploy/app/night_lamp/runtime.py'
        if not frozen.is_file():self.skipTest('Local original-video reference is not shipped')
        spec=importlib.util.spec_from_file_location('night_lamp._completed_night_runtime',frozen)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        f=fixture.RuntimeEvidence();f.setUp();self.addCleanup(f.doCleanups)
        old=module.AdaptiveLampController(f.offset,fixture.Backend(),f.root/'old')
        new=f.controller
        for c in [old,new]:c.monitor.add_known([60,60,68,68],0.,lamp_id='Known')
        for i in range(180):
            frame=f.image.copy();frame[60:69,60:69]=230 if i%5 in (1,2) else 10
            frame[-1,-1]=i
            for c in [old,new]:c.on_frame(frame,i*.2,i*.2,str(i),True)
            self.assertEqual(old.monitor.dump(),new.monitor.dump())
            self.assertEqual(old.monitor.accepted,new.monitor.accepted)
            self.assertEqual(old.search_counter.valid_s,new.search_counter.valid_s)


if __name__=='__main__':unittest.main()
