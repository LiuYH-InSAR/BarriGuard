"""Production adapter/owner wiring for day absence and joint optimization."""
from dataclasses import replace
from datetime import datetime,timezone
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
import unittest
import numpy as np
import cv2 as cv
import test_production_wiring as wiring
import test_adaptive_runtime as evidence
from test_lamp_spatial import scene
from camera_motion.contracts import Feedback
from server.contracts import OffsetSpec
from night_lamp.landmarks import capture


class PhysicalPresenceWiring(unittest.TestCase):
    def setUp(self):
        f=wiring.ProductionWiring();f.setUp();self.addCleanup(f.doCleanups)
        self.f=f;self.w=f.worker(water=False);self.ad=self.w._lamp_adapter;self.c=self.ad.controller
        self.body,self.empty,self.box=scene();offset=self.c.offset
        offset.inspect(self.body,0.,'start');offset.inspect(self.body,.2,'aligned')
        self.c.monitor.reference=offset.references.view;self.c._view=offset.references.view
        cfg=replace(self.c.monitor.params,effective_s=2.,presence_period_s=.5,missing_min_checks=10)
        self.c.monitor.params=cfg
        self.key=self.c.monitor.add_known(self.box,0.,lamp_id='DayLamp')
        self.lamp=self.c.monitor.lamps[self.key]
        self.lamp.spatial=dict(reference=self.lamp.reference,anchor_box=self.box[:],anchor_body_box=self.box[:],
            position_verified=True,profiles=dict(day=capture(self.body,self.box,np.eye(3),self.lamp.reference,'day',0.)))
        self.c.detector.predict=Mock(return_value=[])
        self.c.pipeline.detector.geometry_valid=True;self.c.pipeline.detector.applied_transform=np.eye(3)
        self.c.pipeline.step=lambda *args,**kwargs:dict(evidence.result([]),motion='aligned')
        self.c.quality=NS(observe=lambda *args:dict(fresh=True,continuous=True,clear=True))

    def frame(self,t,image=None,night=False):
        self.f.clock.t=t
        self.c.offset.latest=replace(self.c.offset.latest,status='aligned',matrix=np.eye(3).tolist(),checked_at=t,frame_id=str(t))
        self.c.on_frame(self.empty if image is None else image,t,t,str(t),night)
        self.ad._last_image=(self.empty if image is None else image).copy()
        self.ad._last_image_reference=self.c.monitor.reference

    def absent_window(self,start):
        for i in range(15):self.frame(start+i*.5)

    def test_actual_day_verifier_produces_red_formal_report_after_two_windows(self):
        self.absent_window(1.);self.assertNotEqual(self.lamp.presence['status'],'missing')
        self.absent_window(1801.)
        report=self.ad.finalize(self.empty,datetime.fromtimestamp(1808.,timezone.utc))
        self.assertEqual(report.metadata['lamps'][0]['status'],'missing')
        self.assertEqual(report.metadata['lamps'][0]['health_basis'],'confirmed_day_body_absence')
        self.assertEqual(report.metadata['image_rendering']['counts']['failed'],1)
        self.assertTrue(report.image_jpeg)
        self.assertGreater(self.c.detector.predict.call_count,10)
        image=cv.imdecode(np.frombuffer(report.image_jpeg,np.uint8),cv.IMREAD_COLOR)
        self.assertEqual(image.shape[0],self.empty.shape[0]+76)

    def test_reappearing_template_clears_missing_without_inventing_healthy_flashes(self):
        self.absent_window(1.);self.absent_window(1801.)
        self.frame(1809.,self.body)
        self.assertEqual(self.lamp.presence['status'],'present')
        report=self.ad.finalize(self.body,datetime.fromtimestamp(1810.,timezone.utc))
        self.assertNotIn(report.metadata['lamps'][0]['status'],['normal','missing'])

    def test_foreground_cover_and_night_non_emission_cannot_prove_body_missing(self):
        covered=self.empty.copy();covered[60:78,60:78]=0
        for start in [1.,1801.]:
            for i in range(15):self.frame(start+i*.5,covered)
        self.assertNotEqual(self.lamp.presence.get('status'),'missing')
        for start in [3601.,5401.]:
            for i in range(15):self.frame(start+i*.5,night=True)
        self.assertNotEqual(self.lamp.presence.get('status'),'missing')

    def test_targeted_low_confidence_body_candidate_vetoes_negative_evidence(self):
        self.c.detector.predict.return_value=[self.box+[.2]]
        self.absent_window(1.);self.absent_window(1801.)
        self.assertNotEqual(self.lamp.presence.get('status'),'missing')
        self.assertEqual(self.lamp.presence['last_evidence']['reason'],'targeted_detector_body_candidate')

    def test_unlit_but_visible_body_is_present_even_when_detector_misses_it(self):
        for start in [1.,1801.]:
            for i in range(15):self.frame(start+i*.5,self.body)
        self.assertEqual(self.lamp.presence['status'],'present')
        self.assertEqual(self.lamp.presence['history'],[])

    def test_lost_alignment_cannot_publish_old_missing_as_current_red(self):
        self.absent_window(1.);self.absent_window(1801.)
        self.c.offset.latest=replace(self.c.offset.latest,status='unavailable',matrix=None)
        report=self.ad.finalize(self.empty,datetime.fromtimestamp(1808.,timezone.utc))
        self.assertNotEqual(report.metadata['lamps'][0]['status'],'missing')

    def test_old_night_background_does_not_show_new_view_coordinates(self):
        self.absent_window(1.);self.absent_window(1801.)
        self.ad._last_image_reference='different-night-view'
        report=self.ad.finalize(self.empty,datetime.fromtimestamp(1808.,timezone.utc))
        self.assertFalse(report.metadata['lamps'][0]['position_current'])
        self.assertEqual(report.metadata['image_rendering']['unplotted_ids'],[self.key])
        self.assertEqual(report.metadata['image_reference'],'different-night-view')

    def test_overlapping_old_id_sites_cannot_be_counted_as_two_missing_lamps(self):
        key=self.c.monitor.add_known(self.box,0.,lamp_id='Duplicate')
        from copy import deepcopy
        self.c.monitor.lamps[key].spatial=deepcopy(self.lamp.spatial)
        self.absent_window(1.);self.absent_window(1801.)
        self.assertNotEqual(self.lamp.presence.get('status'),'missing')
        self.assertNotEqual(self.c.monitor.lamps[key].presence.get('status'),'missing')


class JointFeedbackWiring(unittest.TestCase):
    def setUp(self):
        f=wiring.ProductionWiring();f.setUp();self.addCleanup(f.doCleanups)
        self.f=f;self.owner=f.owner('cam',OffsetSpec(period_s=1,stable_s=1),f.clock)
        self.owner.process(f.frame,100.,0.,'start');self.owner.process(f.frame,101.,1.,'aligned')
        self.changed=np.full_like(f.frame,20)

    def begin(self,timeout=120.):
        result=self.owner.offset.begin_migration(self.changed,102.,'candidate',timeout,source_ts=2.)
        self.owner._candidate_frame=self.changed.copy()
        return result

    def consumer(self,name,ready=True):
        old=np.float32([(10,10),(70,10),(10,70),(70,70)])
        def feedback(result,frame):
            observed=old+[5,0]
            return Feedback('cam',result.task_id,name,result.old_reference,result.new_reference,
                result.version,result.frame_id,tuple(map(tuple,old)),tuple(map(tuple,observed)))
        return NS(offset_reference=self.owner.offset.references.view,offset_feedback=feedback,
            offset_feedback_ready=Mock(return_value=ready),observe_offset_candidate=Mock(),apply_shared_offset=Mock())

    def test_waiting_collects_real_source_frames_without_burning_iterations(self):
        lamp=self.consumer('night_lamp',False);water=self.consumer('water_gap')
        self.owner.attach('night_lamp',lamp);self.owner.attach('water_gap',water);self.begin()
        for i in range(20):self.owner.process(self.changed,102.2+i*.2,2.2+i*.2,str(i),light_hint='night',source_ts=2.2+i*.2)
        self.assertEqual(self.owner.offset._iterations,0)
        self.assertEqual(len(self.owner.offset.commits),0)
        self.assertEqual(lamp.observe_offset_candidate.call_count,20)
        self.assertAlmostEqual(lamp.observe_offset_candidate.call_args.args[4],6.)
        lamp.offset_feedback_ready.return_value=True
        self.owner.process(self.changed,108.,8.,'ready',source_ts=8.)
        self.owner.process(self.changed,109.,9.,'commit',source_ts=9.)
        self.assertEqual(len(self.owner.offset.commits),1)
        result=self.owner.offset.commits[-1]
        self.assertEqual(set(result.details['feedback']),{'water_gap','night_lamp'})
        self.assertEqual(set(result.details['group_errors']),{'water_gap','night_lamp'})
        lamp.apply_shared_offset.assert_called_once();water.apply_shared_offset.assert_called_once()

    def test_waiting_has_a_deadline_and_fails_to_unknown_without_a_map(self):
        lamp=self.consumer('night_lamp',False);self.owner.attach('night_lamp',lamp);self.begin(timeout=2.)
        self.owner.offset._background=None
        self.owner.process(self.changed,105.,5.,'expired',source_ts=5.)
        self.assertEqual(len(self.owner.offset.commits),1)
        self.assertIsNone(self.owner.offset.commits[-1].matrix)

    def test_business_group_cannot_be_outvoted_by_many_background_points(self):
        result=self.begin();old=np.float32([(10,10),(70,10),(10,70),(70,70)])
        observed=old+[15,0];observed[0]=old[0]+[5,0]
        self.owner.offset.submit(Feedback('cam',result.task_id,'night_lamp',result.old_reference,result.new_reference,
            result.version,result.frame_id,tuple(map(tuple,old)),tuple(map(tuple,observed))))
        refined=self.owner.offset.refine(103.)
        self.assertFalse(refined.details['accepted'])
        self.assertGreater(refined.details['group_errors']['night_lamp']['median_px'],3.)

    def test_background_scaled_ransac_inliers_are_rechecked_in_original_pixels(self):
        self.begin();a,b,old,observed=self.owner.offset._background
        changed=observed.copy();changed[0,1]+=4.5
        self.owner.offset._background=(a,b,old,changed)
        result=self.owner.offset.refine(103.)
        self.assertTrue(result.details['accepted'])
        self.assertLess(result.details['background_filter']['kept'],result.details['background_filter']['total'])
        self.assertLessEqual(result.details['holdout_max_px'],3.)

    def test_valid_initial_mapping_survives_a_failed_numerical_optimization(self):
        lamp=self.consumer('night_lamp');result=self.begin()
        self.owner.offset.submit(lamp.offset_feedback(result,self.changed))
        wrong=np.array([[1.,0.,50.],[0.,1.,0.]])
        with patch('camera_motion.service.cv.estimateAffinePartial2D',return_value=(wrong,np.ones((1,1),np.uint8))):
            refined=self.owner.offset.refine(103.)
        self.assertTrue(refined.details['accepted'])
        self.assertEqual(refined.details['solution'],'initial_independently_validated')
        self.assertLessEqual(refined.details['holdout_max_px'],3.)

    def test_sparse_background_proposal_needs_independent_business_support(self):
        from camera_motion.vision import Alignment
        vision=self.f.vision;points=np.float32([(x,y) for x in (10,30,50,70) for y in (10,30,50,70)])
        m=np.eye(3);m[0,2]=5
        details=dict(method='orb',inliers=16,inlier_ratio=1.,error_px=.2,cells=3,destination_cells=3,scaling=1.)
        # The production minimum is18; supply24 independent background points.
        points=np.float32([(x,y) for x in (10,22,34,46,58,70) for y in (10,30,50,70)])
        details['inliers']=24
        vision.align=lambda *a:Alignment(False,m.copy(),.1,details.copy())
        def feature(a,b,*args):
            vision.last_matches=(a,b,points,points+[5,0])
            return Alignment(False,m.copy(),.1,details.copy())
        vision._feature_align=feature
        result=self.begin()
        self.assertTrue(result.details['requires_business_feedback'])
        self.assertIsNotNone(result.matrix)
        self.assertFalse(self.owner.offset.refine(103.).details['accepted'])
        self.assertEqual(self.owner.offset._iterations,0)
        lamp=self.consumer('night_lamp');self.owner.offset.submit(lamp.offset_feedback(result,self.changed))
        self.assertTrue(self.owner.offset.refine(103.2).details['accepted'])


if __name__=='__main__':unittest.main()
