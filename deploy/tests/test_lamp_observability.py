import copy
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from night_lamp.adaptive.adaptive_tracking_v5 import BackgroundAnchor
from night_lamp.adaptive.source_separation import SourceSeparation,separation_verified


class BackgroundTests(unittest.TestCase):
    def setUp(self):
        self.frame=np.full((300,400),50,np.uint8);rng=np.random.default_rng(16)
        self.frame[100:200,90:115]=rng.integers(20,120,(100,25),dtype=np.uint8)
        self.frame[100:200,190:215]=rng.integers(20,120,(100,25),dtype=np.uint8)
        self.box=[146,146,154,154]
    def test_frozen_wider_background_can_verify_a_textureless_local_patch(self):
        anchor=BackgroundAnchor(self.frame,self.box);before=anchor.wider[0].reference.copy()
        now=self.frame.copy();now[146:155,146:155]=250
        result=anchor.evaluate(now,self.box)
        self.assertTrue(result['ok']);self.assertTrue(result['expanded_context']);self.assertLess(result['local_tiles'],3)
        self.assertGreaterEqual(result['stable_tiles'],3);np.testing.assert_array_equal(before,anchor.wider[0].reference)
    def test_nearby_occlusion_cannot_be_hidden_by_stable_distant_background(self):
        anchor=BackgroundAnchor(self.frame,self.box);now=self.frame.copy();now[125:175,125:175]=200
        result=anchor.evaluate(now,self.box);self.assertFalse(result['ok']);self.assertEqual(result['reason'],'background_changed_or_occluded')
    def test_completely_uniform_or_outside_frame_stays_unobservable(self):
        uniform=np.full_like(self.frame,50)
        self.assertFalse(BackgroundAnchor(uniform,self.box).evaluate(uniform,self.box)['ok'])
        self.assertFalse(BackgroundAnchor(self.frame,[0,0,8,8]).evaluate(self.frame,[0,0,8,8])['ok'])


def track(key,box,known=True):
    return NS(id=key,box=box,body_box=box,confidence=.9,hits=10,confirmed=known,source='night_flashes',
        pulses=NS(count=20 if known else 0),position_observable=True,locator=NS(locked=True),flash=NS(support_mask=np.ones((15,15),bool)))


class SourceTests(unittest.TestCase):
    def sequence(self,mode='good'):
        resolver=SourceSeparation();tracks={1:track(1,[65,84,100,104],False),2:track(2,[68,88,80,100]),3:track(3,[84,88,96,100])}
        if mode=='cold':tracks[3].pulses.count=0
        epochs={k:0 for k in tracks};offers=[]
        for i in range(32):
            image=np.full((180,200,3),40,np.uint8)
            for key,t in tracks.items():
                if key==1:continue
                x=int((t.box[0]+t.box[2])/2);y=int((t.box[1]+t.box[3])/2)
                if (i%5 in [1,2] or mode=='steady') and not(mode=='single' and key==3):image[y-2:y+2,x-2:x+2]=[0,240,250]
            if mode=='single_halo' and i%5 in [1,2]:image[92:96,72:92]=[0,240,250]
            offers.extend(resolver.observe(image,tracks,epochs,i*.2,mode!='bad_frame'))
        return resolver,tracks,epochs,offers
    def test_two_sources_in_one_spurious_group_are_proved_without_merging_children(self):
        resolver,tracks,epochs,offers=self.sequence();self.assertTrue(offers)
        parent,children,proof=offers[-1];self.assertEqual(parent.id,1);self.assertEqual({t.id for t in children},{2,3})
        records={t.id:dict(box=t.box,body_box=t.body_box,observation_epoch=0) for t in children}
        self.assertTrue(separation_verified(proof,records))
        resolver.groups.append(proof['children']);self.assertTrue(resolver.suppresses(parent.box,tracks,epochs))
        self.assertFalse(resolver.suppresses(children[0].box,tracks,epochs))
        epochs[2]=1;self.assertFalse(resolver.suppresses(parent.box,tracks,epochs))
        wrong=copy.deepcopy(proof);wrong['children'][1]['optical']['point']=wrong['children'][0]['optical']['point']
        self.assertFalse(separation_verified(wrong,records))
    def test_single_or_steady_cold_and_invalid_sequences_cannot_release_children(self):
        for mode in ['single','steady','cold','bad_frame','single_halo']:
            self.assertFalse(self.sequence(mode)[-1],mode)

    def test_auditor_requires_joint_proof_zero_counters_and_absent_group(self):
        from night_lamp.adaptive.adaptive_v9 import ObservationAuditV9
        _,tracks,_,offers=self.sequence();proof=offers[-1][2]
        for mode in ['good','missing_joint','nonzero','parent_present']:
            p=copy.deepcopy(proof);records=[]
            for key in [2,3]:
                t=tracks[key]
                records.append(dict(id=key,box=t.box,body_box=t.body_box,confidence=.9,hits=10,
                    observation_epoch=0,pulses=0,valid_s=0.,confirmed=False,aliases=[],position_observable=True,
                    quality_details=dict(reasons=[]),flash_signal=dict(valid=True)))
            if mode=='missing_joint':p.pop('group_core_count')
            if mode=='nonzero':records[0]['pulses']=1
            if mode=='parent_present':records.append(dict(records[0],id=1,box=tracks[1].box,body_box=tracks[1].box))
            audit=ObservationAuditV9();audit.quarantined={2,3}
            out=audit.apply(dict(tracks=records,events=[dict(event='independent_sources_resolved',discarded_group=1,proof=p,observation_reset=True)]))
            self.assertEqual(all(t['evidence_admitted'] for t in out['tracks'] if t['id'] in [2,3]),mode=='good',mode)


class TrustedSamplerTests(unittest.TestCase):
    def merge(self,changed=False,known=True,ordinary_proof=False):
        from night_lamp.adaptive.adaptive_v9 import AdaptiveV9
        from night_lamp.adaptive.adaptive_v5 import LampTrack
        detector=AdaptiveV9(NS(predict=lambda *a:[]));detector.current_frame=np.random.default_rng(7).integers(20,100,(220,240,3),dtype=np.uint8)
        parent=LampTrack(2,[90,90,120,122],1,5,hits=1,confidence=.9);child=LampTrack(1,[101,101,111,111],0,5,hits=20,confidence=.9)
        import cv2 as cv
        for t in [parent,child]:
            t.body_box=t.box[:];t.anchor=BackgroundAnchor(cv.cvtColor(detector.current_frame,cv.COLOR_BGR2GRAY),t.box)
        child.flash=detector.make_flash_signal();child.flash.last=dict(ready=True,valid=True);child.locator=NS(locked=True)
        child.confirmed=True;child.position_observable=True;child.source='night_flashes';child.pulses.count=20;child.pulses.valid_s=100
        if changed:child.box=[102,101,112,111]
        detector.tracks={1:child,2:parent};detector.frame_boxes={1:child.box,2:parent.box}
        detector.established_emitters={1:0} if known else {}
        proof=dict(proof_kind='established_emitter_body_variant',prior_fixture_id=1,prior_observation_epoch=0,
            parent_hits=1,child_hits=20,recent_family_detection_updates=3,family_detection_updates=3,
            samples=20,span_s=4.,cycles=3,amplitude=80.,unique_child=True,unique_emitter=True,
            peak_in_child=True,inside_variation_fraction=.9,minimum_child_edge=10,neighbor_veto_clear=True,peak=[106,106])
        if ordinary_proof:proof.update(proof_kind='unique_temporal_core',parent_hits=6)
        detector.nested.observe=lambda *a:[(parent,child,proof)]
        old_flash=child.flash;detector.merge_tracks(5)
        return detector,child,old_flash,detector.events[-1]
    def test_preserve_only_a_previously_admitted_exact_sampler(self):
        detector,child,flash,event=self.merge()
        self.assertIs(child.flash,flash);self.assertEqual((child.pulses.count,child.pulses.valid_s),(20,100))
        self.assertEqual(event['evidence_policy'],'unchanged_trusted_sampler');self.assertFalse(event['observation_reset'])
        _,same,_,event=self.merge(ordinary_proof=True)
        self.assertFalse(event['observation_reset']);self.assertEqual(same.pulses.count,20)
        for kwargs in [dict(changed=True),dict(known=False)]:
            _,child,flash,event=self.merge(**kwargs)
            self.assertIsNot(child.flash,flash);self.assertEqual((child.pulses.count,child.pulses.valid_s),(0,0))
            self.assertTrue(event['observation_reset'])
    def test_auditor_requires_its_own_prior_sampler_epoch(self):
        from night_lamp.adaptive.adaptive_v9 import ObservationAuditV9
        _,child,_,event=self.merge()
        for known in [True,False]:
            auditor=ObservationAuditV9();auditor.quarantined={1,2};auditor.established_emitters={1:0} if known else {}
            t=dict(id=1,box=child.box,body_box=child.body_box,confidence=.9,hits=20,pulses=20,valid_s=100,confirmed=True,
                observation_epoch=0,aliases=[2],position_observable=True,flash_signal=dict(ready=True,valid=True),quality_details=dict(reasons=[]))
            out=auditor.apply(dict(tracks=[t],events=[event]))
            self.assertEqual(out['tracks'][0]['evidence_admitted'],known)


if __name__=='__main__':unittest.main()
