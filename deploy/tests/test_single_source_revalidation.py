import copy
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from night_lamp.adaptive.single_source import SingleSourceResolver,overlapping_fixture,ownership_verified
from night_lamp.adaptive.adaptive_v9 import ObservationAuditV9


def track(key,box):
    return NS(id=key,box=box[:],body_box=box[:],confidence=.9,hits=10,
        position_observable=True,confirmed=True,pulses=NS(count=20),locator=NS(locked=True),
        flash=NS(last=dict(ready=True,emission_supported=True)),quality_details=dict(background=dict(ok=True)))


class OpticalTests(unittest.TestCase):
    def sequence(self,mode='pair'):
        resolver=SingleSourceResolver()
        tracks={1:track(1,[80,80,110,110]),2:track(2,[82,77,112,109])}
        if mode.startswith('orphan'):
            tracks.pop(2)
        if mode=='orphan_competitor':tracks[2]=track(2,[85,85,102,102])
        offers=[];ts=0.
        for i in range(50):
            frame=np.full((190,200,3),40,np.uint8)
            if mode=='gaps' and i%12==0 and i:
                ts+=1.5;resolver.invalidate()
            if mode=='broken_cycles':resolver.invalidate()
            if mode=='steady' or i%5 in [1,2]:
                frame[91:97,91:97]=[0,240,250]
                if mode=='two_sources':frame[91:97,102:108]=[0,240,250]
            for t in tracks.values():
                t.hits+=1
                t.flash.last['emission_supported']=mode=='steady' or i%5 in [1,2]
            quarantined={1} if mode.startswith('orphan') else {1,2}
            offers.extend(resolver.observe(frame,tracks,{k:0 for k in tracks},quarantined,ts,mode!='bad_frame'))
            ts+=.2
        return resolver,tracks,offers

    def test_shifted_high_overlap_is_a_hypothesis_but_not_itself_a_merge(self):
        self.assertTrue(overlapping_fixture([80,80,110,110],[82,77,112,109]))
        self.assertFalse(overlapping_fixture([80,80,110,110],[111,80,141,110]))
        self.assertTrue(self.sequence()[2])
        for mode in ['steady','two_sources','bad_frame','broken_cycles']:
            self.assertFalse(self.sequence(mode)[2],mode)

    def test_valid_segments_accumulate_but_no_gap_can_complete_a_cycle(self):
        offers=self.sequence('gaps')[2];self.assertTrue(offers)
        proof=offers[-1][1];self.assertGreaterEqual(proof['effective_s'],3)
        self.assertLess(proof['effective_s'],proof['span_s'])
        self.assertTrue(proof['gap_cycles_excluded'])

    def test_orphan_requires_fresh_optical_proof_and_no_competitor(self):
        self.assertTrue(self.sequence('orphan')[2])
        self.assertFalse(self.sequence('orphan_competitor')[2])

    def test_consumer_requires_previous_epoch_geometry_and_real_metrics(self):
        _,tracks,offers=self.sequence();proof=offers[-1][1]
        previous={t.id:dict(id=t.id,box=t.box,body_box=t.body_box,observation_epoch=0) for t in tracks.values()}
        self.assertTrue(ownership_verified(proof,{1:previous[1]},previous))
        for field,value in [('effective_s',2.9),('cycles',1),('detection_updates',2),('emission_samples',2),('point',[140,140]),('gap_cycles_excluded',False)]:
            bad=copy.deepcopy(proof);bad[field]=value
            self.assertFalse(ownership_verified(bad,{1:previous[1]},previous),field)
        previous[2]['observation_epoch']=1
        self.assertFalse(ownership_verified(proof,{1:previous[1]},previous))

    def test_audit_does_not_release_on_flag_or_disappearance_alone(self):
        _,tracks,offers=self.sequence();proof=offers[-1][1]
        for mode in ['good','nonzero','missing_prior','wrong_epoch']:
            auditor=ObservationAuditV9()
            def record(t,epoch=0):
                return dict(id=t.id,box=t.box,body_box=t.body_box,confidence=.9,hits=20,pulses=20,valid_s=60.,
                    confirmed=True,aliases=[],observation_epoch=epoch,position_observable=True,
                    flash_signal=dict(valid=True),quality_details=dict(reasons=[]))
            initial=[record(t) for t in tracks.values()]
            auditor.apply(dict(tracks=initial,events=[]))
            if mode=='missing_prior':auditor.previous_records.pop(2)
            if mode=='wrong_epoch':auditor.previous_records[2]['observation_epoch']=1
            selected=record(tracks[1],1);selected.update(pulses=1 if mode=='nonzero' else 0,valid_s=0.,confirmed=False,aliases=[2])
            event=dict(event='same_source_resolved',retained=1,aliases=[2],proof=proof,
                observation_reset=True,before_observation_epoch=0,evidence_policy='new_sampler_epoch')
            out=auditor.apply(dict(tracks=[selected],events=[event]))
            self.assertEqual(out['tracks'][0]['evidence_admitted'],mode=='good',mode)
            self.assertEqual(out['events'][0]['consumer_verified'],mode=='good',mode)


if __name__=='__main__':unittest.main()
