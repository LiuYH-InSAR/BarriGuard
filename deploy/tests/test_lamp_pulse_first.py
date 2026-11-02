"""Positive evidence and negative observation time are independent gates."""
import unittest
import test_adaptive_runtime as fixtures
from night_lamp.settings import LampParams
from night_lamp.tracked import PulseCounter


class PulseFirst(unittest.TestCase):
    def setUp(self):
        f=fixtures.RuntimeEvidence();f.setUp();self.addCleanup(f.doCleanups)
        self.f=f;self.c=f.controller;self.m=f.monitor

    def feed(self,start,stop,blink=False):
        for i in range(start,stop):
            self.f.step(i*.2,[fixtures.record(signal=1. if blink and i%5 in (1,2) else 0.)])

    def test_fifteen_cycles_pass_before_five_minutes_but_search_continues(self):
        self.feed(0,100,True);key=next(iter(self.m.lamps))
        self.assertEqual(self.m.observations[key].count,15)
        self.assertLess(self.m.observations[key].valid_s,30)
        self.assertEqual(self.m.status(key,20),'normal')
        self.assertIn(key,self.m.accepted);self.assertEqual(self.m.bucket,0)
        self.feed(100,1503,False)
        self.assertIsNone(self.m.bucket);self.assertEqual(self.m.results[-1]['lamps'][0]['status'],'normal')

    def test_fourteenth_cycle_can_be_followed_by_fifteenth_after_300_valid_seconds(self):
        self.feed(0,70,True);self.feed(70,1510,False)
        key=next(iter(self.m.lamps));counter=self.m.observations[key]
        self.assertEqual(counter.count,14);self.assertGreater(counter.valid_s,300)
        self.assertEqual(self.m.bucket,0);self.assertNotIn(key,self.m.accepted)
        self.feed(1510,1516,True)
        self.assertIn(key,self.m.accepted);self.assertGreater(self.m.accepted[key]['accepted_at'],300)
        self.assertEqual(counter.count,15)

    def test_zero_flash_is_not_finalized_at_300_and_can_recover_late(self):
        self.feed(0,1551,False);key=next(iter(self.m.lamps))
        self.assertEqual(self.m.bucket,0);self.assertGreater(self.m.observations[key].valid_s,300)
        self.feed(5000,5080,True)
        self.assertEqual(self.m.results[-1]['lamps'][0]['status'],'normal')
        self.assertLess(self.m.results[-1]['ended'],1500)

    def test_deadline_keeps_below_threshold_distinct_from_zero(self):
        self.feed(0,70,True);self.feed(70,1551,False)
        self.c.tick(1500,True)
        row=self.m.results[-1]['lamps'][0]
        self.assertEqual(row['status'],'below_threshold');self.assertEqual(row['pulses'],14)
        self.assertGreater(row['valid_s'],300);self.assertIsNone(self.m.bucket)
        self.c.tick(1800,True);self.assertEqual(self.m.bucket,1800)
        self.assertEqual(self.m.accepted,{});self.assertEqual(next(iter(self.m.observations.values())).count,0)

    def test_same_identity_recalibration_does_not_erase_accepted_normal(self):
        self.feed(0,100,True);key=next(iter(self.m.lamps));sealed=dict(self.m.accepted[key])
        old=self.m.observations[key]
        self.f.step(20,[fixtures.record(confirmed=False,observation_epoch=1,flash_signal=dict(valid=False,signal=0.))],
            events=[dict(event='signal_reference_changed',id=1,observation_reset=True)])
        self.assertIs(self.m.observations[key],old);self.assertEqual(self.m.accepted[key],sealed)
        self.assertEqual(old.count,15)
        self.f.step(20.2,[fixtures.record(observation_epoch=1,evidence_admitted=False,position_observable=False)])
        self.assertTrue(self.c.report_identity_ambiguous(key));self.assertEqual(self.m.accepted[key],sealed)

    def test_unregistered_geometry_still_invalidates_accepted_coordinates(self):
        self.feed(0,100,True);self.assertTrue(self.m.accepted)
        self.f.step(20,[],events=[dict(event='geometry_reset')])
        self.assertEqual(self.m.accepted,{})

    def test_new_source_selection_does_not_inherit_accepted_normal(self):
        self.feed(0,100,True);self.assertTrue(self.m.accepted)
        self.f.step(20,[fixtures.record(confirmed=False,observation_epoch=1,signal=1.)],
            events=[dict(event='independent_source_reset',id=1,observation_reset=True)])
        self.assertEqual(self.m.accepted,{})
        self.assertEqual(next(iter(self.m.observations.values())).count,0)

    def test_two_persistent_ids_merge_without_adding_accepted_streams(self):
        for i in range(100):
            signal=1. if i%5 in (1,2) else 0.
            self.f.step(i*.2,[fixtures.record(1,signal=signal),fixtures.record(2,signal=signal,box=[80,60,88,68],body_box=[80,60,88,68])])
        self.assertEqual(len(self.m.accepted),2)
        self.f.step(20,[fixtures.record(1,aliases=[2],observation_epoch=1)],events=[dict(event='nested_identity_resolved',retained=1,aliases=[2])])
        self.assertEqual(len(self.m.lamps),1);self.assertEqual(self.m.accepted,{})
        self.assertEqual(next(iter(self.m.observations.values())).count,0)

    def test_raw_counter_keeps_counting_after_five_minutes_and_rejects_gap_cycle(self):
        counter=PulseCounter();params=LampParams()
        for i in range(1601):counter.sample(i*.2,0.,True,params)
        self.assertGreater(counter.valid_s,300)
        counter.sample(320.2,40.,True,params);counter.sample(500.,0.,True,params)
        self.assertEqual(counter.count,0)
        counter.sample(500.2,40.,True,params);counter.sample(500.4,0.,True,params)
        self.assertEqual(counter.count,1)


if __name__=='__main__':unittest.main()
