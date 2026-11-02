"""Production lifecycle acceptance with synthetic time; no streams or callbacks."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import Mock
import unittest

import test_production_wiring as wiring
import test_adaptive_runtime as evidence
from server.contracts import ScheduleSpec
from server.schedule import Scheduler, RuntimeEventType
from server.worker import Runtime

HK=timezone(timedelta(hours=8))


class ReleaseLifecycle(unittest.TestCase):
    def setUp(self):
        f=wiring.ProductionWiring();f.setUp();self.addCleanup(f.doCleanups)
        self.f=f;self.worker=f.worker(water=False)
        self.clock=f.clock
        self.cfg=NS(schedule=ScheduleSpec(),cameras=[self.worker.camera])
        self.scheduler=Scheduler(self.cfg,self.clock)
        self.worker._scheduler=self.scheduler
        self.runtime=NS(workers={'cam':self.worker})
        self.worker._reader.start=Mock(side_effect=AssertionError('No RTSP allowed'))

    def advance(self,when):
        self.clock.t=when.timestamp()
        events=self.scheduler.poll()
        for event in events:Runtime.dispatch(self.runtime,event)
        self.worker._tick_night(when)
        return events

    def simulate_night(self,recovery=False,incomplete=False):
        self.advance(datetime(2026,11,1,20,tzinfo=HK))
        controller=self.worker._lamp_adapter.controller
        midnight=datetime(2026,11,2,tzinfo=HK)
        for window in range(6):
            start=midnight+timedelta(seconds=1800*window)
            self.advance(start)
            # Feed business-level signals into the real adaptive consumer.
            # Detector lifetime fields deliberately carry impossible totals.
            samples=200 if incomplete and window==2 else 1602
            for i in range(samples):
                wall=(start+timedelta(seconds=i*.2)).timestamp()
                signal=1. if recovery and window==5 and i%5 in (1,2) else 0.
                self.clock.t=wall
                controller.consume(evidence.result([evidence.record(signal=signal)]),wall,wall,True)
            self.advance(start+timedelta(minutes=25))
        self.worker._lamp_adapter._last_image=self.f.frame.copy()
        self.advance(datetime(2026,11,2,3,tzinfo=HK))
        self.assertIsNone(controller.monitor.bucket)
        self.advance(datetime(2026,11,2,7,tzinfo=HK))
        self.f.reporter.submit.assert_called_once()
        report=self.f.reporter.submit.call_args.args[0]
        self.assertTrue(report.image_jpeg)
        self.assertEqual(report.created_at,'2026-11-02T07:00:00+08:00')
        self.assertEqual(report.metadata['observation_policy']['normal_min_effective_s'],0)
        self.assertEqual(self.worker._night,{})
        self.advance(datetime(2026,11,2,7,tzinfo=HK)+timedelta(seconds=1))
        self.f.reporter.submit.assert_called_once()
        self.worker._reader.start.assert_not_called()
        return report.metadata['lamps'][0]

    def test_six_real_business_zero_windows_reach_red_at_07(self):
        lamp=self.simulate_night()
        self.assertEqual(lamp['status'],'not_flashing')
        self.assertEqual(lamp['observation']['pulses'],0)

    def test_final_window_recovery_is_green_at_07_without_300_seconds(self):
        lamp=self.simulate_night(recovery=True)
        self.assertEqual(lamp['status'],'normal')
        self.assertEqual(lamp['observation']['pulses'],15)
        self.assertLess(lamp['observation']['valid_s'],300)

    def test_short_quality_window_breaks_three_hour_failure_proof(self):
        lamp=self.simulate_night(incomplete=True)
        self.assertNotEqual(lamp['status'],'not_flashing')

    def test_restart_in_frozen_interval_still_emits_one_morning_report(self):
        events=self.advance(datetime(2026,11,2,4,tzinfo=HK))
        self.assertEqual([e.type for e in events],[RuntimeEventType.NIGHT_START,RuntimeEventType.NIGHT_FREEZE])
        self.assertTrue(self.worker._lamp_adapter._frozen)
        self.advance(datetime(2026,11,2,7,tzinfo=HK))
        self.f.reporter.submit.assert_called_once()
        self.assertEqual(self.f.reporter.submit.call_args.args[0].status,'degraded')

    def test_boot_at_exact_report_boundary_does_not_lose_the_morning_report(self):
        self.advance(datetime(2026,11,2,7,tzinfo=HK))
        self.f.reporter.submit.assert_called_once()
        self.advance(datetime(2026,11,2,7,tzinfo=HK)+timedelta(seconds=1))
        self.f.reporter.submit.assert_called_once()

    def test_connected_but_stalled_decoder_is_lost_until_a_fresh_frame_arrives(self):
        health=dict(connected=True,last_frame_mono=0.,started_mono=0.)
        self.worker._reader=NS(health=lambda:health)
        self.worker._check_stream_health(10.)
        self.assertEqual(self.worker.status(),'STREAM_LOST')
        self.worker._check_stream_health(11.)
        self.assertTrue(self.worker._stream_lost)
        health['last_frame_mono']=11.
        self.worker._check_stream_health(11.)
        self.assertEqual(self.worker.status(),'OK')
        self.assertEqual([e['payload']['status'] for e in self.worker.events()],['stream_lost','stream_restored'])

    def test_disconnected_zero_monotonic_time_still_reports_loss_and_heartbeat(self):
        self.worker._reader=NS(health=lambda:dict(connected=False,last_frame_mono=0.,started_mono=0.))
        self.worker._check_stream_health(10.)
        self.assertEqual(self.worker.status(),'STREAM_LOST')
        self.worker._check_stream_health(610.)
        self.assertEqual(len(self.worker.events()),2)


if __name__=='__main__':unittest.main()
