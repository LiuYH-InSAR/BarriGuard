from copy import deepcopy
from datetime import datetime,timedelta,timezone
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from night_lamp.tracked import LampMonitor
from night_lamp.controller import LampController
from night_lamp.runtime import AdaptiveLampController


class ReportHealth(unittest.TestCase):
    def fixture(self):
        stamp=datetime(2026,10,31,0,tzinfo=timezone(timedelta(hours=8))).timestamp()
        monitor=LampMonitor('cam','ref');key=monitor.add_known([40,40,60,60],stamp,lamp_id='LampOne')
        lamp=monitor.lamps[key];lamp.observable=False
        controller=NS(monitor=monitor,offset=NS(pending=None),report_identity_ambiguous=lambda key:False)
        return stamp,controller,key,lamp

    def row(self,stamp,lamp,status='normal',valid=300.,pulses=30):
        row=dict(bucket=stamp,ended=stamp+300,valid_s=valid,pulses=pulses,status=status,reference=lamp.reference,lamp_id=lamp.lamp_id)
        lamp.history.append(row);return row

    def report(self,controller,key,lamp,stamp):
        return LampController.report_lamp(controller,key,lamp,stamp,stamp-24*3600)

    def test_last_image_obstruction_cannot_erase_this_windows_completed_normal(self):
        stamp,c,key,lamp=self.fixture();self.row(stamp,lamp)
        before=deepcopy(c.monitor.dump())
        self.assertEqual(c.monitor.status(key,stamp+400),'unobservable')
        row=self.report(c,key,lamp,stamp+400)
        self.assertEqual(row['status'],'normal');self.assertEqual(row['current_observability'],'unobservable')
        self.assertEqual(row['health_basis'],'completed_valid_window_despite_current_visibility')
        self.assertEqual(c.monitor.dump(),before)

    def test_later_empty_window_is_not_painted_green_from_an_earlier_normal(self):
        stamp,c,key,lamp=self.fixture();self.row(stamp,lamp)
        self.row(stamp+1800,lamp,'insufficient_observation',0.,0)
        self.assertEqual(self.report(c,key,lamp,stamp+2200)['status'],'insufficient_observation')

    def test_ambiguous_identity_or_unmapped_reference_remains_unknown(self):
        stamp,c,key,lamp=self.fixture();self.row(stamp,lamp)
        c.report_identity_ambiguous=lambda key:True
        self.assertEqual(self.report(c,key,lamp,stamp+400)['status'],'unobservable')
        c.report_identity_ambiguous=lambda key:False;c.monitor.reference='another-view'
        self.assertEqual(self.report(c,key,lamp,stamp+400)['status'],'unobservable')
        c.monitor.reference=lamp.reference;c.offset.pending=object()
        self.assertEqual(self.report(c,key,lamp,stamp+400)['status'],'unobservable')

    def test_real_adaptive_binding_rejects_quarantined_and_unbound_records(self):
        stamp,c,key,lamp=self.fixture()
        c.bindings={1:key};c.last_result=dict(tracks=[dict(id=1,evidence_admitted=False)])
        self.assertTrue(AdaptiveLampController.report_identity_ambiguous(c,key))
        c.last_result['tracks'][0]['evidence_admitted']=True
        self.assertFalse(AdaptiveLampController.report_identity_ambiguous(c,key))
        c.bindings={}
        self.assertTrue(AdaptiveLampController.report_identity_ambiguous(c,key))

    def test_07_report_uses_final_night_windows_for_six_valid_zero_proof(self):
        stamp,c,key,lamp=self.fixture()
        for i in range(6):self.row(stamp+i*1800,lamp,'no_flash',300.,0)
        self.assertEqual(self.report(c,key,lamp,stamp+7*3600)['status'],'not_flashing')
        lamp.history[-1].update(status='normal',pulses=30)
        self.assertEqual(self.report(c,key,lamp,stamp+7*3600)['status'],'normal')

    def test_incomplete_or_foreign_reference_window_cannot_prove_failure(self):
        stamp,c,key,lamp=self.fixture()
        for i in range(6):self.row(stamp+i*1800,lamp,'no_flash',300.,0)
        lamp.history[2]['valid_s']=0
        self.assertEqual(self.report(c,key,lamp,stamp+7*3600)['status'],'unobservable')
        lamp.history[2]['valid_s']=300.;lamp.history[2]['reference']='old-view'
        self.assertEqual(self.report(c,key,lamp,stamp+7*3600)['status'],'unobservable')


if __name__=='__main__':unittest.main()
