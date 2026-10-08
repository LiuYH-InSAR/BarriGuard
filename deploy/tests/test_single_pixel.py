"""Supplement admits isolated recurrent signals without altering opening."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from night_lamp import single_pixel as sp
from night_lamp.tools import nightly_map as nm


class SinglePixelTests(unittest.TestCase):
    def test_candidates_preserve_original_opening(self):
        duty=np.zeros((20,20),np.float32);duty[10,10]=.25
        opened=nm.split_comps(duty)[1]>0
        self.assertFalse(opened.any())
        count=np.zeros_like(duty);count[10,10]=20
        valid=np.zeros_like(duty,bool)
        self.assertEqual(sp.candidates(duty,count,valid,opened),[(10,10)])
        count[10,10]=19
        self.assertEqual(sp.candidates(duty,count,valid,opened),[])
        count[10,10]=20;valid[10,10]=True
        self.assertEqual(sp.candidates(duty,count,valid,opened),[])

    def test_pixel_gate_without_region_dilution(self):
        t=np.arange(400)*.16;gray=20+(np.arange(400)%4==0)*80
        period=SimpleNamespace(onset_min=20,peak_floor=.2,min_peaks=3,gap_tol=1)
        with patch.dict('night_lamp.disturbance.PARAMETERS',enabled=False):
            result=sp.analyse(gray,t,period,2,44)
            self.assertTrue(result['flashing'])
            self.assertEqual(result['evidence_signal'],'pixel_gray')
            masked=sp.analyse(gray,t,period,2,44,np.ones(400,bool))
            self.assertFalse(masked['flashing'])
            self.assertEqual(masked['samples'],0)
            short=sp.analyse(gray[:80],t[:80],period,2,44)
            self.assertFalse(short['flashing'])
            self.assertEqual(short['reason'],'insufficient_effective_duration')


if __name__=='__main__':unittest.main()
