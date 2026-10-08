import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from night_lamp.disturbance import detect,detect_matrix,periodic_features

class DisturbanceTests(unittest.TestCase):
    def test_up_and_down_excursions_and_normal_pulses(self):
        t=np.arange(0,100,.16)
        x=100+60*(np.arange(len(t))%4==0)
        x[(t>=40)&(t<50)]+=50
        x[(t>=70)&(t<80)]-=45
        mask,mean,info=detect(x,t)
        self.assertFalse(mask[t<30].any())
        self.assertTrue(mask[(t>=40)&(t<50)].all())
        self.assertTrue(mask[(t>=70)&(t<80)].all())
        self.assertEqual(len(info['intervals']),2)
        self.assertFalse(mask[t>85].any())
        batch=detect_matrix(np.vstack([x,x+50]),t)
        np.testing.assert_array_equal(batch[0],mask)
        np.testing.assert_array_equal(batch[1],mask)

    def test_time_is_not_compressed_and_gap_does_not_make_onset(self):
        on=np.tile([False,False,True,False],50)
        valid=np.ones(len(on),bool);valid[40:80]=False
        period=SimpleNamespace(peak_floor=.2,min_peaks=3,gap_tol=1)
        n,peak,lag,curve,passed,peaks=periodic_features(on,valid,2,20,period)
        self.assertTrue(passed)
        self.assertEqual(lag,4)
        self.assertEqual(n,40)

if __name__=='__main__':unittest.main()
