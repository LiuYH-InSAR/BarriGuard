"""Fixed candidate identity survives changes in the highest-duty pixel."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'deploy/app'))
from night_lamp.session import CandidateDiscovery, TemporalSeries
from night_lamp.tools import nightly_map as nm


class BindingTests(unittest.TestCase):
    def test_original_opening_still_removes_single_pixel(self):
        duty=np.zeros((20,20),np.float32);duty[10,10]=.25
        self.assertEqual(len(nm.split_comps(duty)[3]),0)

    def test_representative_change_keeps_exact_original_row(self):
        series=TemporalSeries(30);discovery=CandidateDiscovery(series,10)
        duty=np.zeros((30,30),np.float32);duty[10:14,10:18]=.1
        duty[11,11]=.3
        on=np.full(duty.shape,500,np.float32);swing=duty.copy()
        discovery.observe(duty,on,500,swing)
        labels=nm.split_comps(duty)[1]
        original=discovery.binding(labels,labels[11,11]).copy()
        self.assertEqual((original['y'],original['x']),(11,11))
        duty[11,11]=.2;duty[12,16]=.4
        discovery.observe(duty,on,750,swing)
        labels=nm.split_comps(duty)[1]
        self.assertEqual(discovery.binding(labels,labels[12,16]),original)
        self.assertEqual(discovery.n_candidate_cores,1)
        row=original['series_row']
        self.assertEqual((int(series._pys[row]),int(series._pxs[row])),(11,11))
        # An unrelated component must not borrow this identity by proximity.
        labels[11,11]=0
        self.assertIsNone(discovery.binding(labels,labels[12,16]))
        discovery.release()
        self.assertEqual(discovery._bindings,[])


if __name__=='__main__':unittest.main()
