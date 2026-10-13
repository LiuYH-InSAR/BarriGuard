import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from night_lamp.stutter import shared_branch


class SharedTests(unittest.TestCase):
    def test_untriggered_and_dynamic_regions_do_not_spread(self):
        branch=np.zeros((12,12),bool)
        count=np.ones((12,12),np.float32);total=np.full((12,12),100)
        valid=np.ones((12,12),bool)
        self.assertFalse(shared_branch(branch,count,total,valid,2,.01,20).any())
        branch[5,5]=True
        actual=shared_branch(branch,count,total,valid,2,.01,20)
        np.testing.assert_array_equal(actual,branch)

    def test_same_component_and_lookup_share_path(self):
        branch=np.zeros((20,30),bool);branch[10,10]=True
        count=np.zeros((20,30),np.float32);count[10,10:15]=10
        total=np.full(count.shape,100);valid=np.ones(count.shape,bool)
        actual=shared_branch(branch,count,total,valid,5,.01,2000)
        self.assertTrue(actual[10,14])
        self.assertTrue(actual[10,19])
        self.assertFalse(actual[10,20])
        self.assertFalse(actual[0,29])
        valid[10,19]=False
        self.assertFalse(shared_branch(branch,count,total,valid,5,.01,2000)[10,19])


if __name__=='__main__':unittest.main()
