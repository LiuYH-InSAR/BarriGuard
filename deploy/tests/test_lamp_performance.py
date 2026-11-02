"""Performance changes must preserve masks, quality state and optical rules."""
from pathlib import Path
import importlib.util
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import cv2 as cv
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'deploy/app'))
from night_lamp.quality import StreamQuality
from night_lamp.adaptive.adaptive_tracking_v5 import BackgroundAnchor
from night_lamp.adaptive.optical_fixture_v9 import potential_fixture,_potential_fixture
from server.performance import configure_native_threads


class NativeThreads(unittest.TestCase):
    def test_unconfigured_policy_does_not_import_or_change_libraries(self):
        with patch('server.performance.import_module',side_effect=AssertionError('unexpected import')):
            self.assertIsNone(configure_native_threads({}))

    def test_invalid_policy_fails_before_importing_models(self):
        for value in ('0','-1','bad'):
            with self.assertRaises(ValueError):configure_native_threads({'BARRIGUARD_OPENCV_THREADS':value})

    def test_import_side_effect_happens_before_the_explicit_budget(self):
        state=dict(cv=28,torch=20)
        cv=SimpleNamespace(getNumThreads=lambda:state['cv'],setNumThreads=lambda n:state.update(cv=n))
        torch=SimpleNamespace(get_num_threads=lambda:state['torch'],set_num_threads=lambda n:state.update(torch=n))
        def imported(name):
            if name=='ultralytics':state['cv']=0;return SimpleNamespace()
            return cv if name=='cv2' else torch
        with patch('server.performance.import_module',side_effect=imported):
            actual=configure_native_threads({'BARRIGUARD_OPENCV_THREADS':'8','BARRIGUARD_TORCH_THREADS':'8'})
        self.assertEqual(actual,dict(opencv_threads=8,torch_threads=8))


class QualityReuse(unittest.TestCase):
    def test_shared_pixel_metrics_preserve_independent_stream_continuity(self):
        outer=StreamQuality();inner=StreamQuality();oracle=StreamQuality()
        base=np.random.default_rng(4).integers(0,255,(90,160,3),dtype=np.uint8)
        for i,ts in enumerate([0.,.2,1.,1.2,1.4,2.8,3.]):
            frame=base.copy();frame[0,0]=i
            outer.observe(frame,ts,i)
            if i<2:continue  # Inner stream starts later than the outer one.
            inner.prepared_metrics=outer.last_metrics
            with patch('night_lamp.quality.cv.Laplacian',side_effect=AssertionError('duplicate pixel work')):
                actual=inner.observe(frame,ts,i)
            inner.prepared_metrics=None
            self.assertEqual(actual,oracle.observe(frame,ts,i))
        self.assertFalse(actual['continuous'] is None)

    def test_metrics_do_not_apply_to_another_frame_or_timestamp(self):
        q=StreamQuality();image=np.random.default_rng(4).integers(0,255,(90,160,3),dtype=np.uint8)
        q.observe(image,0,0);q.prepared_metrics=q.last_metrics
        dark=np.zeros_like(image)
        self.assertFalse(q.observe(dark,.2,1)['clear'])

    def test_reuse_preserves_the_receiving_streams_own_thresholds(self):
        image=np.random.default_rng(22).integers(0,255,(90,160,3),dtype=np.uint8)
        outer=StreamQuality();outer.observe(image,0,0)
        inner=StreamQuality(blur_floor=1e10);inner.prepared_metrics=outer.last_metrics
        self.assertFalse(inner.observe(image,0,0)['clear'])


class ExactGeometry(unittest.TestCase):
    @unittest.skipUnless((ROOT/'training/lamps/adaptive_v7.py').exists(),'local offline oracle not deployed')
    def test_far_neighbor_filter_preserves_every_support_pixel(self):
        sys.path.insert(0,str(ROOT/'training/lamps'))
        from adaptive_v7 import SupportEpochSignal as Original
        from night_lamp.adaptive.adaptive_v7 import SupportEpochSignal
        rng=np.random.default_rng(14)
        for radius in range(2,19):
            old,new=Original(),SupportEpochSignal();old.radius=new.radius=radius
            box=[100.,100.,100.+2*radius,100.+2*radius]
            points=rng.uniform(-300,500,(100,2)).tolist()
            old.neighbor_centers=new.neighbor_centers=points
            np.testing.assert_array_equal(old.desired_support(box),new.desired_support(box))

    @unittest.skipUnless((ROOT/'training/lamps/adaptive_tracking_v5.py').exists(),'local offline oracle not deployed')
    def test_vectorized_background_exclusions_equal_original_on_many_shapes(self):
        sys.path.insert(0,str(ROOT/'training/lamps'))
        from adaptive_tracking_v5 import BackgroundAnchor as Original
        rng=np.random.default_rng(79);image=rng.integers(0,255,(220,320),dtype=np.uint8)
        for i in range(80):
            box=[100.25,90.5,108.25,100.5]
            neighbors=[]
            for _ in range(i):
                x,y=rng.uniform(-50,350),rng.uniform(-50,270);w,h=rng.uniform(2,90,2)
                neighbors.append([x,y,x+w,y+h])
            new,old=BackgroundAnchor(image,box),Original(image,box)
            frame=np.clip(image.astype(float)+rng.normal(0,3,image.shape),0,255).astype(np.uint8)
            self.assertEqual(new.evaluate(frame,box,neighbors),old.evaluate(frame,box,neighbors))

    @unittest.skipUnless((ROOT/'training/lamps/optical_fixture_v9.py').exists(),'local offline oracle not deployed')
    def test_cached_geometry_is_the_same_rule_and_bounded(self):
        sys.path.insert(0,str(ROOT/'training/lamps'))
        from optical_fixture_v9 import potential_fixture as original
        rng=np.random.default_rng(11)
        for _ in range(1000):
            x,y=rng.uniform(-20,200,2);w,h=rng.uniform(4,30,2)
            outer=[x,y,x+w,y+h];inner=[x+2,y+2,x+w-2,y+h-2]
            self.assertEqual(potential_fixture(outer,inner),original(outer,inner))
        self.assertLessEqual(_potential_fixture.cache_info().currsize,16384)


if __name__=='__main__':unittest.main()
