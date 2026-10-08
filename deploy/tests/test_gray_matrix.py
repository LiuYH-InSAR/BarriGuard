"""Spatial matrix and arbitrary valid coordinates survive the detector rollback."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'deploy/app'))
sys.path.insert(0, str(ROOT / 'deploy/tools'))
from gray_matrix import GrayRecorder
from inspect_gray import export_position
from gray_stages import stage_details


class MatrixTests(unittest.TestCase):
    def test_saved_background_and_colour_query(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temp:
            recorder = GrayRecorder(temp)
            recorder.mref = 30
            recorder.parameters = {'on_delta':40,'baseline_frames':60,'onset_min':20}
            mask = np.zeros((5,5),bool)
            base = np.full((5,5),30,np.float32)
            for i in range(64):
                value = 80 if i%2 else 30
                gray = np.full((5,5),value,np.uint8)
                recorder.append(gray,mask,0,i*.16,
                    bgr=np.repeat(gray[:,:,None],3,axis=2),background=base,update=(0,0))
            recorder.finish()
            out = export_position(temp,2,2)
            data = np.load(out/'diagnostics.npz')
            self.assertEqual(int(data['pixel_on_hit'].sum()),32)
            np.testing.assert_equal(data['pixel_background'],30)
            self.assertEqual(data['region_bgr_corrected'].shape,(64,3))
            import json
            stages = json.loads((out/'stage_details.json').read_text())
            self.assertEqual(stages['pixel']['discovery']['exceed_count'],32)
            self.assertEqual(stages['region']['states']['yellow_score'],0)
            data.close()

    def test_stages_distinguish_exceed_samples_from_onsets(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temp:
            folder = Path(temp) / 'gray_matrix'
            folder.mkdir()
            count = np.zeros((8,9), np.float32); count[4,5] = 40
            np.save(Path(temp) / 'on_count.npy', count)
            (Path(temp) / 'duty_stats.json').write_text('{"sample_total":120}')
            signal = np.tile([30,30,80,80,30,30],20).astype(float)
            meta = {'parameters':{'on_delta':40,'onset_min':20}}
            result, lines = stage_details(folder,meta,signal,signal,signal,30,4,5)
            self.assertEqual(result['discovery']['exceed_count'],40)
            self.assertEqual(result['periodicity']['onsets'],20)
            self.assertEqual(len(result['states']['weights']),2)
            self.assertTrue(any('BIC gain' in line for line in lines))
            self.assertTrue(any('threshold>=20' in line for line in lines))

    def test_matrix_and_coordinate_query(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as folder:
            recorder = GrayRecorder(folder)
            mask = np.ones((8, 9), bool)
            mask[2:7, 3:8] = False
            for i in range(64):
                frame = np.full((8, 9), 30 + i % 2 * 40, np.uint8)
                recorder.append(frame, mask, 3, i * .16)
            recorder.finish()
            matrix = np.load(Path(folder) / 'gray_raw.npy', mmap_mode='r')
            self.assertEqual(matrix.shape, (5, 5, 64))
            out = export_position(folder, 4, 5)
            saved = np.load(out / 'series.npz')
            np.testing.assert_allclose(saved['pixel_gray_raw'], 30 + np.arange(64) % 2 * 40)
            np.testing.assert_allclose(saved['pixel_gray_corrected'], saved['pixel_gray_raw'] - 3)
            np.testing.assert_allclose(saved['region_gray_corrected'], saved['pixel_gray_corrected'])
            np.testing.assert_allclose(saved['time_s'], np.arange(64) * .16)
            self.assertTrue((out / 'pixel_gray.svg').exists())
            self.assertTrue((out / 'position.jpg').exists())
            saved.close()
            del matrix
            with self.assertRaises(ValueError):
                export_position(folder, 1, 5)


if __name__ == '__main__':
    unittest.main()
