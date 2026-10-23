"""Regression: an unreadable recording must not abort the remaining batch."""
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import crop_poly_videos as crop


class BatchTest(unittest.TestCase):
    def test_bad_video_continues(self):
        errors = io.StringIO()
        jobs = [('bad.mp4', '1749', None), ('good.mp4', '1749', None)]
        with patch.object(crop, 'extract', side_effect=[RuntimeError('Cannot open'), 3]):
            total, failures = crop.run_jobs(jobs, SimpleNamespace(), io.StringIO(), errors)
        self.assertEqual((total, failures), (3, 1))
        self.assertIn('bad.mp4', errors.getvalue())


if __name__ == '__main__':
    unittest.main()
