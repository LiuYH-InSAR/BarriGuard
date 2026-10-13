"""Optional acceleration must not stop replay when Windows blocks its DLL."""
import builtins
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class BackendFallbackTests(unittest.TestCase):
    def test_blocked_numba_uses_numpy(self):
        original_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == 'numba':
                raise OSError('[WinError 4551] DLL blocked by application policy')
            return original_import(name, *args, **kwargs)

        path = Path(__file__).resolve().parents[1] / 'tools/background_kernels.py'
        with patch('builtins.__import__', side_effect=blocked_import):
            module = runpy.run_path(str(path))

        self.assertEqual(module['BACKEND'], 'numpy')
        x = np.array([[100, 150, 100, 150]], dtype=np.float32)
        mask = np.array([[False, False, True, False]])
        initial = np.array([100], dtype=np.float32)
        updates = np.column_stack((np.zeros(4), np.full(4, .01)))
        count, total, peak, background = module['replay'](
            x, mask, initial, updates, 40)
        np.testing.assert_array_equal(count, [2])
        np.testing.assert_array_equal(total, [3])
        np.testing.assert_array_equal(peak, [50])
        np.testing.assert_array_equal(background, [100])
