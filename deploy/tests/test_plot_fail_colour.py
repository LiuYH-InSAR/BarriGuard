import sys,tempfile,unittest
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'deploy/tools'))
sys.path.insert(0,str(ROOT/'deploy/app'))
from night_intermediates import _plot_svg


class PlotColourTests(unittest.TestCase):
    def test_table_columns_and_cell_colour(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
            path=Path(temp)/'plot.svg'
            table=[[dict(text='Step 3',failed=False),dict(text='gap=2 <12 FAIL',failed=True),dict(text='BIC=20 >=10 PASS',failed=False)],
                   [dict(text='Step 4',failed=False),dict(text='onsets=22 PASS',failed=False)]]
            _plot_svg(path,np.arange(80)*.16,np.tile([40.,80.],40),None,None,dict(stage_table=table,stage_lines=['supplement']))
            nodes=[n for n in ET.parse(path).iter() if n.tag.endswith('text')]
            failed=next(n for n in nodes if n.text=='gap=2 <12 FAIL')
            passed=next(n for n in nodes if n.text=='BIC=20 >=10 PASS')
            aligned=next(n for n in nodes if n.text=='onsets=22 PASS')
            self.assertEqual(failed.get('fill'),'#c62828')
            self.assertNotEqual(passed.get('fill'),'#c62828')
            self.assertEqual(failed.get('x'),aligned.get('x'))

    def test_failed_parameter_red_and_active_mask_named(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
            path=Path(temp)/'plot.svg'
            _plot_svg(path,np.arange(80)*.16,np.tile([40.,80.],40),None,None,
                      dict(stage_lines=['separation=1; FAIL','ratio=.3; PASS'],
                           mask_intervals=[[2.,4.]],mask_label='fallback 20s final mask'))
            nodes=[n for n in ET.parse(path).iter() if n.tag.endswith('text')]
            failed=next(n for n in nodes if n.text=='separation=1; FAIL')
            self.assertEqual(failed.get('fill'),'#c62828')
            self.assertTrue(any('fallback 20s final mask' in (n.text or '') for n in nodes))


if __name__=='__main__':unittest.main()
