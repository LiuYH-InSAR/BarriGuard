import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'deploy/tools'))
from inspect_gray_batch import read_points, export_table


class BatchTests(unittest.TestCase):
    def test_txt_different_dates_use_exact_video_folder(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
            temp = Path(temp)
            text = temp/'points.txt'
            text.write_text('1002490_20260918_1900 1138 428\n1002490_20260921_1900 1138,428\n',encoding='utf-8-sig')
            seen=[]
            for date in ('20260918','20260921'):
                folder=temp/'results'/f'1002490_{date}_1900'/'gray_matrix'
                folder.mkdir(parents=True)
                (folder/'manifest.json').write_text('{}')
            def fake(folder,row,col,out):
                seen.append(folder.parent.name)
                out.mkdir(parents=True)
                for name in ('pixel_gray.svg','region_gray.svg'):
                    (out/name).write_text('<svg/>')
                return out
            with patch('inspect_gray_batch.export_position',side_effect=fake):
                result=export_table(text,temp/'results',temp/'checks')
            self.assertEqual(seen,['1002490_20260918_1900','1002490_20260921_1900'])
            self.assertEqual(result['ok'],2)
            import json
            manifest=json.loads((temp/'checks/export_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['dates'],['20260918','20260921'])

    def test_shared_strings_and_error_continue(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temp:
            temp = Path(temp)
            xlsx = temp / 'points.xlsx'
            with zipfile.ZipFile(xlsx, 'w') as z:
                z.writestr('xl/workbook.xml', '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="points" r:id="r1"/></sheets></workbook>')
                z.writestr('xl/_rels/workbook.xml.rels', '<Relationships><Relationship Id="r1" Target="worksheets/sheet1.xml"/></Relationships>')
                z.writestr('xl/sharedStrings.xml', '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><si><t>1002490_20260918_2200</t></si></sst>')
                z.writestr('xl/worksheets/sheet1.xml', '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="2"><c r="A2" t="s"><v>0</v></c><c r="B2" t="inlineStr"><is><t>bad</t></is></c></row><row r="4"><c r="A4" t="s"><v>0</v></c><c r="B4" t="inlineStr"><is><t>1138，428</t></is></c></row></sheetData></worksheet>')
            self.assertEqual(read_points(xlsx)[1], (4, '1002490_20260918_2200', '1138，428'))
            matrix = temp / 'results/1002490_20260918_2200/gray_matrix'
            matrix.mkdir(parents=True)
            (matrix / 'manifest.json').write_text('{}')

            def fake_export(folder, row, col, out):
                self.assertEqual((row, col), (428, 1138))
                out.mkdir(parents=True)
                for name in ('pixel_gray.svg', 'region_gray.svg'):
                    (out / name).write_text('<svg/>')
                return out

            with patch('inspect_gray_batch.export_position', side_effect=fake_export):
                result = export_table(xlsx, temp / 'results', temp / 'checks')
            self.assertEqual(result, {'total': 2, 'ok': 1, 'failed': 1})
            self.assertEqual(len(list((temp / 'checks/plots').glob('*.svg'))), 2)
            self.assertTrue((temp / 'checks/index.html').exists())


if __name__ == '__main__':
    unittest.main()
