"""Read a TXT or XLSX point list and export Gray diagnostics in one folder."""
import argparse
import csv
import html
import hashlib
import json
import posixpath
import re
import shutil
import sys
import zipfile
from pathlib import Path
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

from inspect_gray import export_position

NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'


def read_points(path, sheet_name=None):
    """Return actual Excel row numbers and first two cell values (no dependencies)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Input file not found: {path.resolve()}')
    if path.suffix.lower() == '.txt':
        result = []
        for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(maxsplit=1)
            video = parts[0]
            coordinate = parts[1].strip() if len(parts) > 1 else ''
            pair = re.fullmatch(r'(\d+)\s+(\d+)', coordinate)
            if pair:
                coordinate = ','.join(pair.groups())
            result.append((number, video, coordinate))
        return result
    if path.suffix.lower() != '.xlsx':
        raise ValueError('Input must be a UTF-8 .txt or .xlsx file')
    with zipfile.ZipFile(path) as archive:
        book = ET.fromstring(archive.read('xl/workbook.xml'))
        sheets = book.findall('s:sheets/s:sheet', NS)
        sheet = next((s for s in sheets if s.get('name') == sheet_name), None) if sheet_name else sheets[0]
        if sheet is None:
            raise ValueError(f'Worksheet not found: {sheet_name}')
        relationships = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        relation = next(r for r in relationships if r.get('Id') == sheet.get(f'{{{REL}}}id'))
        if relation.get('TargetMode') == 'External':
            raise ValueError('External worksheet is unsupported')
        target = relation.get('Target')
        member = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
        if not member.startswith('xl/'):
            raise ValueError('Invalid worksheet path')
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared = [''.join(t.text or '' for t in item.findall('.//s:t', NS))
                      for item in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        result = []
        for row in ET.fromstring(archive.read(member)).findall('s:sheetData/s:row', NS):
            cells = {}
            for cell in row.findall('s:c', NS):
                column = re.sub(r'\d', '', cell.get('r', ''))
                if column not in ('A', 'B'):
                    continue
                value = cell.findtext('s:v', default='', namespaces=NS)
                if cell.find('s:f', NS) is not None:
                    value = '[formula unsupported; enter text]'
                elif cell.get('t') == 's':
                    value = shared[int(value)]
                elif cell.get('t') == 'inlineStr':
                    value = ''.join(t.text or '' for t in cell.findall('.//s:t', NS))
                cells[column] = value.strip()
            a, b = cells.get('A', ''), cells.get('B', '')
            if not a and not b:
                continue
            if a in ('影像', '影像编号', '视频', '视频编号', 'video', 'video_id') and ('像素' in b or 'col,row' in b.lower() or b.lower() == 'pixel'):
                continue
            result.append((int(row.get('r')), a, b))
        return result


def export_table(xlsx, results, output, sheet=None):
    source = Path(xlsx).resolve()
    points = read_points(source, sheet)
    source_bytes = source.read_bytes()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plots = output / 'plots'
    plots.mkdir(exist_ok=True)
    records = []
    dates = sorted({video.split('_')[1] for _,video,_ in points if re.fullmatch(r'\d+_\d{8}_\d{4}(?:\.mp4)?',video)})
    print(f'Input file: {source}; rows={len(points)}; dates={dates}')
    for number, video, coordinate in points:
        record = dict(excel_row=number, video=video, col='', row='', status='error', error='', pixel_plot='', region_plot='', data_folder='')
        try:
            video = video.removesuffix('.mp4')
            if not re.fullmatch(r'\d+_\d{8}_\d{4}', video):
                raise ValueError('影像编号应为完整编号，例如 1002490_20260918_2200')
            match = re.fullmatch(r'\s*\(?\s*(\d+)\s*[,，]\s*(\d+)\s*\)?\s*', coordinate)
            if not match:
                raise ValueError('像素号应为 col,row，例如 1138,428；不支持公式')
            col, row = map(int, match.groups())
            record.update(video=video, col=col, row=row)
            matrix = Path(results) / video / 'gray_matrix'
            if not (matrix / 'manifest.json').exists():
                raise FileNotFoundError(f'未找到保存的 Gray 矩阵: {matrix / "manifest.json"}')
            key = f'{video}_col_{col}_row_{row}_excel_{number}'
            data = export_position(matrix, row, col, output / 'data' / key)
            for kind in ('pixel', 'region'):
                target = plots / f'{key}_{kind}_gray.svg'
                shutil.copy2(data / f'{kind}_gray.svg', target)
                record[f'{kind}_plot'] = target.relative_to(output).as_posix()
            record.update(status='ok', data_folder=data.relative_to(output).as_posix())
        except (ValueError, OSError, KeyError) as exc:
            record['error'] = str(exc)
        records.append(record)
        message = f'Input row {number}: {record["status"]} {video} {coordinate} {record["error"]}'
        encoding = getattr(sys.stdout, 'encoding', None) or 'utf-8'
        print(message.encode(encoding, errors='backslashreplace').decode(encoding))
    fields = list(records[0]) if records else ['excel_row', 'video', 'col', 'row', 'status', 'error', 'pixel_plot', 'region_plot', 'data_folder']
    with (output / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    body = []
    for item in records:
        links = ' '.join(f'<a href="{html.escape(item[k], quote=True)}">{label}</a>' for k, label in [('pixel_plot', '单像素 plot'), ('region_plot', '邻域 plot'), ('data_folder', '数据')] if item[k])
        body.append('<tr>' + ''.join(f'<td>{html.escape(str(item[k]))}</td>' for k in ['excel_row', 'video', 'col', 'row', 'status', 'error']) + f'<td>{links}</td></tr>')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>漏检点检查</title><style>body{font-family:sans-serif}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style><h1>漏检点检查</h1><p>坐标为原视频零起始 col,row；plot 沿用现有单点导出逻辑。</p><table><tr><th>Excel行</th><th>影像</th><th>col</th><th>row</th><th>状态</th><th>错误</th><th>结果</th></tr>' + ''.join(body) + '</table>', encoding='utf-8')
    summary = {'total': len(records), 'ok': sum(r['status'] == 'ok' for r in records), 'failed': sum(r['status'] != 'ok' for r in records)}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'export_manifest.json').write_text(json.dumps({
        'input_file':str(source), 'input_sha256':hashlib.sha256(source_bytes).hexdigest(),
        'input_modified_utc':datetime.fromtimestamp(source.stat().st_mtime,timezone.utc).isoformat(),
        'exported_utc':datetime.now(timezone.utc).isoformat(),
        'results_root':str(Path(results).resolve()), 'sheet':sheet,
        'dates':dates, 'videos':sorted({r['video'] for r in records}),
        'summary':summary,
        'current_plots':[r[k] for r in records for k in ('pixel_plot','region_plot') if r[k]],
        'note':'index.html and summary.csv list this export only; older unlisted files may remain in plots/data'
    },indent=2,ensure_ascii=False),encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', '--txt', '--xlsx', dest='input_file', required=True,
                        help='UTF-8 TXT (video col row) or XLSX point list')
    parser.add_argument('--results', required=True, help='Batch result root containing video folders')
    parser.add_argument('--out', help='Unified inspection folder; default: RESULTS/point_checks')
    parser.add_argument('--sheet', help='Worksheet name; default: first worksheet')
    args = parser.parse_args()
    try:
        summary = export_table(args.input_file, args.results, args.out or Path(args.results) / 'point_checks', args.sheet)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    print(summary)
    raise SystemExit(1 if summary['failed'] else 0)


if __name__ == '__main__':
    main()
