"""One bounded-memory result summary for every optional local replay."""
import html
import json
from pathlib import Path

import cv2 as cv
import numpy as np

POLY_ALPHA = .20
POLY_COLOUR = (150, 230, 150)


def draw_label(image, text, x, y, colour):
    font, scale = cv.FONT_HERSHEY_SIMPLEX, .55
    (width, height), baseline = cv.getTextSize(text, font, scale, 1)
    x = max(0, min(x, image.shape[1]-width-8))
    y = max(height+6, min(y, image.shape[0]-baseline-4))
    cv.rectangle(image, (x, y-height-5), (x+width+8, y+baseline+3), (25, 25, 25), -1)
    cv.putText(image, text, (x+4, y), font, scale, colour, 1, cv.LINE_AA)


class RunSummary:
    def __init__(self):
        self.frame = None
        self.rank = (-1, -1)
        self.selected = {}
        self.active = {}
        self.tracks = {}
        self.serial = 0
        self.frames = 0
        self.alarm_frames = 0
        self.suspected_frames = 0

    def add(self, frame, index, now, algo):
        live = {id(t) for t in algo._tracks}
        self.active = {key: value for key, value in self.active.items() if key in live}
        boxes = []
        for track in algo._tracks:
            key = id(track)
            if key not in self.active:
                self.serial += 1
                self.active[key] = self.serial
            token = self.active[key]
            if track['state'] not in ('ALARM', 'SUSPECTED'):
                continue
            item = self.tracks.setdefault(token, dict(id=token, row=track['row_id'],
                first_s=now, last_s=now, peak_state=track['state']))
            item.update(last_s=now, box=list(track['box']))
            if track['state'] == 'ALARM':
                item['peak_state'] = 'ALARM'
            boxes.append(dict(id=token, row=track['row_id'], state=track['state'],
                              box=list(track['box']), rer=float(track['rer'])))
        alarms = sum(b['state'] == 'ALARM' for b in boxes)
        suspects = len(boxes) - alarms
        self.alarm_frames += bool(alarms)
        self.suspected_frames += bool(suspects)
        self.frames += 1
        rank = (alarms, suspects)
        if rank > self.rank:
            self.rank = rank
            self.frame = frame.copy()
            self.selected = dict(frame=index, time_s=now, boxes=boxes)

    def finish(self, out, video, camera, fps, polys=None, block_spec=None, trim=.04):
        out = Path(out)
        result = dict(video=str(video), camera=camera, frames=self.frames,
            duration_s=(self.frames-1)/fps, alarm_frames=self.alarm_frames,
            suspected_frames=self.suspected_frames, tracks=list(self.tracks.values()),
            representative=self.selected, polys=polys or [])
        image = self.frame.copy()
        # Block estimation runs only on the representative original frame, after detection.
        if block_spec is not None:
            from water_blocks import estimate_blocks
            result['block_windows'] = estimate_blocks(self.frame, polys or [], trim, block_spec)
        if not cv.imwrite(str(out/'summary_raw.png'), self.frame):
            raise OSError('Cannot write summary_raw.png')
        # Draw calibrated coverage first so anomaly boxes remain unobscured.
        overlay = image.copy()
        height, width = image.shape[:2]
        polygons = [(row['id'], np.rint(np.asarray(row['poly']) * [width, height]).astype(np.int32))
                    for row in polys or []]
        for _rid, points in polygons:
            cv.fillPoly(overlay, [points], POLY_COLOUR)
        image = cv.addWeighted(overlay, POLY_ALPHA, image, 1-POLY_ALPHA, 0)
        for rid, points in polygons:
            cv.polylines(image, [points], True, POLY_COLOUR, 1, cv.LINE_AA)
            x, y = points.mean(axis=0).astype(int)
            draw_label(image, f'poly: {rid}', int(x), int(y), POLY_COLOUR)
        for block in result.get('block_windows', {}).get('blocks', []):
            x0, y0, x1, y1 = map(int, block['box'])
            cv.rectangle(image, (x0, y0), (x1, y1), (255, 100, 30), 2)
            label = f"{block['colour'][0].upper()}{block['id']}"
            cv.putText(image, label, (x0+2, max(12, y1-5)), cv.FONT_HERSHEY_SIMPLEX,
                       .40, (255, 100, 30), 1, cv.LINE_AA)
        if block_spec is not None and not cv.imwrite(str(out/'blocks.png'), image):
            raise OSError('Cannot write blocks.png')
        for box in self.selected['boxes']:
            x0, y0, x1, y1 = map(int, box['box'])
            colour = (0, 0, 255) if box['state'] == 'ALARM' else (0, 165, 255)
            cv.rectangle(image, (x0, y0), (x1, y1), colour, 3)
            label = f"#{box['id']} {box['row']} {box['state']} RER={box['rer']:.3f}"
            y = max(18, y0-8)
            draw_label(image, label, x0, y, colour)
        label = f"t={self.selected['time_s']:.2f}s | ALARM={self.rank[0]} SUSPECTED={self.rank[1]}"
        draw_label(image, label, 10, 25, (255, 255, 255))
        if not cv.imwrite(str(out/'summary.png'), image):
            raise OSError('Cannot write summary.png')
        (out/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        alarms = sum(t['peak_state'] == 'ALARM' for t in self.tracks.values())
        rows = ''.join(f"<tr><td>{t['id']}</td><td>{html.escape(t['row'])}</td>"
            f"<td>{t['peak_state']}</td><td>{t['first_s']:.2f}–{t['last_s']:.2f}s</td></tr>"
            for t in self.tracks.values())
        block_detail = ''
        if 'block_windows' in result:
            preview = result['block_windows']
            params = preview['parameters']
            entries = ''.join(f"<tr><td>{html.escape(b['row'])}/B{b['id']}</td>"
                f"<td>{'红' if b['colour'] == 'red' else '白'}</td>"
                f"<td>{b['start']}–{b['end']}</td><td>{b['width']}</td><td>{b['width_px']:.1f}</td>"
                f"<td>{b['score']:.3f}</td><td>{b['valid_ratio']:.1%}</td></tr>" for b in preview['blocks'])
            block_detail = f'''<h2>蓝框：下部红白连续分块（{len(preview['blocks'])}个）</h2>
<p>R=下部红色优势段；W=下部白色优势段。两条信号分别滤波，红色门槛≥{params['score_min']}，白色门槛≥{params['white_score_min']}，连续同色站合为一段。
有效比例≥{params['valid_ratio']:.0%}；高斯sigma={params['smooth_sigma']}；局部参考宽=poly高度×{params['width_ratio']}，仅供对比、不强制切段。
两色都弱或相等时不分块；不参与缺口报警，仅分析代表帧。站范围0–{params['ref']-1}。</p>
<details><summary>每个块的站范围与参数</summary><table><tr><th>编号</th><th>颜色</th><th>站范围</th>
<th>站数</th><th>像素宽</th><th>优势颜色均值</th><th>有效比例</th></tr>{entries}</table></details>'''
        body = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>水马检测总结果</title>
<style>body{{font:16px 'Microsoft YaHei',sans-serif;margin:24px}}img{{max-width:100%}}td,th{{padding:8px;border:1px solid #ddd}}table{{border-collapse:collapse}}</style>
<h1>本次水马检测总结果</h1><p>{html.escape(str(video))}；相机{html.escape(camera)}；
检查0–{result['duration_s']:.2f}s，共{self.frames}帧。</p>
<p>算法报警轨道{alarms}条；仅疑似轨道{len(self.tracks)-alarms}条。
有报警的帧{self.alarm_frames}；有疑似的帧{self.suspected_frames}（可同时出现）。轨道数不等于真实缺口数。</p>
<p>浅绿色透明区域=poly检测范围；红框=算法报警；橙框=疑似。图为报警框最多、其次疑似框最多的一帧（{self.selected['time_s']:.2f}s），下表汇总整个测试片段。
没有框表示本次未产生报警或疑似，不能据此评价准确率。</p><img src="summary.png">
<table><tr><th>轨道</th><th>区域</th><th>最高状态</th><th>异常首次–末次出现</th></tr>{rows}</table>{block_detail}</html>'''
        (out/'summary.html').write_text(body, encoding='utf-8')
        print(f"Result: ALARM tracks={alarms}; suspected-only tracks={len(self.tracks)-alarms}; "
              f"tested={result['duration_s']:.2f}s; summary={out.resolve() / 'summary.html'}")
