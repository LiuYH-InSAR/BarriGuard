"""Offline water traces and plots. No service imports or callback side effects."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
import hashlib
import html
import json
from pathlib import Path

import cv2 as cv
import numpy as np

from water_barrier import signal as S
from water_barrier import track as T

FAIL = "#c62828"
SKIP = "#777777"
WIDTH = 1500
COLOURS = ("#222222", "#2474b5", "#8653a8", "#e58b20", "#239466")


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def pack_tree(value, arrays, key="trace"):
    if isinstance(value, np.ndarray):
        arrays[key] = value
        return {"array": key}
    if isinstance(value, dict):
        return {str(k): pack_tree(v, arrays, f"{key}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [pack_tree(v, arrays, f"{key}.{i}") for i, v in enumerate(value)]
    if isinstance(value, np.generic):
        return pack_tree(value.item(), arrays, key)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def unpack_tree(value, arrays):
    if isinstance(value, dict):
        if set(value) == {"array"}:
            return arrays[value["array"]].copy()
        return {k: unpack_tree(v, arrays) for k, v in value.items()}
    if isinstance(value, list):
        return [unpack_tree(v, arrays) for v in value]
    return value


def image_file(path, image):
    if not cv.imwrite(str(path), image):
        raise OSError(f"Cannot write {path}")
    return str(path.relative_to(path.parents[1])).replace("\\", "/")


def parameters(calib, spec):
    """Freeze effective settings, including fixed rules used outside config.yaml."""
    return dict(
        detect=calib["detect"], track=calib["track"], rows=calib["rows"], road_rois=calib["road_rois"],
        sampling=dict(step=S.STEP, min_pixels=S.MINH, ref=S.REF, top_fraction=.6,
                      bottom_fraction=.4, red_weight=.6, white_weight=.4, erosion_width=7,
                      lower_colours="red_or_white"),
        colour=dict(red_h=[[0, 12], [165, 180]], red_s_min=70, red_v_min=50,
                    sun_s_max=70, sun_v_min=150, shadow_s=(70, 120), shadow_v=(80, 150),
                    shadow_h_min=85, foreign_s_min=100),
        quality=dict(block=32, sobel_kernel=3, gray_divisor=255, pmax=S.PMAX_TH,
                     foreign=S.FOREIGN_TH, foreign_span=S.FOREIGN_SPAN,
                     dark=S.DARK_TH, dark_span=S.DARK_SPAN),
        signal=dict(sigma=S.SMOOTH_SIGMA, kernel=S.SMOOTH_SIGMA * 6 + 1,
                    baseline_window=S.BASE_WIN, entry=.25, extension=.125, valid_ratio=.9,
                    unit_lo=.5, unit_hi=2.),
        rer=dict(min_area=200, profile_patch=8, l_sigma=2.5, ab_sigma_min=.1,
                 distance_percentile=95, distance_margin=.5, downgrade_updates=2),
        reporting=asdict(spec))


class Recorder:
    def __init__(self, out, video, fps, shape, calib, spec, page_every, at_s, max_pages):
        self.out = Path(out)
        for name in ("data", "images", "pages"):
            (self.out / name).mkdir(exist_ok=True)
        source_dir = Path(S.__file__).parent
        self.manifest = dict(schema=2, complete=False, video=str(video), fps=fps,
                             frame_shape=list(shape), time_source="frame_index/FPS (constant frame rate)",
                             parameters=parameters(calib, spec), snapshots=[], frames=0,
                             selection=dict(page_every=page_every, at_s=at_s, max_pages=max_pages),
                             source_sha256={name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                                            for name in ("signal.py", "track.py", "pipeline.py")})
        self.file = (self.out / "frames.jsonl").open("w", encoding="utf-8")
        self.period = page_every
        self.requested = sorted(at_s)
        self.next_page = 0.
        self.limit = max_pages
        self.previous = None
        self.serial = 0
        self.ids = {}
        self.frame_times = {}
        self.last_rer = {}
        self.suppressed = 0
        dump_json(self.out / "manifest.json", self.manifest)

    def add(self, index, now, frame, result, trace, algo):
        # Tokens are stable per run, even when Python later reuses an object address.
        previous_times = {token: list(times) for token, times in self.frame_times.items()}
        for match in trace["matches"]:
            token = match["token"]
            if token not in self.ids:
                self.serial += 1
                self.ids[token] = self.serial
                self.frame_times[token] = deque(maxlen=algo._tcfg["median_window"])
            self.frame_times[token].append(now)
        for tr in trace["tracks"]:
            token = tr["token"]
            tr["median_times"] = list(self.frame_times.get(token, []))
            tr["rer_frame"] = self.last_rer.get(token)
            tr["quality_skip"] = trace["skip"].get("all", trace["skip"].get(tr["row_id"]))
            tr["fresh_hit"] = any(
                rid == tr["row_id"] and tuple(box) == tuple(tr["box"])
                and any(a <= sta[1] and b >= sta[0] for a, b in trace["kept"].get(rid, []))
                for rid, sta, box, _sup in trace["gaps"])
        for check in trace["rer_checks"]:
            token = check["token"]
            self.last_rer[token] = index
            times = deque(previous_times.get(token, []), maxlen=algo._tcfg["median_window"])
            for match in trace["matches"][:check["push_ordinal"]]:
                if match["token"] == token:
                    times.append(now)
            check["median_times"] = list(times)
            for tr in trace["tracks"]:
                if tr["token"] == token:
                    tr["rer_frame"] = index
            for key, ext in (("median", "jpg"), ("foreground", "png"),
                             ("missing", "png"), ("road_sample", "jpg")):
                if key in check:
                    image = check.pop(key)
                    check[f"{key}_image"] = image_file(
                        self.out / "images" / f"rer_{index:08d}_{self.ids[token]}_{key}.{ext}", image)
        trace["support"] = {rid: dict(values=v.copy(), maximum=float(np.max(v)), hits=algo._sup["acc"][rid].copy(),
                                      observations=algo._sup["n_obs"][rid])
                            for rid, v in T.support_of(algo._sup).items()}
        self.remap(trace)
        signature = (tuple(sorted(trace["skip"].items())),
                     tuple((rid, len(row["kept"]), tuple(c["reason"] for c in row["candidates"]))
                           for rid, row in trace["rows"].items()),
                     tuple((t["token"], t["state"], t["matched"], t["fresh_hit"], t["removed"])
                           for t in trace["tracks"]))
        reasons = []
        if signature != self.previous:
            reasons.append("状态/候选/质量门变化")
        if now >= self.next_page:
            reasons.append("定时检查")
            self.next_page = now + self.period
        while self.requested and now >= self.requested[0]:
            reasons.append(f"指定时间 {self.requested.pop(0):g}s")
        selected = bool(reasons) and len(self.manifest["snapshots"]) < self.limit
        record = dict(index=index, time_s=now, trace=trace, selected=selected)
        if selected:
            record["image"] = image_file(self.out / "images" / f"frame_{index:08d}.jpg", frame)
            # Masks are computed on the original frame, never from the JPEG preview.
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            red, white = S.red_m(frame), S.white_m(frame)
            masks = np.zeros_like(frame)
            masks[red] = (0, 0, 255)
            masks[white] = (255, 255, 255)
            foreign = (hsv[:, :, 1] > 100) & (hsv[:, :, 0] > 12) & (hsv[:, :, 0] < 165)
            record["pixel_data"] = dict(hsv=hsv, red=red, white=white, foreign=foreign)
            masks[foreign] = (255, 0, 255)
            record["masks"] = image_file(self.out / "images" / f"masks_{index:08d}.png", masks)
            vis = render_frame(frame, algo, result, trace)
            record["overlay"] = image_file(self.out / "images" / f"overlay_{index:08d}.jpg", vis)
            self.manifest["snapshots"].append(dict(index=index, time_s=now, reasons=reasons))
        elif reasons:
            self.suppressed += 1
        self.previous = signature
        arrays = {}
        packed = pack_tree(record, arrays)
        packed["arrays_file"] = f"data/frame_{index:08d}.npz"
        np.savez_compressed(self.out / packed["arrays_file"], **arrays)
        self.file.write(json.dumps(packed, ensure_ascii=False, allow_nan=False) + "\n")
        self.manifest["frames"] = index + 1
        # Drop only diagnostic identifiers; algorithm state is untouched.
        removed = [token for token, serial in self.ids.items()
                   if any(t["token"] == serial and t["removed"] for t in trace["tracks"])]
        for token in removed:
            self.ids.pop(token, None)
            self.frame_times.pop(token, None)
            self.last_rer.pop(token, None)

    def remap(self, value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "token":
                    value[key] = self.ids[child]
                else:
                    self.remap(child)
        elif isinstance(value, list):
            for child in value:
                self.remap(child)

    def finish(self):
        self.manifest.update(complete=True, suppressed_snapshots=self.suppressed,
                             pending_requested_times=self.requested)
        self.close()
        dump_json(self.out / "manifest.json", self.manifest)
        replot(self.out)

    def close(self):
        if not self.file.closed:
            self.file.close()
        dump_json(self.out / "manifest.json", self.manifest)


def render_frame(frame, algo, result, trace):
    from server import render
    vis = render.draw_annots(render.overlay(frame, algo.fg), result.annots)
    for x0, y0, x1, y1 in algo._road_rects:
        cv.rectangle(vis, (x0, y0), (x1, y1), (0, 200, 255), 2)
    block = trace["quality"].get("block_yx")
    if block is not None:
        y, x = block
        cv.rectangle(vis, (int(x) * 32, int(y) * 32), (int(x + 1) * 32, int(y + 1) * 32), (255, 255, 0), 2)
    for rid, row in trace["rows"].items():
        for c in row["candidates"]:
            box = S.pixel_box(row["pack"], *c["env"], frame.shape[1], frame.shape[0])
            cv.rectangle(vis, box[:2], box[2:], (0, 140, 255), 1)
        pts = S.sta_to_xy(row["pack"], [0, S.REF - 1], frame.shape[1], frame.shape[0]).astype(int)
        cv.arrowedLine(vis, tuple(pts[0]), tuple(pts[1]), (255, 255, 0), 2)
    return vis


def esc(value):
    return html.escape(str(value))


def gate(name, value, limit, op=">="):
    if value is None:
        return (name, "SKIP", f"未执行；要求 {op} {limit}")
    passed = {">=": lambda: value >= limit, ">": lambda: value > limit,
              "<": lambda: value < limit, "<=": lambda: value <= limit}[op]()
    return (name, "PASS" if passed else "FAIL", f"{value:.6g} {op} {limit:.6g}")


def table(rows):
    return '<table><tr><th>步骤 / 参数</th><th>结果</th><th>实际值与判断条件</th></tr>' + ''.join(
        f'<tr style="color:{FAIL if status == "FAIL" else SKIP if status == "SKIP" else "#222"}">'
        f'<td>{esc(name)}</td><td>{esc(status)}</td><td>{esc(value)}</td></tr>'
        for name, status, value in rows) + '</table>'


def chart(title, xs, curves, limits=(), excluded=None, width=WIDTH, tick=None):
    """Fixed axes and labels; NaNs break paths, excluded regions keep curves visible."""
    xs = np.asarray(xs, float)
    lo, hi = float(xs[0]), float(xs[-1])
    if hi <= lo:
        hi = lo + 1
    finite = [np.asarray(v, float)[np.isfinite(v)] for _name, v in curves]
    values = np.concatenate([v for v in finite if len(v)] + [np.asarray([v for _n, v in limits] or [0., 1.])])
    low, high = min(0., float(values.min())), max(1., float(values.max()))
    high += (high - low) * .08
    left, right, top, bottom = 90, width - 35, 65, 255
    xmap = lambda v: left + (v - lo) / (hi - lo) * (right - left)
    ymap = lambda v: bottom - (v - low) / (high - low) * (bottom - top)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="320" viewBox="0 0 {width} 320">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="Arial,Microsoft YaHei,sans-serif" font-size="13">',
             f'<text x="90" y="20">{esc(title)}</text>']
    if excluded is not None:
        excluded = np.asarray(excluded, bool)
        for i in np.flatnonzero(excluded):
            a = xs[i]
            b = xs[i + 1] if i + 1 < len(xs) else hi
            parts.append(f'<rect x="{xmap(a):.2f}" y="{top}" width="{max(1., xmap(b)-xmap(a)):.2f}" height="{bottom-top}" fill="#e57373" opacity=".22"/>')
    for value in np.linspace(low, high, 6):
        y = ymap(value)
        parts.extend([f'<path d="M85 {y:.2f} H{right}" stroke="#ddd"/>',
                      f'<text x="78" y="{y+4:.2f}" text-anchor="end">{value:.3g}</text>'])
    ticks = np.arange(np.ceil(lo / tick) * tick, hi + 1e-6, tick) if tick else np.linspace(lo, hi, 9)
    for value in ticks:
        x = xmap(value)
        parts.extend([f'<path d="M{x:.2f} {top} V{bottom+4}" stroke="#eee"/>',
                      f'<text transform="translate({x:.2f},{bottom+13}) rotate(90)" font-size="11">{value:g}</text>'])
    for i, (name, curve) in enumerate(curves):
        colour = COLOURS[i % len(COLOURS)]
        path, pen = [], "M"
        for x, y in zip(xs, curve):
            if not np.isfinite(y):
                pen = "M"
                continue
            path.append(f"{pen}{xmap(x):.2f},{ymap(y):.2f}")
            pen = "L"
        parts.extend([f'<text x="{90+i*270}" y="43" fill="{colour}">{esc(name)}</text>',
                      f'<path d="{" ".join(path)}" stroke="{colour}" fill="none" stroke-width="1.3"/>'])
        finite_mask = np.isfinite(curve)
        for j in np.flatnonzero(finite_mask):
            if (j == 0 or not finite_mask[j-1]) and (j == len(xs)-1 or not finite_mask[j+1]):
                parts.append(f'<circle cx="{xmap(xs[j]):.2f}" cy="{ymap(curve[j]):.2f}" r="3" fill="{colour}"/>')
    for i, (name, value) in enumerate(limits):
        parts.extend([f'<path d="M{left} {ymap(value):.2f} H{right}" stroke="{FAIL}" stroke-dasharray="7 5"/>',
                      f'<text x="{90+i*330}" y="310" fill="{FAIL}">{esc(name)}={value:g}</text>'])
    parts.append('</g></svg>')
    return ''.join(parts)


def document(title, body):
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            f'<title>{esc(title)}</title><style>body{{font:15px Arial,"Microsoft YaHei",sans-serif;margin:24px;color:#222}}'
            'h2{margin-top:32px}table{border-collapse:collapse;min-width:1000px}td,th{border:1px solid #ddd;padding:7px 12px;text-align:left}'
            'pre{white-space:pre-wrap;background:#f6f6f6;padding:12px}.scroll{overflow-x:auto}img{max-width:100%}'
            '.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:10px 0}a{color:#2474b5}</style>'
            f'<h1>{esc(title)}</h1>{body}</html>')


def picture(path, caption, prefix="../"):
    return f'<figure><figcaption>{esc(caption)}</figcaption><img src="{prefix}{esc(path)}"></figure>'


def row_checks(row, sup, p, unit):
    d, q = p["detect"], p["quality"]
    checks = [("③ 采样", "INFO", f"STEP={p['sampling']['step']}; MINH={p['sampling']['min_pixels']}; REF={p['sampling']['ref']}"),
              ("④ 上下区域 / 权重", "INFO", "上部60%查白；下部40%查红；红权重0.6，白权重0.4"),
              ("⑤ 有效区域", "INFO", f"trim={d['trim']}; 收缩核宽7，两处各执行一次")]
    checks.extend([gate("⑥ 异色最长段", row.get("foreign_run"), q["foreign_span"], "<="),
                   gate("⑥ 极低信号最长段", row.get("dark_run"), q["dark_span"], "<="),
                   ("⑥ 比例门", "INFO", f"异色>{q['foreign']}; 信号<{q['dark']}"),
                   ("⑦ 平滑 / 基线", "SKIP" if "smooth" not in row else "INFO",
                    f"sigma={p['signal']['sigma']}; kernel={p['signal']['kernel']}; baseline={p['signal']['baseline_window']}"),
                   ("⑧ 低谷包络门", "SKIP" if "smooth" not in row else "INFO",
                    f"启动>0.25；向右延续>0.125；有效比例>=0.9；宽度>={d['min_width']}"),
                   ("⑧ 核心门", "SKIP" if "smooth" not in row else "INFO",
                    f"信号<{d['floor_max']}; 退出<{d['floor_max']+d['core_exit']}; 核心宽度>={d['core_min_width']}"),
                   ("⑨ U宽度门", "INFO", f"U={unit}; 允许[{unit*.5:g}, {unit*2:g}]原生站；conf_th={d['conf_th']}"),
                   ("⑩ 历史累计", "INFO", f"有效帧={sup['observations']}; 全历史累计，无滑窗，无自动遗忘"),
                   gate("⑩ 最大support（区间仍需宽度门）", float(np.max(sup["values"])), d["support_th"]),
                   ("⑩ 成框宽度门", "INFO", f"连续宽度>={d['min_box_width']}归一化位置")])
    if not row.get("candidates"):
        checks.append(("⑧ 候选", "SKIP" if "smooth" not in row else "INFO", "未执行低谷检测" if "smooth" not in row else "没有启动包络的低谷"))
    for i, c in enumerate(row.get("candidates", [])):
        label = f"⑧ 候选{i} {c['env']}"
        checks.extend([gate(label + " 包络宽", c["env_width"], d["min_width"]),
                       gate(label + " 有效比例", c["valid_ratio"], .9)])
        for j, trial in enumerate(c["trials"]):
            checks.extend([gate(f"{label} 核心尝试{j}最低值", trial["floor"], d["floor_max"], "<"),
                           gate(f"{label} 核心尝试{j}宽", trial["width"], d["core_min_width"])])
        checks.append((label + " 原因", "PASS" if c.get("kept") else "FAIL", c["reason"]))
        if "native_width" in c:
            checks.extend([gate(label + " 原生宽下限", c["native_width"], .5 * unit),
                           gate(label + " 原生宽上限", c["native_width"], 2 * unit, "<="),
                           gate(label + " 分数", c["confidence"], d["conf_th"])])
    return checks


def frame_details(out, record, p):
    trace, index = record["trace"], record["index"]
    ref = p["sampling"]["ref"]
    q = trace["quality"]
    body = '<p><a href="../index.html">返回总览</a>；红字=FAIL；灰字=SKIP；INFO=设置或中间值。</p>'
    body += table([gate("② 整帧pmax", q["pmax"], p["quality"]["pmax"], "<"),
                   ("② 最大块(y,x)", "INFO", str(q["block_yx"])),
                   ("当前质量 / 历史候选", "INFO", f"skip={trace['skip']}; 历史输出框数={len(trace['gaps'])}")])
    body += '<div class="pair">' + picture(record["overlay"], "原图定位：黄色=路面参考；青色=轴向/最大梯度块；橙色=所有低谷包络")
    body += picture(record["masks"], "原帧颜色掩膜：红=红色；白=白色；紫=异色（可能与白色重叠）") + '</div>'
    for r in p["rows"]:
        rid = r["id"]
        row = trace["rows"].get(rid, {})
        sup = trace["support"][rid]
        body += f'<h2>③—⑩ {esc(rid)}：{esc(row.get("status", "INVALID / SKIP"))}</h2>'
        body += table(row_checks(row, sup, p, r["U"]))
        if "pack" in row:
            pack, valid = row["pack"], row["valid"]
            raw = cv.imread(str(out / record["image"]))
            pts = (np.asarray(r["poly"]) * [raw.shape[1], raw.shape[0]]).astype(int)
            x0, y0 = np.maximum(pts.min(axis=0), 0)
            x1, y1 = np.minimum(pts.max(axis=0) + 1, [raw.shape[1], raw.shape[0]])
            crop = raw[y0:y1, x0:x1]
            crop_path = image_file(out / "images" / f"crop_{index:08d}_{rid}.jpg", crop)
            split = row.get("split")
            if split is not None:
                colour = np.zeros((*split.shape, 3), np.uint8)
                colour[split == 1] = (255, 180, 0)
                colour[split == 2] = (0, 140, 255)
                split_path = image_file(out / "images" / f"split_{index:08d}_{rid}.png", colour)
                lower = "红白" if 'lower_white_ratio' in row else "红"
                body += '<div class="pair">' + picture(crop_path, "水马行放大图") + picture(split_path, f"采样分区：蓝=上部查白；橙=下部查{lower}；黑=ROI外") + '</div>'
            x = np.arange(ref)
            body += '<div class="scroll">' + chart("颜色信号 / 异色比例；横轴=归一化位置；红底=无效位置", x,
                [("原始S", pack["s"]), ("异色比例", pack["f"])],
                [("异色门", p["quality"]["foreign"]), ("极低信号门", p["quality"]["dark"])], ~valid)
            if "red_ratio" in row:
                ratios = [("下部红色", row["red_ratio"]), ("上部白色", row["white_ratio"])]
                if 'lower_white_ratio' in row:
                    ratios += [("下部白色（扣除红色重叠）", row['lower_white_ratio'])]
                body += chart("原生采样站：红白比例", np.arange(pack["M"]), ratios)
                body += chart("原生采样站像素数量", np.arange(pack["M"]), [("像素数", row["counts"])], [("MINH", p["sampling"]["min_pixels"])])
            if "smooth" in row:
                body += chart("平滑 / 局部基线", x, [("原始S", pack["s"]), ("平滑S", row["smooth"]),
                    ("最终基线", row["baseline"]), ("局部均值", row["local_mean"])],
                    [("floor_max", p["detect"]["floor_max"]), ("floor+core_exit", p["detect"]["floor_max"]+p["detect"]["core_exit"])], ~valid)
                envelope, core = np.zeros(ref), np.zeros(ref)
                for c in row["candidates"]:
                    a, b = c["env"]
                    envelope[a:b+1] = 1
                    if c.get("core"):
                        a, b = c["core"][:2]
                        core[a:b+1] = 1
                body += chart("低谷下降幅度 / 包络 / 核心（含淘汰候选）", x,
                              [("下降幅度", row["baseline"]-row["smooth"]), ("包络", envelope), ("核心", core)],
                              [("启动门", .25), ("延续门", .125)], ~valid)
            body += '</div>'
        runs = trace.get("support_runs", {}).get(rid)
        body += table([gate(f"⑩ support区间{run['start']}—{run['end']}宽度", run["width"], p["detect"]["min_box_width"])
                       for run in (runs or [])] or [("⑩ support区间", "SKIP" if runs is None else "INFO",
                           "本帧整帧跳过，沿用旧框" if runs is None else "没有达到support门的区间")])
        fresh, stable = np.zeros(ref), np.zeros(ref)
        for a, b in trace["kept"].get(rid, []):
            fresh[a:b+1] = 1
        for row_id, (a, b), _box, _support in trace["gaps"]:
            if row_id == rid:
                stable[a:b+1] = 1
        body += '<div class="scroll">' + chart("历史support与候选；横轴=归一化位置", np.arange(ref),
                    [("support", sup["values"]), ("本帧保留核心", fresh), ("历史成框区间", stable)],
                    [("support_th", p["detect"]["support_th"])]) + '</div>'
    body += '<h2>⑪ 轨道匹配</h2>'
    for match in trace["matches"]:
        checks = [(f"轨道{match['token']}", "INFO", f"{'新建' if match['new'] else '匹配'}；row={match['row']}；box={match['box']}")]
        for attempt in match["attempts"]:
            checks.extend([gate(f"与轨道{attempt['token']} IoU", attempt["iou"], p["track"]["match_iou"]),
                           gate("中心距离", attempt["center_distance"], attempt["center_limit"], "<")])
        body += table(checks)
    if not trace["matches"]:
        body += '<p>SKIP：本帧没有历史support候选框。</p>'
    body += '<h2>⑫—⑯ 时间状态与最终确认</h2>'
    for tr in trace["tracks"]:
        t = p["track"]
        checks = [(f"轨道{tr['token']} / {tr['state']}", "INFO",
                   f"历史框匹配={tr['matched']}；本帧低谷重叠={tr['fresh_hit']}；质量跳过={tr['quality_skip']}；删除={tr['removed']}"),
                  gate("时间累计", tr["accum"], t["alarm_hold_s"]),
                  ("正常恢复计时", "INFO", f"{tr['intact_sec']:.3f}s / reset={t['intact_reset_s']}s；decay={t['decay_rate']}"),
                  ("过期清理", "INFO", f"距最后匹配={record['time_s']-tr['last_seen']:.3f}s；stale={t['track_stale_s']}s；仅NORMAL可删"),
                  ("中值帧时间", "INFO", f"{tr['median_times']}；容量={t['median_window']}"),
                  gate("RER", tr["rer"] if tr["rer_t"] is not None else None, t["rer_threshold"]),
                  ("RER重算 / 降级", "INFO", f"上次重算={tr['rer_t']}；间隔={t['reconfirm_s']}s；低值更新计数={tr['rer_below']} / 2（按更新调用计数）")]
        body += table(checks)
        if tr["rer_frame"] is not None:
            body += f'<p>实际RER证据：<a href="../rer_{tr["rer_frame"]:08d}_{tr["token"]}.html">打开确认页</a></p>'
    if not trace["tracks"]:
        body += '<p>SKIP：没有轨道，未执行时间累计及RER。</p>'
    (out / "pages" / f"frame_detail_{index:08d}.html").write_text(
        document(f"水马过程检查：帧{index} / {record['time_s']:.3f}s", body), encoding="utf-8")


def rer_details(out, index, check, p):
    t = p["track"]
    checks = [("实际确认状态", "FAIL" if check["status"] != "OK" else "INFO", check["status"]),
              gate("非红白剩余面积", check.get("missing_pixels"), p["rer"]["min_area"]),
              ("patch设置", "INFO", f"检测patch={t['patch_size']}；purity>={t['rer_purity']}；拟合patch固定8"),
              gate("有效patch数量", check.get("valid_count"), 1),
              ("路面patch数量", "INFO", str(check.get("road_count"))),
              gate("RER", check["rer"], t["rer_threshold"]),
              ("中值帧时间", "INFO", str(check["median_times"]))]
    body = '<p><a href="index.html">返回总览</a></p>' + table(checks)
    if "error" in check:
        body += f'<p style="color:{FAIL}">{esc(check["error"])}</p>'
    for key, title in (("median", "实际参与确认的中值图"), ("road_sample", "实际用于拟合的路面采样图（多框裁成共同宽度后拼接）"),
                       ("foreground", "红白前景"), ("missing", "缺口框内非红白区域")):
        if f"{key}_image" in check:
            body += picture(check[f"{key}_image"], title, prefix="")
    if "labels" in check:
        labels = check["labels"]
        canvas = np.zeros((*labels.shape, 3), np.uint8)
        canvas[labels == 1] = (0, 0, 255)
        canvas[labels == 2] = (0, 180, 0)
        canvas = cv.resize(canvas, None, fx=t["patch_size"], fy=t["patch_size"], interpolation=cv.INTER_NEAREST)
        path = image_file(out / "images" / f"patch_{index:08d}_{check['token']}.png", canvas)
        body += picture(path, "patch分类：绿=路面；红=有效但非路面；黑=无效/未纳入。网格与原图左上角对齐。", prefix="")
        rows = []
        for i, ((y, x), mean, distance) in enumerate(zip(np.argwhere(check["valid"]), check["means"], check["distances"])):
            rows.extend([gate(f"patch{i} ({x},{y})纯度", check["counts"][y, x] / t["patch_size"] ** 2, t["rer_purity"]),
                         gate("亮度L", mean[0], check["profile"]["l_max"], "<="),
                         gate("色度距离（亮度超限时为999）", distance, check["profile"]["dist_th"], "<=")])
        body += '<h2>逐patch判断</h2>' + table(rows)
    if "profile" in check:
        body += '<h2>本次实际路面模型</h2>'
        body += table([(k, "INFO", str(v)) for k, v in check["profile"].items()])
    body += '<h2>路面拟合固定规则</h2><pre>' + esc(json.dumps(p["rer"], indent=2)) + '</pre>'
    (out / f"rer_detail_{index:08d}_{check['token']}.html").write_text(
        document(f"实际RER确认：轨道{check['token']} / 帧{index}", body), encoding="utf-8")


def timeline_page(out, token, samples, p):
    xs = np.asarray([s["time"] for s in samples])
    begin, end = xs[0], xs[-1]
    starts = [begin, max(begin, (begin + end - 10) / 2), max(begin, end - 10)]
    panels = [("全程", begin, end)] + [(label, a, min(end, a + 10)) for label, a in zip(("开头10秒", "中间10秒", "结尾10秒"), starts)]
    changes = [i for i, s in enumerate(samples) if i == 0 or s["state"] != samples[i-1]["state"]]
    panels += [(f"状态变化 {samples[i]['state']} @ {xs[i]:.3f}s", max(begin, xs[i]-5), min(end, xs[i]+5)) for i in changes[1:]]
    body = '<p><a href="index.html">返回总览</a>；时间轴刻度2秒；红底=当前质量门跳过；缺失值断线；不抽样。</p>'
    body += '<p>状态编码：0=NORMAL，1=SUSPECTED，2=ALARM。RER仅在实际重算帧画点；缓存值不冒充新确认。</p>'
    body += '<h2>状态变化</h2>' + table([(f"{xs[i]:.3f}s", "INFO", samples[i]["state"]) for i in changes])
    t = p["track"]
    body += table([(k, "INFO", str(v)) for k, v in t.items()])
    for title, lo, hi in panels:
        ids = np.flatnonzero((xs >= lo) & (xs <= hi))
        if not len(ids):
            continue
        values = [samples[i] for i in ids]
        x = xs[ids]
        width = max(WIDTH, int(np.ceil((hi - lo) / 2)) * 14 + 140)
        excluded = np.array([s["quality_skip"] is not None for s in values])
        get = lambda key: np.asarray([s[key] for s in values], float)
        body += f'<h2>{esc(title)}：{lo:.3f}—{hi:.3f}s</h2><div class="scroll">'
        body += chart("命中与状态；横轴=视频时间(s)", x,
                      [("本帧低谷重叠", get("fresh_hit")), ("历史框匹配", get("matched")), ("状态", get("state_code"))], excluded=excluded, width=width, tick=2)
        body += chart("历史support", x, [("轨道support", get("severity"))],
                      [("support_th", p["detect"]["support_th"])], excluded, width, 2)
        body += chart("累计及恢复时间(s)", x, [("累计", get("accum")), ("未匹配", get("intact_sec"))],
                      [("hold", t["alarm_hold_s"]), ("reset", t["intact_reset_s"])], excluded, width, 2)
        body += chart("RER：蓝线缓存值；黑点实际确认值", x,
                      [("实际确认", get("rer_check")), ("缓存RER", get("rer"))],
                      [("rer_threshold", t["rer_threshold"])], excluded, width, 2)
        # Single-point updates need visible markers, not a connected line.
        checks = [(s["time"], s["rer_check"]) for s in values if np.isfinite(s["rer_check"])]
        body += '<p>实际确认点：' + esc('; '.join(f'{a:.3f}s={b:.4f}' for a, b in checks) or 'SKIP') + '</p></div>'
    (out / f"track_{token:04d}.html").write_text(document(f"轨道{token}时间过程", body), encoding="utf-8")


def replot(out, frame_limit=None, stop_after=None):
    """Read only saved arrays/images/settings; no imports of replay or video decoding."""
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    p = manifest["parameters"]
    timelines, snapshots, confirmations = {}, [], []
    from water_plot import frame_plot, road_page
    available = [item["index"] for item in manifest["snapshots"]
                 if stop_after is None or item["time_s"] <= stop_after]
    if frame_limit is not None and len(available) > frame_limit:
        available = [available[i] for i in np.linspace(0, len(available)-1, frame_limit, dtype=int)]
    selected = set(available)
    latest_rer = {}
    rendered_frames = 0
    row_history = {r["id"]: [] for r in p["rows"]}
    with (out / "frames.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            saved = json.loads(line)
            if stop_after is not None and saved["time_s"] > stop_after:
                break
            rendered_frames += 1
            needs_arrays = saved["index"] in selected or saved["trace"]["rer_checks"]
            if needs_arrays:
                with np.load(out / saved["arrays_file"], allow_pickle=False) as arrays:
                    record = unpack_tree(saved, arrays)
                for check in record["trace"]["rer_checks"]:
                    check["frame_index"] = record["index"]
                    latest_rer[check["token"]] = check
                    rer_details(out, record["index"], check, p)
                    road_page(out, record["index"], check, p)
                    confirmations.append((record["index"], check["token"], record["time_s"]))
                if saved["index"] in selected:
                    frame_details(out, record, p)
                    checks = [latest_rer[tr["token"]] for tr in record["trace"]["tracks"]
                              if tr["token"] in latest_rer]
                    frame_plot(out, record, p, checks)
                    snapshots.append((record["index"], record["time_s"]))
            for rid in row_history:
                sup = saved["trace"]["support"][rid]
                row_history[rid].append((saved["time_s"], sup["observations"],
                                         saved["trace"]["skip"].get("all", saved["trace"]["skip"].get(rid)),
                                         sup["maximum"], len(saved["trace"]["kept"].get(rid, [])),
                                         sum(g[0] == rid for g in saved["trace"]["gaps"])))
            checked = {c["token"]: c["rer"] for c in saved["trace"]["rer_checks"]}
            for tr in saved["trace"]["tracks"]:
                sample = {k: tr[k] for k in ("state", "severity", "accum", "intact_sec", "rer", "matched", "fresh_hit", "quality_skip")}
                sample.update(time=saved["time_s"], state_code={"NORMAL": 0, "SUSPECTED": 1, "ALARM": 2}[tr["state"]],
                              rer_check=checked.get(tr["token"], float("nan")))
                timelines.setdefault(tr["token"], []).append(sample)
    for token, samples in timelines.items():
        timeline_page(out, token, samples, p)
    body = f'<p>视频：{esc(manifest["video"])}；帧数={manifest["frames"]}；FPS={manifest["fps"]}；完整结束={manifest["complete"]}</p>'
    if (out / "summary.html").is_file():
        body += '<h2><a href="summary.html">本次检测总结果</a></h2><img style="max-width:100%" src="summary.png">'
        result = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        if 'block_windows' in result and (out / "blocks.png").is_file():
            body += '<h2>水马红白分块预览</h2><p>蓝框：R=红色优势段，W=白色优势段；仅供检查，不参与报警。</p><img style="max-width:100%" src="blocks.png">'
    body += f'<p>本次重画范围：{rendered_frames}帧；主页面仅保留图示，原始判断表在补充详情中。</p>'
    body += '<p>离线检查独立运行；不发送回调。时间=帧号/FPS；从视频开头累计；适用于固定帧率视频。</p>'
    body += f'<p>详细快照上限={manifest["selection"]["max_pages"]}；超过上限未导出快照次数={manifest.get("suppressed_snapshots",0)}；逐帧trace仍完整。</p>'
    body += '<h2>按代码顺序的单帧过程</h2><ul>' + ''.join(
        f'<li><a href="pages/frame_{i:08d}.html">帧{i} / {t:.3f}s</a></li>' for i, t in snapshots) + '</ul>'
    body += '<h2>轨道全程 / 开头 / 中间 / 结尾 / 状态变化</h2><ul>' + ''.join(
        f'<li><a href="track_{token:04d}.html">轨道{token}</a></li>' for token in timelines) + '</ul>'
    body += '<h2>实际RER确认</h2><ul>' + ''.join(
        f'<li><a href="rer_{i:08d}_{token}.html">轨道{token} / {t:.3f}s</a></li>' for i, token, t in confirmations) + '</ul>'
    body += '<h2>每行有效帧累计与质量门</h2><div class="scroll">'
    for rid, values in row_history.items():
        if values:
            body += chart(rid + "有效观测帧数（红底=质量跳过）", [v[0] for v in values],
                          [("有效帧数", np.array([v[1] for v in values]))], excluded=[v[2] is not None for v in values])
            body += chart(rid + "全程support与候选数", [v[0] for v in values],
                          [("最大support", np.array([v[3] for v in values])),
                           ("本帧候选数", np.array([v[4] for v in values])),
                           ("历史框数", np.array([v[5] for v in values]))],
                          [("support_th", p["detect"]["support_th"])], [v[2] is not None for v in values])
    body += '</div><details><summary>本次保存的全部参数</summary><pre>' + esc(json.dumps(p, ensure_ascii=False, indent=2)) + '</pre></details>'
    body += '<h2>已知实现行为</h2><p>support无滑窗；质量跳过保留历史框，上层仍可累计时间并存入中值帧；'
    body += 'rer_below按更新调用计数；轨道匹配未按row_id额外过滤。此工具记录这些行为，不修改判断。</p>'
    (out / "index.html").write_text(document("水马缺口中间过程检查", body), encoding="utf-8")
    print(f"Plots: {(out / 'index.html').resolve()}")
