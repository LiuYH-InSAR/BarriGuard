"""Offline-only exports of tracked lamp regions and their chronological samples."""
import csv
import html
import json
import shutil
from pathlib import Path

import numpy as np
import cv2 as cv

from night_lamp.evidence import REGION_RADIUS, state_sequence

PLOT_LEFT = 90
DETAIL_SECONDS = 10.0
TIME_TICK_S = 2.0
GRAY_TICK = 20.0
PLOT_HEIGHT = 1340
SELECTED_PLOTS = "selected_gray_plots"


def _lamp_ids(checks, item):
    if checks is None:
        return None
    point = (float(item["x"]), float(item["y"]))
    return [name for name, poly in checks.polys.items()
            if cv.pointPolygonTest(poly, point, False) >= 0]


def write_plot_index(out_dir):
    """Browse the original SVG plots and evidence; never rerun detection."""
    out = Path(out_dir)
    manifests = ([out / "manifest.json"] if (out / "manifest.json").exists()
                 else sorted(out.rglob("manifest.json")))
    rows = []
    first_plot = ""
    for path in manifests:
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("candidates", []):
            candidate = item["candidate_id"]
            plot = path.parent / f"candidate_{candidate:03d}.svg"
            if not plot.exists():
                continue
            relative = plot.relative_to(out).as_posix()
            first_plot = first_plot or relative
            lamps = ','.join(item.get('lamp_ids', []))
            title = f"{path.parent.parent.name} / {lamps} / {candidate:03d} / ({item.get('x')}, {item.get('y')})"
            reason = str(item.get("reason", ""))
            rows.append(f'<tr><td><a target="plot" href="{html.escape(relative, quote=True)}">'
                        f'{html.escape(title)}</a></td><td>{html.escape(reason)}</td>'
                        f'<td>{item.get("flashing", False)}</td></tr>')
    page = ('<!doctype html><meta charset="utf-8"><title>Night diagnostic plots</title>'
            '<style>body{font:14px sans-serif;margin:15px}.layout{display:flex;gap:15px}'
            '.list{width:35%;height:88vh;overflow:auto}table{border-collapse:collapse}'
            'td,th{padding:6px;border:1px solid #ccc}iframe{width:65%;height:88vh}'
            'input{width:90%;padding:8px}</style><h2>候选位置诊断曲线</h2>'
            '<p>点击左侧候选，在右侧查看原始plot；包含未通过项。曲线过宽时可横向滚动。</p>'
            '<div class="layout"><div class="list"><input placeholder="搜索小时、坐标或失败原因" '
            'oninput="for(const r of document.querySelectorAll(\'tbody tr\'))'
            'r.hidden=!r.textContent.toLowerCase().includes(this.value.toLowerCase())">'
            '<table><thead><tr><th>小时／编号／坐标</th><th>原因</th><th>通过</th></tr>'
            '</thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
            f'<iframe name="plot" src="{html.escape(first_plot, quote=True)}"></iframe></div>')
    (out / "index.html").write_text(page, encoding="utf-8")
    return len(rows)


def replot_saved(out_dir):
    """Regenerate four-panel Gray plots without replaying already-saved sequences."""
    out = Path(out_dir)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    selected = out.parent / SELECTED_PLOTS
    selected.mkdir(exist_ok=True)
    count = 0
    with np.load(out / "series.npz", allow_pickle=False) as arrays:
        if "pixel_gray_corrected" not in arrays:
            raise ValueError(f"{out}: original pixel Gray is absent; replay the video to export it")
        for item in manifest["candidates"]:
            row = item.get("series_row")
            if row is None or row >= len(arrays["pixel_gray_corrected"]):
                continue
            name = f"candidate_{item['candidate_id']:03d}.svg"
            _plot_svg(out / name, arrays["time_s"], arrays["gray_corrected"][row],
                      None, None, item)
            if item.get("flashing") and (out / name).exists():
                shutil.copy2(out / name, selected / name)
                count += 1
    return count


def _plot_svg(path, times, gray, rgb, states, evidence):
    valid = np.flatnonzero(np.isfinite(gray) & np.isfinite(times))
    if not valid.size:
        return
    begin, end = float(times[valid[0]]), float(times[valid[-1]])
    # Wide overview keeps every 2-second tick readable, including long clips.
    width = max(1400, int(np.ceil((end - begin) / TIME_TICK_S)) * 14 + 140)
    table=evidence.get('stage_table',[])
    columns=max((len(row) for row in table),default=0)
    # Fixed tab stops keep A and B aligned even when their values differ.
    column_widths=([100]+[360]*(columns-1)) if columns else []
    width=max(width,int(sum(column_widths))+50)
    right = width - 30
    stage_lines = evidence.get('stage_lines', [])
    shift = (len(stage_lines)+len(table)) * 22 + 30 if stage_lines or table else 0
    height = PLOT_HEIGHT + shift
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             f'<rect width="{width}" height="{height}" fill="white"/>',
             '<g font-family="sans-serif" font-size="14">',
             f'<text x="25" y="25">{html.escape(path.stem)} | {html.escape(evidence.get("reason", "tracked"))}</text>',
             '<text x="25" y="48">Gray (exposure-corrected). X tick: 2 seconds; Y tick: 20 gray levels.</text>']
    starts = [begin, max(begin, (begin + end - DETAIL_SECONDS) / 2), max(begin, end - DETAIL_SECONDS)]
    panels = [("Full tracked sequence", begin, end)] + [
        (f"{label}: {start:.2f} to {min(end, start + DETAIL_SECONDS):.2f} seconds",
         start, min(end, start + DETAIL_SECONDS))
        for label, start in zip(("First 10s", "Middle 10s", "Last 10s"), starts)]
    reference = evidence.get('mref')
    bounds = gray[valid]
    if reference is not None and np.isfinite(reference):
        bounds = np.r_[bounds, reference]
        parts.append(f'<text x="25" y="68" fill="#8653a8">'
                     f'Mref = {reference:.2f} Gray: global exposure reference; '
                     'not the local background or ON threshold.</text>')
    low = np.floor(float(bounds.min()) / GRAY_TICK) * GRAY_TICK
    high = np.ceil(float(bounds.max()) / GRAY_TICK) * GRAY_TICK
    high = max(high, low + GRAY_TICK)
    mean_lines = [(key, label, colour, evidence.get(key)) for key, label, colour in
                  (("on_mean", "ON mean", "#d9534f"),
                   ("off_mean", "OFF mean", "#2474b5"),
                   ("mref", "Mref (global reference)", "#8653a8"))
                  if evidence.get(key) is not None and np.isfinite(evidence[key])]
    for i,row in enumerate(table):
        x=25
        for c,item in enumerate(row):
            colour='#c62828' if item['failed'] else '#222222'
            parts.append(f'<text x="{x:.1f}" y="{92+i*22}" fill="{colour}" font-family="monospace">{html.escape(item["text"])}</text>')
            x+=column_widths[c]
    for i, line in enumerate(stage_lines):
        colour='#c62828' if 'FAIL' in line else '#222222'
        parts.append(f'<text x="25" y="{92 + (i+len(table)) * 22}" fill="{colour}">{html.escape(line)}</text>')
    intervals=evidence.get('mask_intervals',evidence.get('disturbance',{}).get('intervals',[]))
    for panel, (title, lo, hi) in enumerate(panels):
        top, bottom = 110 + shift + panel * 300, 290 + shift + panel * 300
        ids = np.flatnonzero((times >= lo) & (times <= hi))
        parts += [f'<text x="90" y="{top - 12}">{html.escape(title)}</text>',
                  f'<path d="M90 {top} V{bottom} H{right}" fill="none" stroke="#777"/>']
        for start,stop in intervals:
            start,stop=max(start,lo),min(stop,hi)
            if stop<start:continue
            x=PLOT_LEFT+(start-lo)/max(hi-lo,1e-9)*(right-PLOT_LEFT)
            w=(stop-start)/max(hi-lo,1e-9)*(right-PLOT_LEFT)
            parts.append(f'<rect data-disturbance-mask="true" x="{x:.2f}" y="{top}" width="{w:.2f}" height="{bottom-top}" fill="#e57373" opacity="0.25"/>')
        if intervals:parts.append(f'<text x="90" y="{bottom+88}" fill="#b03a2e">Red shading: {html.escape(evidence.get("mask_label","active-path final mask"))}; original signal remains visible.</text>')
        for value in np.arange(low, high + GRAY_TICK / 2, GRAY_TICK):
            y = bottom - (value - low) / (high - low) * (bottom - top)
            parts += [f'<path d="M85 {y:.2f} H{right}" stroke="#e2e2e2"/>',
                      f'<text x="75" y="{y + 5:.2f}" text-anchor="end">{value:.0f}</text>']
        ticks = np.arange(np.ceil(lo / TIME_TICK_S) * TIME_TICK_S, hi + 1e-6, TIME_TICK_S)
        for tick in ticks:
            x = PLOT_LEFT + (tick - lo) / max(hi - lo, 1e-9) * (right - PLOT_LEFT)
            parts += [f'<path d="M{x:.2f} {top} V{bottom + 5}" stroke="#e2e2e2"/>',
                      f'<text transform="translate({x:.2f},{bottom + 12}) rotate(90)" font-size="11">{tick:.0f}</text>']
        commands = []
        pen = "M"
        for index in ids:
            value = gray[index]
            if not np.isfinite(value):
                pen = "M"
                continue
            x = PLOT_LEFT + (times[index] - lo) / max(hi - lo, 1e-9) * (right - PLOT_LEFT)
            y = bottom - (value - low) / (high - low) * (bottom - top)
            commands.append(f"{pen}{x:.2f},{y:.2f}")
            pen = "L"
        parts += [f'<path d="{" ".join(commands)}" fill="none" stroke="#333333" stroke-width="1.2"/>',
                  f'<text x="{width // 2}" y="{bottom + 65}" text-anchor="middle">Time since replay start (s)</text>']
        for number, (key, label, colour, value) in enumerate(mean_lines):
            y = bottom - (value - low) / (high - low) * (bottom - top)
            parts += [f'<path data-state-mean="{key}" d="M90 {y:.2f} H{right}" '
                      f'fill="none" stroke="{colour}" stroke-width="1.8" stroke-dasharray="8 5"/>',
                      f'<text x="{90 + number * 300}" y="{top - 32}" fill="{colour}">'
                      f'{label}: {value:.2f} Gray ({html.escape(evidence.get("fit_scope","full sequence"))})</text>']
    parts += [f'<text x="25" y="{height - 15}">No subsampling. Missing values are gaps. Under 30 observed seconds, detail windows may overlap or be shorter.</text>',
              '</g></svg>']
    path.write_text("\n".join(parts), encoding="utf-8")


def save_duty_maps(folder, duty, valid):
    """Full-video discovery duty, fixed 0..100% scale, before final verdict."""
    from night_lamp.tools import nightly_map as nm
    folder = Path(folder)
    np.save(folder / 'duty.npy', duty)
    colour = cv.applyColorMap((np.clip(duty, 0, 1) * 255).astype(np.uint8), cv.COLORMAP_TURBO)
    colour[~valid] = 0
    _, labels, _, _ = nm.split_comps(duty)
    np.save(folder / 'duty_candidate_labels.npy', labels)
    ranges = np.zeros_like(colour)
    ranges[(labels > 0) & valid] = 255
    contours, _ = cv.findContours(valid.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for filename, canvas, caption in (
        ('duty_map.jpg', colour, 'Discovery duty: fixed 0-100%; cyan = valid ROI; black = excluded'),
        ('duty_candidate_mask.jpg', ranges, f'White = duty >= {nm.CAND_DUTY_LO:g} after 3x3 opening; not lamp verdict')):
        cv.drawContours(canvas, contours, -1, (255, 255, 0), 1)
        canvas = cv.copyMakeBorder(canvas, 0, 65, 0, 0, cv.BORDER_CONSTANT, value=0)
        cv.putText(canvas, caption, (12, canvas.shape[0] - 42), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        if filename == 'duty_map.jpg':
            width = max(1, min(400, canvas.shape[1] - 24))
            gradient = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (12, 1))
            canvas[-28:-16, 12:12 + width] = cv.applyColorMap(gradient, cv.COLORMAP_TURBO)
            cv.putText(canvas, '0%                         50%                         100%', (12, canvas.shape[0] - 3), cv.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        if not cv.imwrite(str(folder / filename), canvas):
            raise RuntimeError(f'Cannot write {filename}')


def save_intermediates(session, report, out_dir):
    """Save before a burst reset or session release; never reconstruct old videos."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = session.intermediate_arrays()
    if 'duty' in arrays:
        save_duty_maps(out.parent, arrays['duty'], arrays['valid_mask'])
        np.save(out.parent / 'on_count.npy', arrays['on_count'])
        effective=getattr(session,'_valid_sample_count',None)
        if effective is not None:np.save(out.parent/'valid_sample_count.npy',effective)
        (out.parent / 'duty_stats.json').write_text(json.dumps({
            'sample_total': int(arrays['sample_total'])}), encoding='utf-8')
    gray = arrays["gray_corrected"]
    from night_lamp.disturbance import detect, PARAMETERS
    parameters=getattr(getattr(session,'recorder',None),'parameters',None) or {}
    disturbance_parameters=parameters.get('disturbance',PARAMETERS)
    masks=[]; windows=[]; disturbances=[]
    from night_lamp.stutter import detect_path
    for i,signal in enumerate(gray):
        force=getattr(session,'_stutter_for',lambda row:None)(i)
        mask,mean,info=detect_path(signal,arrays['time_s'],disturbance_parameters,parameters.get('stutter'),force)
        masks.append(mask);windows.append(mean);disturbances.append(info)
    from night_lamp.disturbance import describe_mask
    for i,mask in enumerate(masks):
        recorded=getattr(session,'_record_disturbance_mask',lambda row:None)(i)
        if recorded is not None:
            start=len(mask)-len(recorded);mask[start:]|=recorded
            disturbances[i].update(describe_mask(mask,arrays['time_s']))
    arrays['disturbance_mask']=np.asarray(masks,bool).reshape(gray.shape)
    arrays['disturbance_window_mean']=np.asarray(windows,float).reshape(gray.shape)
    (out/'disturbance_intervals.json').write_text(json.dumps({'parameters':disturbance_parameters,'rows':disturbances},indent=2),encoding='utf-8')
    states = np.array([state_sequence(row) for row in gray], dtype=np.float32).reshape(gray.shape)
    metadata = (report.metadata or {}) if report else {}
    pixel_gray = arrays["pixel_gray_corrected"]
    arrays["on_state"] = states
    np.savez_compressed(out / "series.npz", **arrays)
    scope = metadata.get("lamp_method", {}).get("evidence_scope", "unavailable")
    evidence = metadata.get("lamp_evidence", []) if scope == "current_window" else []
    mapping = []
    selected_plots = out.parent / SELECTED_PLOTS
    selected_plots.mkdir(exist_ok=True)
    position_image = None
    if report and report.image_jpeg:
        position_image = cv.imdecode(np.frombuffer(report.image_jpeg, np.uint8), cv.IMREAD_COLOR)
    for candidate, item in enumerate(evidence):
        lamps = _lamp_ids(getattr(session, "_lamp_checks", None), item)
        if lamps == []:
            continue
        row = item.get("series_row")
        name = f"candidate_{candidate:03d}"
        info = {"candidate_id": candidate, **item}
        if lamps is not None:
            info["lamp_ids"] = lamps
        mapping.append(info)
        if position_image is not None:
            x, y = int(item["x"]), int(item["y"])
            colour = (0, 255, 0) if item.get("flashing") else (0, 165, 255)
            cv.circle(position_image, (x, y), 5, colour, 1)
            cv.putText(position_image, str(candidate), (x + 5, y - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv.LINE_AA)
        if row is None or row >= len(gray):
            continue
        values = np.column_stack((arrays["time_s"], arrays["sample_index"],
                                  arrays["gray_raw"][row], arrays["gray_corrected"][row],
                                  pixel_gray[row], states[row]))
        np.savetxt(out / (name + ".csv"), values, delimiter=",", fmt="%.6f",
                   header="time_s,sample_index,region_gray_raw,region_gray_corrected,pixel_gray_corrected,on_state",
                   comments="")
        _plot_svg(out / (name + ".svg"), arrays["time_s"],
                  gray[row], None, states[row], item)
        if item.get("flashing") and (out / (name + ".svg")).exists():
            shutil.copy2(out / (name + ".svg"), selected_plots / (name + ".svg"))
    if position_image is not None:
        cv.imwrite(str(out / "candidate_positions.jpg"), position_image)
    with (out / "candidates.csv").open("w", newline="", encoding="utf-8-sig") as file:
        fields = ["candidate_id", "series_row", "x", "y", "kind", "flashing", "reason"]
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(mapping)
    manifest = {"format_version": 2, "channel_order": "Gray", "region_radius_px": REGION_RADIUS,
                "report_evidence_scope": scope,
                "time_basis": "seconds_since_replay_start (actual accepted sample timestamps)",
                "missing": "NaN: not yet tracked, warmup has no regional Gray, or unavailable samples",
                "state_semantics": "on_state: two-Gaussian high-state assignment, NOT final verdict",
                "decision_signal": "gray_corrected: exposure-corrected 5x5 regional Gray",
                "gray_plot_layout": "full tracked sequence + first/middle/last 10s",
                "gray_plot_ticks": {"time_s": TIME_TICK_S, "gray": GRAY_TICK},
                "gray_semantics": "weighted regional BGR mean: 0.114 B + 0.587 G + 0.299 R",
                "exposure_semantics": "corrected = raw - exposure_offset",
                "matrix_rows": "tracked locations including candidate cores and reflection hosts; use candidates.csv to map",
                "shapes": {k: list(v.shape) for k, v in arrays.items()},
                "candidates": mapping}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_plot_index(out)
    return str(out)
