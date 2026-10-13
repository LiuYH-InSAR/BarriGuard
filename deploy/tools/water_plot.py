"""Visual water diagnostics; thresholds come only from the saved run."""
from pathlib import Path
import textwrap

import cv2 as cv
import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt, font_manager
from matplotlib.patches import Rectangle

FONT_SIZE = 11
RED = "#c62828"
BLUE = "#2474b5"
GREEN = "#238443"
GREY = "#999999"
NORMAL_GAP_PX = 150

font_path = Path("C:/Windows/Fonts/msyh.ttc")
if font_path.is_file():
    font_manager.fontManager.addfont(str(font_path))
    matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
matplotlib.rcParams.update({"font.size": FONT_SIZE, "axes.unicode_minus": False,
                           "svg.fonttype": "none"})


def runs(mask):
    mask = np.asarray(mask, bool)
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)-1))


def save_plot(out, name, fig):
    folder = Path(out) / "plots"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / f"{name}.svg", facecolor="white")
    fig.savefig(folder / f"{name}.png", dpi=110, facecolor="white")
    plt.close(fig)
    return f"plots/{name}.svg"


def anomaly_sheet(out, name, lines):
    if not lines:
        return
    rows = int(np.ceil(len(lines)/3))
    fig, ax = plt.subplots(figsize=(15, 1 + rows*.55))
    ax.axis("off")
    fig.subplots_adjust(left=.03, right=.98, top=.9, bottom=.05)
    ax.set_title("全部异常标注：编号对应主图；不省略异常块", loc="left", fontsize=13)
    for i, line in enumerate(lines):
        row, col = divmod(i, 3)
        ax.text(col/3, 1-(row+.4)/(rows+1), line, transform=ax.transAxes, fontsize=10,
                color=RED, va="top", linespacing=1.7)
    save_plot(out, name+"_notes", fig)


def sparse_labels(ax, xs, ys, blocked=None):
    """Normal labels share one upper rail; pixel spacing scales with font size."""
    xs, ys = np.asarray(xs), np.asarray(ys)
    if not len(xs):
        return
    blocked = np.zeros(len(xs), bool) if blocked is None else np.asarray(blocked, bool)
    count = max(2, int(ax.figure.get_figwidth() * 90 / NORMAL_GAP_PX))
    last = -NORMAL_GAP_PX
    width = ax.figure.get_figwidth() * 90 * .86
    for i in np.unique(np.linspace(0, len(xs)-1, count, dtype=int)):
        x = (xs[i]-xs[0]) / max(float(xs[-1]-xs[0]), 1) * width
        if blocked[i] or not np.isfinite(ys[i]) or x-last < NORMAL_GAP_PX:
            continue
        ax.annotate(f"{ys[i]:.3g}", xy=(xs[i], ys[i]), xytext=(xs[i], .97),
                    textcoords=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9,
                    color=BLUE, bbox=dict(facecolor="white", edgecolor="none", alpha=.85),
                    arrowprops=dict(arrowstyle="-", color=BLUE, lw=.6))
        last = x


def series_plot(out, name, title, xs, curves, params, limits=(), annotations=(), blocked=None):
    """Every abnormal interval gets a separate callout rail; none are capped."""
    notes = list(annotations)
    wrapped = [textwrap.wrap(f"[{a:g}—{b:g}] {label}", width=100) for a, b, label, colour in notes]
    reserved = .7 + sum(.24 * len(lines) + .18 for lines in wrapped)
    fig = plt.figure(figsize=(14, 4.3 + reserved))
    grid = fig.add_gridspec(2, 1, height_ratios=[3.1, reserved], hspace=.42,
                           left=.07, right=.97, top=.86, bottom=.055)
    ax, rail = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    fig.suptitle(title, x=.07, ha="left", fontsize=14, y=.98)
    fig.text(.07, .92, params, fontsize=10, va="top")
    xs = np.asarray(xs)
    for label, values, colour in curves:
        ax.plot(xs, values, label=label, color=colour, lw=1.3)
    for label, value in limits:
        ax.axhline(value, color=RED, ls="--", lw=1)
        ax.text(1, value, f" {label}={value:g}", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=10, color=RED,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
    blocked = np.zeros(len(xs), bool) if blocked is None else np.asarray(blocked, bool)
    for a, b in runs(blocked):
        ax.axvspan(xs[a], xs[b]+.5, color=RED, alpha=.14)
    for a, b, text, colour in notes:
        ax.axvspan(a, b+.5, color=colour, alpha=.09)
        if curves and colour == RED:
            selected = (xs >= a) & (xs <= b)
            ax.scatter(xs[selected], np.asarray(curves[0][1])[selected], color=RED, s=8, zorder=4)
    sparse_labels(ax, xs, curves[0][1], blocked | np.array([
        any(a <= x <= b and colour == RED for a, b, _text, colour in notes) for x in xs]))
    ax.legend(loc="lower right", fontsize=10, framealpha=.9)
    ax.grid(alpha=.2)
    ax.set_xlim(xs[0], max(float(xs[-1]), float(xs[0])+1))
    ax.set_xlabel("沿水马行的位置")
    rail.set_xlim(ax.get_xlim())
    rail.set_ylim(0, reserved)
    rail.axis("off")
    if not notes:
        rail.text(0, .6, "没有触发异常区间；蓝色数值按显示间距疏排。", transform=rail.transAxes, color=BLUE)
    y = reserved - .1
    for (a, b, label, colour), lines in zip(notes, wrapped):
        rail.plot([a, b], [y, y], color=colour, lw=3)
        rail.text(.005, y-.09, "\n".join(lines), transform=rail.get_yaxis_transform(),
                  color=colour, fontsize=10, va="top", linespacing=1.3)
        y -= .24 * len(lines) + .18
    return save_plot(out, name, fig)


def masks_from_hsv(hsv, p):
    c = p["colour"]
    h, s, v = np.moveaxis(hsv, -1, 0)
    red_h = np.zeros(h.shape, bool)
    for lo, hi in c["red_h"]:
        red_h |= (h >= lo) & (h <= hi)
    red = red_h & (s >= c["red_s_min"]) & (v >= c["red_v_min"])
    sun = (s <= c["sun_s_max"]) & (v >= c["sun_v_min"])
    shadow = ((s > c["shadow_s"][0]) & (s <= c["shadow_s"][1])
              & (v >= c["shadow_v"][0]) & (v < c["shadow_v"][1]) & (h >= c["shadow_h_min"]))
    return dict(hsv=hsv, red=red, white=sun | shadow, foreign=(s > c["foreign_s_min"]) & ~red_h)


def colour_plot(out, name, image, pixels, expected, kind, params, thresholds, exact):
    """Separate classifiers preserve white/foreign overlap; histogram includes every pixel."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.8))
    fig.subplots_adjust(left=.06, right=.97, top=.83, bottom=.08, hspace=.5, wspace=.2)
    fig.suptitle(f"{kind}识别：独立掩膜 + HSV分布", x=.06, ha="left", y=.98, fontsize=14)
    fig.text(.06, .93, params, fontsize=10)
    fig.text(.06, .875, "原始掩膜" if exact else "旧数据：从保存JPEG重建的颜色示意，不冒充原始像素判定", color=BLUE if exact else RED, fontsize=10)
    mask = pixels[{"红色": "red", "白色": "white", "异色": "foreign"}[kind]]
    matched, excluded = expected & mask, expected & ~mask
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB).copy()
    rgb[~expected] = (rgb[~expected] * .22).astype(np.uint8)
    tint = np.array([200, 40, 40] if kind == "红色" else [40, 160, 200] if kind == "白色" else [180, 40, 170])
    rgb[matched] = (.35 * rgb[matched] + .65 * tint).astype(np.uint8)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"着色=命中；未命中={excluded.sum()}像素（不直接等于缺口）", fontsize=11)
    axes[0, 0].axis("off")
    # Excluded histogram bins are red; every excluded pixel contributes to its bin.
    for channel, ax in enumerate((axes[0, 1], axes[1, 0], axes[1, 1])):
        maximum = 180 if channel == 0 else 255
        values = pixels["hsv"][:, :, channel]
        good = np.bincount(values[matched], minlength=maximum+1)[:maximum+1]
        bad = np.bincount(values[excluded], minlength=maximum+1)[:maximum+1]
        ax.bar(np.arange(maximum+1), good, color="#ae3d9e" if kind == "异色" else GREEN, width=1, label="命中像素")
        ax.bar(np.arange(maximum+1), bad, bottom=good, color=GREY if kind == "异色" else RED, alpha=.65, width=1, label="未命中像素")
        for value in thresholds[channel]:
            ax.axvline(value, color=BLUE, ls="--")
            ax.text(value, .94, f"{value:g}", transform=ax.get_xaxis_transform(), color=BLUE,
                    ha="center", va="top", fontsize=10,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
        ax.set_title(f"{'HSV'[channel]}：{'紫色异色 / 灰色非异色' if kind == '异色' else '绿色命中 / 红色未命中'}，联合HSV条件判断", fontsize=10)
        ax.set_xlim(0, maximum)
        ax.set_ylabel("像素数")
        ax.grid(alpha=.15)
    axes[0, 1].legend(fontsize=9)
    return save_plot(out, name, fig)


def quality_plot(out, record, p):
    blocks = record["trace"]["quality"]["blocks"]
    limit = p["quality"]["pmax"]
    failed = blocks >= limit
    notes = [(int(y), int(x), float(blocks[y, x])) for y, x in np.argwhere(failed)]
    height = 5.8 + len(notes) * .19
    fig = plt.figure(figsize=(15, height))
    grid = fig.add_gridspec(2, 2, height_ratios=[6, max(.2, len(notes)*.19)],
                           left=.05, right=.97, top=.88, bottom=.05)
    original, heat, rail = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
    raw = cv.imread(str(out / record["image"]))
    original.imshow(cv.cvtColor(raw, cv.COLOR_BGR2RGB))
    original.axis("off")
    for y, x, value in notes:
        original.add_patch(Rectangle((x*p["quality"]["block"], y*p["quality"]["block"]),
                                    p["quality"]["block"], p["quality"]["block"], fill=False, ec=RED))
    by, bx = record["trace"]["quality"]["block_yx"]
    original.add_patch(Rectangle((bx*32, by*32), 32, 32, fill=False, ec=BLUE, lw=2))
    im = heat.imshow(blocks, cmap="YlOrRd", vmin=0, vmax=max(limit, float(blocks.max())))
    for y in np.linspace(0, blocks.shape[0]-1, 4, dtype=int):
        for x in np.linspace(0, blocks.shape[1]-1, 5, dtype=int):
            if not failed[y, x] and (y, x) != (by, bx):
                heat.text(x, y, f"{blocks[y,x]:.2f}", ha="center", va="center", fontsize=8, color=BLUE,
                          bbox=dict(facecolor="white", edgecolor="none", alpha=.8))
    for y, x in np.argwhere(failed):
        heat.add_patch(Rectangle((x-.5, y-.5), 1, 1, fill=False, ec=RED, lw=1.5))
    heat.annotate(f"最大值={blocks[by,bx]:.4f}", (bx, by), xytext=(.02, 1.02),
                  textcoords="axes fraction", color=RED if failed[by,bx] else BLUE,
                  arrowprops=dict(arrowstyle="->", color=BLUE))
    fig.colorbar(im, ax=heat, shrink=.7, label="32×32块平均边缘强度")
    heat.set_xlabel("块列号")
    heat.set_ylabel("块行号")
    fig.suptitle("整帧质量：原图位置与块热图", x=.05, ha="left", fontsize=14)
    fig.text(.05, .93, f"Sobel={p['quality']['sobel_kernel']}×{p['quality']['sobel_kernel']}；块={p['quality']['block']}×{p['quality']['block']}；"
             f"最大块值必须 < {limit:g}；{'FAIL：本帧跳过' if notes else 'PASS'}", color=RED if notes else BLUE)
    rail.axis("off")
    for i, (y, x, value) in enumerate(notes):
        rail.text(0, 1-(i+1)/(len(notes)+1), f"异常块 行{y}/列{x}：{value:.4f} >= {limit:g}", color=RED, fontsize=10)
    return save_plot(out, f"quality_{record['index']:08d}", fig)


def validity_plot(out, name, row, p):
    pack = row["pack"]
    counts = row.get("counts", np.zeros(pack["M"]))
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.4))
    fig.subplots_adjust(top=.80, bottom=.12, left=.07, right=.97, hspace=.6)
    fig.suptitle("有效区域：采样像素数与裁剪结果", x=.07, ha="left", fontsize=14)
    fig.text(.07, .90, f"MINH={p['sampling']['min_pixels']}；STEP={p['sampling']['step']}px；REF={p['sampling']['ref']}；"
             f"trim={p['detect']['trim']:.0%}/端；收缩宽={p['sampling']['erosion_width']}，执行两次", fontsize=10)
    axes[0].plot(counts, color=BLUE)
    axes[0].axhline(p["sampling"]["min_pixels"], ls="--", color=RED)
    axes[0].scatter(np.flatnonzero(counts < p["sampling"]["min_pixels"]), counts[counts < p["sampling"]["min_pixels"]], color=RED, s=12)
    sparse_labels(axes[0], np.arange(len(counts)), counts, counts < p["sampling"]["min_pixels"])
    axes[0].set_ylabel("站内像素数")
    axes[0].set_xlabel("原生采样站")
    states = np.vstack([pack["v"], row["valid"]])
    axes[1].imshow(states, aspect="auto", cmap=matplotlib.colors.ListedColormap([RED, "#d9edf7"]), vmin=0, vmax=1)
    axes[1].set_yticks([0, 1], ["首次收缩后", "端部裁剪+再次收缩后"])
    axes[1].set_xlabel("归一化位置；红色=被排除的位置，不直接等于画面异常")
    # Every excluded run is identified by endpoints, even if it is narrow.
    excluded = runs(~row["valid"])
    lines = textwrap.wrap("排除区间：" + "；".join(f"{a}—{b}" for a, b in excluded), width=100)
    if len(lines) > 1:
        fig.set_figheight(6.4 + .22*(len(lines)-1))
    fig.text(.07, .02, "\n".join(lines), color=RED, fontsize=10)
    return save_plot(out, name, fig)


def road_plot(out, name, check, p):
    if check is None:
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.axis("off")
        ax.text(.02, .65, "路面判断：SKIP，当前轨道尚无实际RER确认", fontsize=14)
        ax.text(.02, .35, f"累计门={p['track']['alarm_hold_s']}s；重算间隔={p['track']['reconfirm_s']}s；"
                f"RER门={p['track']['rer_threshold']}；不补算未经算法执行的路面结果", fontsize=11)
        return save_plot(out, name, fig)
    labels = check.get("labels")
    valid = check.get("valid")
    rejected = [] if labels is None else [tuple(x) for x in np.argwhere(labels == 1)]
    impure = [] if valid is None else [tuple(x) for x in np.argwhere((check["counts"] > 0) & ~valid)]
    fig = plt.figure(figsize=(15, 10))
    grid = fig.add_gridspec(3, 2, height_ratios=[4.5, 2.5, .6],
                           top=.84, bottom=.04, left=.06, right=.97, hspace=.65)
    preview, classes = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    distance, luminance = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    rail = fig.add_subplot(grid[2, :])
    ps = p["track"]["patch_size"]
    fig.suptitle(f"路面判断：实际确认帧{check.get('frame_index','?')} / 轨道{check['token']}", x=.06, ha="left", fontsize=14)
    fig.text(.06, .93, f"patch={ps}×{ps}；purity≥{p['track']['rer_purity']:.0%}；非红白面积≥{p['rer']['min_area']}px；"
             f"RER={check['rer']:.4f} / 门{p['track']['rer_threshold']}；{check['status']}", fontsize=11,
             color=RED if check['rer'] < p['track']['rer_threshold'] else BLUE)
    if "median_image" in check:
        original = cv.imread(str(out / check["median_image"]))
        x0, y0, x1, y1 = check["box"]
        preview.imshow(cv.cvtColor(original[y0:y1, x0:x1], cv.COLOR_BGR2RGB))
        preview.set_title("实际中值图中的候选框")
    preview.axis("off")
    rail.axis("off")
    if labels is None:
        classes.axis("off")
        classes.text(0, .5, f"SKIP：{check['status']}；剩余面积={check.get('missing_pixels','未执行')}", color=RED)
        distance.axis("off")
        luminance.axis("off")
        return save_plot(out, name, fig)
    active = np.argwhere(check["counts"] > 0)
    ylo, xlo = active.min(axis=0)
    yhi, xhi = active.max(axis=0)+1
    classes.imshow(labels[ylo:yhi, xlo:xhi], interpolation="nearest", aspect="auto", vmin=0, vmax=2,
                   cmap=matplotlib.colors.ListedColormap(["#dddddd", RED, GREEN]), extent=(xlo, xhi, yhi, ylo))
    classes.set_title("灰=未纳入；红=非路面；绿=路面", fontsize=11)
    positions = np.argwhere(valid)
    failed = labels[valid] == 1
    for y, x in impure:
        classes.add_patch(Rectangle((x, y), 1, 1, fill=False, ec=RED, lw=.8))
    # IDs map every failing patch to the graphs and the full callout rail.
    normal_ids = np.flatnonzero(~failed)
    for i in np.flatnonzero(failed):
        y, x = positions[i]
        classes.text(x+.5, y+.5, str(i), ha="center", va="center", fontsize=8, color="white")
    for i in normal_ids[::max(1, len(normal_ids)//5)]:
        y, x = positions[i]
        classes.text(x+.5, y+.5, str(i), ha="center", va="center", fontsize=8, color="white")
    profile = check["profile"]
    values = check["distances"]
    # 999 is a brightness-failure sentinel, not an actual measured colour distance.
    visible_dist = np.where(check["means"][:, 0] > profile["l_max"], np.nan, values)
    ids = np.arange(len(values))
    distance.scatter(ids[~failed], visible_dist[~failed], color=GREEN, s=10)
    distance.scatter(ids[failed], visible_dist[failed], color=RED, s=16)
    distance.axhline(profile["dist_th"], color=RED, ls="--")
    distance.set_title(f"色度距离：门≤{profile['dist_th']:.4f}；亮度超限999不作真实距离画点", fontsize=10)
    light = check["means"][:, 0]
    luminance.scatter(ids, light, c=np.where(light > profile["l_max"], RED, GREEN), s=12)
    luminance.axhline(profile["l_max"], color=RED, ls="--")
    luminance.set_title(f"亮度L：门≤{profile['l_max']:.4f}；亮度系数={p['rer']['l_sigma']}", fontsize=10)
    distance.set_xlabel("patch编号（对应分类图）")
    luminance.set_xlabel("patch编号（对应分类图）")
    for ax in (distance, luminance):
        ax.grid(alpha=.2)
    note_lines = []
    for i in np.flatnonzero(failed):
        y, x = positions[i]
        ratio = check["counts"][y, x]/ps**2
        note_lines.append(f"异常 #{i} ({x},{y})；纯度={ratio:.1%}\nL={light[i]:.2f}/≤{profile['l_max']:.2f}；距离={values[i]:.2f}/≤{profile['dist_th']:.2f}")
    for y, x in impure:
        note_lines.append(f"未纳入 ({x},{y})\n纯度={check['counts'][y,x]/ps**2:.1%} < {p['track']['rer_purity']:.0%}；未执行颜色判断")
    rail.text(0, .8, f"非路面异常块={len(rejected)}；纯度不足块={len(impure)}。主图全部标红，具体值在下方“全部异常标注图”展开。", color=RED, fontsize=11)
    anomaly_sheet(out, name, note_lines)
    fig.text(.06, .885, f"路面拟合：8×8；色度标准差下限={p['rer']['ab_sigma_min']}；"
             f"距离第{p['rer']['distance_percentile']}百分位+{p['rer']['distance_margin']}；"
             f"路面块={check.get('road_count',0)}/{check.get('valid_count',0)}", fontsize=10)
    return save_plot(out, name, fig)


def figure_html(path, title, prefix="../"):
    from water_intermediates import esc
    return f'<section><h2>{esc(title)}</h2><img style="width:100%;min-width:1000px" src="{prefix}{esc(path)}"></section>'


def frame_plot(out, record, p, checks):
    from water_intermediates import document, esc
    out = Path(out)
    index, trace = record["index"], record["trace"]
    raw = cv.imread(str(out / record["image"]))
    exact = "pixel_data" in record
    pixels = record.get("pixel_data")
    if pixels is None:
        pixels = masks_from_hsv(cv.cvtColor(raw, cv.COLOR_BGR2HSV), p)
    body = '<p><a href="../index.html">返回总览</a>；红色=阈值未通过/排除项，蓝色=正常疏排标记；异常区间不省略。</p>'
    if not exact:
        body += '<p style="color:#c62828">旧数据的颜色像素图为JPEG重建示意；比例、分数、质量、support及RER使用真实trace。</p>'
    body += figure_html(quality_plot(out, record, p), "整帧质量")
    for r in p["rows"]:
        rid = r["id"]
        row = trace["rows"].get(rid, {})
        body += f'<h1>{esc(rid)}：{esc(row.get("status", "INVALID / SKIP"))}</h1>'
        if "pack" not in row:
            body += '<p>SKIP：整帧质量未通过，本行未执行采样及颜色检测。</p>'
            continue
        stem = f"f{index:08d}_{rid}"
        x = np.arange(p["sampling"]["ref"])
        pack = row["pack"]
        notes = []
        for c in row["candidates"]:
            a, b = c["env"]
            text = f"{c['reason']}；包络宽={c['env_width']}/≥{p['detect']['min_width']}"
            if c.get("core"):
                text += f"；核心最低值={c['core'][3]}/<{p['detect']['floor_max']}；核心宽={c['core'][2]}/≥{p['detect']['core_min_width']}；原生宽={c['native_width']:.2f}/[{.5*r['U']:.2f},{2*r['U']:.2f}]；分数={c['confidence']}/≥{p['detect']['conf_th']}"
            text = ("缺口候选；" if c.get("kept") else "筛选未通过；") + text
            notes.append((a, b, text, RED))
        if "red_ratio" in row:
            ratios = [("下部红色比例", row["red_ratio"], RED), ("上部白色比例", row["white_ratio"], BLUE)]
            if "lower_white_ratio" in row:
                ratios += [("下部白色比例（扣除与红色重叠）", row["lower_white_ratio"], GREEN),
                           ("下部红白合计", row["red_ratio"] + row["lower_white_ratio"], "#8653a8")]
            path = series_plot(out, stem+"_ratio", "红白比例", np.arange(pack["M"]),
                ratios, f"上部{p['sampling']['top_fraction']:.0%}计白色；下部{p['sampling']['bottom_fraction']:.0%}计"
                + ("红色+白色" if 'lower_white_ratio' in row else "红色（旧记录）")
                + "；比例没有独立通过门")
            body += figure_html(path, "红白比例")
        curves = [("颜色分数", pack["s"], "#222222")]
        if "smooth" in row:
            curves += [("平滑分数", row["smooth"], BLUE), ("局部基线", row["baseline"], "#8653a8")]
        path = series_plot(out, stem+"_score", "颜色分数与低谷判断", x, curves,
            f"下部{'红白' if p['sampling'].get('lower_colours') == 'red_or_white' else '红色'}权重={p['sampling']['red_weight']}；上部白权重={p['sampling']['white_weight']}；sigma={p['signal']['sigma']}；"
            f"基线窗口={p['signal']['baseline_window']}；下降启动>{p['signal']['entry']} / 延续>{p['signal']['extension']}",
            [("核心floor", p["detect"]["floor_max"]), ("核心退出", p["detect"]["floor_max"]+p["detect"]["core_exit"])], notes, ~row["valid"])
        body += figure_html(path, "颜色分数")
        origin = row.get("origin")
        split = row.get("split")
        if split is not None:
            ox, oy = origin
            h, w = split.shape
            image = raw[oy:oy+h, ox:ox+w]
            cropped = {k: a[oy:oy+h, ox:ox+w] for k, a in pixels.items()}
            c = p["colour"]
            specs = [("红色", split == 2, f"下部红色：H∈{c['red_h']}；S≥{c['red_s_min']}；V≥{c['red_v_min']}",
                      ([v for pair in c["red_h"] for v in pair], [c["red_s_min"]], [c["red_v_min"]])),
                     ("白色", split > 0 if 'lower_white_ratio' in row else split == 1, f"{'上下部' if 'lower_white_ratio' in row else '上部'}白色：日光S≤{c['sun_s_max']},V≥{c['sun_v_min']}；"
                      f"阴影{c['shadow_s'][0]}<S≤{c['shadow_s'][1]},{c['shadow_v'][0]}≤V<{c['shadow_v'][1]},H≥{c['shadow_h_min']}",
                      ([c["shadow_h_min"]], [c["sun_s_max"], c["shadow_s"][1]], list(c["shadow_v"]))),
                     ("异色", split > 0, f"异色：S>{c['foreign_s_min']}且非红色色相；白色与异色可能同时命中，独立显示",
                      ([v for pair in c["red_h"] for v in pair], [c["foreign_s_min"]], []))]
            for kind, expected, params, thresholds in specs:
                path = colour_plot(out, stem+"_"+kind, image, cropped, expected, kind, params, thresholds, exact)
                body += figure_html(path, kind+"识别")
        else:
            body += '<p>颜色图SKIP：该行没有有效采样分区。</p>'
        foreign_notes = []
        for a, b in runs(row["valid"] & (pack["f"] > p["quality"]["foreign"])):
            bad = b-a+1 > p["quality"]["foreign_span"]
            foreign_notes.append((a, b, f"异色连续长度={b-a+1} / 允许≤{p['quality']['foreign_span']}；{'遮挡门失败' if bad else '未达到遮挡长度'}", RED if bad else BLUE))
        path = series_plot(out, stem+"_foreign", "异色比例与遮挡区域", x, [("异色比例", pack["f"], "#8653a8")],
            f"比例>{p['quality']['foreign']}且连续长度>{p['quality']['foreign_span']}判遮挡；"
            f"另一个遮挡门：颜色分数<{p['quality']['dark']}且连续>{p['quality']['dark_span']}；实际最长={row['dark_run']}",
            [("异色比例门", p["quality"]["foreign"])], foreign_notes, ~row["valid"])
        body += figure_html(path, "异色区域判断")
        body += figure_html(validity_plot(out, stem+"_valid", row, p), "有效区域")
        sup = trace["support"][rid]
        support_notes = []
        for run in trace.get("support_runs", {}).get(rid, []):
            support_notes.append((run["start"], run["end"], f"成框宽度={run['width']} / ≥{p['detect']['min_box_width']}；{'缺口候选，保留' if run['kept'] else '过窄，淘汰'}", RED))
        path = series_plot(out, stem+"_support", "历史累计support", x, [("历史命中比例", sup["values"], BLUE)],
            f"有效帧数={sup['observations']}；support门≥{p['detect']['support_th']}；成框宽≥{p['detect']['min_box_width']}；全历史，无滑窗；"
            f"当前行{'跳过更新，保留历史' if row['status'] != 'OK' else '正常更新'}", [("support_th", p["detect"]["support_th"])], support_notes)
        body += figure_html(path, "历史累计参数")
        tokens = {tr["token"] for tr in trace["tracks"] if tr["row_id"] == rid}
        available = [c for c in checks if c["token"] in tokens]
        for j, check in enumerate(available or [None]):
            path = road_plot(out, stem+f"_road{j}", check, p)
            body += figure_html(path, "路面判断参数")
            notes_path = path.replace(".svg", "_notes.svg")
            if (out / notes_path).is_file():
                body += '<details><summary>全部异常标注图</summary>' + figure_html(notes_path, "异常值逐块标注") + '</details>'
    body += f'<details><summary>补充：原始判断表和轨道信息</summary><a href="frame_detail_{index:08d}.html">打开详情</a></details>'
    (out / "pages" / f"frame_{index:08d}.html").write_text(
        document(f"水马图示：帧{index} / {record['time_s']:.3f}s", body), encoding="utf-8")


def road_page(out, index, check, p):
    from water_intermediates import document
    path = road_plot(out, f"rer_{index:08d}_{check['token']}", check, p)
    body = '<p><a href="index.html">返回总览</a></p>' + figure_html(path, "路面判断", prefix="")
    notes_path = path.replace(".svg", "_notes.svg")
    if (out / notes_path).is_file():
        body += '<details><summary>全部异常标注图</summary>' + figure_html(notes_path, "异常值逐块标注", prefix="") + '</details>'
    body += f'<details><summary>原始逐patch数据</summary><a href="rer_detail_{index:08d}_{check["token"]}.html">打开详情</a></details>'
    (out / f"rer_{index:08d}_{check['token']}.html").write_text(document("实际路面判断图", body), encoding="utf-8")
