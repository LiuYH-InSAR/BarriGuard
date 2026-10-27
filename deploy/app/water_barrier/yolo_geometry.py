"""坐标、唯一匹配与水马排列；不读旧 poly。"""
import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a, b):
    low = np.maximum(a[:2], b[:2])
    high = np.minimum(a[2:4], b[2:4])
    area = np.maximum(high - low, 0).prod()
    aa = np.maximum(np.asarray(a[2:4]) - a[:2], 0).prod()
    bb = np.maximum(np.asarray(b[2:4]) - b[:2], 0).prod()
    return float(area / max(aa + bb - area, np.finfo(float).eps))


def warp_boxes(boxes, matrix):
    if not len(boxes):
        return []
    out = []
    for box in boxes:
        x0, y0, x1, y1 = box[:4]
        corners = np.float32([[x0,y0], [x1,y0], [x1,y1], [x0,y1]])
        pts = cv.perspectiveTransform(corners[None], matrix)[0]
        out.append([*pts.min(axis=0).tolist(), *pts.max(axis=0).tolist()])
    return out


def match_boxes(expected, detected, cfg, error_ratio=0.0):
    if not len(expected) or not len(detected):
        return {}, list(range(len(expected)))
    tolerance = min(cfg.max_center_ratio, cfg.center_ratio + error_ratio)
    cost = np.ones((len(expected), len(detected))) * np.inf
    for a, box in enumerate(expected):
        size = np.maximum(np.asarray(box[2:4]) - box[:2], np.finfo(float).eps)
        center = (np.asarray(box[:2]) + box[2:4]) / 2
        for b, other in enumerate(detected):
            osize = np.maximum(np.asarray(other[2:4]) - other[:2], np.finfo(float).eps)
            ratios = osize / size
            if np.any(ratios < cfg.min_size_ratio) or np.any(ratios > cfg.max_size_ratio):
                continue
            distance = np.linalg.norm(((np.asarray(other[:2]) + other[2:4]) / 2 - center) / size)
            overlap = iou(box, other)
            if overlap >= cfg.min_iou or distance <= tolerance:
                cost[a, b] = 1 - overlap + distance
    finite = np.isfinite(cost)
    if not finite.any():
        return {}, list(range(len(expected)))
    safe = np.where(finite, cost, np.finfo(np.float32).max)
    rows, cols = linear_sum_assignment(safe)
    matches = {int(a): int(b) for a,b in zip(rows,cols) if finite[a,b]}
    return matches, [a for a in range(len(expected)) if a not in matches]


def tile_regions(shape, cfg):
    height, width = shape[:2]
    stride = max(1, round(cfg.tile_px * (1 - cfg.overlap)))
    def starts(length):
        last = max(0, length - cfg.tile_px)
        return sorted(set([*range(0, last + 1, stride), last]))
    return [(x,y,min(x+cfg.tile_px,width),min(y+cfg.tile_px,height))
            for y in starts(height) for x in starts(width)]


def merge_boxes(boxes, threshold):
    kept = []
    for box in sorted(boxes, key=lambda b: b[4], reverse=True):
        if not any(iou(box, old) >= threshold for old in kept):
            kept.append(box)
    return kept


def build_rows(boxes, cfg):
    """局部邻接生成排；不同尺度或明显不同行的目标不连接。"""
    groups = [{i} for i in range(len(boxes))]
    for i,a in enumerate(boxes):
        ac = (np.array(a[:2]) + a[2:4]) / 2
        size = np.maximum(np.array(a[2:4]) - a[:2], 1)
        for j,b in enumerate(boxes[:i]):
            bc = (np.array(b[:2]) + b[2:4]) / 2
            bs = np.maximum(np.array(b[2:4]) - b[:2], 1)
            delta = np.abs(ac - bc) / np.maximum(size, bs)
            if delta[1] > cfg.row_vertical_ratio or np.linalg.norm(delta) > cfg.row_distance_ratio:
                continue
            g1 = next(g for g in groups if i in g)
            g2 = next(g for g in groups if j in g)
            if g1 is not g2:
                g1.update(g2)
                groups.remove(g2)
    rows = []
    for group in groups:
        indices = list(group)
        centers = np.array([(np.array(boxes[i][:2])+boxes[i][2:4])/2 for i in indices])
        if len(indices) > 1:
            _, _, axes = np.linalg.svd(centers - centers.mean(axis=0), full_matrices=False)
            indices = [indices[k] for k in np.argsort(centers @ axes[0])]
        rows.append(indices)
    return rows


def missing_groups(indices, rows):
    missing = set(indices)
    groups = []
    for row in rows:
        current = []
        for index in row:
            if index in missing:
                current.append(index)
            elif current:
                groups.append(current)
                current = []
        if current:
            groups.append(current)
    return groups


def union_box(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def infer_initial_gaps(boxes, rows, cfg):
    """只推断排内异常间隔，不延长排的端点；结果是待核实空位。"""
    inferred = []
    for row in rows:
        if len(row) < cfg.min_row_observations:
            continue
        values = np.asarray([boxes[i][:4] for i in row], dtype=float)
        centers = (values[:, :2] + values[:, 2:]) / 2
        sizes = values[:, 2:] - values[:, :2]
        vectors = np.diff(centers, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        # 以局部框宽消除近大远小，避免用全排统一像素间距。
        widths = (sizes[:-1, 0] + sizes[1:, 0]) / 2
        normalized = lengths / np.maximum(widths, np.finfo(float).eps)
        for k, distance in enumerate(normalized):
            peers = [j for j in range(max(0,k-cfg.local_neighbors),
                     min(len(normalized),k+cfg.local_neighbors+1)) if j != k]
            if len(peers) < cfg.min_neighbor_intervals or lengths[k] <= 0:
                continue
            direction = vectors[k] / lengths[k]
            peers = [j for j in peers if lengths[j] > 0 and
                     np.dot(direction, vectors[j]/lengths[j]) >= cfg.min_direction_cosine]
            if len(peers) < cfg.min_neighbor_intervals:
                continue
            scale = np.median(normalized[peers])
            if scale <= 0:
                continue
            ratio = distance / scale
            intervals = round(ratio)
            size_ratio = sizes[k+1] / np.maximum(sizes[k], np.finfo(float).eps)
            if (not cfg.min_gap_ratio <= ratio <= cfg.max_gap_ratio
                    or abs(ratio-intervals) > cfg.integer_tolerance
                    or not 1 <= intervals-1 <= cfg.max_inferred_per_gap
                    or np.any(size_ratio > cfg.max_size_ratio)
                    or np.any(size_ratio < 1/cfg.max_size_ratio)):
                continue
            for n in range(1, intervals):
                fraction = n / intervals
                box = ((1-fraction)*values[k] + fraction*values[k+1]).tolist()
                if any(iou(box, b) >= cfg.duplicate_iou for b in boxes+inferred):
                    continue
                if len(boxes)+len(inferred) >= cfg.max_slots:
                    return inferred
                inferred.append(box)
    return inferred
