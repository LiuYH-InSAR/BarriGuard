"""Shared coordinate transforms and one-to-one box association."""
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


