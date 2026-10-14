"""Fixed lamp polygons: hourly flash evidence and a separate morning status."""
import hashlib
import json
from pathlib import Path

import cv2 as cv
import numpy as np

COLORS = {"FLASHING": (0, 200, 0), "SUSPECT": (0, 165, 255),
          "NOT_FLASHING": (0, 0, 255), "UNDETERMINED": (160, 160, 160)}
CIRCLE_VERTICES = 64


def hotspot_region(duty, top, keep, dim, gate, excluded, gamma, floor, threshold):
    color_level = np.clip(duty / max(top, np.finfo(np.float32).eps), 0, 1) ** gamma
    # Normalize out the fixed blend opacity; retain both existing dim weights.
    strength = color_level * keep * np.clip(duty / floor, 0, 1) * dim * dim * gate
    return (strength >= threshold) & ~excluded


class LampChecks:
    def __init__(self, spec):
        self.spec = spec
        raw = Path(spec.polygons).read_bytes()
        data = json.loads(raw)
        self.size = (int(data["imageHeight"]), int(data["imageWidth"]))
        self.polys = {}
        for shape in data.get("shapes", []):
            label = str(shape.get("label", ""))
            points = np.asarray(shape.get("points", []), np.float64)
            kind = shape.get("shape_type")
            if kind == "circle" and points.shape == (2, 2) and np.isfinite(points).all():
                center = points[0]
                radius = float(np.linalg.norm(points[1] - center))
                if radius <= 0:
                    raise ValueError(f"Lamp circle has zero radius: {label}")
                angles = np.linspace(0, 2 * np.pi, CIRCLE_VERTICES, endpoint=False)
                points = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
            if (not label or label in self.polys or kind not in ("polygon", "circle")
                    or points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                    or not np.isfinite(points).all()):
                raise ValueError("Each lamp needs a unique label and a valid polygon or circle")
            if (points[:, 0].min() < 0 or points[:, 1].min() < 0
                    or points[:, 0].max() >= self.size[1] or points[:, 1].max() >= self.size[0]):
                raise ValueError(f"Lamp polygon outside image: {label}")
            self.polys[label] = np.rint(points).astype(np.int32)
        if not self.polys:
            raise ValueError("No lamp polygons found")
        self.signature = hashlib.sha256(raw).hexdigest()
        self.history = []

    def dump(self):
        return json.dumps({"version": 3, "signature": self.signature, "history": self.history})

    def restore(self, value):
        data = json.loads(str(value))
        if data.get("version") == 3 and data.get("signature") == self.signature:
            self.history = data.get("history", [])[-48:]

    def observe(self, heatmap, excluded, coverage, usable, stamp, *, overflow=False):
        if heatmap.shape != self.size:
            raise ValueError("Lamp label image dimensions differ from stream")
        rows = []
        for label, poly in self.polys.items():
            mask = np.zeros(self.size, np.uint8)
            cv.fillPoly(mask, [poly], 1)
            area = int(mask.sum())
            valid = (mask > 0) & ~excluded
            support = int(valid.sum())
            hits = int(np.count_nonzero(heatmap & (mask > 0)))
            ratio = hits / max(area, 1)
            status = "FLASHING" if ratio > self.spec.heatmap_ratio else "NOT_FLASHING"
            rows.append(dict(lamp_id=label, status=status, heatmap_pixels=hits,
                             heatmap_ratio=round(ratio, 4), polygon_pixels=area,
                             valid_pixels=support, sample_coverage=round(coverage, 4),
                             candidate_overflow=bool(overflow)))
        item = dict(hour=str(stamp)[:13], lamps=rows)
        if self.history and self.history[-1]["hour"] == item["hour"]:
            self.history[-1] = item
        else:
            self.history.append(item)
        self.history = self.history[-48:]
        return rows

    def summary(self):
        rows = []
        for label in self.polys:
            statuses = [next(r["status"] for r in h["lamps"] if r["lamp_id"] == label)
                        for h in self.history]
            lit = statuses.count("FLASHING")
            final = "FLASHING" if lit >= self.spec.lit_hours else "NOT_FLASHING"
            rows.append(dict(lamp_id=label, status=final, lit_hours=lit,
                             unavailable_hours=statuses.count("UNDETERMINED"),
                             observed_hours=len(statuses)))
        return rows

    def unknown(self, stamp):
        empty = np.zeros(self.size, bool)
        rows = self.observe(empty, ~empty, 0.0, False, stamp)
        for row in rows:
            row["status"] = "UNDETERMINED"
        return rows

    def draw(self, image, rows):
        for row in rows:
            poly = self.polys[row["lamp_id"]]
            color = COLORS["UNDETERMINED"] if row["status"] == "SUSPECT" else COLORS[row["status"]]
            cv.polylines(image, [poly], True, color, 2)
        return image
