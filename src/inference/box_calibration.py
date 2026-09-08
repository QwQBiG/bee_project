"""Optional center-preserving HBB calibration after detection NMS."""
import math


def calibrate_detections(rows, factor, width, height):
    factor = float(factor)
    if not math.isfinite(factor) or not 0.5 <= factor <= 1.5:
        raise ValueError("detection box scale must be finite and within [0.5, 1.5]")
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError("image dimensions must be finite and positive")
    result = []
    for row in rows:
        x, y, w, h = row["bbox"]
        if not all(math.isfinite(v) for v in (x, y, w, h)) or min(w, h) <= 0:
            raise ValueError("box coordinates must be finite with positive size")
        left, top = max(0, x+w*(1-factor)/2), max(0, y+h*(1-factor)/2)
        right, bottom = min(width, x+w*(1+factor)/2), min(height, y+h*(1+factor)/2)
        box = [round(v, 2) for v in (left, top, right-left, bottom-top)]
        if min(box[2:]) <= 0:
            raise ValueError("calibrated box is empty")
        result.append({**row, "bbox": box})
    return result
