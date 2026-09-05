"""Optional overlapping crops with global NMS in original image coordinates."""
import time

import numpy as np

from inference.algorithm_cli import multiclass_nms, run_detection_array


def detect_array(image, config, *, conf_override=None, topk=300):
    scene = config.get("_scene")
    options = config.get("tiling", {})
    if not options.get("enabled", False):
        return run_detection_array(image, config, conf_override=conf_override, topk=topk)
    if scene not in ("outside", "inside"):
        raise ValueError("tiled detection requires an explicit scene")
    overlap = float(options.get("overlap", .2))
    if not 0 <= overlap < .5:
        raise ValueError("tile overlap must be within [0, .5)")
    started = time.perf_counter()
    height, width = image.shape[:2]
    tile_h, tile_w = int(np.ceil(height / (2 - overlap))), int(np.ceil(width / (2 - overlap)))
    regions = [(x, y, tile_w, tile_h)
               for y in sorted({0, height - tile_h})
               for x in sorted({0, width - tile_w})]
    if options.get("include_full", True):
        regions.insert(0, (0, 0, width, height))
    candidates = []
    for x, y, w, h in regions:
        rows, _ = run_detection_array(
            image[y:y+h, x:x+w], config, conf_override=conf_override, topk=max(600, topk))
        for row in rows:
            left, top, bw, bh = row["bbox"]
            candidates.append([left + x, top + y, bw, bh, row["confidence"]])
    if not candidates:
        return [], int((time.perf_counter() - started) * 1000)
    array = np.asarray(candidates, dtype=np.float32)
    cfg = config["detector"][scene]
    threshold = cfg.get("conf", .25) if conf_override is None else conf_override
    kept = multiclass_nms(array[:, :4], array[:, 4], float(cfg.get("iou", .45)),
                          threshold, topk=topk)
    rows = [{"bbox": [round(float(x), 2) for x in row[:4]], "class_id": 0,
             "label": "bee", "confidence": round(float(row[4]), 6)} for row in kept]
    return rows, int((time.perf_counter() - started) * 1000)
