"""Feed ONNX detections to installed Ultralytics BYTETracker, without YOLO loading."""
from types import SimpleNamespace

import numpy as np


class DetectionBatch:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32).reshape(-1, 6)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return DetectionBatch(self.data[index])

    @property
    def conf(self):
        return self.data[:, 4]

    @property
    def cls(self):
        return self.data[:, 5]

    @property
    def xywh(self):
        boxes = self.data[:, :4].copy()
        boxes[:, :2] += boxes[:, 2:] / 2
        return boxes


class OnnxByteTracker:
    def __init__(self, options=None):
        # Fail before Ultralytics has a chance to auto-install dependencies.
        try:
            import lap  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("ByteTrack requires installed lap; offline runtime must bundle it") from exc
        from ultralytics.trackers.byte_tracker import BYTETracker
        options = options or {}
        self.low = float(options.get("track_low_thresh", .05))
        high = float(options.get("track_high_thresh", .1))
        new = float(options.get("new_track_thresh", high))
        if not 0 <= self.low < high <= new <= 1:
            raise ValueError("require 0 <= low < high <= new <= 1")
        self.backend = BYTETracker(SimpleNamespace(
            track_low_thresh=self.low, track_high_thresh=high,
            new_track_thresh=new, track_buffer=int(options.get("track_buffer", 10)),
            match_thresh=float(options.get("match_thresh", .8)),
            fuse_score=bool(options.get("fuse_score", True))), frame_rate=30)

    def update(self, rows):
        data = [[*row["bbox"], row["confidence"], 0] for row in rows]
        results = self.backend.update(DetectionBatch(data))
        output = []
        for row in results:
            x1, y1, x2, y2, track_id = row[:5]
            bbox = [float(x1), float(y1), float(x2-x1), float(y2-y1)]
            if round(bbox[2], 2) <= 0 or round(bbox[3], 2) <= 0:
                continue
            output.append({"track_id": int(track_id), "bbox": bbox})
        return output
