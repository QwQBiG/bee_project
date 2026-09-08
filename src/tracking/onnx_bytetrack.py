"""Lightweight two-stage ByteTrack-style association for ONNX detections.

The submission runtime deliberately avoids importing Ultralytics or PyTorch.
Detection is performed by ONNX Runtime; this module only needs NumPy and lap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class _Track:
    track_id: int
    bbox: np.ndarray
    velocity: np.ndarray
    missed: int = 0

    def predicted_bbox(self) -> np.ndarray:
        predicted = self.bbox.copy()
        predicted[:2] += self.velocity * max(1, self.missed + 1)
        return predicted


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, aw, ah = (float(v) for v in a)
    bx1, by1, bw, bh = (float(v) for v in b)
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    union = max(aw, 0.0) * max(ah, 0.0) + max(bw, 0.0) * max(bh, 0.0) - intersection
    return intersection / union if union > 0 else 0.0


def _associate(
    tracks: List[_Track], detections: List[Dict], cost_limit: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))
    import lap

    cost = np.empty((len(tracks), len(detections)), dtype=np.float64)
    for track_index, track in enumerate(tracks):
        predicted = track.predicted_bbox()
        for detection_index, detection in enumerate(detections):
            cost[track_index, detection_index] = 1.0 - _iou(
                predicted, detection["bbox"])
    _, track_to_detection, detection_to_track = lap.lapjv(
        cost, extend_cost=True, cost_limit=float(cost_limit))
    matches = [(track_index, int(detection_index))
               for track_index, detection_index in enumerate(track_to_detection)
               if detection_index >= 0]
    unmatched_tracks = [index for index, value in enumerate(track_to_detection)
                        if value < 0]
    unmatched_detections = [index for index, value in enumerate(detection_to_track)
                            if value < 0]
    return matches, unmatched_tracks, unmatched_detections


class OnnxByteTracker:
    """Two-stage high/low-score tracker with motion-aware IoU association."""

    def __init__(self, options=None):
        try:
            import lap  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "ByteTrack requires bundled lap; offline runtime cannot install it") from exc
        options = options or {}
        self.low = float(options.get("track_low_thresh", 0.05))
        self.high = float(options.get("track_high_thresh", 0.25))
        self.new = float(options.get("new_track_thresh", self.high))
        self.buffer = int(options.get("track_buffer", 10))
        self.match_cost = float(options.get("match_thresh", 0.8))
        if not 0 <= self.low < self.high <= self.new <= 1:
            raise ValueError("require 0 <= low < high <= new <= 1")
        if self.buffer < 0 or not 0 < self.match_cost <= 1:
            raise ValueError("track_buffer/match_thresh is invalid")
        self._tracks: List[_Track] = []
        self._next_id = 1

    @staticmethod
    def _valid(rows: List[Dict], low: float) -> List[Dict]:
        output = []
        for row in rows:
            box = np.asarray(row["bbox"], dtype=np.float64)
            confidence = float(row["confidence"])
            if (box.shape == (4,) and np.isfinite(box).all()
                    and box[2] > 0 and box[3] > 0
                    and np.isfinite(confidence) and confidence >= low):
                output.append({"bbox": box, "confidence": confidence})
        return output

    @staticmethod
    def _update_track(track: _Track, detection: Dict) -> None:
        new_box = np.asarray(detection["bbox"], dtype=np.float64)
        old_center = track.bbox[:2] + track.bbox[2:] / 2
        new_center = new_box[:2] + new_box[2:] / 2
        measured_velocity = new_center - old_center
        track.velocity = 0.65 * measured_velocity + 0.35 * track.velocity
        track.bbox = new_box
        track.missed = 0

    def update(self, rows):
        detections = self._valid(list(rows), self.low)
        high = [row for row in detections if row["confidence"] >= self.high]
        low = [row for row in detections if row["confidence"] < self.high]
        emitted: List[_Track] = []

        matches, unmatched_track_indexes, unmatched_high = _associate(
            self._tracks, high, self.match_cost)
        for track_index, detection_index in matches:
            track = self._tracks[track_index]
            self._update_track(track, high[detection_index])
            emitted.append(track)

        remaining_tracks = [self._tracks[index]
                            for index in unmatched_track_indexes]
        low_matches, still_unmatched, _ = _associate(
            remaining_tracks, low, min(self.match_cost, 0.5))
        for track_index, detection_index in low_matches:
            track = remaining_tracks[track_index]
            self._update_track(track, low[detection_index])
            emitted.append(track)

        for index in still_unmatched:
            remaining_tracks[index].missed += 1

        for detection_index in unmatched_high:
            detection = high[detection_index]
            if detection["confidence"] < self.new:
                continue
            track = _Track(
                track_id=self._next_id,
                bbox=np.asarray(detection["bbox"], dtype=np.float64),
                velocity=np.zeros(2, dtype=np.float64),
            )
            self._next_id += 1
            self._tracks.append(track)
            emitted.append(track)

        self._tracks = [track for track in self._tracks
                        if track.missed <= self.buffer]
        emitted.sort(key=lambda track: track.track_id)
        return [{"track_id": track.track_id,
                 "bbox": [float(value) for value in track.bbox]}
                for track in emitted]
