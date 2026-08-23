"""巢外视频的携粉候选、进巢采集量和营养风险分析。

默认方案以可见光中后足附近的黄/橙色像素作为保守候选，适合工程联调，
不等同于经花粉团标注模型确认的生物学结论。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


@dataclass
class _TrackPollenState:
    samples: List[float]
    positions: List[Tuple[float, float]]
    entered: bool = False


class OutsidePollenAnalyzer:
    """从巢外 RGB 视频中统计携粉进巢候选事件。"""

    def __init__(self, config: Dict | None = None):
        config = config or {}
        self.enabled = bool(config.get("enabled", True))
        self.min_color_ratio = float(config.get("min_color_ratio", 0.12))
        self.min_samples = int(config.get("min_samples", 3))
        self.positive_sample_ratio = float(config.get("positive_sample_ratio", 0.60))
        self.min_inbound_events = int(config.get("min_inbound_events", 8))
        self.min_pollen_ratio = float(config.get("min_pollen_ratio", 0.15))
        self.entrance_region = config.get("entrance_region")
        self.hsv_lower = np.array(config.get("hsv_lower", [15, 70, 70]), dtype=np.uint8)
        self.hsv_upper = np.array(config.get("hsv_upper", [45, 255, 255]), dtype=np.uint8)
        self.tracks: Dict[int, _TrackPollenState] = defaultdict(lambda: _TrackPollenState([], []))
        self.inbound_events: List[Dict] = []
        self.frame_shape: Tuple[int, int] | None = None

    def _in_entrance(self, center: Tuple[float, float]) -> bool:
        if not self.entrance_region or not self.frame_shape:
            return False
        x, y, width, height = self.entrance_region
        image_height, image_width = self.frame_shape
        left, top = x * image_width, y * image_height
        right, bottom = (x + width) * image_width, (y + height) * image_height
        return left <= center[0] <= right and top <= center[1] <= bottom

    def _pollen_color_score(self, frame: np.ndarray, bbox: List[float]) -> float:
        x, y, width, height = [int(round(value)) for value in bbox]
        # 花粉团通常位于后足；仅检查检测框下半部，减少身体背景色的干扰。
        y += max(height // 2, 0)
        height = max(height // 2, 1)
        crop = frame[max(0, y):min(frame.shape[0], y + height),
                     max(0, x):min(frame.shape[1], x + width)]
        if crop.size == 0:
            return 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        # 去除极小噪点；分辨率过低时保留原掩码。
        if min(crop.shape[:2]) >= 5:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return float(np.count_nonzero(mask) / mask.size)

    def _loaded(self, state: _TrackPollenState) -> Tuple[bool, float]:
        if len(state.samples) < self.min_samples:
            return False, 0.0
        positive = sum(score >= self.min_color_ratio for score in state.samples)
        confidence = positive / len(state.samples)
        return confidence >= self.positive_sample_ratio, confidence

    def update(self, frame: np.ndarray, tracks: Iterable, frame_id: int) -> None:
        if not self.enabled:
            return
        self.frame_shape = frame.shape[:2]
        for track in tracks:
            track_id = int(track.track_id)
            state = self.tracks[track_id]
            state.samples.append(self._pollen_color_score(frame, track.bbox))
            # 限制单轨迹样本数量，使长视频内存稳定。
            if len(state.samples) > 120:
                state.samples.pop(0)
            center = tuple(float(value) for value in track.center)
            was_inside = self._in_entrance(state.positions[-1]) if state.positions else False
            is_inside = self._in_entrance(center)
            state.positions.append(center)
            if len(state.positions) > 120:
                state.positions.pop(0)
            # 从入口外进入预设入口区域，记录一次进巢事件。
            if self.entrance_region and not state.entered and state.positions[:-1] and not was_inside and is_inside:
                loaded, confidence = self._loaded(state)
                self.inbound_events.append({
                    "frame": int(frame_id), "track_id": track_id,
                    "pollen_candidate": loaded,
                    "candidate_confidence": round(confidence, 3),
                })
                state.entered = True

    def build_report(self) -> Dict:
        observed = [state for state in self.tracks.values() if len(state.samples) >= self.min_samples]
        candidates = sum(self._loaded(state)[0] for state in observed)
        inbound_total = len(self.inbound_events)
        inbound_pollen = sum(event["pollen_candidate"] for event in self.inbound_events)
        pollen_ratio = inbound_pollen / inbound_total if inbound_total else None
        if not self.enabled:
            nutrition_status, nutrition_text = "unknown", "花粉分析已禁用。"
        elif not self.entrance_region:
            nutrition_status, nutrition_text = "unknown", "未配置蜂箱入口区域，无法将携粉候选转化为进巢采集量。"
        elif inbound_total < self.min_inbound_events:
            nutrition_status, nutrition_text = "unknown", "进巢样本不足，不能据此做营养判断。"
        elif pollen_ratio < self.min_pollen_ratio:
            nutrition_status, nutrition_text = "warning", "携粉进巢比例偏低；请结合连续多日数据、天气和花源现场调查复核。"
        else:
            nutrition_status, nutrition_text = "normal", "本时间窗内携粉进巢比例未低于设定阈值。"
        return {
            "method": "HSV 花粉颜色候选（非专用花粉团模型确认）",
            "pollen_candidates": candidates,
            "analyzable_tracks": len(observed),
            "inbound_events": inbound_total,
            "pollen_inbound_events": inbound_pollen,
            "pollen_inbound_ratio": round(pollen_ratio, 3) if pollen_ratio is not None else None,
            "entrance_region": self.entrance_region,
            "nutrition_assessment": {"status": nutrition_status, "message": nutrition_text},
            "limitations": "颜色会受光照、花粉颜色、背景和分辨率影响。用于正式评估前，必须以“携粉/未携粉”标注数据训练并验证专用模型。",
        }
