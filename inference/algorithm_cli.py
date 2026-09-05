"""Standard competition evaluation CLI — single-image inference to stdout.

This module implements the **刚性交付约束** from section 3.3 of the
official evaluation specification for SY-202601:

* Model format: **.onnx only** (no .pt / .pth / .pb).
* Runtime: ONNX Runtime with ``log_severity_level = 3`` (Errors only) so
  no Info/Warning lines pollute stdout.
* One-shot per-process invocation:
      algorithm.exe --image_path "<path>"
    → stdout exactly **one line** with the detection JSON.
* Exit codes:
    code=1  → success, JSON printed with detections.
    code=0  → handled user error (file missing / invalid format), JSON
              printed with an ``error`` field and ``message``.
* Per-image wall-clock timeout in the evaluator is 10 seconds; we keep
  the session warm via lazy module-level caching but everything required
  by the grader works in a fresh process.
* Path handling: supports long paths and Chinese file names on Windows
  (CVAT exports routinely contain such characters).

The CLI supports both DETECTION (``--task detect``, default) and
TRACKING mode.  For tracking, the program must hold identity state
across frames; since the competition contract specifies a separate
per-frame invocation for detection, we keep tracking as a side-mode
driven by a small ``--sequence_id`` / ``--frame_id`` pair and a
file-backed track cache under ``%TEMP%``.

Usage (detection, competition default):

    algorithm_cli.py --image_path "C:\\demo\\0001.jpg"

Usage (tracking, local development — sequence cache persisted):

    algorithm_cli.py --task track --sequence_id seq001 --frame_id 7
                     --image_path "C:\\demo\\seq001_f007.jpg"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Config discovery — shipped alongside algorithm.exe in the compiled onedir
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_BASENAME = "algorithm_config.json"


def _default_search_paths(prog_path: Path) -> List[Path]:
    """Where we look for the compiled-onedir config & onnx weights."""
    candidates = [
        prog_path.parent / DEFAULT_CONFIG_BASENAME,          # alongside exe
        prog_path.parent / "configs" / DEFAULT_CONFIG_BASENAME,
        Path.cwd() / DEFAULT_CONFIG_BASENAME,                 # working dir
        Path(__file__).resolve().parent.parent / "configs" / DEFAULT_CONFIG_BASENAME,
    ]
    return [c for c in candidates if c.is_file()]


def load_runtime_config(explicit: Optional[str] = None) -> Dict[str, Any]:
    """Load the JSON config that selects onnx weights / image sizes / etc.

    The config schema is::

        {
          "detector": {
            "outside": {"model": "artifacts/models/hive_entrance.onnx",
                        "imgsz": 1280, "conf": 0.25, "iou": 0.45,
                        "scene_hint": "outside"},
            "inside":  {"model": "artifacts/models/honey_bee_detector.onnx",
                        "imgsz": 640,  "conf": 0.25, "iou": 0.45,
                        "scene_hint": "inside"}
          },
          "labels": {"0": "bee"}
        }
    """
    prog = Path(sys.argv[0]).resolve()
    if explicit:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise FileNotFoundError(404, f"config not found: {path}")
    else:
        hits = _default_search_paths(prog)
        if not hits:
            raise FileNotFoundError(404, (
                f"missing {DEFAULT_CONFIG_BASENAME}; expected next to "
                f"{prog} or under ./configs/"))
        path = hits[0]
    with path.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["_config_path"] = str(path)
    cfg["_config_dir"] = str(path.parent)
    return cfg


# ---------------------------------------------------------------------------
# Scene auto-detection (filepath heuristics → image saturation fallback)
# ---------------------------------------------------------------------------

_IMAGE_PATH_HINTS_OUTSIDE = ("outside", "entrance", "vnbee", "巢外", "门口",
                            "a-5-1", "a-5-2", "a-5-3", "a-5-4", "visible")
_IMAGE_PATH_HINTS_INSIDE = ("inside", "ir", "infrared", "巢内", "红外",
                           "b-5-1", "b-5-2", "b-5-3", "b-5-4")


def infer_scene_from_path(path: str) -> Optional[str]:
    text = Path(path).name.lower() + "/" + Path(path).parent.name.lower()
    if any(token in text for token in _IMAGE_PATH_HINTS_OUTSIDE):
        return "outside"
    if any(token in text for token in _IMAGE_PATH_HINTS_INSIDE):
        return "inside"
    return None


def infer_scene_from_image(bgr_image: np.ndarray) -> str:
    """Hive-entrance images are colourful; inside-IR are nearly grey."""
    try:
        import cv2  # type: ignore
    except ImportError:
        return "outside"  # best-effort fallback when cv2 missing
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    sat_mean = float(hsv[:, :, 1].mean())
    return "inside_ir" if sat_mean < 18.0 else "outside_entrance"


# ---------------------------------------------------------------------------
# Image pre- + post-processing (YOLOv8 detection ONNX layout)
# ---------------------------------------------------------------------------

def letterbox(
    image: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, Tuple[float, float]]:
    h, w = image.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = int(round(w * r)), int(round(h * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    try:
        import cv2  # type: ignore
        if (h, w) != new_unpad[::-1]:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=color)
    except ImportError:  # pragma: no cover
        pass
    return image, (r, (dw, dh))


def scale_boxes(
    xywh: np.ndarray,
    ratio_pad: Tuple[float, Tuple[float, float]],
    orig_shape: Tuple[int, int],
) -> np.ndarray:
    r, (dw, dh) = ratio_pad
    boxes = xywh.copy()
    # Convert xywH to xyxy before scaling.
    half_w = boxes[:, 2] / 2.0
    half_h = boxes[:, 3] / 2.0
    x1 = (boxes[:, 0] - half_w - dw) / r
    y1 = (boxes[:, 1] - half_h - dh) / r
    x2 = (boxes[:, 0] + half_w - dw) / r
    y2 = (boxes[:, 1] + half_h - dh) / r
    # Clip to original image bounds.
    oh, ow = orig_shape[:2] if len(orig_shape) > 2 else orig_shape
    x1 = np.clip(x1, 0, ow)
    x2 = np.clip(x2, 0, ow)
    y1 = np.clip(y1, 0, oh)
    y2 = np.clip(y2, 0, oh)
    # Back to xywh.
    boxes[:, 0] = x1
    boxes[:, 1] = y1
    boxes[:, 2] = x2 - x1
    boxes[:, 3] = y2 - y1
    return boxes


def multiclass_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
    conf_threshold: float = 0.25,
    topk: int = 300,
) -> np.ndarray:
    """Simple numpy NMS (single-class detector → only one label)."""
    mask = ((scores >= conf_threshold) & np.isfinite(scores)
            & np.isfinite(boxes).all(axis=1) & (boxes[:, 2:] > 0).all(axis=1))
    boxes = boxes[mask]
    scores = scores[mask]
    if boxes.size == 0:
        return np.empty((0, 5), dtype=np.float32)
    order = np.argsort(-scores, kind="stable")
    boxes = boxes[order]
    scores = scores[order]
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    keep: List[int] = []
    # order here is indices inside the already-filtered (boxes, scores).
    order = np.arange(boxes.shape[0], dtype=np.int64)
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1 or len(keep) >= topk:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        ww = np.maximum(0.0, xx2 - xx1)
        hh = np.maximum(0.0, yy2 - yy1)
        inter = ww * hh
        ovr = inter / (areas[i] + areas[rest] - inter + 1e-12)
        inds = np.where(ovr <= iou_threshold)[0]
        order = rest[inds]
    kept_boxes = boxes[keep]
    kept_scores = scores[keep]
    return np.concatenate(
        [kept_boxes, kept_scores[:, np.newaxis].astype(np.float32)],
        axis=1,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# ONNX session (singleton, log_severity_level=3 required by spec)
# ---------------------------------------------------------------------------

_SESSIONS: Dict[str, Tuple[Any, int, int]] = {}  # key → (session, imgsz, outputs)


def _get_session(model_path: Path, imgsz: int, config_dir: str) -> Tuple[Any, int]:
    """Build/retrieve an ONNX Runtime session with INFO+WARNING muted."""
    resolved = (Path(config_dir) / model_path).resolve()
    if resolved.suffix.lower() != ".onnx":
        raise ValueError("inference requires an ONNX model")
    key = f"{resolved}|{imgsz}"
    if key in _SESSIONS:
        sess, sz, _ = _SESSIONS[key]
        return sess, sz
    import onnxruntime as ort
    opts = ort.SessionOptions()
    # Mandatory per evaluation spec §3.3-4: only ERROR-level logs on stdout.
    opts.log_severity_level = 3
    opts.log_verbosity_level = 0
    # Use CUDA if the driver exposes nvcuda.dll; fall back to CPU otherwise.
    providers: List[str] = []
    try:
        available = ort.get_available_providers()
    except Exception:
        available = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    if not model_path.is_absolute():
        resolved = (Path(config_dir) / str(model_path)).resolve()
    else:
        resolved = model_path
    if not resolved.is_file():
        raise FileNotFoundError(404, f"onnx model not found: {resolved}")
    sess = ort.InferenceSession(str(resolved), sess_options=opts,
                                 providers=providers)
    _SESSIONS[key] = (sess, imgsz, len(sess.get_outputs()))
    return sess, imgsz


def run_detection(image_path: Path, config: Dict[str, Any], *,
                  conf_override: Optional[float] = None,
                  topk: int = 300) \
        -> Tuple[List[Dict[str, Any]], int]:
    """Return (detections_list, processing_time_ms)."""
    import cv2  # type: ignore

    # Scene selection: path hint first, image saturation fallback.
    scene_key = config.get("_scene") or infer_scene_from_path(str(image_path))
    if scene_key is None:
        raw = cv2.imdecode(
            np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if raw is None:
            raise OSError(501, f"cannot decode image: {image_path}")
        scene_key = ("outside" if "outside" in infer_scene_from_image(raw)
                     else "inside")
    else:
        raw = cv2.imdecode(
            np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if raw is None:
            raise OSError(501, f"cannot decode image: {image_path}")

    det_cfg = (config["detector"].get("outside")
               if scene_key.startswith("outside") else
               config["detector"].get("inside"))
    if not det_cfg:
        raise RuntimeError(f"config missing detector.{scene_key}")

    model_rel = Path(det_cfg["model"])
    imgsz = int(det_cfg.get("imgsz", 640))
    conf_thr = (float(conf_override) if conf_override is not None else
                float(det_cfg.get("conf", 0.25)))
    iou_thr = float(det_cfg.get("iou", 0.45))
    config_dir = config["_config_dir"]

    t0 = time.perf_counter()
    sess, _ = _get_session(model_rel, imgsz, config_dir)
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape[-2:]  # (H, W)
    sz = (int(input_shape[0]), int(input_shape[1]))

    letterboxed, ratio_pad = letterbox(raw, sz)
    blob = cv2.dnn.blobFromImage(
        letterboxed, 1 / 255.0, sz[::-1], (0, 0, 0), swapRB=True, crop=False)
    pred = sess.run(None, {input_name: blob})[0]

    # ultralytics YOLOv8 detect .onnx → shape [1, 4+num_classes, 8400]
    # We transpose → [8400, 4+cls], take cx/cy/w/h + max-class probability.
    if pred.ndim == 3:
        pred = pred[0]
    pred = pred.T
    if pred.ndim != 2 or pred.shape[1] != 5:
        raise RuntimeError(f"expected single-class YOLO detection output, got {pred.shape}; pose models require a separate decoder")
    cx_cy_w_h = pred[:, :4]
    cls_scores = pred[:, 4:].max(axis=1)
    # NMS expects left/top/width/height, whereas YOLO emits center coordinates.
    scaled_boxes = scale_boxes(cx_cy_w_h, ratio_pad, raw.shape[:2])
    keep = multiclass_nms(
        scaled_boxes, cls_scores, iou_thr, conf_thr, topk=topk)
    keep_boxes = keep[:, :4]
    raw_scores = keep[:, 4].copy() if keep.size else np.empty((0,), dtype=np.float32)
    boxes = keep_boxes

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    label_map = config.get("labels", {"0": "bee"})
    out: List[Dict[str, Any]] = []
    # scale_boxes keeps xywh rows; attach the matched score column from NMS.
    for row, conf in zip(boxes, raw_scores):
        x, y, w, h = [float(v) for v in row[:4]]
        if round(w, 2) <= 0 or round(h, 2) <= 0:
            continue
        cid = 0
        out.append({
            "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
            "label": label_map.get(str(cid), "bee"),
            "class_id": cid,
            "confidence": round(float(conf), 6),
        })
    return out, elapsed_ms


# ---------------------------------------------------------------------------
# Tracking-mode helper (per-sequence file cache)
# ---------------------------------------------------------------------------

_CACHE_VERSION = 1


def _track_cache_path(sequence_id: str) -> Path:
    h = hashlib.sha1(sequence_id.encode("utf-8")).hexdigest()[:12]
    base = Path(os.environ.get("TEMP", "/tmp")) / "bee_algo_track_cache"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{sequence_id}_{h}.json"


def _load_tracks(sequence_id: str) -> Dict[str, Any]:
    path = _track_cache_path(sequence_id)
    if not path.is_file():
        return {"version": _CACHE_VERSION, "next_id": 1, "frames": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": _CACHE_VERSION, "next_id": 1, "frames": {}}


def _save_tracks(sequence_id: str, store: Dict[str, Any]) -> None:
    _track_cache_path(sequence_id).write_text(
        json.dumps(store, ensure_ascii=False), encoding="utf-8")


def match_ious(current_boxes: List[List[float]],
               last_boxes: List[List[float]]) -> Dict[int, int]:
    """Greedy IoU matcher used as a tiny tracking baseline in CLI mode."""
    def _iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
        ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
        inter = iw * ih
        union = max(aw, 0) * max(ah, 0) + max(bw, 0) * max(bh, 0) - inter
        return inter / union if union > 0 else 0.0
    matches: Dict[int, int] = {}
    used = set()
    pairs = sorted(
        ((_iou(cb, lb), ci, li)
         for ci, cb in enumerate(current_boxes)
         for li, lb in enumerate(last_boxes)),
        reverse=True,
    )
    for score, ci, li in pairs:
        if score < 0.2 or ci in matches or li in used:
            continue
        matches[ci] = li
        used.add(li)
    return matches


def run_tracking(image_path: Path, frame_id: int, sequence_id: str,
                 config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    detections, ms = run_detection(image_path, config)
    store = _load_tracks(sequence_id)
    last_frame = max((int(k) for k in store["frames"].keys()), default=None)
    last_boxes: List[List[float]] = []
    last_ids: List[int] = []
    if last_frame is not None:
        for entry in store["frames"][str(last_frame)]:
            last_boxes.append(entry["bbox"])
            last_ids.append(entry["track_id"])
    cur_boxes = [d["bbox"] for d in detections]
    matches = match_ious(cur_boxes, last_boxes)
    next_id = int(store.get("next_id", 1))
    frame_rows: List[Dict[str, Any]] = []
    for ci, det in enumerate(detections):
        if ci in matches:
            tid = last_ids[matches[ci]]
        else:
            tid = next_id
            next_id += 1
        row = {"track_id": tid, **det}
        frame_rows.append(row)
    store["next_id"] = next_id
    store["frames"][str(frame_id)] = frame_rows
    _save_tracks(sequence_id, store)
    return frame_rows, ms


# ---------------------------------------------------------------------------
# Stdout + exit code compliance
# ---------------------------------------------------------------------------

def _emit(payload: Dict[str, Any], *, code: int) -> int:
    # Print EXACTLY one line of JSON (no trailing whitespace before newline).
    try:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False,
                                     separators=(", ", ": ")) + "\n")
        sys.stdout.flush()
    except Exception:
        # Last resort — avoid Windows error dialogs at all cost.
        sys.stdout.write("{\"error\":\"json_encode_failed\"}\n")
        sys.stdout.flush()
    return code


# ---------------------------------------------------------------------------
# argparse entrypoint
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="algorithm", description=__doc__)
    p.add_argument("--image_path", type=str, required=True,
                   help="Absolute or relative path to an input image file. "
                        "Accepts Chinese characters and spaces.")
    p.add_argument("--task", choices=("detect", "track"), default="detect",
                   help="Evaluation mode (default: detect — one-shot box "
                        "output; track requires --frame_id and --sequence_id).")
    p.add_argument("--frame_id", type=int, default=1,
                   help="Required when --task=track; 1-based frame index.")
    p.add_argument("--sequence_id", type=str, default="seq001",
                   help="Required when --task=track; used as identity cache "
                        "bucket.")
    p.add_argument("--image_id", type=str, default=None,
                   help="Overrides the JSON image_id field (default: input "
                        "file stem).")
    p.add_argument("--config", type=str, default=None,
                   help="Path to algorithm_config.json (auto-discovered next "
                        "to the executable when omitted).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        # argparse calls sys.exit() on --help / errors; swallow it to avoid
        # Windows popups and emit a compliant JSON instead.
        msg = {"code": 0, "error": "invalid_arguments",
               "message": "see algorithm --help"}
        return _emit(msg, code=0)

    # --- Image existence check (return code=0 for missing/invalid) ----------
    try:
        image_path = Path(args.image_path)
    except Exception as exc:
        return _emit({"code": 0, "error": "invalid_path",
                       "message": str(exc)}, code=0)
    if not image_path.is_file():
        return _emit({
            "code": 0,
            "error": "file_not_found",
            "message": f"--image_path does not exist: {image_path}",
        }, code=0)
    if image_path.stat().st_size == 0:
        return _emit({
            "code": 0, "error": "file_empty",
            "message": f"file is 0 bytes: {image_path}",
        }, code=0)

    # --- Runtime config ----------------------------------------------------
    try:
        config = load_runtime_config(args.config)
    except FileNotFoundError as exc:
        errno, msg = exc.args if len(exc.args) == 2 else (501, str(exc))
        return _emit({"code": 0, "error": "missing_config",
                       "message": str(msg)}, code=0)
    except json.JSONDecodeError as exc:
        return _emit({"code": 0, "error": "bad_config_json",
                       "message": str(exc)}, code=0)

    # --- Inference ---------------------------------------------------------
    try:
        if args.task == "track":
            tracks, elapsed_ms = run_tracking(
                image_path, args.frame_id, args.sequence_id, config)
            payload: Dict[str, Any] = {
                "code": 1,
                "frame_id": args.frame_id,
                "sequence_id": args.sequence_id,
                "tracks": tracks,
                "processing_time_ms": elapsed_ms,
            }
            return _emit(payload, code=1)
        else:
            detections, elapsed_ms = run_detection(image_path, config)
            image_id = args.image_id or image_path.stem
            payload = {
                "code": 1,
                "image_id": image_id,
                "detections": detections,
                "processing_time_ms": elapsed_ms,
            }
            return _emit(payload, code=1)
    except MemoryError:
        return _emit({
            "code": 0, "error": "out_of_memory",
            "message": "OOM during inference (peak VRAM must stay <= 16GB).",
        }, code=0)
    except Exception as exc:  # pragma: no cover - defensive catch-all
        return _emit({
            "code": 0,
            "error": "inference_failure",
            "message": f"{type(exc).__name__}: {exc}",
        }, code=0)


if __name__ == "__main__":
    raise SystemExit(main())
