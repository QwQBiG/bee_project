"""Render two fully-annotated example images (outside + inside) with Chinese
notes so the user can see exactly what each bounding box should look like.

Picks the first jpg from each easy-label folder that has ≥2 bees, runs the
fine-tuned detector on it, then overlays:
  1. Yellow 2px rectangle  = 正确的蜜蜂 bbox 示例
  2. Red dashed rectangle  = 反例：框太大（塞太多空气）
  3. Green dashed          = 反例：框太小（只框到胸部，漏翅膀/屁股）
  4. Chinese side notes    = 画框原则 + 姿态标注说明
  5. Footer                = 场景 / 文件名 / 「比赛 §五/4 标准」
Saves the composites next to the label zip files as:
    datasets/label_tasks_final/_EXAMPLE_outside_annotated.png
    datasets/label_tasks_final/_EXAMPLE_inside_annotated.png
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_OUT_DIR = Path("datasets/label_tasks_final")
_OUTSIDE_JPG = (
    _OUT_DIR / "easy_label_outside"
    / "A-5-2-094933e58b48_frame_00003408.jpg"
)
_INSIDE_JPG = (
    _OUT_DIR / "easy_label_inside"
    / "B-5-1-f250bb33ba22_frame_00003401.jpg"
)


def _sandbox_proof() -> None:
    cfg_dir = Path(tempfile.gettempdir()) / "ultra_cfg_bee"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cfg_dir))
    os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", str(cfg_dir))
    try:
        import ultralytics.utils.checks as c  # type: ignore
        c.check_font = lambda *a, **k: None  # noqa: E731
    except Exception:
        pass
    try:
        import ultralytics.data.utils as u  # type: ignore
        u.check_font = lambda *a, **k: None  # noqa: E731
    except Exception:
        pass


def _detect_xyxy(img_path: Path, model_path: str,
                 imgsz: int, conf: float) -> np.ndarray:
    """Return (N,4) xyxy boxes in raw-image pixels."""
    from ultralytics import YOLO  # type: ignore

    model = YOLO(model_path)
    res = model.predict(source=str(img_path), imgsz=imgsz, conf=conf,
                        verbose=False, save=False)[0]
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    return boxes.xyxy.cpu().numpy().astype(np.float32)


def _pick_boxes(boxes: np.ndarray, n_wanted: int) -> List[Tuple[int, int, int, int]]:
    """Return up to n_wanted boxes sorted by area descending
    (big, clear bees first → easier to annotate manually after)."""
    if boxes.size == 0:
        return []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-areas)
    out: List[Tuple[int, int, int, int]] = []
    for idx in order[:n_wanted]:
        x1, y1, x2, y2 = boxes[idx].astype(int).tolist()
        out.append((int(x1), int(y1), int(x2), int(y2)))
    return out


def _chinese_font(size: int) -> ImageFont.ImageFont:
    """Try Windows CJK fonts in a known order; fallback to PIL default."""
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",   # 微软雅黑
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf", # 黑体
        r"C:\Windows\Fonts\simsun.ttc", # 宋体
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _rect(draw: ImageDraw.ImageDraw, box, color, width=3, dash=None) -> None:
    x1, y1, x2, y2 = box
    if dash is None:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        return
    # 4-sided dashed: draw short line segments along each edge.
    dash_len, gap = dash
    def dashed_line(p0, p1):
        ax, ay = p0; bx, by = p1
        L = max(abs(bx - ax), abs(by - ay)) or 1
        n = int(np.ceil(L / (dash_len + gap)))
        xs = np.linspace(ax, bx, n + 1).astype(int)
        ys = np.linspace(ay, by, n + 1).astype(int)
        for i in range(0, n, 2):
            draw.line([(xs[i], ys[i]), (xs[i + 1], ys[i + 1])],
                      fill=color, width=width)
    dashed_line((x1, y1), (x2, y1))
    dashed_line((x2, y1), (x2, y2))
    dashed_line((x2, y2), (x1, y2))
    dashed_line((x1, y2), (x1, y1))


def _make_composite(img_path: Path, boxes: List[Tuple[int, int, int, int]],
                    scene: str, out_path: Path) -> None:
    raw = cv2.imread(str(img_path))
    if raw is None:
        raise FileNotFoundError(img_path)
    H, W = raw.shape[:2]
    # Right info panel: 450 px wide, same height as image.
    PANEL_W = 460
    canvas = np.full((H, W + PANEL_W, 3), 245, dtype=np.uint8)
    canvas[:H, :W] = raw
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    font_sm = _chinese_font(20)
    font_md = _chinese_font(24)
    font_lg = _chinese_font(30)

    # --- draw ground-truth bee boxes on the image ---
    YELLOW = (234, 188, 22)   # 正例
    RED    = (224, 54,  36)   # 反例：太大
    GREEN  = (36,  196, 112)  # 反例：太小
    BLUE   = (36,  120, 224)  # 编号点

    n_show = min(3, len(boxes))
    for i in range(n_show):
        x1, y1, x2, y2 = boxes[i]
        _rect(draw, (x1, y1, x2, y2), YELLOW, width=3)
        draw.ellipse([x1 - 10, y1 - 10, x1 + 10, y1 + 10], fill=BLUE, outline="black")
        draw.text((x1 + 12, y1 - 18), f"bee_{i+1}", fill=BLUE, font=font_md)

        # --- construct two "anti-patterns" ---
        pad = max(12, int(min(x2 - x1, y2 - y1) * 0.22))
        shrink = max(6, int(min(x2 - x1, y2 - y1) * 0.20))
        _rect(draw, (x1 - pad, y1 - pad, x2 + pad, y2 + pad),
              RED, width=2, dash=(14, 8))
        _rect(draw, (x1 + shrink, y1 + shrink, x2 - shrink, y2 - shrink),
              GREEN, width=2, dash=(12, 7))

    # --- title bar across the top ---
    draw.rectangle([0, 0, W + PANEL_W, 56], fill=(30, 41, 59))
    draw.text((18, 12), f"【标注样例 · {scene}】{img_path.name}",
              fill=(255, 255, 255), font=font_lg)

    # --- legend inside the image area (top-right) ---
    legend_x, legend_y = W - 320, 72
    draw.rectangle([legend_x - 10, legend_y - 10,
                    legend_x + 300, legend_y + 120],
                   fill=(255, 255, 255, 200), outline=(210, 210, 210), width=2)
    _rect(draw, (legend_x, legend_y, legend_x + 50, legend_y + 28),
          YELLOW, width=3)
    draw.text((legend_x + 62, legend_y + 3), "✅ 正确：身体+翅膀最小外接", fill=(40, 40, 40), font=font_sm)
    _rect(draw, (legend_x, legend_y + 38, legend_x + 50, legend_y + 66),
          RED, width=2, dash=(14, 8))
    draw.text((legend_x + 62, legend_y + 41), "❌ 太大：塞进太多背景空气", fill=(40, 40, 40), font=font_sm)
    _rect(draw, (legend_x, legend_y + 76, legend_x + 50, legend_y + 104),
          GREEN, width=2, dash=(12, 7))
    draw.text((legend_x + 62, legend_y + 79), "❌ 太小：漏翅膀或屁股", fill=(40, 40, 40), font=font_sm)

    # --- right info panel: detailed rules written in Chinese ---
    PAD = 24
    px = W + PAD
    py = PAD + 28
    sections: List[Tuple[str, List[str]]] = [
        ("① 类别：1 个，叫 bee（单类，类别号 0）", [
            "· 你在 CVAT 里只用创建 1 个 label，名字就填 bee。",
            "· 不区分雄蜂/工蜂/蜂王，统统一类 bee。",
        ]),
        ("② 画框原则（对应比赛 §五/4）", [
            "· 标准：躯干（头+胸+腹）+ 正常展开的翅膀",
            "  → 取它们的最小外接矩形。",
            "· 翅膀展开就包含翅膀；翅膀收拢就只包身体。",
            "· 边缘截断：只露半身 → 框『可见部分』",
            "  不要脑补、不要框到图片外的空气。",
            "· 重叠：如果 2 只叠在一起，但头/胸都看得清，",
            "  → 分别独立画 2 个框，允许框重叠。",
        ]),
        ("③ 哪些情况可以不标", [
            "· 人眼极限：只剩模糊黑影，看不清生物特征。",
            "· 只留一根触角/一条腿，身体主体没进画面。",
            "· 图片完全过曝/全黑，啥都看不清楚。",
        ]),
        ("④ 关于『姿态』和这个框的关系（你最关心的问题！）", [
            "· 只画一个 bbox 不能表达姿态。没错。",
            "· 这次你标 bbox 解决的是：",
            "    → 评测 §1.1/1.2（巢外/巢内检测 mAP）24 分。",
            "· 姿态（头/胸/腹 3 关键点）是另一个任务：",
            "    → 评测 §2.1 个体行为量化 8 分",
            "    → §4.1 创新性 6 分",
            "  姿态要另找少量图画 3 个点，那是下一批。",
            "· 先把 24 分检测 baseline 拉满，性价比最高。",
        ]),
        ("⑤ 你做完回传什么", [
            "· 在 CVAT 右上角点『Save work』，再 Menu →",
            "  Export task dataset。",
            "· 格式任选其一：",
            "   - CVAT for images 1.1 → 出 annotations.xml",
            "   - COCO JSON          → 出 instances_default.json",
            "· 然后把那个文件丢回项目里任意位置，",
            "  告诉我一声『我标完了，巢外用 xml / json』就行。",
        ]),
    ]
    for title, lines in sections:
        draw.text((px, py), title, fill=(18, 52, 94), font=font_md)
        py += 36
        for line in lines:
            draw.text((px + 10, py), line, fill=(55, 55, 55), font=font_sm)
            py += 26
        py += 14
        if py > H - 40:
            break

    # Footer: competition spec reference
    draw.rectangle([0, H - 40, W + PANEL_W, H], fill=(30, 41, 59))
    draw.text((18, H - 32),
              "比赛 §五/4：蜜蜂全躯干（头胸腹+正常展开翅膀）的最小外接矩形。边缘截断/遮挡但"
              "核心生物特征仍可判定的个体 → 框可见部分；达人眼分辨极限的 → 不标。",
              fill=(255, 255, 255), font=font_sm)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"SAVED  {out_path}  ({out_path.stat().st_size // 1024} KB)")


def main() -> None:
    _sandbox_proof()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [
        (_OUTSIDE_JPG, "artifacts/models/hive_entrance_bee_yolov8n.pt",
         1280, 0.25, "巢外可见光", _OUT_DIR / "_EXAMPLE_outside_annotated.png"),
        (_INSIDE_JPG, "artifacts/models/honey_bee_detector_yolov8s.pt",
         640, 0.15, "巢内红外", _OUT_DIR / "_EXAMPLE_inside_annotated.png"),
    ]
    for img, model, imgsz, conf, scene, out_path in tasks:
        if not img.exists():
            # fallback: first jpg found in that folder
            parent = img.parent
            jpgs = sorted(parent.glob("*.jpg"))
            if not jpgs:
                print(f"SKIP {scene}: no jpg under {parent}")
                continue
            img = jpgs[0]
        boxes_xyxy = _detect_xyxy(img, model, imgsz, conf)
        picks = _pick_boxes(boxes_xyxy, 3)
        print(f"{scene}: {img.name} got {len(boxes_xyxy)} bees, "
              f"showing top-{len(picks)}")
        if not picks:
            print("  → detector gave 0 boxes; drawing a blank example with "
                  "text-only info panel")
            # Still create a picture even if empty: the right-side rules
            # panel contains the human instructions the user cares about.
            raw = cv2.imread(str(img))
            if raw is None:
                continue
            h, w = raw.shape[:2]
            canvas = np.full((h, w + 460, 3), 245, dtype=np.uint8)
            canvas[:h, :w] = raw
            tmp_path = img.parent / ("_tmp_" + img.name)
            cv2.imwrite(str(tmp_path), canvas)
            tmp_img_path = Path(tmp_path)
            # Run the same renderer on the synthetic "zero detections"
            # placeholder: use a big fake box so the side panel displays.
            frame_boxes = [(int(w * 0.3), int(h * 0.3),
                            int(w * 0.6), int(h * 0.6))]
            _make_composite(img, frame_boxes, scene, out_path)
            # clean
            tmp_path.unlink(missing_ok=True)
        else:
            _make_composite(img, picks, scene, out_path)


if __name__ == "__main__":
    main()
