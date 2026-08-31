"""Pick N 'easy to label' bee images for the outside and inside scenes.

Strategy
--------
Run the teacher detector (the same .pt used to bootstrap pseudo-labels)
across the candidate pool, then sort images by the number of detected
bees. We select frames whose bee count sits in a user-specified window
(default 1 to 6 bees per frame):

* zero-bee frames are skipped (nothing to label, a background-only
  sample has very little value for hand-labeling)
* crowded frames (> ``max_bees``) are skipped because the user asked
  for images with "not a lot of content per picture"
* the survivors are de-duplicated by source video id (so we do not
  return 50 frames that all came from a single short clip) and then
  copied into a flat output folder ready for CVAT upload.

The script also emits:
* ``index.csv``  – one row per picked image: image,bees,video
* ``README.txt`` – concise labeling rules translated from §五/4 of the
  competition specification, so the user can label directly without
  re-reading the entire competition document.

Usage::

    py -3.13 tools/pick_easy_label_images.py
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

_OUTSIDE_ROOT = (
    "datasets/official_work/diverse_60/round2_300_per_scene/"
    "cvat_upload/outside_300"
)
_INSIDE_ROOT = (
    "datasets/official_work/diverse_60/round2_300_per_scene/"
    "cvat_upload/inside_300"
)

_README_TEMPLATE = """蜜蜂人工标注简易规则（对应比赛文档 §五/4）
==============================================================
一、标什么
  * 只标 1 个类别：bee（单类，类别编号 0）
  * 凡是"核心生物特征（头/胸/核心轮廓）仍能判定"的个体都要标。
    包含：
      - 完整显示的蜜蜂（头+胸+腹/翅膀正常展开）→ 标最小外接矩形
      - 边缘截断只露半身，但头或胸还看得见 → 标可见部分的矩形
      - 遮挡重叠但头部/胸部明显的个体 → 分别独立标框（允许重叠）
  * 完全看不见、只剩模糊黑影、人眼已无法判断 → 不标。

二、怎么画框（CVAT 矩形工具）
  1. 从左上角按住鼠标拉到右下角，矩形包围"身体躯干"。
  2. 框边不要贴到像素外，也不要留太多空气（背景别塞太多进框）；
     翅膀展开就包含翅膀，翅膀收拢就只包身体。
  3. 每只蜜蜂一个独立框，不要把 2 只蜜蜂画在一个大框里。
  4. 边缘截断：框包含"可见部分"即可，不要脑补框出去。

三、多人多轮校验（你可拉同学交叉标）
  * 如果同一张图你标了第 1 轮，隔 2 天再标第 2 轮，前后差异
    大于 1 只蜜蜂的图，建议在 CVAT 里手动 review。
  * 确实"人眼都看不清"的区域，直接不标，不要硬画，会害模型。

四、导出 / 回传
  你标完后在 CVAT 导出，二选一，随便一种都行：
    1) CVAT for images 1.1  → 得到 annotations.xml
    2) COCO JSON            → 得到 instances_default.json 或 similar
  然后把导出文件 + 你用的是哪种格式告诉我即可，我这边自动转 YOLO。
"""


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


def _video_id(img: Path) -> str:
    """Extract A-5-1 / B-5-2 etc. from the competition file-name scheme.

    Example: ``A-5-1-3922c3762a68_frame_00000012.jpg`` → ``A-5-1``
    """
    stem = img.stem
    parts = stem.split("_")
    if parts and parts[0]:
        seg = parts[0]  # A-5-1-3922c3762a68
    else:
        return stem[:8]
    # A-5-1-xxx → first 3 dash-delimited tokens (len=4 before hash)
    toks = seg.split("-")
    if len(toks) >= 3:
        return "-".join(toks[:3])
    return seg[:6]


def list_images(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in _IMG_EXTS)


def count_boxes(
    images: List[Path],
    model_path: str,
    imgsz: int,
    conf: float,
    iou: float,
) -> List[Tuple[Path, int]]:
    """Infer once per image; return (image, number of detected bees)."""
    from ultralytics import YOLO  # type: ignore

    model = YOLO(model_path)
    out: List[Tuple[Path, int]] = []
    for img in images:
        res = model.predict(
            source=str(img), imgsz=imgsz, conf=conf, iou=iou,
            verbose=False, save=False, save_txt=False,
        )[0]
        boxes = getattr(res, "boxes", None)
        n = 0 if boxes is None else int(len(boxes))
        out.append((img, n))
    return out


def select_easy(
    scored: List[Tuple[Path, int]],
    *,
    n_wanted: int,
    min_bees: int,
    max_bees: int,
    per_video_cap: int,
) -> List[Tuple[Path, int]]:
    """Diversified pick: max ``per_video_cap`` images per source video id.

    We first keep only frames in the [min_bees, max_bees] bee-count
    window, then sort by bee count ascending (so the cleanest, 1-2 bee
    frames come first), and finally round-robin sample across videos so
    the 50 picked frames cover all four A-5-X / B-5-X clips evenly.
    """
    eligible = [(p, n) for p, n in scored if min_bees <= n <= max_bees]
    eligible.sort(key=lambda t: t[1])  # fewest bees first

    by_video: Dict[str, List[Tuple[Path, int]]] = defaultdict(list)
    for p, n in eligible:
        by_video[_video_id(p)].append((p, n))

    picks: List[Tuple[Path, int]] = []
    counts: Dict[str, int] = defaultdict(int)
    # Round-robin: iterate video buckets, pop the least-crowded head,
    # stop if all buckets are empty or we hit n_wanted.
    while len(picks) < n_wanted and any(by_video.values()):
        progress = False
        for vid in sorted(by_video.keys()):
            if len(picks) >= n_wanted:
                break
            q = by_video[vid]
            if not q:
                continue
            if counts[vid] >= per_video_cap:
                continue
            item = q.pop(0)
            picks.append(item)
            counts[vid] += 1
            progress = True
        if not progress:
            # Caps are blocking. Ignore caps on remaining slots — we
            # prefer anything eligible to handing back fewer images.
            for vid in sorted(by_video.keys()):
                while by_video[vid] and len(picks) < n_wanted:
                    picks.append(by_video[vid].pop(0))
    return picks


def _copy_and_pack(
    scene: str,
    picks: List[Tuple[Path, int]],
    out_dir: Path,
) -> Tuple[Path, Path]:
    """Copy picks to out_dir flatly, write index.csv and readme, zip it."""
    scene_dir = out_dir / f"easy_label_{scene}"
    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    scene_dir.mkdir(parents=True)
    for path, n in picks:
        dst = scene_dir / path.name
        # Flat copy; collisions between sub-folders are impossible under
        # the competition naming scheme (each frame has a unique hash).
        if dst.exists():
            dst = scene_dir / f"{path.parent.name}_{path.name}"
        shutil.copy2(path, dst)
    (scene_dir / "README.txt").write_text(
        _README_TEMPLATE + f"\n本批次：{scene.upper()}，共 {len(picks)} 张\n",
        encoding="utf-8",
    )
    with (scene_dir / "index.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "detected_bees", "video_id", "absolute_path"])
        for p, n in picks:
            w.writerow([p.name, n, _video_id(p), str(p.resolve())])
    # Zip the scene folder (next to it, no nested junk).
    zip_path = out_dir / f"easy_label_{scene}"
    if zip_path.with_suffix(".zip").exists():
        zip_path.with_suffix(".zip").unlink()
    shutil.make_archive(str(zip_path), "zip", root_dir=str(out_dir),
                        base_dir=f"easy_label_{scene}")
    return scene_dir, zip_path.with_suffix(".zip")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pick easy bee-label images")
    ap.add_argument("--outside_root", default=_OUTSIDE_ROOT)
    ap.add_argument("--inside_root", default=_INSIDE_ROOT)
    ap.add_argument("--outside_model",
                    default="artifacts/models/hive_entrance_bee_yolov8n.pt")
    ap.add_argument("--inside_model",
                    default="artifacts/models/honey_bee_detector_yolov8s.pt")
    ap.add_argument("--out", default="datasets/label_tasks")
    ap.add_argument("--n", type=int, default=50,
                    help="images wanted per scene")
    ap.add_argument("--min_bees", type=int, default=1)
    ap.add_argument("--max_bees", type=int, default=6,
                    help="upper bound on bees per frame")
    ap.add_argument("--per_video_cap", type=int, default=20,
                    help="max images taken from any single A-5-X / B-5-X clip")
    args = ap.parse_args()

    _sandbox_proof()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out.resolve()}")

    total_summary: Dict[str, dict] = {}
    for scene, root_str, model, imgsz, conf in (
        ("outside", args.outside_root, args.outside_model, 1280, 0.25),
        ("inside",  args.inside_root,  args.inside_model,  640,  0.15),
    ):
        root = Path(root_str)
        imgs = list_images(root)
        print(f"\n[{scene}] scanning {len(imgs)} images with {model} "
              f"(imgsz={imgsz}, conf={conf}) ...")
        scored = count_boxes(imgs, model, imgsz, conf, 0.45)
        eligible = sum(1 for _, n in scored
                       if args.min_bees <= n <= args.max_bees)
        zero = sum(1 for _, n in scored if n == 0)
        crowded = sum(1 for _, n in scored if n > args.max_bees)
        print(f"  eligible={eligible}  zero-bee={zero}  crowded(>{args.max_bees})={crowded}")
        picks = select_easy(
            scored,
            n_wanted=args.n,
            min_bees=args.min_bees,
            max_bees=args.max_bees,
            per_video_cap=args.per_video_cap,
        )
        scene_dir, zip_path = _copy_and_pack(scene, picks, out)
        print(f"  picked {len(picks)} -> {scene_dir}")
        print(f"  zip: {zip_path}  ({zip_path.stat().st_size / 1024:.0f} KB)")
        total_summary[scene] = {
            "zip": str(zip_path),
            "folder": str(scene_dir),
            "picked": len(picks),
            "avg_bees": round(sum(n for _, n in picks) / max(1, len(picks)), 2),
            "eligible": eligible,
        }
    print("\n=== FINAL SUMMARY ===")
    for s, d in total_summary.items():
        print(f"  {s}: {d}")


if __name__ == "__main__":
    main()
