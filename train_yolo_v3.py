"""
train_yolo_v3.py
================
Retrain dragon-fruit lesion detection with YOLOv8s.

The current weights are YOLOv8n (width 0.25). This run:
  - starts from COCO-pretrained YOLOv8s
  - oversamples train images that contain the rare classes
    (Anthracnose, Stem_Canker, Soft_Rot)
  - keeps the original validation split unchanged
  - replaces models/yolo_dragon_best.pt only when val mAP50 improves
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC_ROOT = Path(r"C:\Users\prady\OneDrive\Desktop\Dragon_fruit_dataset\dataset\yolo_dragon_lesions")
OUT_ROOT = BASE_DIR / "dataset" / "yolo_v3"
DATA_YAML = BASE_DIR / "data_dragon_lesions_v3.yaml"
MODEL_DIR = BASE_DIR / "models"
OLD_WEIGHTS = MODEL_DIR / "yolo_dragon_best.pt"
BACKUP_WEIGHTS = MODEL_DIR / "yolo_dragon_nano_backup.pt"
RUN_NAME = "dragon_lesions_v3"

# Extra copies of a training image, chosen from the rarest class present.
EXTRA_COPIES = {0: 6, 1: 4, 2: 2}
CLASS_NAMES = {
    0: "Anthracnose",
    1: "Stem_Canker",
    2: "Soft_Rot",
    3: "Brown_Stem_Spot",
    4: "Gray_Blight",
}


def _classes_in_label(path: Path) -> set[int]:
    classes = set()
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        parts = line.split()
        if parts:
            classes.add(int(float(parts[0])))
    return classes


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os_link = getattr(__import__("os"), "link")
        os_link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_dataset() -> Counter:
    train_dir = OUT_ROOT / "images" / "train"
    if train_dir.exists() and len(list(train_dir.glob("*"))) >= 1200 and DATA_YAML.exists():
        print("[OK] reusing existing oversampled dataset")
        return Counter(reused=len(list(train_dir.glob("*"))))

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    counts = Counter()
    for split in ("train", "val", "test"):
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
        images = sorted((SRC_ROOT / "images" / split).glob("*"))
        images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        for img in images:
            label = SRC_ROOT / "labels" / split / f"{img.stem}.txt"
            if not label.exists():
                continue
            _link_or_copy(img, OUT_ROOT / "images" / split / img.name)
            _link_or_copy(label, OUT_ROOT / "labels" / split / label.name)
            counts[split] += 1
            if split != "train":
                continue
            present = _classes_in_label(label)
            rare = [c for c in (0, 1, 2) if c in present]
            extra = EXTRA_COPIES[min(rare)] if rare else 0
            for i in range(extra):
                name = f"{img.stem}__os{i}"
                _link_or_copy(img, OUT_ROOT / "images" / "train" / f"{name}{img.suffix.lower()}")
                _link_or_copy(label, OUT_ROOT / "labels" / "train" / f"{name}.txt")
                counts["train_extra"] += 1

    DATA_YAML.write_text(
        "\n".join(
            [
                "# Oversampled train split for YOLOv8s. Val/test match the original set.",
                f"path: {OUT_ROOT.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 5",
                "names:",
                *[f"  {i}: {name}" for i, name in CLASS_NAMES.items()],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return counts


def _metric(results, key: str) -> float:
    value = results.results_dict.get(key, 0.0)
    return float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    if not SRC_ROOT.exists():
        print(f"[ERROR] dataset not found: {SRC_ROOT}")
        sys.exit(1)

    counts = build_dataset()
    print("[OK] dataset built")
    for key, value in counts.items():
        print(f"   {key}: {value}")

    device = "0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"[OK] CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] CUDA not available, training on CPU will be slow")

    baseline = {
        "mAP50": 0.6677,
        "mAP50-95": 0.4001,
        "precision": 0.7106,
        "recall": 0.6257,
    }
    if args.skip_baseline:
        print(
            f"[BASELINE] using measured mAP50={baseline['mAP50']:.4f}  "
            f"mAP50-95={baseline['mAP50-95']:.4f}"
        )
    elif OLD_WEIGHTS.exists():
        if not BACKUP_WEIGHTS.exists():
            shutil.copy2(OLD_WEIGHTS, BACKUP_WEIGHTS)
            print(f"[OK] backed up current weights to {BACKUP_WEIGHTS}")
        print("\n=== Baseline validation (current YOLOv8n) ===")
        old = YOLO(str(OLD_WEIGHTS))
        base_res = old.val(data=str(DATA_YAML), split="val", imgsz=640, batch=16, device=device, plots=False, verbose=True)
        baseline = {
            "mAP50": _metric(base_res, "metrics/mAP50(B)"),
            "mAP50-95": _metric(base_res, "metrics/mAP50-95(B)"),
            "precision": _metric(base_res, "metrics/precision(B)"),
            "recall": _metric(base_res, "metrics/recall(B)"),
        }
        print(
            f"[BASELINE] mAP50={baseline['mAP50']:.4f}  "
            f"mAP50-95={baseline['mAP50-95']:.4f}  "
            f"P={baseline['precision']:.4f}  R={baseline['recall']:.4f}"
        )

    print("\n=== Training YOLOv8s ===")
    model = YOLO("yolov8s.pt")
    train_kwargs = dict(
        data=str(DATA_YAML),
        epochs=40,
        imgsz=640,
        batch=8,
        device=device,
        workers=2,
        cache="disk",
        project=str(BASE_DIR / "runs" / "detect"),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        cos_lr=True,
        box=7.5,
        cls=1.0,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=12.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.3,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.15,
        close_mosaic=15,
        erasing=0.1,
        patience=30,
        amp=device != "cpu",
        plots=True,
        save=True,
        seed=42,
        verbose=True,
    )

    try:
        results = model.train(**train_kwargs)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        print("[WARN] CUDA OOM at batch 8, retrying with batch 4")
        torch.cuda.empty_cache()
        train_kwargs["batch"] = 4
        results = model.train(**train_kwargs)

    new_metrics = {
        "mAP50": _metric(results, "metrics/mAP50(B)"),
        "mAP50-95": _metric(results, "metrics/mAP50-95(B)"),
        "precision": _metric(results, "metrics/precision(B)"),
        "recall": _metric(results, "metrics/recall(B)"),
    }
    print(
        f"[NEW] mAP50={new_metrics['mAP50']:.4f}  "
        f"mAP50-95={new_metrics['mAP50-95']:.4f}  "
        f"P={new_metrics['precision']:.4f}  R={new_metrics['recall']:.4f}"
    )

    best_src = BASE_DIR / "runs" / "detect" / RUN_NAME / "weights" / "best.pt"
    summary = {
        "baseline": baseline,
        "new": new_metrics,
        "weights": str(best_src),
        "replaced_best": False,
    }

    if best_src.exists():
        dest_v3 = MODEL_DIR / "yolo_dragon_v3.pt"
        MODEL_DIR.mkdir(exist_ok=True)
        shutil.copy2(best_src, dest_v3)
        improved = baseline is None or new_metrics["mAP50"] > baseline["mAP50"]
        if improved:
            shutil.copy2(best_src, OLD_WEIGHTS)
            summary["replaced_best"] = True
            print(f"[OK] improved weights saved to {OLD_WEIGHTS}")
        else:
            print("[OK] new run did not beat the current weights; kept yolo_dragon_best.pt")
            print(f"[OK] new weights kept at {dest_v3}")
    else:
        print(f"[WARN] best.pt missing at {best_src}")

    summary_path = MODEL_DIR / "yolo_v3_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] summary written to {summary_path}")


if __name__ == "__main__":
    main()
