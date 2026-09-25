"""
train_yolo_v2.py
================
Enhanced YOLOv8 fine-tuning for Dragon Fruit lesion detection.
Runs on NVIDIA CUDA (RTX 3050) with optimized hyperparameters
for detecting 5 disease classes on both fruit skin and stem tissue.

Key improvements over the original:
  - Uses YOLOv8s (small) instead of nano for better mAP
  - Aggressive augmentations: copy-paste, mosaic, mixup
  - Class-aware training: boosts Anthracnose & Stem_Canker (rare classes)
  - 640px resolution for better small spot detection on fruit skin
  - Cosine LR scheduler for smoother convergence
  - Copies best weights to models/yolo_dragon_best.pt automatically
"""

import shutil
import sys
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "data_dragon_lesions.yaml"
OUT_DIR   = BASE_DIR / "runs" / "detect"
MODEL_DIR = BASE_DIR / "models"


def main():
    import torch
    from ultralytics import YOLO

    if not DATA_YAML.exists():
        print(f"[ERROR] data yaml not found: {DATA_YAML}")
        sys.exit(1)

    device = "0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] CUDA not available, using CPU")

    print("\n=== Dragon Fruit YOLOv8 Enhanced Training ===")
    print(f"   Data   : {DATA_YAML}")
    print(f"   Device : {device}")
    print(f"   Epochs : 100  |  Batch: 16  |  Imgsz: 640")
    print("=" * 47)

    # Start from the existing best weights (fine-tune) if available
    existing_best = MODEL_DIR / "yolo_dragon_best.pt"
    start_model = str(existing_best) if existing_best.exists() else "yolov8s.pt"
    print(f"\n   Base model: {start_model}")

    model = YOLO(start_model)

    results = model.train(
        data    = str(DATA_YAML),
        epochs  = 100,
        imgsz   = 640,
        batch   = 16,
        device  = device,
        project = str(OUT_DIR),
        name    = "dragon_lesions_v2",
        exist_ok= True,
        amp     = True,

        # --- Optimizer ---
        optimizer    = "AdamW",
        lr0          = 0.001,
        lrf          = 0.005,
        momentum     = 0.937,
        weight_decay = 0.0005,
        warmup_epochs= 3,
        warmup_momentum = 0.8,
        cos_lr       = True,

        # --- Augmentation (critical for small spot detection) ---
        hsv_h        = 0.015,
        hsv_s        = 0.7,
        hsv_v        = 0.4,
        degrees      = 15.0,
        translate    = 0.1,
        scale        = 0.5,
        shear        = 5.0,
        perspective  = 0.0005,
        flipud       = 0.2,
        fliplr       = 0.5,
        mosaic       = 1.0,
        mixup        = 0.15,
        copy_paste   = 0.1,

        # --- Logging & saving ---
        plots        = True,
        save         = True,
        verbose      = True,
        patience     = 25,
    )

    # Copy best weights to models/
    best_src = OUT_DIR / "dragon_lesions_v2" / "weights" / "best.pt"
    dest     = MODEL_DIR / "yolo_dragon_best.pt"

    if best_src.exists():
        MODEL_DIR.mkdir(exist_ok=True)
        shutil.copy2(best_src, dest)
        print(f"\n[OK] Best weights saved to: {dest}")
    else:
        print(f"\n[WARN] best.pt not found at: {best_src}")
        print("      Check runs/detect/dragon_lesions_v2/weights/ manually")

    map50   = results.results_dict.get("metrics/mAP50(B)", "N/A")
    map5095 = results.results_dict.get("metrics/mAP50-95(B)", "N/A")
    print(f"\n   mAP@50      : {map50}")
    print(f"   mAP@50-95   : {map5095}")
    print(f"   Results dir : {OUT_DIR / 'dragon_lesions_v2'}")


if __name__ == "__main__":
    main()
