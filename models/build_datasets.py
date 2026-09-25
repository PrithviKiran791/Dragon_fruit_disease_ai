"""
Build Enhanced Datasets for ResNet50 and ConViTX
=================================================
Creates:
1. dataset/fruit_healthy_defect/
     Healthy/             (~1,600 fruit images)
     Defective_Diseased/  (~2,000 fruit images with spots, rot, scabs)
   -> Used for ResNet50 healthy vs defect/spotted skin classification.

2. dataset/merged_6class_v2/
     Anthracnose/
     Brown_Stem_Spot/
     Gray_Blight/
     Healthy/
     Soft_Rot/
     Stem_Canker/
   -> Used for ConViTX 6-class diagnosis (balanced across both fruits and stems).
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DESKTOP_DATASET = Path(r"C:\Users\prady\OneDrive\Desktop\Dragon_fruit_dataset\dataset")
TARGET_DATASET_ROOT = PROJECT_ROOT / "dataset"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def verify_image(p: Path) -> bool:
    try:
        with Image.open(p) as img:
            img.verify()
        return True
    except Exception:
        return False


def build_fruit_healthy_defect(target_dir: Path):
    print("\n" + "=" * 60)
    print("  BUILDING: fruit_healthy_defect dataset for ResNet50")
    print(f"  Target: {target_dir}")
    print("=" * 60)

    classes = ["Healthy", "Defective_Diseased"]
    for c in classes:
        (target_dir / c).mkdir(parents=True, exist_ok=True)

    sources = {
        "Healthy": [
            # 1. Quality grading fresh fruit
            DESKTOP_DATASET / "Dragon Fruit Quality Grading Dataset" / "Original Dataset" / "Fresh Dragon Fruit",
            # 2. Dragon healthy
            DESKTOP_DATASET / "dragon" / "healthy",
            # 3. Archive fruit healthy
            DESKTOP_DATASET / "archive" / "oversample" / "Fruit" / "Healthy",
            # 4. Bangladesh good fruit
            DESKTOP_DATASET / "Dragon fruit & leaf Dataset from Bangladesh for Classification and Ecological Research" / "Dragon leaf and fruit" / "Dragon fruit and leaf" / "Good fruit",
        ],
        "Defective_Diseased": [
            # 1. Quality grading defect fruit
            DESKTOP_DATASET / "Dragon Fruit Quality Grading Dataset" / "Original Dataset" / "Defect Dragon Fruit",
            # 2. Dragon diseased
            DESKTOP_DATASET / "dragon" / "diseased",
            # 3. Archive fruit diseases (spots, scabs, rots)
            DESKTOP_DATASET / "archive" / "oversample" / "Fruit" / "Anthracnose",
            DESKTOP_DATASET / "archive" / "oversample" / "Fruit" / "Brown Spot",
            DESKTOP_DATASET / "archive" / "oversample" / "Fruit" / "Fruit Rot",
            DESKTOP_DATASET / "archive" / "oversample" / "Fruit" / "Soft Rot",
            DESKTOP_DATASET / "archive" / "oversample" / "Fruit" / "White Spot",
            # 4. Bangladesh bad fruit
            DESKTOP_DATASET / "Dragon fruit & leaf Dataset from Bangladesh for Classification and Ecological Research" / "Dragon leaf and fruit" / "Dragon fruit and leaf" / "Bad fruit",
        ]
    }

    counts = defaultdict(int)
    for cls_name, paths in sources.items():
        out_cls_dir = target_dir / cls_name
        idx = 0
        for src_path in paths:
            if not src_path.exists():
                print(f"  [WARN] Source path not found: {src_path}")
                continue

            src_tag = src_path.parent.name if src_path.parent.name != "dataset" else src_path.name
            src_tag = src_tag.replace(" ", "_")[:15]

            imgs = [p for p in src_path.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
            for img in imgs:
                dst = out_cls_dir / f"{src_tag}_{cls_name}_{idx:05d}{img.suffix.lower()}"
                if not dst.exists():
                    try:
                        shutil.copy2(img, dst)
                        idx += 1
                        counts[cls_name] += 1
                    except Exception as e:
                        print(f"  [ERR] {img} -> {e}")
                else:
                    idx += 1
                    counts[cls_name] += 1

    print(f"  [OK] Completed fruit_healthy_defect:")
    for c, cnt in counts.items():
        print(f"    - {c:20s}: {cnt:5d} images")


def build_merged_6class_v2(target_dir: Path):
    print("\n" + "=" * 60)
    print("  BUILDING: merged_6class_v2 dataset for ConViTX")
    print(f"  Target: {target_dir}")
    print("=" * 60)

    classes = [
        "Anthracnose",
        "Brown_Stem_Spot",
        "Gray_Blight",
        "Healthy",
        "Soft_Rot",
        "Stem_Canker"
    ]
    for c in classes:
        (target_dir / c).mkdir(parents=True, exist_ok=True)

    # Class mappings: (source_dir, target_class, prefix_tag)
    mappings = [
        # Pitahaya stems (all 6 classes)
        (DESKTOP_DATASET / "Dragon Fruit (Pitahaya)" / "Dragon Fruit (Pitahaya)" / "Converted Images" / "Anthracnose", "Anthracnose", "Pitahaya"),
        (DESKTOP_DATASET / "Dragon Fruit (Pitahaya)" / "Dragon Fruit (Pitahaya)" / "Converted Images" / "Brown_Stem_Spot", "Brown_Stem_Spot", "Pitahaya"),
        (DESKTOP_DATASET / "Dragon Fruit (Pitahaya)" / "Dragon Fruit (Pitahaya)" / "Converted Images" / "Gray_Blight", "Gray_Blight", "Pitahaya"),
        (DESKTOP_DATASET / "Dragon Fruit (Pitahaya)" / "Dragon Fruit (Pitahaya)" / "Converted Images" / "Healthy", "Healthy", "Pitahaya"),
        (DESKTOP_DATASET / "Dragon Fruit (Pitahaya)" / "Dragon Fruit (Pitahaya)" / "Converted Images" / "Soft_Rot", "Soft_Rot", "Pitahaya"),
        (DESKTOP_DATASET / "Dragon Fruit (Pitahaya)" / "Dragon Fruit (Pitahaya)" / "Converted Images" / "Stem_Canker", "Stem_Canker", "Pitahaya"),

        # Archive Fruit classes
        (DESKTOP_DATASET / "archive" / "oversample" / "Fruit" / "Anthracnose", "Anthracnose", "Archive_Fruit"),
        (DESKTOP_DATASET / "archive" / "oversample" / "Fruit" / "Brown Spot", "Brown_Stem_Spot", "Archive_Fruit"),
        (DESKTOP_DATASET / "archive" / "oversample" / "Fruit" / "Fruit Rot", "Soft_Rot", "Archive_Fruit"),
        (DESKTOP_DATASET / "archive" / "oversample" / "Fruit" / "Soft Rot", "Soft_Rot", "Archive_Fruit"),
        (DESKTOP_DATASET / "archive" / "oversample" / "Fruit" / "Healthy", "Healthy", "Archive_Fruit"),

        # Archive Leaf classes
        (DESKTOP_DATASET / "archive" / "oversample" / "Leaf" / "Anthracnose", "Anthracnose", "Archive_Leaf"),
        (DESKTOP_DATASET / "archive" / "oversample" / "Leaf" / "Brown Spot", "Brown_Stem_Spot", "Archive_Leaf"),
        (DESKTOP_DATASET / "archive" / "oversample" / "Leaf" / "Stem_Canker", "Stem_Canker", "Archive_Leaf"),
        (DESKTOP_DATASET / "archive" / "oversample" / "Leaf" / "Stem Rot", "Stem_Canker", "Archive_Leaf"),
        (DESKTOP_DATASET / "archive" / "oversample" / "Leaf" / "Twig Blight", "Gray_Blight", "Archive_Leaf"),
        (DESKTOP_DATASET / "archive" / "oversample" / "Leaf" / "Healthy", "Healthy", "Archive_Leaf"),

        # Dragon healthy & diseased fruit
        (DESKTOP_DATASET / "dragon" / "healthy", "Healthy", "Dragon_Fruit"),
    ]

    counts = defaultdict(int)
    for src_dir, tgt_cls, tag in mappings:
        if not src_dir.exists():
            print(f"  [WARN] Mapping src not found: {src_dir}")
            continue

        out_cls_dir = target_dir / tgt_cls
        imgs = [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
        for img in imgs:
            dst = out_cls_dir / f"{tag}_{img.name}"
            if not dst.exists():
                try:
                    shutil.copy2(img, dst)
                    counts[tgt_cls] += 1
                except Exception as e:
                    print(f"  [ERR] {img} -> {e}")
            else:
                counts[tgt_cls] += 1

    print(f"  [OK] Completed merged_6class_v2:")
    total = 0
    for c in classes:
        cnt = counts[c]
        total += cnt
        bar = "#" * (cnt // 25)
        print(f"    - {c:20s}: {cnt:5d}  {bar}")
    print(f"    {'TOTAL':20s}: {total:5d}")


if __name__ == "__main__":
    TARGET_DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    build_fruit_healthy_defect(TARGET_DATASET_ROOT / "fruit_healthy_defect")
    build_merged_6class_v2(TARGET_DATASET_ROOT / "merged_6class_v2")
