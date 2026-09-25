"""Resume the in-progress YOLOv8s run. Epoch cap is stored in last.pt."""

from pathlib import Path

from ultralytics import YOLO

LAST = Path(__file__).resolve().parent / "runs" / "detect" / "dragon_lesions_v3" / "weights" / "last.pt"


if __name__ == "__main__":
    YOLO(str(LAST)).train(resume=True, workers=0)
