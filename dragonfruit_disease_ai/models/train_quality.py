"""
Train ResNet50 for Dragon Fruit Quality Grading.

Default dataset layout expected:
  dataset/Dragon Fruit Quality Grading Dataset/Augmented Dataset/<class_name>/*.jpg

Outputs:
  - quality_resnet50.pth
  - quality_training_curves.png
  - quality_classes.txt
"""

import argparse
import copy
import os
import time
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
try:
    import torch_directml
    _HAS_DML = True
except ImportError:
    _HAS_DML = False
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.transforms import functional as TF


class RandomBrightnessContrast:
    """Torchvision equivalent of RandomBrightnessContrast augmentation."""

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, p: float = 0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() > self.p:
            return img

        brightness_factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.brightness
        contrast_factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.contrast
        img = TF.adjust_brightness(img, brightness_factor)
        img = TF.adjust_contrast(img, contrast_factor)
        return img


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.normpath(
        os.path.join(
            script_dir,
            "..",
            "dataset",
            "Dragon Fruit Quality Grading Dataset",
            "Augmented Dataset",
        )
    )

    parser = argparse.ArgumentParser(description="Train quality grading model (ResNet50)")
    parser.add_argument("--data-dir", type=str, default=default_data_dir, help="ImageFolder data directory")
    parser.add_argument("--save-dir", type=str, default=script_dir, help="Output directory for model and plots")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=7)
    return parser.parse_args()


def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        RandomBrightnessContrast(brightness=0.25, contrast=0.25, p=0.7),
        transforms.ColorJitter(saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return train_tf, val_tf


def main():
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif _HAS_DML:
        device = torch_directml.device()
    else:
        device = torch.device("cpu")

    data_dir = os.path.normpath(args.data_dir)
    save_dir = os.path.normpath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    train_tf, val_tf = build_transforms(args.img_size)

    full_dataset = datasets.ImageFolder(data_dir)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    if num_classes < 2:
        raise ValueError(f"At least 2 classes required, found: {class_names}")

    train_size = int(len(full_dataset) * args.train_split)
    val_size = len(full_dataset) - train_size
    if train_size == 0 or val_size == 0:
        raise ValueError("Invalid split. Adjust --train-split so train and val contain samples.")

    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_subset.dataset = copy.copy(full_dataset)
    train_subset.dataset.transform = train_tf
    val_subset.dataset = copy.copy(full_dataset)
    val_subset.dataset.transform = val_tf

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = models.resnet50(weights="DEFAULT")
    for name, param in model.named_parameters():
        if "layer3" not in name and "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes),
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    # Start below any real accuracy so epoch 1 is always checkpointed.
    best_val_acc = -1.0
    best_state = None
    patience_counter = 0

    model_path = os.path.join(save_dir, "quality_resnet50.pth")
    curve_path = os.path.join(save_dir, "quality_training_curves.png")
    class_map_path = os.path.join(save_dir, "quality_classes.txt")

    print(f"Device       : {device}")
    print(f"Dataset      : {data_dir}")
    print(f"Classes      : {class_names}")
    print(f"Train/Val    : {train_size}/{val_size}")

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum, train_correct = 0.0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_loss = train_loss_sum / train_size
        train_acc = train_correct / train_size

        model.eval()
        val_loss_sum, val_correct = 0.0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * images.size(0)
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_loss = val_loss_sum / val_size
        val_acc = val_correct / val_size
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, model_path)
            patience_counter = 0
            print(f"  New best model saved: {model_path} (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    elapsed = time.time() - start
    print(f"Training complete in {elapsed / 60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    with open(class_map_path, "w", encoding="utf-8") as f:
        for idx, cls in enumerate(class_names):
            f.write(f"{idx}\t{cls}\n")
    print(f"Saved class mapping: {class_map_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train Accuracy")
    axes[1].plot(history["val_acc"], label="Val Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves: {curve_path}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.eval()
        final_preds, final_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                preds = model(images).argmax(1).cpu().tolist()
                final_preds.extend(preds)
                final_labels.extend(labels.tolist())

        print("\nValidation classification report:")
        print(classification_report(final_labels, final_preds, target_names=class_names))


if __name__ == "__main__":
    main()
