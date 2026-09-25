"""
Train ResNet50 for Dragon Fruit Skin Health & Defect Detection
==============================================================
Binary classification:
  Class 0: Defective_Diseased (Skin spots, fungal lesions, rot, scabs)
  Class 1: Healthy (Clean, fresh dragon fruit)

Uses torchvision ResNet50 (ImageNet pretrained) with 2-phase training,
mixed precision (AMP), differential learning rates, and early stopping.
"""

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def plot_metrics(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-s", label="Val Loss")
    ax1.set_title("ResNet50 Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Acc")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "r-s", label="Val Acc")
    ax2.set_title("ResNet50 Accuracy Curve (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix — ResNet50 Fruit Quality",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold"
            )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def build_model(num_classes=2):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes),
    )
    return model


def train_resnet50(
    data_dir,
    save_dir=_SCRIPT_DIR,
    epochs=30,
    freeze_epochs=4,
    batch_size=32,
    lr_head=1e-3,
    lr_backbone=2e-5,
    seed=42,
    patience=8,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"  Training ResNet50 on Fruit Health & Defects")
    print(f"  Device   : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Data dir : {data_dir}")
    print(f"  Epochs   : {epochs} (Freeze head warmup: {freeze_epochs})")
    print(f"=======================================================\n")

    train_tf, val_tf = get_transforms(224)

    # Full dataset without transform to extract targets
    raw_dataset = datasets.ImageFolder(data_dir)
    class_names = raw_dataset.classes
    print(f"Classes: {class_names} (total images: {len(raw_dataset)})")

    # Stratified split 80% train, 20% val
    targets = [s[1] for s in raw_dataset.samples]
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=0.20,
        random_state=seed,
        stratify=targets,
    )

    # Class balance weights
    train_targets = [targets[i] for i in train_idx]
    class_counts = np.bincount(train_targets, minlength=len(class_names))
    print(f"Training counts: {dict(zip(class_names, class_counts))}")

    class_weights = 1.0 / (class_counts.astype(np.float32) + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_names)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)

    # Datasets with appropriate transforms
    train_ds = Subset(datasets.ImageFolder(data_dir, transform=train_tf), train_idx)
    val_ds = Subset(datasets.ImageFolder(data_dir, transform=val_tf), val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True if torch.cuda.is_available() else False
    )

    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # ── Phase 1: Freeze Backbone, Train Head ──
    print("\n[Phase 1] Training custom classifier head (backbone frozen)...", flush=True)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer_head = torch.optim.AdamW(model.fc.parameters(), lr=lr_head, weight_decay=1e-3)

    for epoch in range(1, freeze_epochs + 1):
        model.train()
        running_loss, running_corrects, total_samples = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer_head.zero_grad()
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer_head)
            scaler.update()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)

        ep_loss = running_loss / total_samples
        ep_acc = running_corrects / total_samples
        print(f"  Warmup Epoch {epoch:2d}/{freeze_epochs:2d} - Loss: {ep_loss:.4f} - Acc: {ep_acc * 100:.2f}%", flush=True)

    # ── Phase 2: Full Fine-Tuning with Differential LRs ──
    print("\n[Phase 2] Joint fine-tuning (backbone unfrozen, differential LR)...")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if not n.startswith("fc")], "lr": lr_backbone, "weight_decay": 1e-4},
        {"params": model.fc.parameters(), "lr": lr_head * 0.2, "weight_decay": 1e-3},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - freeze_epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(freeze_epochs + 1, epochs + 1):
        # Train step
        model.train()
        train_loss, train_corrects, train_total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(outputs, 1)
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data).item()
            train_total += inputs.size(0)

        scheduler.step()
        ep_train_loss = train_loss / train_total
        ep_train_acc = train_corrects / train_total

        # Validation step
        model.eval()
        val_loss, val_corrects, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()
                val_total += inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        ep_val_loss = val_loss / val_total
        ep_val_acc = val_corrects / val_total
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)

        history["train_loss"].append(ep_train_loss)
        history["train_acc"].append(ep_train_acc)
        history["val_loss"].append(ep_val_loss)
        history["val_acc"].append(ep_val_acc)

        print(
            f"Epoch {epoch:2d}/{epochs:2d} | "
            f"Train Loss: {ep_train_loss:.4f} Acc: {ep_train_acc*100:5.2f}% | "
            f"Val Loss: {ep_val_loss:.4f} Acc: {ep_val_acc*100:5.2f}% F1: {f1:.4f}",
            flush=True,
        )

        # Track best model
        if ep_val_acc > best_val_acc or (ep_val_acc == best_val_acc and f1 > best_val_f1):
            best_val_acc = ep_val_acc
            best_val_f1 = f1
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_preds = all_preds
            best_labels = all_labels
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] No improvement in {patience} epochs. Stopping.")
                break

    # Save best model
    out_model_path = Path(save_dir) / "best_resnet50_fruit.pth"
    torch.save(best_model_wts, out_model_path)
    print(f"\n[OK] Best model saved -> {out_model_path}")
    print(f"  Best Validation Accuracy: {best_val_acc*100:.2f}% | Macro F1: {best_val_f1:.4f}")

    # Generate curves and confusion matrix
    plot_metrics(history, Path(save_dir) / "resnet50_fruit_curves.png")
    cm = confusion_matrix(best_labels, best_preds)
    plot_confusion_matrix(cm, class_names, Path(save_dir) / "resnet50_fruit_cm.png")

    report = classification_report(best_labels, best_preds, target_names=class_names, output_dict=True)
    summary = {
        "best_val_accuracy": round(best_val_acc * 100, 2),
        "best_macro_f1": round(best_val_f1, 4),
        "classes": class_names,
        "classification_report": report,
    }
    with open(Path(save_dir) / "resnet50_fruit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nClassification Report:")
    print(classification_report(best_labels, best_preds, target_names=class_names))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "dataset" / "fruit_healthy_defect"))
    parser.add_argument("--save-dir", type=str, default=str(_SCRIPT_DIR))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    train_resnet50(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )
