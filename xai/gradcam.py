"""
Grad-CAM XAI module for Dragon Fruit Disease Detection.
Highlights the visual regions that influenced the model's prediction.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")  # headless backend — no plt.show() popups
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from models.convitx import build_convitx_base
from models.convitx_pretrained import build_convitx_pretrained

IMG_SIZE = 224
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── TRANSFORMS ──────────────────────────────────────────────────────────────
infer_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# TTA augmentation set — same as evaluate_tta.py (6 passes → +1% accuracy)
_TTA_TRANSFORMS = [
    transforms.Compose([  # original
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([  # H-flip
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([  # V-flip
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([  # 90°
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation((90, 90)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([  # 180°
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation((180, 180)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([  # 270°
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation((270, 270)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
]

# ─── GRAD-CAM ────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Computes Grad-CAM heatmap for any CNN with a named target layer.
    Works with both timm EfficientNet and torchvision ResNet.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model.eval().to(DEVICE)
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor: torch.Tensor, class_idx: int = None):
        """
        Args:
            image_tensor: [1, 3, H, W] normalized tensor
            class_idx:    target class index (None → use predicted class)
        Returns:
            heatmap (np.ndarray, float32, shape [H, W], range [0,1])
            predicted class index (int)
            prediction probabilities (np.ndarray)
        """
        image_tensor = image_tensor.to(DEVICE).requires_grad_(True)

        # Forward pass
        logits = self.model(image_tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().detach().numpy()

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Pool gradients across channels
        weights  = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        cam      = (weights * self.activations).sum(dim=1).squeeze()  # [H, W]
        cam      = F.relu(cam).detach().cpu().numpy()

        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, probs


# ─── OVERLAY ─────────────────────────────────────────────────────────────────
def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Superimposes Grad-CAM heatmap on the original image."""
    h, w = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)   # BGR

    if original_image.shape[2] == 3:
        img_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = original_image

    overlay = cv2.addWeighted(img_bgr, 1 - alpha, colored_heatmap, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


# ─── INTELLIGENT XAI LESION ANALYZER ─────────────────────────────────────────
def analyze_lesions(heatmap: np.ndarray, original_image: np.ndarray, threshold: float = 0.45) -> dict:
    """
    Intelligent XAI Diagnostic Engine:
    Detects focal activation hotspots, computes bounding boxes for skin spots,
    calculates affected surface area percentage, and grades pathology severity.
    """
    h, w = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Binary mask of high activation
    mask = (heatmap_resized >= threshold).astype(np.uint8) * 255

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    # Detect distinct lesion clusters
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = (h * w) * 0.0006  # 0.06% area threshold
    valid_boxes = []
    total_lesion_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            x, y, bw, bh = cv2.boundingRect(c)
            patch = heatmap_resized[y:y+bh, x:x+bw]
            peak = float(patch.max()) if patch.size > 0 else 0.0
            valid_boxes.append({
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "area": float(area),
                "peak_activation": round(peak * 100, 1),
            })
            total_lesion_area += area

    valid_boxes.sort(key=lambda b: b["area"], reverse=True)
    affected_pct = round((total_lesion_area / (h * w)) * 100, 2)
    peak_act = round(float(heatmap_resized.max()) * 100, 1)

    # Clinical severity determination
    if affected_pct < 0.6:
        severity = "Healthy / Low Risk"
    elif affected_pct < 3.5:
        severity = "Mild (Early Stage)"
    elif affected_pct < 10.0:
        severity = "Moderate"
    else:
        severity = "Severe / Advanced"

    return {
        "lesion_count": len(valid_boxes),
        "lesions": valid_boxes,
        "affected_area_pct": affected_pct,
        "peak_activation": peak_act,
        "severity": severity,
    }


def draw_intelligent_overlay(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    lesion_info: dict,
    predicted_class: str,
    confidence: float,
    alpha: float = 0.40,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Renders an agronomic diagnostic overlay:
      - Grad-CAM heatmap blending
      - Bounding boxes around identified skin spots / lesions
      - Diagnostic HUD banner summarizing pathology and severity
    """
    h, w = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)

    if original_image.shape[2] == 3:
        img_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = original_image.copy()

    overlay = cv2.addWeighted(img_bgr, 1 - alpha, colored_heatmap, alpha, 0)

    # Box colors
    is_healthy = "healthy" in predicted_class.lower()
    box_color = (0, 200, 50) if is_healthy else (0, 75, 255)  # Orange-Red or Green

    # Draw bounding boxes on detected spots
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, min(0.65, w / 900.0))
    for idx, lesion in enumerate(lesion_info.get("lesions", []), 1):
        bx, by, bw, bh = lesion["bbox"]
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), box_color, 2)
        label = f"Spot {idx} ({lesion['peak_activation']}%)"
        (tw, th), _ = cv2.getTextSize(label, font, scale, 1)

        cv2.rectangle(
            overlay,
            (bx, max(0, by - th - 6)),
            (bx + tw + 6, max(th + 6, by)),
            (20, 20, 25),
            -1
        )
        cv2.putText(
            overlay,
            label,
            (bx + 3, max(th + 2, by - 4)),
            font,
            scale,
            box_color,
            1,
            cv2.LINE_AA
        )

    # Draw diagnostic HUD banner at top
    banner_h = max(45, int(h * 0.085))
    hud_bg = overlay[:banner_h, :].copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (12, 16, 28), -1)
    overlay[:banner_h, :] = cv2.addWeighted(hud_bg, 0.25, overlay[:banner_h, :], 0.75, 0)

    hud_font = cv2.FONT_HERSHEY_SIMPLEX
    hud_scale = max(0.42, min(0.68, w / 950.0))

    diag_text = f"Diagnosis: {predicted_class.replace('_', ' ')} ({confidence*100:.1f}%)"
    meta_text = (
        f"Severity: {lesion_info['severity']}  |  "
        f"Spots: {lesion_info['lesion_count']}  |  "
        f"Affected Area: {lesion_info['affected_area_pct']}%"
    )

    cv2.putText(overlay, diag_text, (15, int(banner_h * 0.45)), hud_font, hud_scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(overlay, meta_text, (15, int(banner_h * 0.85)), hud_font, hud_scale * 0.85, (0, 215, 255), 1, cv2.LINE_AA)

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def generate_xai_explanation(predicted_class: str, confidence: float, lesion_info: dict) -> str:
    """Produces agronomic natural-language diagnostic rationale based on XAI metrics."""
    cls_clean = predicted_class.replace("_", " ")
    count = lesion_info.get("lesion_count", 0)
    area = lesion_info.get("affected_area_pct", 0.0)
    severity = lesion_info.get("severity", "Unknown")
    peak = lesion_info.get("peak_activation", 0.0)

    if "healthy" in predicted_class.lower() or count == 0:
        return (
            f"Explainable AI (Grad-CAM) analysis detected uniform epidermal integrity with minimal anomalous activation "
            f"(surface coverage: {area}%). No pathogenic lesion clusters were identified, verifying healthy tissue."
        )
    else:
        return (
            f"Explainable AI (Grad-CAM) identified {count} distinct anomalous spot cluster(s) covering {area}% of the visible "
            f"surface with a peak neural activation of {peak}%. The visual attention pattern confirms focal necrotic "
            f"tissue characteristic of {cls_clean} (rated {severity})."
        )


# ─── CONVENIENCE FUNCTION ────────────────────────────────────────────────────
def run_gradcam(
    model:        torch.nn.Module,
    target_layer: torch.nn.Module,
    image_path:   str,
    class_names:  list,
    save_path:    str = None,
    use_tta:      bool = True,
) -> dict:
    """
    End-to-end Grad-CAM pipeline with Intelligent XAI analysis.
    """
    pil_img = Image.open(image_path).convert("RGB")
    orig_np = np.array(pil_img)

    # ── TTA: average probabilities over 6 augmented views ────────────────
    if use_tta:
        all_probs = []
        for tf in _TTA_TRANSFORMS:
            t = tf(pil_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(t)
                p = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            all_probs.append(p)
        probs    = np.stack(all_probs, axis=0).mean(axis=0)   # avg over 6 passes
        pred_idx = int(np.argmax(probs))
    else:
        tensor_img = infer_transforms(pil_img).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor_img.to(DEVICE))
            probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs))

    # ── Grad-CAM heatmap ─────────────────────────────────────────────────
    tensor_img = infer_transforms(pil_img).unsqueeze(0)
    gradcam    = GradCAM(model, target_layer)
    heatmap, _, _ = gradcam.generate(tensor_img, class_idx=pred_idx)
    raw_overlay = overlay_heatmap(orig_np, heatmap)

    # ── Intelligent Lesion & Severity Analysis ───────────────────────────
    pred_cls = class_names[pred_idx]
    conf_score = float(probs[pred_idx])
    lesion_info = analyze_lesions(heatmap, orig_np)
    annotated_overlay = draw_intelligent_overlay(
        orig_np, heatmap, lesion_info, pred_cls, conf_score
    )
    xai_explanation = generate_xai_explanation(pred_cls, conf_score, lesion_info)

    low_confidence = conf_score < 0.50

    result = {
        "predicted_class":   pred_cls,
        "confidence":        conf_score,
        "probabilities":     {c: float(p) for c, p in zip(class_names, probs)},
        "heatmap":           heatmap,
        "overlay":           annotated_overlay,
        "raw_overlay":       raw_overlay,
        "lesion_info":       lesion_info,
        "xai_explanation":   xai_explanation,
        "low_confidence":    low_confidence,
    }

    # Visualise
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig_np);   axes[0].set_title("Original Image");        axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title(f"Grad-CAM Heatmap (Peak {lesion_info['peak_activation']}%)"); axes[1].axis("off")
    conf_label = f"{pred_cls} ({conf_score:.1%})\nSeverity: {lesion_info['severity']}"
    if low_confidence:
        conf_label += "\n[WARN] Low confidence"
    axes[2].imshow(annotated_overlay); axes[2].set_title(conf_label); axes[2].axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)

    return result


# ─── LAYER HELPERS ───────────────────────────────────────────────────────────
def get_target_layer_efficientnet(model) -> torch.nn.Module:
    """Returns the last convolutional block of a timm EfficientNet."""
    return model.blocks[-1]


def get_target_layer_resnet50(model) -> torch.nn.Module:
    """Returns the last bottleneck block of ResNet50 layer4 for Grad-CAM."""
    return model.layer4[-1]


def get_target_layer_convitx(model) -> torch.nn.Module:
    """Returns an architecture-aware target layer for ConViTX Grad-CAM hooks."""
    if hasattr(model, "fusion_conv"):
        return model.fusion_conv[0]

    if hasattr(model, "cnn_branch"):
        try:
            return model.cnn_branch[12]   # Last InvertedResidual in MobileNetV3-Small
        except (IndexError, TypeError):
            return model.cnn_branch[-2]   # fallback

    raise AttributeError("Unsupported ConViTX architecture: missing Grad-CAM target layer")


def load_convitx_model(
    model_path: str,
    num_classes: int = 6,
    device: torch.device = DEVICE,
) -> torch.nn.Module:
    """Load ConViTX checkpoint, auto-selecting architecture from checkpoint keys."""
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        state = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format in: {model_path}")

    is_pretrained_convitx = any(k.startswith("head.0") for k in state.keys())
    if is_pretrained_convitx:
        model = build_convitx_pretrained(num_classes=num_classes)
    else:
        model = build_convitx_base(num_classes=num_classes, enforce_budget=False)

    model.load_state_dict(state)
    model.eval().to(device)
    return model


def load_resnet50_model(
    model_path: str,
    num_classes: int = 2,
    device: torch.device = DEVICE,
) -> torch.nn.Module:
    """Load trained ResNet50 fruit model."""
    from torchvision import models
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes),
    )
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        state = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval().to(device)
    return model
