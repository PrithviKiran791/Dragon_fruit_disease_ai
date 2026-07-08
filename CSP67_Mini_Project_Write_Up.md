# Dragon Fruit Disease Detection & Quality Assessment System
## VI Semester Mini Project Write-Up (CSP67) — Official Academic Submission

---

## 1. PROBLEM STATEMENT & OBJECTIVES [2 Marks]

### 1.1 Problem Statement

Dragon fruit (*Pitahaya*) cultivation faces significant agronomic challenges that impact yield quality and marketability:

- **Agricultural Impact:** Multiple foliar (leaf) and stem diseases (e.g., Anthracnose, Brown Stem Spot, Soft Rot, Gray Blight, Stem Canker) cause economic losses, with symptoms that are visually similar and difficult for farmers to distinguish in-field.
- **Current Bottleneck:** Manual disease identification is time-consuming, subjective, and requires expert agronomists who are not always accessible in remote farming regions.
- **Computational Constraints:** Edge deployment (mobile/IoT) demands lightweight models; cloud-only solutions introduce latency and connectivity issues unsuitable for real-time field use.
- **Data Scarcity:** Limited annotated dragon fruit disease datasets with high-resolution lesion labels and diverse environmental conditions complicate model training and generalization.

### 1.2 Primary Objectives

The system is designed to achieve six core goals:

1. **Automated Disease Classification:** Multiclass classification of six disease categories + healthy status from fruit and leaf images with ≥ 90% validation accuracy.
2. **Lesion Localization:** Precise object detection to identify and bound infected regions, enabling severity estimation (low/medium/high) based on lesion count and coverage.
3. **Visual Explainability:** Generate Grad-CAM heatmaps to highlight regions contributing to classification decisions, improving user trust and agronomist verification.
4. **Produce Quality Grading:** Secondary classification task for fruit maturity (Fresh, Mature, Immature, Defect) to assist post-harvest sorting.
5. **Actionable Advisory:** Map predicted disease + severity → treatment/prevention recommendations via a chatbot interface (using LLM or knowledge-base fallback).
6. **Web-Based Accessibility:** Flask-backed single-page application with upload and camera capture interfaces to support farmers without machine-learning expertise.

---

## 2. DESIGN & SYSTEM ARCHITECTURE [3 Marks]

### 2.1 High-Level System Architecture

The application follows a **three-tier architectural pattern:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRESENTATION TIER (Frontend)                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Flask Templates & Static Assets                           │ │
│  │  • Home, Disease, Quality, Detection, Camera, VQA Pages   │ │
│  │  • Real-time Camera Integration (WebRTC / getUserMedia)   │ │
│  │  • Image Upload, Result Display, Grad-CAM Overlays        │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                 ↓ HTTP/JSON
┌─────────────────────────────────────────────────────────────────┐
│                APPLICATION TIER (API & Orchestration)           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Flask Application Server (app/main.py, app/app.py)       │ │
│  │  ├─ Route Handlers (GET/POST)                             │ │
│  │  ├─ Input Validation & File Upload Management             │ │
│  │  ├─ Request Preprocessing & Artifact Generation           │ │
│  │  ├─ Advisory/VQA Orchestration (chatbot/)                │ │
│  │  └─ Response Composition & Error Handling                 │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                 ↓ Python API Calls
┌─────────────────────────────────────────────────────────────────┐
│              AI INFERENCE TIER (Model Runtime)                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Classification Branch                                     │ │
│  │  └─ ConViTX-Pretrained Hybrid (CNN+ViT)                   │ │
│  │     • PyTorch checkpoint: models/best_convitx_pretrained  │ │
│  │     • Inference: image → 6-class logits → softmax probs   │ │
│  │     • TTA (6-pass augmentation) for robustness            │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │  Detection Branch                                          │ │
│  │  └─ YOLOv8 Lesion Detector                                │ │
│  │     • PyTorch checkpoint: models/yolo_dragon_best.pt      │ │
│  │     • Output: bounding boxes + confidences                │ │
│  │     • Heuristic: severity = f(box_count, box_area)        │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │  Explainability Branch                                     │ │
│  │  └─ Grad-CAM (xai/gradcam.py)                             │ │
│  │     • Backprop through classification model                │ │
│  │     • Generate class-activation map + overlay image       │ │
│  │     • Store in results/ for client download               │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│               DATA & ARTIFACT STORAGE                            │
│  ├─ dataset/          [Training data, multiple sources]         │
│  ├─ models/           [Checkpoints, summaries, ONNX exports]    │
│  ├─ results/          [Generated heatmaps, test artifacts]      │
│  ├─ app/static/uploads/ [Ephemeral user uploads]               │
│  └─ chatbot/          [Advisory knowledge base & LLM client]    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Components

| **Component**         | **Files**                            | **Responsibility**                                          |
|-----------------------|--------------------------------------|-------------------------------------------------------------|
| **Frontend**          | `app/templates/`, `app/static/`      | HTML/CSS/JS pages; image upload; real-time camera          |
| **API Server**        | `app/main.py`, `app/app.py`         | Flask routes, request handling, orchestration              |
| **Classification**    | `models/convitx_pretrained.py`       | CNN+ViT hybrid; 6-class disease inference                  |
| **Detection**         | `models/yolo_dragon_best.pt`        | YOLOv8 lesion localization & severity heuristics           |
| **Explainability**    | `xai/gradcam.py`                    | Grad-CAM heatmap generation & visualization                |
| **Advisory**          | `chatbot/advisor.py`, `knowledge_base.py` | Treatment recommendations based on prediction             |
| **Data Prep**         | `prepare_yolo_dataset.py`, `build_merged_dataset.py` | Dataset construction & augmentation |
| **Validation**        | `validate_yolo_dataset.py`          | Sanity checks on image/label pairs                         |

### 2.3 Request-Response Dataflow

A typical classification request flows as follows:

```
1. Client  → POST /predict_disease (multipart form with image)
2. Flask   → Validate file extension & MIME type
3. Flask   → Save to app/static/uploads/{uuid}.jpg
4. Preprocess → Resize, normalize (ImageNet stats)
5. Classification Model → Forward pass → 6 logits
6. TTA Loop (6 augmentations) → Average logits → Final probs
7. Severity Estimation → YOLOv8 forward pass → boxes → heuristic
8. Grad-CAM → Backprop activations → heatmap image
9. Advisory → Disease label + severity → LLM/KB text
10. Response → JSON {label, confidence, boxes, heatmap_url, advice}
11. Client  ← Display results & overlays
```

### 2.4 Design Rationale

- **Hybrid CNN+ViT:** Combines local feature extraction (CNN) with global contextual modeling (ViT), balancing computational cost (~18 MB) and accuracy (~95% validation).
- **Dual-branch (classification + detection):** Separate pathways avoid single-point failure; detection provides spatial context for severity grading.
- **TTA for robustness:** Six augmented passes (rotate, flip, crop variations) reduce overfitting to training distribution; improves edge deployment reliability.
- **Grad-CAM for trust:** Visual explanations allow agronomists to validate predictions and build confidence in automated recommendations.
- **Modular architecture:** Flask orchestrates independent components; easy to swap models or add new tasks (e.g., segmentation) without rewriting routing logic.

---

## 3. METHODOLOGY [3 Marks]

### 3.1 Data Preparation

**Dataset Sources:**
- **Multi-source collection:** Consolidated datasets from public repositories and field surveys:
  - Dragon Fruit (Pitahaya) classification dataset (Bangladesh, ecological focus)
  - Dragon Fruit Stem Disease annotated high-resolution dataset (segmentation-ready)
  - Dragon Fruit Quality Grading dataset (Fresh, Mature, Immature, Defect)
  - YOLO lesion detection dataset with bounding-box annotations
  
**Processing Pipeline:**
1. **Inventory Creation:** Audit all source folders and generate `dataset_image_inventory.csv` with image paths, dimensions, and label associations.
2. **Merging:** Consolidate heterogeneous class labels into a unified 6-class taxonomy (Anthracnose, Brown Stem Spot, Gray Blight, Healthy, Soft Rot, Stem Canker) via `build_merged_dataset.py`.
3. **Augmentation:** Apply composition augmentations—RandomCrop, HorizontalFlip, VerticalFlip, Rotate (±15°), ColorJitter (brightness, contrast, saturation), RandomErasing—to address class imbalance and increase effective training set size.
4. **Train/Val/Test Split:** Stratified split by class and source to minimize data leakage; validation set reserved for early stopping and TTA evaluation.

**Dataset Statistics:**
- **Training set:** ~800–1000 images across 6 disease classes + healthy.
- **Validation set:** ~150–200 images (per-class support: 20–27 samples).
- **Test set:** ~100 images for final evaluation.
- **Class distribution:** Imbalanced; oversampled/augmented rare classes (Gray Blight, Stem Canker) to improve recall.

### 3.2 Model Architecture: ConViTX (CNN + Vision Transformer Hybrid)

**Architecture Overview:**

```
Input Image (H, W, 3)
         ↓
┌──────────────────────────────────┐
│   CNN Backbone (MobileNetV3-Small) │ ← Pretrained on ImageNet
│   [BN] → [Conv] → [ReLU] blocks   │    (feature extraction)
└──────────────────────────────────┘
         ↓
   Feature Map (C, H', W')
         ↓
┌──────────────────────────────────┐
│  Patch Embedding & ViT Head      │
│  • Reshape patches                 │
│  • Learnable class token           │
│  • 12-layer Transformer stack      │
│  • Multi-head self-attention       │
│  • Position encoding + dropout     │
└──────────────────────────────────┘
         ↓
     [CLS] Token Representation
         ↓
┌──────────────────────────────────┐
│   Classification Head             │
│   [Linear(768) → 6-way softmax]  │
└──────────────────────────────────┘
         ↓
   Class Probabilities (6,)
```

**Key Design Choices:**

1. **CNN Backbone:** MobileNetV3-Small (pretrained ImageNet) for efficient feature extraction; reduces parameters by 90% vs. ResNet50 while maintaining accuracy.
2. **ViT Encoder:** Adds global receptive field and long-range dependencies; enables Grad-CAM on transformer attention maps.
3. **Two-phase Training:**
   - **Phase 1 (0 epochs):** CNN frozen; only ViT head + classification layer trained (rapid convergence).
   - **Phase 2 (25 epochs):** Joint fine-tuning; CNN learning rate scaled 10× lower to preserve pretrained features.
4. **Differential Learning Rates:** CNN LR = 3e-5; ViT/head LR = 3e-4. Prevents catastrophic forgetting while adapting to domain.

**Hyperparameters:**
```
Optimizer:          AdamW
Base LR:            3e-5 (CNN), 3e-4 (ViT)
Batch Size:         32
Epochs:             25
Loss Function:      Focal CrossEntropy (γ=2.0) + Label Smoothing (0.05)
EMA Decay:          0.9995
Augmentation:       RandomCrop, Flip, Rotate, ColorJitter, RandomErasing
```

### 3.3 Training Strategy

**Loss Function — Focal CrossEntropy:**
$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where:
- $p_t$ = model confidence for true class
- $\gamma = 2.0$ (focusing parameter, emphasizes hard negatives)
- $\alpha_t$ = class weighting to balance imbalanced data

**Label Smoothing:**
$$q_i = (1 - \epsilon) \cdot \mathbb{1}[y = i] + \frac{\epsilon}{K}$$

where $\epsilon = 0.05$, $K = 6$ classes. Prevents overconfident predictions on noisy labels.

**Exponential Moving Average (EMA):**
$$\theta_{\text{ema}} \leftarrow 0.9995 \cdot \theta_{\text{ema}} + 0.0005 \cdot \theta_{\text{current}}$$

Maintains a slower-moving shadow model for inference stability.

**Early Stopping:** Monitor validation loss; stop after 12 epochs without improvement.

### 3.4 Test-Time Augmentation (TTA)

During inference, six augmented variants of the input are passed through the model:
1. Original image
2. Horizontal flip
3. Vertical flip
4. 90° rotation
5. 180° rotation
6. RandomCrop + center crop

**TTA Aggregation:**
$$\hat{y}_{\text{TTA}} = \text{softmax}\left(\frac{1}{6} \sum_{i=1}^{6} z_i\right)$$

where $z_i$ = logits from augmented pass $i$. Improves robustness by averaging predictions across different image perspectives.

### 3.5 Grad-CAM for Visual Explainability

**Gradient-weighted Class Activation Mapping:**

For a classification model and target class $c$:

1. **Forward pass:** Compute activations $A^k$ from a chosen layer (e.g., ViT token logits).
2. **Backward pass:** Compute gradients $\frac{\partial s^c}{\partial A^k}$ where $s^c$ is the score for class $c$.
3. **Importance weights:**
   $$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial s^c}{\partial A_{ij}^k}$$
4. **Activation map:**
   $$L^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
5. **Overlay:** Superimpose colormap on original image; lighter regions indicate stronger contribution to predicted class.

**Implementation:** See `xai/gradcam.py` lines 248–264 for layer targeting logic.

### 3.6 Lesion Detection & Severity Heuristic

**YOLOv8 Object Detector:**
- **Input:** Lesion bounding-box annotated dataset (YOLO format).
- **Output:** Bounding boxes $(x, y, w, h)$, class (lesion), and confidence $\geq 0.5$.
- **Training:** Standard YOLOv8 training pipeline with data augmentation and class weighting.

**Severity Estimation:**
$$\text{Severity} = \begin{cases} \text{Low} & \text{if } n_{\text{boxes}} \leq 2 \text{ or } A_{\text{total}} < 5\% \\ \text{Medium} & \text{if } 2 < n_{\text{boxes}} \leq 5 \text{ or } 5\% \leq A_{\text{total}} < 15\% \\ \text{High} & \text{if } n_{\text{boxes}} > 5 \text{ or } A_{\text{total}} \geq 15\% \end{cases}$$

where $n_{\text{boxes}}$ = count of detected lesions, $A_{\text{total}}$ = percentage of image area covered by bounding boxes.

---

## 4. RESULTS & EVALUATION [2 Marks]

### 4.1 Classification Performance

**ConViTX-Pretrained (Best Model):**

| Metric              | Value    |
|---------------------|----------|
| **Best Epoch**      | 9        |
| **Validation Acc.** | **94.62%** |
| **Macro F1-Score**  | **0.9363** |
| **TTA Accuracy**    | 95.76%   |
| **Inference Time**  | ~0.35s   |
| **Model Size**      | ~18 MB   |
| **Parameters**      | 2,993,254 |

**Per-Class Metrics (Validation Set):**

| Disease Class     | Precision | Recall | F1-Score | Support |
|-------------------|-----------|--------|----------|---------|
| Anthracnose       | 0.769     | 1.000  | 0.870    | 13      |
| Brown_Stem_Spot   | 0.984     | 0.939  | 0.961    | 33      |
| Gray_Blight       | 0.903     | 1.000  | 0.949    | 21      |
| Healthy           | 1.000     | 0.955  | 0.977    | 22      |
| Soft_Rot          | 0.991     | 0.982  | 0.987    | 56      |
| Stem_Canker       | 0.913     | 0.840  | 0.875    | 25      |
| **Macro Avg.**    | **0.927** | **0.953** | **0.936** | 170 |

### 4.2 Detection Performance (YOLOv8 Lesion Detector)

| Metric           | Value     |
|------------------|-----------|
| **Inference Time** | ~0.33s   |
| **Detected Lesions (per image)** | 1–10 (avg. 3–4) |
| **Severity Assignment** | Functional (Low/Medium/High) |
| **Sample Output** | "1 box, severity=Low" (from test TC-06) |

### 4.3 Quality Grading Performance

**Test Case Results (TC-05 Re-run):**

| Grade Class       | Accuracy | Avg. Confidence | Notes                  |
|-------------------|----------|-----------------|------------------------|
| Fresh             | PASS     | 0.402           | Conservative predict   |
| Defect            | PASS     | 0.877           | High confidence        |
| Immature          | PASS     | 0.961           | Strong prediction      |
| Mature            | PASS     | 0.583           | Borderline case        |

### 4.4 System-Level Test Results

| Test Case | Module Tested                | Status | Notes                         |
|-----------|------------------------------|--------|-------------------------------|
| TC-01     | Image Upload (JPG/PNG)       | **PASS** | Supports .jpg, .jpeg, .png |
| TC-02     | Unsupported Files (.pdf)     | **PASS** | Rejects safely              |
| TC-05     | Quality Grading              | **PASS** | Confidence > 0.58           |
| TC-06     | YOLOv8 Lesion Detection      | **PASS** | 1 box, severity assigned    |
| TC-08     | Advisory Generation          | **PASS** | Treatment text returned     |
| TC-10     | System Performance (latency) | **PASS** | Grad-CAM 0.87s, YOLO 0.33s |

### 4.5 Model Comparison

| Model                | Params (M) | Size (MB) | Val. Acc. | Inference (s) | Notes               |
|----------------------|------------|-----------|-----------|---------------|---------------------|
| **ConViTX-Pretrained** | **2.99**  | **18**    | **94.62%** | **0.35**     | ✓ Selected          |
| ConViTX-Finetuned    | 1.21       | 12        | 46.21%    | 0.31          | Data scarcity issue |
| ResNet50             | 25.6       | 142       | ~92%      | 0.40          | 8× larger params    |

**Key Insight:** ConViTX-Pretrained achieves competitive accuracy (94.62%) with **90% fewer parameters** than ResNet50, making it suitable for edge deployment (mobile, IoT).

### 4.6 Grad-CAM Visual Explanations

**Artifact Storage:** Heatmap overlays saved to `results/` directory on-demand.

**Example Output:** For a Stem_Canker prediction, Grad-CAM highlights the lesion region(s) on the fruit surface, enabling agronomist verification.

---

## 5. TECHNICAL ACHIEVEMENTS & CONTRIBUTIONS

### 5.1 Key Innovations

1. **Hybrid CNN+ViT Architecture:** Custom ConViTX design leveraging pretrained CNNs for efficient feature extraction + ViT for global context; achieves 94.62% accuracy with minimal parameters.

2. **Multi-task Learning Pipeline:** Single framework handles classification, detection, quality grading, VQA, and explainability—reducing code duplication and maintaining consistency.

3. **Grad-CAM Integration:** Real-time heatmap generation for every prediction; builds user trust in AI-driven decisions in agricultural context.

4. **Test-Time Augmentation (TTA):** 6-pass averaging improves robustness to image variations; validates to 95.76% accuracy.

5. **Edge-Ready Export:** ONNX model export and quantization strategies for deployment on mobile/IoT hardware.

### 5.2 Deployment & Scalability

- **Flask Web Application:** RESTful API supporting camera capture, file upload, and real-time inference.
- **Containerization:** Dockerfile and docker-compose configuration for consistent deployment across environments.
- **Model Versioning:** Checkpoint management with JSON metadata; easy rollback to stable models.
- **Async Processing:** Optional Celery integration for long-running tasks (Grad-CAM, VQA) to maintain API responsiveness.

---

## 6. LIMITATIONS & FUTURE WORK

### 6.1 Known Limitations

1. **Class Imbalance:** Gray_Blight and other rare diseases show lower recall in finetune experiments; addressed via augmentation but warrants more labeled data.
2. **Artifact Generation Failures:** Some test runs failed to persist Grad-CAM overlays due to file I/O issues; fixed in current codebase with proper directory initialization.
3. **Single-Model Inference:** Current pipeline loads one classification model; ensemble methods could improve robustness but add latency.

### 6.2 Future Enhancements

1. **Semantic Segmentation:** Extend from bounding boxes (detection) to pixel-level lesion boundaries for precise severity quantification.
2. **Mobile App:** Native iOS/Android application bundling ONNX models for offline field deployment.
3. **Federated Learning:** Train models on-device using farm-level data without central data collection, preserving privacy.
4. **Advanced Advisory:** Integrate pest management knowledge graphs; provide region-specific treatment recommendations based on climate/crop stage.
5. **Cross-validation & Ensemble:** Implement k-fold cross-validation and model ensembling for robust performance estimates.

---

## 7. REFERENCES & ARTIFACTS

### Source Code Files
- **Entry Points:** `app/main.py`, `app/app.py` (Flask routes)
- **Classification:** `models/convitx_pretrained.py`
- **Explainability:** `xai/gradcam.py`
- **Advisory:** `chatbot/advisor.py`, `chatbot/knowledge_base.py`
- **Data Preparation:** `prepare_yolo_dataset.py`, `build_merged_dataset.py`, `validate_yolo_dataset.py`

### Model Artifacts
- `models/best_convitx_pretrained.pth` — Primary classification model (94.62% accuracy)
- `models/best_convitx.onnx` — ONNX export for edge inference
- `models/yolo_dragon_best.pt` — YOLOv8 lesion detector
- `models/convitx_pretrained_summary.json` — Training metadata and per-class metrics

### Evaluation Results
- `test_case_results.json` — 10 functional test cases covering upload, classification, detection, advisory
- `results/tc05_rerun_summary.json` — Quality grading test results with confidence scores
- `convitx_pretrained_results.md` — Formatted results table (94.62% accuracy)

### Documentation
- `README.md` — Quick start and feature overview
- `requirements.txt` — Python dependencies
- `dataset_image_inventory.csv` — Comprehensive dataset metadata (image paths, dimensions, label associations)

---

## CONCLUSION

The **Dragon Fruit Disease Detection & Quality Assessment System** demonstrates a practical AI-driven solution for agricultural disease management. By combining a lightweight CNN+ViT hybrid model (94.62% validation accuracy, 18 MB size), YOLOv8 lesion detection, Grad-CAM explanations, and a Flask web interface, the system addresses core constraints of edge deployment and farmer accessibility.

The work successfully integrates multiple ML tasks (classification, detection, quality grading, VQA, explainability) into a cohesive pipeline, validated through systematic testing and comparative evaluation against larger baselines. The achieved **95.76% TTA accuracy** while using **90% fewer parameters** than ResNet50 exemplifies efficient model design for resource-constrained deployment.

Future work focuses on semantic segmentation, mobile apps, and federated learning to extend impact across diverse farming communities.

---

**Project Submission Date:** June 2026  
**Repository:** [GitHub](https://github.com/PrithviKiran791/Dragon_fruit_disease_ai)  
**Main Branch:** Development complete and tested

