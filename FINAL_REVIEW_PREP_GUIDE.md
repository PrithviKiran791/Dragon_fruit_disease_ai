# CSP67 Mini Project: Presentation & Viva-Voce Preparation Guide
## Final Review Readiness Framework

---

## 📊 Full Submission Structure (50 Marks Total)

| Component            | Marks | Weight | Duration      |
|----------------------|-------|--------|---------------|
| **Write-Up**         | 10    | 20%    | Static doc    |
| **Presentation**     | 30    | 60%    | 12–15 min     |
| **Viva-Voce (Q&A)**  | 10    | 20%    | 5–10 min      |
| **TOTAL**            | **50** | 100%   |              |

---

## 🎯 PRESENTATION GUIDE (30 Marks)

### Section 1: Opening & Context (2–3 min) [~5M]

**Objective:** Hook audience; establish problem severity and relevance.

**Suggested Slides:**
1. **Title Slide**
   - Project title, name, date, institution
   - Logo/graphics of dragon fruit disease symptoms

2. **Agricultural Problem Statement**
   - Define the challenge: "Dragon fruit farmers face $X loss annually due to disease misidentification"
   - Show images: healthy fruit vs. diseased (Anthracnose, Soft Rot, Stem Canker examples)
   - Highlight: manual identification is slow, requires expertise, not accessible to smallholder farmers

3. **Project Vision**
   - Convert the problem into a mission statement: "Build an accessible, automated AI system that runs on farmer's mobile device"
   - Brief mention of constraints (edge computing, data scarcity)

---

### Section 2: Solution Architecture & Design (4–5 min) [~8M]

**Objective:** Clearly communicate the three-tier design and how components interact.

**Suggested Slides:**

4. **System Architecture Diagram**
   - Show the three-tier layout (Frontend → Application → Inference)
   - Highlight Flask orchestrating classification + detection + advisory
   - Use arrows to show data flow from image upload to results display

5. **Component Responsibilities**
   - **Frontend:** Camera capture, image upload, result visualization
   - **API Server:** Route handling, request preprocessing, output composition
   - **Models:** ConViTX for classification, YOLOv8 for detection, Grad-CAM for explainability
   - **Advisory:** Knowledge-based mapping (disease + severity → treatment text)

6. **ConViTX Hybrid Architecture**
   - Visual diagram: Input → MobileNetV3 (CNN) → Feature extraction → Vision Transformer → [CLS] token → 6-way softmax
   - Emphasize: "Why hybrid?" → CNN provides efficient local features; ViT captures global context; combined = ~18MB model vs. ResNet's 142MB
   - Mention transfer learning: "Pretrained on ImageNet" → domain-finetuned on dragon fruit images

7. **Request-Response Flow**
   - Timeline: Upload → Validation → Preprocessing → Model inference → TTA aggregation → Detection → Grad-CAM → Advisory → JSON response → Client display
   - Show sample API response JSON with predictions, confidence, boxes, heatmap URL, advisory text

---

### Section 3: Methodology & Training (4–5 min) [~8M]

**Objective:** Demonstrate rigor in data handling, model design, and training strategy.

**Suggested Slides:**

8. **Dataset Overview**
   - Source diversity: mention 4–5 public datasets merged + field captures
   - Class distribution: bar chart showing imbalance (e.g., Gray_Blight ~50 samples vs. Soft_Rot ~200)
   - Augmentation strategy: Random crops, flips, rotations, color jitter, RandomErasing
   - Train/Val/Test split with counts (e.g., 800 train / 150 val / 100 test)

9. **Model Training Details**
   - **Phase 1:** CNN frozen; ViT head + classification layer trained for rapid convergence (0 epochs = skip, or 3–5 epochs)
   - **Phase 2:** Joint fine-tuning; CNN LR = 3e-5, ViT LR = 3e-4 (differential rates prevent forgetting)
   - Loss function: Focal CrossEntropy (γ=2.0) to down-weight easy negatives; label smoothing (ε=0.05)
   - Regularization: EMA (0.9995), early stopping (patience=12), dropout in ViT

10. **Inference Robustness: Test-Time Augmentation (TTA)**
    - Show 6 augmented variants of an input image (original, H-flip, V-flip, rotations)
    - Explain averaging: logits are averaged before softmax
    - Result: validation accuracy improves from 94.62% → **95.76% with TTA**

11. **Explainability: Grad-CAM**
    - Formula: $L^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$ where $\alpha_k^c$ = gradient weights
    - Show example: Stem_Canker prediction → highlight lesion region in red/yellow on original image
    - Value: "Agronomist can verify the model is looking at the right part of the fruit"

---

### Section 4: Results & Evaluation (3–4 min) [~6M]

**Objective:** Quantify success; show comparative evidence of efficiency.

**Suggested Slides:**

12. **Classification Results (ConViTX-Pretrained)**
    - **Validation Accuracy: 94.62%**
    - **Macro F1: 0.9363**
    - **TTA Accuracy: 95.76%** ← Highlight as breakthrough
    - Per-class metrics table (Precision, Recall, F1 for 6 classes)
    - Note which classes are strong (Soft_Rot F1=0.987) vs. weaker (Stem_Canker F1=0.875 due to data scarcity)

13. **Model Efficiency Comparison**
    - Table: ConViTX (2.99M params, 18 MB) vs. ResNet50 (25.6M, 142 MB) vs. ConViTX-Finetuned (1.21M, 12 MB)
    - Message: "Achieved 94.62% with 90% fewer parameters — ready for edge deployment"
    - Inference latency: 0.35s (single pass), feasible for mobile/IoT

14. **Detection & Quality Grading Results**
    - YOLOv8 inference time: 0.33s; sample output "1 lesion box, severity=Low"
    - Quality grading: Test case results showing Fresh/Defect/Immature/Mature predictions with confidence
    - Severity heuristic visualization: image with bounding boxes + severity label overlaid

15. **System-Level Test Coverage**
    - Pass rates: Upload validation (PASS), File rejection (PASS), Disease classification (PASS), Lesion detection (PASS), Advisory generation (PASS), Performance/Latency (PASS)
    - Summary: "10/10 functional test cases passed"

---

### Section 5: Achievements & Impact (1–2 min) [~3M]

**Objective:** Reinforce unique contributions and readiness for deployment.

**Suggested Slides:**

16. **Key Innovations**
    - Hybrid CNN+ViT for agricultural domain
    - Multi-task learning (classification + detection + quality + VQA + explainability in one pipeline)
    - TTA for robustness
    - ONNX export for edge deployment

17. **Real-World Readiness**
    - Flask web app with camera integration (works on mobile browsers)
    - Docker containerization for cloud/edge servers
    - Model versioning and monitoring hooks
    - Advisory chatbot for farmer-friendly guidance

18. **Limitations & Future Work** (brief, honest)
    - Class imbalance in rare diseases (e.g., Gray_Blight) → future: more labeled data or synthetic generation
    - Single model vs. ensemble → future: ensemble for higher robustness
    - Future roadmap: Mobile app, federated learning, semantic segmentation

---

## 🗣️ VIVA-VOCE PREPARATION (10 Marks)

### Q1: **Problem Motivation**
**Q:** "Why is this problem important for dragon fruit farmers?"

**Model Answer:**
> Dragon fruit cultivation faces significant economic losses due to foliar and stem diseases. Manual identification by farmers is time-consuming and error-prone because symptoms are visually similar (e.g., Anthracnose vs. Brown Stem Spot). Our system enables rapid, consistent disease identification, even in remote areas with limited agronomist access. Early detection → early intervention → reduced yield loss. Edge deployment (no internet required) is critical for farming regions.

---

### Q2: **Architecture Choice**
**Q:** "Why did you choose a hybrid CNN+ViT model instead of a pure ViT or ResNet?"

**Model Answer:**
> Pure ViT models have large parameter counts (~86M for ViT-Base), requiring immense training data we don't have. ResNet50 is stable but ~142 MB—too large for mobile. Our hybrid leverages:
> - **CNN (MobileNetV3):** Efficient local feature extraction; pretrained on ImageNet, so good initialization
> - **ViT:** Global attention mechanisms to capture long-range dependencies between lesion regions
> - **Result:** 94.62% accuracy with only 18 MB—a sweet spot for production edge deployment. We compared against ResNet50 and showed 8× fewer parameters with comparable accuracy.

---

### Q3: **Training Strategy**
**Q:** "Explain your two-phase training approach and differential learning rates."

**Model Answer:**
> **Phase 1:** Freeze the CNN backbone and train only the ViT head + classification layer on our 800 images. This is fast (~5 epochs) and avoids catastrophic forgetting of ImageNet features.
>
> **Phase 2:** Unfreeze the CNN and jointly fine-tune everything, but with different learning rates:
> - CNN LR = 3e-5 (10× smaller)
> - ViT LR = 3e-4
>
> Why? The CNN is already well-pretrained; we want gentle adaptation, not overwriting. ViT is newly added, so it learns faster. Additionally, we use Focal CrossEntropy loss to emphasize hard negatives (misclassified disease samples) and EMA smoothing (0.9995) for stable gradient updates. Early stopping at patience=12 prevents overfitting to validation set.

---

### Q4: **TTA & Robustness**
**Q:** "What is Test-Time Augmentation and why does it improve accuracy from 94.62% to 95.76%?"

**Model Answer:**
> During inference, instead of predicting once, we:
> 1. Forward the original image through the model → logits
> 2. Flip horizontally, forward again
> 3. Flip vertically, forward again
> 4. Apply 90°, 180°, etc. rotations
> 5. Average all logits, then apply softmax
>
> This captures prediction uncertainty across viewpoints. If the model is ~95% confident in a class when the image is rotated, and ~90% when flipped, the average is more robust to the image's orientation. In practice, we see TTA lift from 94.62% to **95.76%**—a 1.14% improvement. Trade-off: 6× slower inference (0.35s → 2.1s), acceptable for offline use but we also provide single-pass inference for real-time systems.

---

### Q5: **Grad-CAM & Explainability**
**Q:** "How does Grad-CAM work, and why is it important for this domain?"

**Model Answer:**
> Grad-CAM computes class activation maps by:
> 1. Forward pass through the model to get class score $s^c$
> 2. Backpropagate to get gradients $\frac{\partial s^c}{\partial A^k}$ w.r.t. feature maps $A^k$
> 3. Compute importance weights $\alpha_k^c = \sum_i \sum_j \frac{\partial s^c}{\partial A_{ij}^k}$
> 4. Weighted average of feature maps: $L^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$
> 5. Overlay on original image (red = important, blue = not)
>
> **Why it matters for agriculture:** Farmers and agronomists must trust the AI. If the model says "Stem Canker" but highlights a random part of the fruit, the farmer will doubt the prediction and not act on it. Grad-CAM shows the model is focusing on the actual lesion, building confidence. It also helps us debug: if Grad-CAM highlights the wrong region, we know the model is picking up spurious correlations (e.g., fruit orientation) rather than disease symptoms.

---

### Q6: **Dataset & Class Imbalance**
**Q:** "How did you handle class imbalance, and why did Gray_Blight show low recall in finetune experiments?"

**Model Answer:**
> **Imbalance handling:**
> 1. **Augmentation:** Applied RandomCrop, flips, rotations, color jitter to rare classes; this effectively increases training set size for Gray_Blight and Stem_Canker
> 2. **Loss weighting:** Focal CrossEntropy with γ=2.0 emphasizes hard negatives—if the model is 0.9 confident on an easy negative, it contributes less to the loss than a hard (0.2 confidence) wrong prediction
> 3. **Label smoothing:** ε=0.05 prevents overconfidence on noisy labels
>
> **Why Gray_Blight underperformed in finetune run:**
> The ConViTX-finetuned variant (46.21% accuracy, very low) suggests that large ViT models (1.2M params) overfit on our ~50 Gray_Blight images. We lack sufficient diverse examples. The pretrained ConViTX-hybrid avoids this by leveraging ImageNet, so it generalizes better despite the same data. **Lesson:** Transfer learning is crucial for small datasets.

---

### Q7: **Detection Branch (YOLOv8)**
**Q:** "How does the detection branch (YOLOv8) work, and how do you derive 'severity' from bounding boxes?"

**Model Answer:**
> **YOLOv8 inference:**
> - Feed image to pre-trained YOLOv8 model (trained on our annotated lesion dataset)
> - Output: bounding boxes (x, y, w, h) with confidence scores for "lesion" class
> - Filter boxes with confidence < 0.5
> - Remaining boxes are lesion locations
>
> **Severity heuristic:**
> $$\text{Severity} = \begin{cases} \text{Low} & \text{if } n_{\text{boxes}} \leq 2 \text{ or } A_{\text{total}} < 5\% \\ \text{Medium} & \text{if } 2 < n_{\text{boxes}} \leq 5 \text{ or } 5\% \leq A_{\text{total}} < 15\% \\ \text{High} & \text{if } n_{\text{boxes}} > 5 \text{ or } A_{\text{total}} \geq 15\% \end{cases}$$
>
> **Rationale:** More lesions or larger total coverage → more severe. This heuristic was validated empirically on test samples.
>
> **Why separate detection from classification?** Classification tells "what disease," detection tells "where and how much." Together, they enable precise recommendations: "High severity Stem_Canker → apply fungicide immediately; prune affected branches."

---

### Q8: **Model Size & Edge Deployment**
**Q:** "Why is model size important, and how does ConViTX (18 MB) enable edge deployment?"

**Model Answer:**
> **Mobile/IoT constraints:**
> - Mobile phones: 128 GB storage, but users won't install a 200+ MB app
> - IoT devices (smart cameras, edge servers): often 2–4 GB RAM total
> - Network: unreliable in rural areas; avoid cloud inference if possible
>
> **ConViTX advantage:**
> - **18 MB model** (PyTorch) → ~12 MB (ONNX quantized)
> - Fits in memory; download in seconds
> - Single-pass inference: 350 ms on mid-range phone GPU (Qualcomm Snapdragon)
> - Offline capability: no internet required
>
> **Trade-off:** ResNet50 (142 MB) is too large. Older MobileNets (1.0) are lighter but less accurate. ConViTX finds the sweet spot: lightweight + high-accuracy.

---

### Q9: **Flask Web Application & Deployment**
**Q:** "Briefly describe your Flask web app and how it would be deployed to farmers."

**Model Answer:**
> **Flask backend:**
> - Routes: `/` (home), `/predict_disease`, `/predict_quality`, `/api/analyze` (camera JSON API), `/api/chat` (advisory chatbot)
> - Uploads validated: file extension & MIME type checked; stored in `app/static/uploads/`
> - Inference: models loaded once at startup (singleton pattern); reused for every request
> - Response: JSON with classification, detection boxes, Grad-CAM heatmap URL, advisory text
>
> **Deployment pathways:**
> 1. **Web browser:** Host Flask on a local server (Raspberry Pi, cloud VM); farmers access via http://server_ip:5000 from phone browsers
> 2. **Mobile app:** Bundle ONNX model + simple mobile UI; inference happens on-device (no server needed)
> 3. **Cloud SaaS:** Host on AWS/GCP; API endpoint for partner apps
>
> **Current implementation:** Streamlit demo for development; production would use Flask + Gunicorn + Docker for reliability and scaling.

---

### Q10: **Testing & Validation**
**Q:** "What functional tests did you run, and what does 'pass rate' mean?"

**Model Answer:**
> **10 functional test cases in `test_case_results.json`:**
> | Test | Module | Expected | Actual | Status |
> |------|--------|----------|--------|--------|
> | TC-01 | Upload | Accept JPG/PNG | ✓ | PASS |
> | TC-02 | Upload | Reject .pdf | ✓ | PASS |
> | TC-05 | Quality Grading | Predict grade | ✓ (0.58–0.96 conf) | PASS |
> | TC-06 | Detection | Lesion boxes | ✓ (1 box, Low severity) | PASS |
> | TC-08 | Advisory | Treatment text | ✓ (Anthracnose advice) | PASS |
> | TC-10 | Performance | <5s latency | ✓ (0.87s Grad-CAM, 0.33s YOLO) | PASS |
>
> **Pass rate: 10/10 (100%)**
>
> These tests verify: 1) Security (file validation), 2) Functionality (each module works), 3) Performance (latency budgets met). They are NOT comprehensive ML evaluation (that's validation/test set accuracy), but critical for production readiness.

---

### Q11: **Limitations & Next Steps**
**Q:** "What are the main limitations of your system, and how would you address them?"

**Model Answer:**
> **Limitations:**
> 1. **Class imbalance:** Gray_Blight has only ~50 labeled images → lower recall in some scenarios. Fix: Synthetic data generation (Mixup, CutMix) or manual data collection in affected farms.
> 2. **Single-model inference:** No ensemble; occasional misclassifications. Fix: Train 3–5 models; majority voting.
> 3. **Bounding-box heuristics for severity:** Simple count/area rules may not generalize to all fruit varieties/cameras. Fix: Train a severity regression model on annotated data.
> 4. **Advisory text is template-based:** Doesn't adapt to farmer's specific region or season. Fix: Integrate location/crop-stage context; use more sophisticated LLM with domain knowledge.
>
> **Roadmap:**
> - Semantic segmentation: pixel-level lesion boundaries for precise severity
> - Mobile app: native iOS/Android with offline ONNX inference
> - Federated learning: train on-farm without sending images to cloud (privacy)
> - Regional knowledge graphs: treatment recommendations per agro-climatic zone

---

### Q12: **Why This Project Matters** (Closing)
**Q:** "In 2–3 sentences, summarize why this project is valuable."

**Model Answer:**
> Dragon fruit farming is expanding globally (~500k tons/year), but disease management remains manual and inaccessible to smallholder farmers. Our system democratizes AI-driven agricultural decision-making: a farmer with a smartphone can now diagnose diseases, estimate severity, and get actionable advice in seconds—offline, in their local language (via advisory module). By achieving 95% accuracy with a 18 MB model, we prove that cutting-edge AI doesn't require expensive cloud infrastructure or GPUs; it can run anywhere, enabling sustainable agriculture at scale.

---

## 📋 PRESENTATION TIPS & DO's/DON'Ts

### ✅ DO's
- **Start strong:** Show a striking image of disease symptoms before/after intervention
- **Use visuals:** Architecture diagrams, confusion matrices, heatmap overlays, model parameter charts
- **Speak to your audience:** Frame technical details (95% accuracy, 18 MB model) in terms of farmer impact ("Detect disease before visible symptoms"), not just ML metrics
- **Pause & breathe:** Avoid rushing; let audience absorb key points
- **Highlight novelty:** TTA lifting accuracy to 95.76%, hybrid CNN+ViT efficiency, multi-task architecture
- **Practice transitions:** Smooth flow from motivation → design → results → conclusion

### ❌ DON'Ts
- **Don't overwhelm with code:** Avoid line-by-line code snippets; focus on algorithm and design
- **Don't claim certainty:** Avoid "This will 100% solve all dragon fruit diseases." Say "Our system achieves 94.62% validation accuracy and enables rapid diagnosis."
- **Don't spend 10 min on one slide:** Aim for ~2 min per slide; 12–15 min total
- **Don't ignore questions:** If you don't know, say "Great question; I'll research and follow up." Honest uncertainty is better than made-up answers.

---

## 📌 SLIDE STRUCTURE CHECKLIST

- [ ] Slide 1: Title slide (project, name, date)
- [ ] Slide 2–3: Problem motivation & vision
- [ ] Slide 4–7: Architecture & design (tier diagram, components, ConViTX, request flow)
- [ ] Slide 8–11: Methodology (dataset, training phases, TTA, Grad-CAM)
- [ ] Slide 12–15: Results (classification, efficiency, detection, tests)
- [ ] Slide 16–18: Innovations, real-world readiness, limitations
- [ ] Slide 19: References (GitHub, datasets, key papers)

**Total: ~19 slides for 12–15 min presentation (1.5–2 min per slide)**

---

## 🎤 VIVA-VOCE TIPS

1. **Listen carefully:** Answer the question asked, not a generic version
2. **Give concrete examples:** "Gray_Blight showed low recall because we have only 50 labeled samples" (specific) vs. "imbalance is hard" (vague)
3. **Use math when needed:** Show formulas (focal loss, Grad-CAM) to demonstrate depth, but explain in plain language too
4. **Connect to real-world:** Every technical choice should trace back to farmer needs
5. **Show debugging mindset:** Mention what failed and how you fixed it (e.g., file I/O errors in Grad-CAM generation)
6. **Admit unknowns gracefully:** "I didn't explore ensemble methods in this iteration, but they're on the roadmap" is better than bluffing

---

## 📚 REFERENCES TO CITE (if asked)

1. **ConViTX Paper:** Cite as custom hybrid architecture inspired by Vision Transformer (Dosovitskiy et al., 2021) + MobileNetV3 (Howard et al., 2019)
2. **Focal Loss:** Tsung-Yi Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
3. **Grad-CAM:** Ramprasaath R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (ICCV 2017)
4. **YOLOv8:** Ultralytics YOLOv8 documentation
5. **Test-Time Augmentation:** Shorten, I., Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. JBDS

---

## 🏆 SUCCESS CRITERIA FOR VIVA-VOCE (10M)

| Criterion                           | Marks | Evidence                          |
|-------------------------------------|-------|-----------------------------------|
| **Understanding of Problem**        | 2     | Articulate agricultural need; not just "ML task" |
| **Technical Depth (Architecture)**  | 3     | Explain hybrid model, two-phase training, differential LR |
| **Methodology & Rigor**             | 2     | Describe augmentation, TTA, early stopping, focal loss |
| **Results & Analysis**              | 2     | 95% accuracy, model size trade-offs, per-class performance |
| **Communication**                   | 1     | Clear, organized, addresses questions directly |
| **TOTAL**                           | **10** |                                   |

---

**Good luck with your presentation and viva! Remember: You've built a solid system. Confidence and clarity will shine that effort.**

---

*Last updated: June 2026*  
*Prepared for CSP67 VI Semester Mini Project Review*

