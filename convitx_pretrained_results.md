# ConViTX-Pretrained Results

*Hybrid CNN (MobileNetV3-Small, pretrained) + ViT  |  Params: 2,993,254 trainable*


## Best Validation Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | **91.68%** |
| **Macro F1** | **0.9094** |
| Best Epoch   | 27 |

## Per-Class Metrics

| Class | Precision | Recall | F1 |
|-------|----------:|-------:|---:|
| Anthracnose | 0.7692 | 1.0000 | 0.8696 |
| Brown_Stem_Spot | 0.9744 | 0.8636 | 0.9157 |
| Gray_Blight | 0.8553 | 1.0000 | 0.9220 |
| Healthy | 0.9930 | 0.9156 | 0.9527 |
| Soft_Rot | 0.9823 | 0.9911 | 0.9867 |
| Stem_Canker | 0.8100 | 0.8100 | 0.8100 |

## Training Config

- CNN backbone: MobileNetV3-Small (ImageNet pretrained)
- Phase 1: CNN frozen for `0` epochs → only ViT+Head trained
- Phase 2: Full joint fine-tuning (CNN LR×0.1)
- LR=5e-05  |  Batch=32  |  Epochs=30
- Augmentation: RandomCrop+Flip+Rotate+ColorJitter+RandomErasing (NO MixUp/CutMix)
- Loss: Focal CE (γ=2.0) + label smoothing 0.05
- EMA decay: 0.9995

## Artifacts

- `models/best_convitx_pretrained.pth`
- `models/convitx_pretrained_curves.png`
- `models/convitx_pretrained_cm.png`
- `models/convitx_pretrained_summary.json`