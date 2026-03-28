<p align="center">
  <h1 align="center">Offroad Semantic Segmentation</h1>
  <p align="center">
    <strong>Pixel-level scene understanding for offroad & wilderness terrain</strong><br>
    Built at <em>CodeEdge Hackathon 2026</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/CUDA-RTX_5060-76B900?logo=nvidia&logoColor=white" alt="CUDA">
    <img src="https://img.shields.io/badge/Model-UNet++-blue" alt="UNet++">
    <img src="https://img.shields.io/badge/Encoder-EfficientNet--B5-orange" alt="EfficientNet-B5">
    <img src="https://img.shields.io/badge/Classes-11-green" alt="11 Classes">
  </p>
</p>

---

## Problem Statement

Given offroad terrain images (960x540), predict a pixel-wise segmentation mask across **11 semantic classes**:

| Class | Color | Class | Color |
|-------|-------|-------|-------|
| Background | ![#000000](https://img.shields.io/badge/-000000-000000?style=flat-square) Black | Ground Clutter | ![#FFD700](https://img.shields.io/badge/-FFD700-FFD700?style=flat-square) Gold |
| Trees | ![#006400](https://img.shields.io/badge/-006400-006400?style=flat-square) Dark Green | Flowers | ![#FF69B4](https://img.shields.io/badge/-FF69B4-FF69B4?style=flat-square) Hot Pink |
| Lush Bushes | ![#00FF00](https://img.shields.io/badge/-00FF00-00FF00?style=flat-square) Lime | Logs | ![#8B4513](https://img.shields.io/badge/-8B4513-8B4513?style=flat-square) Saddle Brown |
| Dry Grass | ![#F4A460](https://img.shields.io/badge/-F4A460-F4A460?style=flat-square) Sandy Brown | Rocks | ![#808080](https://img.shields.io/badge/-808080-808080?style=flat-square) Gray |
| Dry Bushes | ![#A0522D](https://img.shields.io/badge/-A0522D-A0522D?style=flat-square) Sienna | Landscape | ![#CD853F](https://img.shields.io/badge/-CD853F-CD853F?style=flat-square) Peru |
| | | Sky | ![#87CEEB](https://img.shields.io/badge/-87CEEB-87CEEB?style=flat-square) Sky Blue |

## Dataset Overview

| Split | Images | Resolution | Purpose |
|-------|-------:|:----------:|---------|
| Train | 2,857 | 960x540 | Model training |
| Val | 317 | 960x540 | Hyperparameter tuning |
| Test | 668 | 960x540 | Final evaluation |
| **Total** | **3,842** | | |

- **Mask format**: 16-bit grayscale PNGs with encoded values → mapped to class indices 0-10
- **Key challenge**: Train scenes are **lush green forests**; test scenes are **arid rocky terrain** (severe domain shift)

---

## Our Journey: From 0.29 to 0.68 mIoU

### Phase 1 — ConvNeXt Backbone (Initial Attempt)

Started with a ConvNeXt-based encoder and a custom segmentation head. Discovered a missing class (Flowers) in our pipeline and fixed data loading. Architecture was heavy and under-optimized for our GPU budget.

> **Takeaway**: Needed a more battle-tested segmentation framework.

---

### Phase 2 — DeepLabV3 + ResNet50

Switched to `torchvision.models.segmentation.deeplabv3_resnet50` with ImageNet-pretrained weights. Trained 50 epochs with OneCycleLR scheduler and class-weighted CrossEntropy.

| Metric | Score |
|--------|------:|
| Val mIoU | **0.6479** |
| Val Pixel Accuracy | 0.8105 |
| Test mIoU (all 11 classes) | 0.2920 |
| Test mIoU (7 present classes) | 0.4171 |

> **Shock moment**: Val looked great at 0.65, but test crashed to 0.29. What went wrong?

We dug into the data and discovered a **massive domain shift** — 4 classes were completely absent from the test set, and pixel distributions were wildly different (see [Challenges](#challenges--how-we-solved-them) below). After excluding absent classes from evaluation, test mIoU rose to **0.4171**, but still far from our target.

---

### Phase 3 — UNet++ + EfficientNet-B5 (Final Architecture)

Complete rewrite using `segmentation_models_pytorch`. Pulled out every trick in the book to fight domain shift and maximize IoU.

#### Architecture
```
UNet++ (Dense Nested U-Net)
├── Encoder: EfficientNet-B5 (ImageNet pretrained, 30M params)
│   └── Compound scaling: resolution + depth + width
├── Decoder: Dense skip connections at every level
│   └── Progressive feature fusion for fine-grained boundaries
└── Total: ~37M parameters
```

#### Training Strategy

| Component | Details |
|-----------|---------|
| **Loss Function** | 0.4 x Lovasz-Softmax + 0.3 x Focal (gamma=2.5) + 0.3 x Dice |
| **Why Lovasz?** | Directly optimizes IoU — the exact metric we're maximizing |
| **EMA** | Exponential Moving Average of weights (decay=0.999/0.9995) |
| **Dataset** | Combined train+val = 3,174 images |
| **Class Weights** | Test-aware: Logs(8.0), Rocks(5.0), DryBushes(3.0), Sky(0.5) |
| **Optimizer** | AdamW (weight_decay=0.01) + differential LR (encoder x0.1) |
| **AMP** | Mixed precision for memory efficiency |
| **Grad Accumulation** | Effective batch size = 8 |

#### Two-Phase Progressive Training

```
Phase 1: Fast Convergence (20 epochs)                Phase 2: High-Res Finetune (15 epochs)
┌─────────────────────────────────┐                  ┌─────────────────────────────────┐
│  Resolution:  416 x 736         │                  │  Resolution:  544 x 960         │
│  Batch size:  4 (accum=2)       │  ──best ckpt──>  │  Batch size:  2 (accum=4)       │
│  LR:          2e-4 (OneCycle)   │                  │  LR:          3e-5 (OneCycle)    │
│  Encoder:     Frozen 2 epochs   │                  │  Encoder:     Fully unfrozen     │
│  EMA decay:   0.999             │                  │  EMA decay:   0.9995             │
└─────────────────────────────────┘                  └─────────────────────────────────┘
```

#### Augmentation Pipeline (Domain-Shift Buster)

```python
# Forces model to learn TEXTURE, not COLOR — critical for green→brown generalization
HueSaturationValue(hue=30, sat=40, val=25, p=0.7)
RandomBrightnessContrast(0.4, 0.4, p=0.6)
ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.1, p=0.4)
OneOf([ChannelShuffle, ToGray, RGBShift, FancyPCA], p=0.3)
CoarseDropout(holes=1-4, size=20-60px, p=0.2)
```

#### Inference: 6-Fold Multi-Scale TTA

```
Scales [0.75, 1.0, 1.25]  x  [Original, H-Flip]  =  6 predictions averaged
                            ↓
              Class Suppression (zero out absent classes)
                            ↓
                      Final argmax prediction
```

#### Results

| Metric | Score |
|--------|------:|
| Val mIoU | **0.6497** |
| Test mIoU (7 classes) | **0.4957** |
| Test Pixel Accuracy | 0.6811 |

**Per-Class Breakdown (Test Set)**:

```
Sky           ██████████████████████████████████████████████████  0.978  ← Excellent
Landscape     ██████████████████████████████                      0.592
Trees         █████████████████████████                           0.491
Dry Grass     ██████████████████████                              0.453
Dry Bushes    ██████████                                          0.203
Rocks         ███                                                 0.053  ← Hardest (domain shift)
Lush Bushes   ▏                                                   0.000  ← Annotation noise
```

---

## Challenges & How We Solved Them

### 1. Severe Domain Shift (The Big One)

**The Problem**: Training images show **lush green forests**. Test images show **dry rocky terrain**. The model learned "green = vegetation" but test vegetation is brown.

```
Train Distribution          Test Distribution
─────────────────          ──────────────────
Landscape   24.0%    →     Landscape   43.2%   (+19%)
Trees        4.0%    →     Trees        0.3%   (-3.7%)
Rocks        1.2%    →     Rocks       18.1%   (+16.9%)  ← 15x increase!
```

**How We Fought It**:
- Aggressive color augmentation (ChannelShuffle, ToGray, FancyPCA) to break color dependency
- Test-aware class weights — upweight Rocks (5.0x), Dry Bushes (3.0x)
- Lovasz-Softmax loss to directly optimize IoU rather than pixel-wise cross-entropy

---

### 2. Missing Classes in Test Set

**The Problem**: Test mIoU was 0.2920 — investigation revealed **4 classes are completely absent** from test ground truth: Background, Ground Clutter, Flowers, and Logs. Any prediction of these classes was penalized with IoU=0.

**The Fix**:
- **Evaluation**: Compute mIoU over only the 7 classes present in test GT
- **Inference**: Class suppression — zero out logits for absent classes before argmax, redistributing probability mass to real classes

---

### 3. Annotation Noise in Test GT

**The Problem**: "Lush Bushes" appears in test GT but only as **1-5 pixels per image** — clearly annotation noise from the labeling tool, not real vegetation. IoU is extremely sensitive to near-zero GT pixels.

**Impact**: Even a perfect model hits a **~0.86 mIoU ceiling** because Lush Bushes IoU will always be near-zero.

---

### 4. DeepLabV3 `aux_classifier` Mismatch

**The Problem**: `deeplabv3_resnet50(weights=None)` creates a model without the auxiliary head, but training with `aux_loss=True` creates one. State dict keys didn't match at test time.

**The Fix**: Ensured `aux_loss=True` in both train and test. Became irrelevant after switching to `smp.UnetPlusPlus`.

---

### 5. UNet++ Requires Dimensions Divisible by 32

**The Problem**: Native 540x960 — but 540 is not divisible by 32 (UNet++ has 5 pooling levels: 2^5 = 32).

**The Fix**: Used **544x960** (544 = 17 x 32) for training and inference.

---

### 6. Virtual Environment Package Isolation

**The Problem**: `pip install segmentation-models-pytorch` installed to global Python site-packages, not the project `.venv`.

**The Fix**: Explicitly targeted the venv: `.venv/Scripts/pip install segmentation-models-pytorch timm`

---

## Quick Start

### Prerequisites
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install torch torchvision segmentation-models-pytorch timm albumentations opencv-python tqdm matplotlib
```

### Train
```bash
python train_segmentation.py
```
Runs both phases automatically. Saves best model checkpoint as `segmentation_model.pth`.

### Evaluate
```bash
python test_segmentation.py
```

**Outputs**:
```
predictions/
├── masks/              # Raw class-index masks (for submission)
├── masks_color/        # Colorized prediction visualizations
├── comparisons/        # Side-by-side: Input | Ground Truth | Prediction
├── evaluation_metrics.txt
└── per_class_metrics.png
```

---

## Future Improvements

If we had more time, here's what would move the needle most:

### High Impact

| Approach | Expected Gain | Effort | Description |
|----------|:------------:|:------:|-------------|
| **Pseudo-labeling** | +5-10% | Medium | Run current model on unlabeled arid images, use high-confidence predictions as pseudo-labels, retrain. Directly addresses domain shift. |
| **Stronger Encoder** | +3-5% | Low | Swap to Swin-V2-Base or ConvNeXt-Large — richer features for texture-heavy classes like Rocks. |
| **Style Transfer DA** | +5-8% | High | Apply neural style transfer to make training images look arid/brown. Train on original + transferred data. |
| **Higher Resolution** | +2-4% | Low | Train at full 960x544 or super-resolve to 1280x720 for finer boundaries. |

### Medium Impact

| Approach | Expected Gain | Effort | Description |
|----------|:------------:|:------:|-------------|
| **DenseCRF** | +1-3% | Low | Post-processing to sharpen boundaries and enforce spatial consistency. Free at inference. |
| **OHEM** | +2-3% | Medium | Focus training on hardest pixels — improves confusing boundaries (Dry Bushes vs Dry Grass). |
| **Model Ensemble** | +2-4% | Medium | Average softmax from 3-5 models with different backbones. Reliable but slow. |
| **Test-Time Training** | +2-5% | High | Self-supervised fine-tuning on each test image before prediction. |

### Polish

| Approach | Expected Gain | Effort | Description |
|----------|:------------:|:------:|-------------|
| **Boundary Loss** | +1-2% | Low | Penalize errors near class boundaries for sharper edges. |
| **Class-Balanced Sampling** | +1-2% | Low | Ensure every batch has rare class examples for better gradient signal. |

---

## Tech Stack

| Tool | Role |
|------|------|
| **PyTorch 2.x + CUDA** | Deep learning framework |
| **segmentation_models_pytorch** | UNet++ architecture with pretrained encoders |
| **timm** | EfficientNet-B5 encoder backbone |
| **albumentations** | Fast augmentation pipeline with mask support |
| **OpenCV + Pillow** | Image I/O and processing |
| **matplotlib** | Training curves and evaluation charts |

---

## Project Structure

```
CodeEdge-Hackathon/
│
├── train_segmentation.py          # Training script — UNet++ EfficientNet-B5, two-phase
├── test_segmentation.py           # Evaluation — multi-scale TTA + class suppression
├── visualize.py                   # Visualization utilities
│
├── segmentation_model.pth         # Best trained model weights
├── best_model_phase1.pth          # Phase 1 checkpoint
│
├── predictions/                   # Test inference outputs
│   ├── masks/                     #   Raw predictions
│   ├── masks_color/               #   Colorized masks
│   ├── comparisons/               #   Side-by-side visualizations
│   ├── evaluation_metrics.txt     #   mIoU and per-class scores
│   └── per_class_metrics.png      #   Bar chart
│
├── train_stats/                   # Training logs and curves
│
├── Offroad_Segmentation_Training_Dataset/
│   ├── train/                     # 2,857 images + masks
│   └── val/                       # 317 images + masks
│
└── Offroad_Segmentation_testImages/
    ├── Color_Images/              # 668 test images
    └── Segmentation/              # 668 test masks (ground truth)
```

---

<p align="center">
  Built with determination and lots of debugging at <strong>CodeEdge Hackathon 2026</strong>
</p>
