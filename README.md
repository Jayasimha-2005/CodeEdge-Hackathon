# Offroad Semantic Segmentation

Semantic segmentation for offroad/wilderness scenes — built during CodeEdge Hackathon.

## Problem Statement

Segment offroad terrain images (960x540) into **11 classes**: Background, Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Flowers, Logs, Rocks, Landscape, and Sky.

## Dataset

| Split | Images | Source |
|-------|--------|--------|
| Train | 2,857 | `Offroad_Segmentation_Training_Dataset/train/` |
| Val | 317 | `Offroad_Segmentation_Training_Dataset/val/` |
| Test | 668 | `Offroad_Segmentation_testImages/` |

- **Image resolution**: 960x540 RGB
- **Mask format**: 16-bit grayscale PNGs with encoded class values (0, 100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000)

## Our Journey

### Iteration 1: ConvNeXt Backbone (Initial Attempt)

Started with a ConvNeXt-based encoder with custom segmentation head. Added the Flowers class (initially missing), experimented with transforms and DataLoader configurations.

**Result**: Moderate val performance, but the architecture was heavy and not well-suited for the task.

### Iteration 2: DeepLabV3-ResNet50

Switched to `torchvision.models.segmentation.deeplabv3_resnet50` with ImageNet-pretrained weights. Trained for 50 epochs with OneCycleLR, class-weighted CrossEntropy loss, and basic augmentations.

**Results**:
| Metric | Value |
|--------|-------|
| Val mIoU | 0.6479 |
| Val Pixel Accuracy | 0.81 |
| Test mIoU (all 11 classes) | 0.2920 |
| Test mIoU (7 present classes) | 0.4171 |

### Iteration 3: UNet++ with EfficientNet-B5 (Final)

Complete rewrite using `segmentation_models_pytorch`. This was the big push to maximize IoU.

**Architecture**: UNet++ with EfficientNet-B5 encoder (ImageNet pretrained, ~37M params)

**Training strategy**:
- **Triple Loss**: 0.4x Lovasz-Softmax + 0.3x Focal + 0.3x Dice
- **EMA** (Exponential Moving Average) of model weights
- **Combined dataset**: train + val = 3,174 images
- **Two-phase training**:
  - Phase 1: 416x736, batch=4, 20 epochs, lr=2e-4, encoder frozen for first 2 epochs
  - Phase 2: 544x960 (near-native), batch=2, 15 epochs, lr=3e-5
- **Aggressive augmentation**: HueSaturationValue, ChannelShuffle, ToGray, ColorJitter, FancyPCA, CoarseDropout
- **Test-aware class weights**: Heavy weight on rare/hard classes (Logs: 8.0, Rocks: 5.0, Dry Bushes: 3.0)
- **Differential LR**: encoder x0.1, decoder x1.0
- **Gradient accumulation**: effective batch size = 8
- **AMP** (mixed precision) for memory efficiency

**Inference**: Multi-scale TTA (3 scales x 2 flips = 6-fold averaging) + class suppression for absent test classes.

**Results**:
| Metric | Value |
|--------|-------|
| Val mIoU | 0.6497 |
| Test mIoU (7 classes) | 0.3957 |
| Test Pixel Accuracy | 0.6811 |

**Test Per-Class IoU**:
| Class | IoU | Notes |
|-------|-----|-------|
| Sky | 0.9779 | Excellent |
| Landscape | 0.5918 | Decent |
| Trees | 0.4906 | Moderate |
| Dry Grass | 0.4531 | Moderate |
| Dry Bushes | 0.2032 | Poor |
| Rocks | 0.0529 | Very poor |
| Lush Bushes | 0.0002 | Annotation noise |

## Problems We Faced & How We Solved Them

### 1. Massive Domain Shift (Train vs Test)

**Problem**: Training scenes are lush, green forests. Test scenes are arid, rocky, brown terrain. This caused a catastrophic drop from 0.65 val mIoU to 0.29 test mIoU.

**Discovery**: Analyzed pixel distributions and found:
- **Rocks**: 1.2% of train pixels -> 18.1% of test pixels (15x increase)
- **Landscape**: 24% of train -> 43% of test
- **Trees**: 4% of train -> 0.27% of test
- The model had barely seen Rocks during training, but Rocks was the 2nd largest class in test.

**Solution**: Aggressive color augmentation (ChannelShuffle, ToGray, HueSaturationValue, FancyPCA) to force the model to learn texture/structure rather than color. Test-aware class weights to upweight underrepresented classes.

### 2. Missing Classes in Test Set

**Problem**: Initial test mIoU was 0.2920 — shockingly low. Investigation revealed 4 classes are **completely absent** from the test ground truth: Background, Ground Clutter, Flowers, and Logs.

**Solution**:
- **Evaluation fix**: Skip absent classes when computing mIoU (7-class mIoU instead of 11-class)
- **Inference fix**: Class suppression — zero out logits for absent classes before argmax, so those predictions are redistributed to classes that actually exist in test

### 3. Lush Bushes Annotation Noise

**Problem**: "Lush Bushes" appears in test GT but only as 1-5 pixels per image in most samples — clearly annotation noise, not real vegetation. This imposes a ~0.14 mIoU error floor (IoU is extremely sensitive when GT has near-zero pixels).

**Impact**: Even a perfect model cannot achieve >0.86 test mIoU due to this noise.

### 4. `aux_classifier` Error with DeepLabV3

**Problem**: Test script crashed with `TypeError` when loading DeepLabV3 — `deeplabv3_resnet50(weights=None)` doesn't create the auxiliary classifier head, causing a state_dict mismatch.

**Solution**: Added `aux_loss=True` to model creation. Later became irrelevant when we switched to `smp.UnetPlusPlus`.

### 5. Virtual Environment Package Issues

**Problem**: `pip install segmentation-models-pytorch` installed to global Python, not the project's `.venv`.

**Solution**: Used `.venv/Scripts/pip install segmentation-models-pytorch timm` to target the correct environment.

### 6. Resolution Must Be Divisible by 32

**Problem**: Native resolution 540x960 — 540 is not divisible by 32, which UNet++ requires (5 levels of 2x downsampling = 2^5 = 32).

**Solution**: Used 544x960 (544 = 17 x 32) for Phase 2 training and TTA inference.

## How to Use

### Setup
```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install torch torchvision segmentation-models-pytorch timm albumentations opencv-python tqdm matplotlib
```

### Training
```bash
python train_segmentation.py
```
Runs Phase 1 (low-res, 20 epochs) then Phase 2 (high-res, 15 epochs). Saves best model as `segmentation_model.pth`.

### Testing
```bash
python test_segmentation.py
```
Runs multi-scale TTA inference on test images. Outputs:
- `predictions/masks/` — raw class-index masks
- `predictions/masks_color/` — colorized prediction masks
- `predictions/comparisons/` — side-by-side comparisons (input | GT | prediction)
- `predictions/evaluation_metrics.txt` — mIoU and per-class IoU
- `predictions/per_class_metrics.png` — bar chart of per-class IoU

## Future Improvements (If More Time Available)

### High Impact

1. **Pseudo-labeling / Self-training on Test Domain**
   - Run current model on unlabeled arid/rocky images, use high-confidence predictions as pseudo-labels, retrain. This directly addresses the domain shift problem.

2. **Stronger Encoder (ConvNeXt-Large or Swin-V2)**
   - EfficientNet-B5 is good but larger vision transformer backbones (Swin-V2-Base, ConvNeXt-Large) extract richer features, especially for texture-heavy classes like Rocks and Dry Bushes.

3. **Domain Adaptation (Style Transfer)**
   - Apply style transfer to make training images look arid/brown like test images. Train on both original + style-transferred data. Directly bridges the green-to-brown domain gap.

4. **Higher Resolution Training**
   - Train at full native 960x544 or even super-resolve to 1280x720. Larger resolution = more detail for small objects (Logs, Ground Clutter) and better boundary precision.

### Medium Impact

5. **CRF Post-processing**
   - Apply DenseCRF on model outputs to sharpen boundaries and enforce spatial consistency. Typically +1-3% mIoU for free at inference time.

6. **Online Hard Example Mining (OHEM)**
   - Focus training on the hardest pixels (highest loss). Forces the model to improve on confusing boundaries between similar classes (Dry Bushes vs Dry Grass, Rocks vs Landscape).

7. **Model Ensemble**
   - Train 3-5 models with different backbones (EfficientNet-B5, ResNet101, ConvNeXt-Base) and average their softmax outputs. Ensembles typically add +2-4% mIoU.

8. **Test-Time Training (TTT)**
   - Fine-tune the model on each test image using a self-supervised loss (e.g., rotation prediction) before making predictions. Adapts features to each image's specific domain.

### Lower Impact (Polish)

9. **Boundary-Aware Loss**
   - Add a boundary loss term that penalizes errors near class boundaries. Improves IoU for classes with complex shapes.

10. **Class-Balanced Sampling**
    - Instead of random sampling, ensure each batch contains examples of all rare classes. Better gradient signal for underrepresented classes.

## Tech Stack

- **PyTorch** + **CUDA** (RTX 5060 Laptop, 8GB VRAM)
- **segmentation_models_pytorch** — UNet++ architecture
- **timm** — EfficientNet-B5 encoder
- **albumentations** — augmentation pipeline
- **OpenCV** + **Pillow** — image I/O
- **matplotlib** — visualization

## Project Structure

```
CodeEdge-Hackathon/
├── train_segmentation.py      # Training script (UNet++ EfficientNet-B5)
├── test_segmentation.py       # Test/evaluation with multi-scale TTA
├── visualize.py               # Visualization utilities
├── segmentation_model.pth     # Best trained model weights
├── predictions/               # Test outputs (masks, comparisons, metrics)
├── train_stats/               # Training curves and logs
├── Offroad_Segmentation_Training_Dataset/
│   ├── train/
│   │   ├── Color_Images/      # 2857 training images
│   │   └── Segmentation/      # 2857 training masks
│   └── val/
│       ├── Color_Images/      # 317 validation images
│       └── Segmentation/      # 317 validation masks
└── Offroad_Segmentation_testImages/
    ├── Color_Images/          # 668 test images
    └── Segmentation/          # 668 test masks
```
