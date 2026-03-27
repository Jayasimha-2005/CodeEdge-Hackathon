"""
Offroad Semantic Segmentation — Test/Evaluation Script
DeepLabV3-ResNet50 with Test-Time Augmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models.segmentation as seg_models
import numpy as np
from PIL import Image
import cv2
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================================
# Class Definitions
# ============================================================================

value_map = {
    0: 0,        # Background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    600: 6,      # Flowers
    700: 7,      # Logs
    800: 8,      # Rocks
    7100: 9,     # Landscape
    10000: 10,   # Sky
}

n_classes = 11

class_names = [
    "Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Flowers", "Logs", "Rocks", "Landscape", "Sky"
]

color_palette = np.array([
    [0, 0, 0],         # Background — black
    [34, 139, 34],     # Trees — forest green
    [0, 255, 0],       # Lush Bushes — lime
    [210, 180, 140],   # Dry Grass — tan
    [139, 90, 43],     # Dry Bushes — brown
    [128, 128, 0],     # Ground Clutter — olive
    [255, 105, 180],   # Flowers — hot pink
    [139, 69, 19],     # Logs — saddle brown
    [128, 128, 128],   # Rocks — gray
    [160, 82, 45],     # Landscape — sienna
    [135, 206, 235],   # Sky — sky blue
], dtype=np.uint8)


# ============================================================================
# Dataset — returns image (transformed) + mask at NATIVE resolution
# ============================================================================

class TestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.img_dir = os.path.join(data_dir, 'Color_Images')
        self.mask_dir = os.path.join(data_dir, 'Segmentation')
        self.has_masks = os.path.isdir(self.mask_dir) and len(os.listdir(self.mask_dir)) > 0
        self.img_names = sorted(os.listdir(self.img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = np.array(Image.open(img_path).convert("RGB"))
        orig_h, orig_w = image.shape[:2]

        # GT mask at native resolution (NOT transformed)
        mask = None
        if self.has_masks:
            mask_path = os.path.join(self.mask_dir, img_name)
            if os.path.exists(mask_path):
                raw_mask = np.array(Image.open(mask_path))
                mask = np.zeros(raw_mask.shape[:2], dtype=np.int64)
                for raw_value, class_id in value_map.items():
                    mask[raw_mask == raw_value] = class_id
                mask = torch.from_numpy(mask)

        # Transform image only (resize + normalize for model input)
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        data_id = os.path.splitext(img_name)[0]

        if mask is None:
            mask = torch.tensor(-1)

        return image, mask, data_id, orig_h, orig_w


# ============================================================================
# Model
# ============================================================================

def create_model(num_classes=11):
    model = seg_models.deeplabv3_resnet50(weights=None)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


# ============================================================================
# TTA Inference — predictions at native resolution
# ============================================================================

def predict_with_tta(model, images, orig_h, orig_w):
    """Inference with horizontal flip TTA, output upsampled to native resolution."""
    with torch.no_grad(), torch.amp.autocast('cuda'):
        # Original
        out = model(images)['out']
        out = F.interpolate(out, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        probs = torch.softmax(out, dim=1)

        # Horizontal flip
        images_flip = torch.flip(images, dims=[3])
        out_flip = model(images_flip)['out']
        out_flip = torch.flip(out_flip, dims=[3])
        out_flip = F.interpolate(out_flip, size=(orig_h, orig_w),
                                 mode='bilinear', align_corners=False)
        probs = probs + torch.softmax(out_flip, dim=1)

    return probs / 2.0


# ============================================================================
# Visualization
# ============================================================================

def mask_to_color(mask):
    """Convert class-ID mask to RGB."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(n_classes):
        color[mask == c] = color_palette[c]
    return color


def denormalize_image(img_tensor):
    """Convert normalized image tensor back to uint8 RGB numpy array."""
    img = img_tensor.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = (img * std + mean) * 255
    return np.clip(img, 0, 255).astype(np.uint8)


def save_comparison(image_np, gt_mask, pred_mask, save_path):
    """Save side-by-side: image | GT | prediction."""
    gt_color = mask_to_color(gt_mask)
    pred_color = mask_to_color(pred_mask)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image_np)
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    axes[1].imshow(gt_color)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    axes[2].imshow(pred_color)
    axes[2].set_title("Prediction")
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(script_dir, 'Offroad_Segmentation_testImages')
    print(f"Loading dataset from {test_dir}...")

    # Model inference resolution (must match Phase 2 training resolution)
    H, W = 512, 896

    test_transform = A.Compose([
        A.Resize(H, W),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    dataset = TestDataset(test_dir, transform=test_transform)
    print(f"Loaded {len(dataset)} samples (GT masks: {dataset.has_masks})")

    loader = DataLoader(dataset, batch_size=2, shuffle=False,
                        num_workers=0, pin_memory=True)

    # Load model
    model = create_model(n_classes).to(device)

    model_path = os.path.join(script_dir, 'segmentation_model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(script_dir, 'segmentation_model_best.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(script_dir, 'best_model_phase1.pth')

    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded successfully!")

    # Output directories
    pred_dir = os.path.join(script_dir, 'predictions')
    masks_dir = os.path.join(pred_dir, 'masks')
    color_dir = os.path.join(pred_dir, 'masks_color')
    comp_dir = os.path.join(pred_dir, 'comparisons')
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(comp_dir, exist_ok=True)

    # Global metric accumulators
    total_inter = torch.zeros(n_classes)
    total_union = torch.zeros(n_classes)
    total_correct = 0
    total_pixels = 0
    has_gt = False
    comparison_count = 0
    max_comparisons = 10

    print(f"\nRunning evaluation with TTA on {len(dataset)} images...")

    pbar = tqdm(loader, desc="Processing", unit="batch")
    for images, masks, data_ids, orig_hs, orig_ws in pbar:
        images = images.to(device)
        batch_size = images.shape[0]

        # All images are 960x540, use first as reference
        orig_h = orig_hs[0].item()
        orig_w = orig_ws[0].item()

        # TTA prediction at native resolution
        probs = predict_with_tta(model, images, orig_h, orig_w)
        preds = probs.argmax(dim=1)  # (B, orig_h, orig_w) on GPU

        # Compute metrics if GT available
        gt_valid = masks.dim() > 1 and masks.shape[-1] > 1
        if gt_valid:
            has_gt = True
            masks_gpu = masks.to(device)

            for c in range(n_classes):
                pred_c = (preds == c)
                target_c = (masks_gpu == c)
                total_inter[c] += (pred_c & target_c).sum().float().cpu()
                total_union[c] += (pred_c | target_c).sum().float().cpu()

            total_correct += (preds == masks_gpu).sum().float().cpu()
            total_pixels += masks_gpu.numel()

            # Running mIoU for progress bar
            ious = []
            for c in range(n_classes):
                if total_union[c] > 0:
                    ious.append((total_inter[c] / total_union[c]).item())
            pbar.set_postfix(miou=f"{np.mean(ious):.3f}" if ious else "N/A")

        preds_np = preds.cpu().numpy()

        # Save predictions
        for i in range(batch_size):
            data_id = data_ids[i]
            pred_mask = preds_np[i].astype(np.uint8)

            # Raw mask (class IDs 0-10)
            cv2.imwrite(os.path.join(masks_dir, f"{data_id}_pred.png"), pred_mask)

            # Colored mask
            color_mask = mask_to_color(pred_mask)
            cv2.imwrite(os.path.join(color_dir, f"{data_id}_pred_color.png"),
                        color_mask[:, :, ::-1])  # RGB → BGR for OpenCV

            # Comparisons (first few with GT)
            if gt_valid and comparison_count < max_comparisons:
                gt_mask = masks[i].numpy().astype(np.uint8)
                img_denorm = denormalize_image(images[i])
                img_resized = cv2.resize(img_denorm, (orig_w, orig_h))
                save_comparison(
                    img_resized, gt_mask, pred_mask,
                    os.path.join(comp_dir, f"sample_{comparison_count}_comparison.png")
                )
                comparison_count += 1

    # ==================================================================
    # Print Results
    # ==================================================================
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    if has_gt:
        iou_per_class = []
        for c in range(n_classes):
            if total_union[c] == 0:
                iou_per_class.append(float('nan'))
            else:
                iou_per_class.append((total_inter[c] / total_union[c]).item())

        miou = float(np.nanmean(iou_per_class))
        pixel_acc = (total_correct / total_pixels).item() if total_pixels > 0 else 0

        print(f"Mean IoU:          {miou:.4f}")
        print(f"Pixel Accuracy:    {pixel_acc:.4f}")
        print("=" * 50)

        print("\nPer-Class IoU:")
        print("-" * 40)
        for i, name in enumerate(class_names):
            iou = iou_per_class[i]
            status = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
            print(f"  {name:20s}: {status}")

        # Save metrics to file
        metrics_path = os.path.join(pred_dir, 'evaluation_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("EVALUATION RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Mean IoU:          {miou:.4f}\n")
            f.write(f"Pixel Accuracy:    {pixel_acc:.4f}\n")
            f.write("=" * 50 + "\n\n")
            f.write("Per-Class IoU:\n")
            f.write("-" * 40 + "\n")
            for i, name in enumerate(class_names):
                iou = iou_per_class[i]
                status = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
                f.write(f"  {name:20s}: {status}\n")
        print(f"\nSaved evaluation metrics to {metrics_path}")

        # Per-class IoU bar chart
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = ['#333333', '#228B22', '#00FF00', '#D2B48C', '#8B5A2B',
                  '#808000', '#FF69B4', '#8B4513', '#808080', '#A0522D', '#87CEEB']
        valid_iou = [x if not np.isnan(x) else 0 for x in iou_per_class]
        bars = ax.bar(range(n_classes), valid_iou, color=colors)
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_ylabel('IoU')
        ax.set_title(f'Per-Class IoU — mIoU: {miou:.4f}')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, valid_iou):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        chart_path = os.path.join(pred_dir, 'per_class_metrics.png')
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"Saved per-class metrics chart to '{chart_path}'")
    else:
        print("No ground truth masks available — predictions only.")

    print(f"\nPrediction complete! Processed {len(dataset)} images.")
    print(f"\nOutputs saved to {pred_dir}/")
    print(f"  - masks/           : Raw prediction masks (class IDs 0-10)")
    print(f"  - masks_color/     : Colored prediction masks (RGB)")
    print(f"  - comparisons/     : Side-by-side comparison images")
    if has_gt:
        print(f"  - evaluation_metrics.txt")
        print(f"  - per_class_metrics.png")


if __name__ == '__main__':
    main()
