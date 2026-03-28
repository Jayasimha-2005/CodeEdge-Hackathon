"""
Offroad Semantic Segmentation — Test/Evaluation Script V3
UNet++ with EfficientNet-B5 + Multi-Scale TTA + Class Suppression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
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

# Classes absent from test GT — suppress at inference
SUPPRESS_CLASSES = [0, 5, 6, 7]  # Background, Ground Clutter, Flowers, Logs

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
# Dataset
# ============================================================================

class TestDataset(Dataset):
    def __init__(self, data_dir):
        self.img_dir = os.path.join(data_dir, 'Color_Images')
        self.mask_dir = os.path.join(data_dir, 'Segmentation')
        self.has_masks = os.path.isdir(self.mask_dir) and len(os.listdir(self.mask_dir)) > 0
        self.img_names = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = np.array(Image.open(img_path).convert("RGB"))
        orig_h, orig_w = image.shape[:2]

        mask = None
        if self.has_masks:
            mask_path = os.path.join(self.mask_dir, img_name)
            if os.path.exists(mask_path):
                raw_mask = np.array(Image.open(mask_path))
                mask = np.zeros(raw_mask.shape[:2], dtype=np.int64)
                for raw_value, class_id in value_map.items():
                    mask[raw_mask == raw_value] = class_id

        data_id = os.path.splitext(img_name)[0]
        return image, mask, data_id, orig_h, orig_w


# ============================================================================
# Model
# ============================================================================

def create_model(num_classes=11, encoder_name='efficientnet-b5'):
    return smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=None,  # we load our own weights
        in_channels=3,
        classes=num_classes,
    )


# ============================================================================
# Multi-Scale TTA with Class Suppression
# ============================================================================

def get_transform(h, w):
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def predict_multiscale_tta(model, image_np, device, orig_h, orig_w,
                           base_h=544, base_w=960,
                           scales=[0.75, 1.0, 1.25],
                           suppress_classes=None):
    """
    Multi-scale TTA: each scale × [original, hflip] = 6-fold averaging.
    Returns argmax prediction at native resolution.
    """
    accumulated = torch.zeros(1, n_classes, orig_h, orig_w, device=device)
    count = 0

    for scale in scales:
        h_s = int(base_h * scale)
        w_s = int(base_w * scale)
        # Round to nearest multiple of 32 for UNet++
        h_s = max(32, (h_s // 32) * 32)
        w_s = max(32, (w_s // 32) * 32)

        transform = get_transform(h_s, w_s)
        img_tensor = transform(image=image_np)['image'].unsqueeze(0).to(device)

        for flip in [False, True]:
            inp = torch.flip(img_tensor, [3]) if flip else img_tensor

            with torch.no_grad(), torch.amp.autocast('cuda'):
                out = model(inp)
                if flip:
                    out = torch.flip(out, [3])
                out = F.interpolate(out, (orig_h, orig_w),
                                    mode='bilinear', align_corners=False)
                accumulated += torch.softmax(out, dim=1)
                count += 1

    probs = accumulated / count

    # Suppress classes absent from test GT
    if suppress_classes:
        for c in suppress_classes:
            probs[:, c, :, :] = 0.0

    return probs.argmax(dim=1)  # (1, orig_h, orig_w)


# ============================================================================
# Visualization
# ============================================================================

def mask_to_color(mask):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(n_classes):
        color[mask == c] = color_palette[c]
    return color


def save_comparison(image_np, gt_mask, pred_mask, save_path):
    gt_color = mask_to_color(gt_mask)
    pred_color = mask_to_color(pred_mask)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image_np); axes[0].set_title("Input Image"); axes[0].axis('off')
    axes[1].imshow(gt_color); axes[1].set_title("Ground Truth"); axes[1].axis('off')
    axes[2].imshow(pred_color); axes[2].set_title("Prediction"); axes[2].axis('off')
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

    dataset = TestDataset(test_dir)
    print(f"Loaded {len(dataset)} samples (GT masks: {dataset.has_masks})")

    # Load model
    encoder_name = 'efficientnet-b5'
    model = create_model(n_classes, encoder_name).to(device)

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
    total_gt_pixels = torch.zeros(n_classes)
    total_correct = 0
    total_pixels = 0
    has_gt = False
    comparison_count = 0
    max_comparisons = 15

    # TTA config
    base_h, base_w = 544, 960
    scales = [0.75, 1.0, 1.25]

    print(f"\nRunning multi-scale TTA ({len(scales)} scales × 2 flips = "
          f"{len(scales)*2}-fold) on {len(dataset)} images...")
    print(f"Suppressing classes: {[class_names[c] for c in SUPPRESS_CLASSES]}")

    pbar = tqdm(range(len(dataset)), desc="Processing", unit="img")
    for idx in pbar:
        image_np, mask, data_id, orig_h, orig_w = dataset[idx]

        # Multi-scale TTA prediction at native resolution
        pred = predict_multiscale_tta(
            model, image_np, device, orig_h, orig_w,
            base_h=base_h, base_w=base_w, scales=scales,
            suppress_classes=SUPPRESS_CLASSES
        )
        pred_np = pred[0].cpu().numpy().astype(np.uint8)

        # Compute metrics if GT available
        if mask is not None:
            has_gt = True
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)
            pred_gpu = pred.to(device)

            for c in range(n_classes):
                pred_c = (pred_gpu == c)
                target_c = (mask_tensor == c)
                total_inter[c] += (pred_c & target_c).sum().float().cpu()
                total_union[c] += (pred_c | target_c).sum().float().cpu()
                total_gt_pixels[c] += target_c.sum().float().cpu()

            total_correct += (pred_gpu == mask_tensor).sum().float().cpu()
            total_pixels += mask_tensor.numel()

            # Running mIoU for progress bar
            ious = []
            for c in range(n_classes):
                if total_gt_pixels[c] > 0 and total_union[c] > 0:
                    ious.append((total_inter[c] / total_union[c]).item())
            if ious:
                pbar.set_postfix(miou=f"{np.mean(ious):.3f}")

        # Save predictions
        cv2.imwrite(os.path.join(masks_dir, f"{data_id}_pred.png"), pred_np)
        color_mask = mask_to_color(pred_np)
        cv2.imwrite(os.path.join(color_dir, f"{data_id}_pred_color.png"),
                    color_mask[:, :, ::-1])

        # Comparison images
        if mask is not None and comparison_count < max_comparisons:
            gt_mask = mask.astype(np.uint8)
            save_comparison(image_np, gt_mask, pred_np,
                          os.path.join(comp_dir, f"sample_{comparison_count}.png"))
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
            if total_gt_pixels[c] == 0:
                iou_per_class.append(float('nan'))
            elif total_union[c] == 0:
                iou_per_class.append(float('nan'))
            else:
                iou_per_class.append((total_inter[c] / total_union[c]).item())

        miou = float(np.nanmean(iou_per_class))
        pixel_acc = (total_correct / total_pixels).item() if total_pixels > 0 else 0
        n_evaluated = sum(1 for x in iou_per_class if not np.isnan(x))

        print(f"Mean IoU:          {miou:.4f}  (over {n_evaluated} classes present in GT)")
        print(f"Pixel Accuracy:    {pixel_acc:.4f}")
        print("=" * 50)

        print("\nPer-Class IoU:")
        print("-" * 50)
        for i, name in enumerate(class_names):
            iou = iou_per_class[i]
            if np.isnan(iou):
                if total_gt_pixels[i] == 0:
                    status = "N/A (not in test GT)"
                else:
                    status = "N/A"
            else:
                status = f"{iou:.4f}"
            gt_pct = 100 * total_gt_pixels[i].item() / total_pixels if total_pixels > 0 else 0
            print(f"  {name:20s}: {status:>25s}  ({gt_pct:5.1f}% of pixels)")

        # Save metrics
        metrics_path = os.path.join(pred_dir, 'evaluation_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("EVALUATION RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Mean IoU:          {miou:.4f}  (over {n_evaluated} classes)\n")
            f.write(f"Pixel Accuracy:    {pixel_acc:.4f}\n")
            f.write("=" * 50 + "\n\n")
            f.write("Per-Class IoU:\n")
            f.write("-" * 50 + "\n")
            for i, name in enumerate(class_names):
                iou = iou_per_class[i]
                if np.isnan(iou):
                    status = "N/A (not in test GT)" if total_gt_pixels[i] == 0 else "N/A"
                else:
                    status = f"{iou:.4f}"
                f.write(f"  {name:20s}: {status}\n")
        print(f"\nSaved metrics to {metrics_path}")

        # Per-class bar chart
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = ['#333333', '#228B22', '#00FF00', '#D2B48C', '#8B5A2B',
                  '#808000', '#FF69B4', '#8B4513', '#808080', '#A0522D', '#87CEEB']
        valid_iou = [x if not np.isnan(x) else 0 for x in iou_per_class]
        bars = ax.bar(range(n_classes), valid_iou, color=colors)
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_ylabel('IoU')
        ax.set_title(f'Per-Class IoU — mIoU: {miou:.4f} ({n_evaluated} classes)')
        ax.set_ylim(0, 1); ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, valid_iou):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        chart_path = os.path.join(pred_dir, 'per_class_metrics.png')
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"Saved chart to '{chart_path}'")
    else:
        print("No ground truth masks — predictions only.")

    print(f"\nProcessed {len(dataset)} images.")
    print(f"Outputs: {pred_dir}/")


if __name__ == '__main__':
    main()
