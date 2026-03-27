"""
Offroad Semantic Segmentation — Training Script
DeepLabV3-ResNet50 (COCO pretrained) with Two-Phase Training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.models.segmentation as seg_models
import numpy as np
from PIL import Image
import cv2
import os
import time
from tqdm import tqdm
from collections import Counter
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


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.img_dir = os.path.join(data_dir, 'Color_Images')
        self.mask_dir = os.path.join(data_dir, 'Segmentation')
        self.img_names = sorted(os.listdir(self.img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        # Map raw pixel values to class IDs 0-10
        new_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        for raw_value, class_id in value_map.items():
            new_mask[mask == raw_value] = class_id

        if self.transform:
            transformed = self.transform(image=image, mask=new_mask)
            image = transformed['image']
            new_mask = transformed['mask'].long()

        return image, new_mask


# ============================================================================
# Loss Functions
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dims)
        cardinality = (probs + targets_one_hot).sum(dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None, focal_weight=0.5, dice_weight=0.5, gamma=2.0):
        super().__init__()
        self.focal = FocalLoss(alpha=class_weights, gamma=gamma)
        self.dice = DiceLoss(num_classes)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.focal_weight * self.focal(logits, targets) + \
               self.dice_weight * self.dice(logits, targets)


# ============================================================================
# Class Weight Computation
# ============================================================================

def compute_class_weights(data_dir, num_classes=11):
    """Scan ALL training masks, compute sqrt-inverse-frequency weights."""
    masks_dir = os.path.join(data_dir, 'Segmentation')
    mask_files = sorted(os.listdir(masks_dir))

    pixel_counts = Counter()
    for fname in tqdm(mask_files, desc="Computing class weights"):
        mask = np.array(Image.open(os.path.join(masks_dir, fname)))
        new_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        for raw_value, class_id in value_map.items():
            new_mask[mask == raw_value] = class_id
        unique, counts = np.unique(new_mask, return_counts=True)
        for u, c in zip(unique, counts):
            pixel_counts[int(u)] += c

    total = sum(pixel_counts.values())
    freqs = []
    for c in range(num_classes):
        freqs.append(pixel_counts.get(c, 0) / total if total > 0 else 0)

    # sqrt-inverse-frequency with median dampening
    positive_freqs = [f for f in freqs if f > 0]
    median_freq = np.median(positive_freqs) if positive_freqs else 1.0
    weights = torch.ones(num_classes)
    for c in range(num_classes):
        if freqs[c] > 0:
            weights[c] = np.sqrt(median_freq / freqs[c])
        else:
            weights[c] = 0.5  # Background or absent class

    weights = weights.clamp(min=0.5, max=20.0)
    print(f"Class weights: {[f'{w:.2f}' for w in weights.tolist()]}")
    return weights


# ============================================================================
# Metrics — Global Accumulation (statistically correct)
# ============================================================================

def compute_metrics_batch(pred_logits, targets, num_classes):
    """Returns per-class intersection and union for a single batch."""
    pred = torch.argmax(pred_logits, dim=1)
    intersections = torch.zeros(num_classes)
    unions = torch.zeros(num_classes)

    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (targets == c)
        intersections[c] = (pred_c & target_c).sum().float().cpu()
        unions[c] = (pred_c | target_c).sum().float().cpu()

    correct = (pred == targets).sum().float().cpu()
    total = targets.numel()
    return intersections, unions, correct, total


def compute_miou_from_accumulated(total_intersection, total_union):
    """Compute per-class IoU and mIoU from accumulated totals."""
    iou_per_class = []
    for c in range(len(total_intersection)):
        if total_union[c] == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((total_intersection[c] / total_union[c]).item())
    miou = float(np.nanmean(iou_per_class))
    return miou, iou_per_class


# ============================================================================
# Model
# ============================================================================

def create_model(num_classes=11):
    model = seg_models.deeplabv3_resnet50(weights='DEFAULT')
    # Replace classification heads for our 11 classes
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


# ============================================================================
# Plotting
# ============================================================================

def save_training_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history['val_miou'], label='Val mIoU', color='green')
    axes[1].set_title('Validation mIoU')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(history['val_pixel_acc'], label='Val Pixel Acc', color='orange')
    axes[2].set_title('Validation Pixel Accuracy')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()

    # Per-class IoU bar chart (final epoch)
    if history.get('val_per_class_iou'):
        last_iou = history['val_per_class_iou'][-1]
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = ['#333333', '#228B22', '#00FF00', '#D2B48C', '#8B5A2B',
                  '#808000', '#FF69B4', '#8B4513', '#808080', '#A0522D', '#87CEEB']
        valid_iou = [x if not np.isnan(x) else 0 for x in last_iou]
        bars = ax.bar(range(n_classes), valid_iou, color=colors)
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_ylabel('IoU')
        ax.set_title(f'Per-Class IoU (Final) — mIoU: {np.nanmean(last_iou):.4f}')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, valid_iou):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_class_iou.png'), dpi=150)
        plt.close()

    print(f"Saved training plots to '{output_dir}'")


# ============================================================================
# Training Phase
# ============================================================================

def get_param_groups(model, lr):
    """Create differential learning rate param groups."""
    encoder_low = []   # layers 1-2, conv1, bn1
    encoder_high = []  # layers 3-4
    decoder = []       # classifier, aux_classifier

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in ['backbone.layer1', 'backbone.layer2',
                                     'backbone.bn1', 'backbone.conv1']):
            encoder_low.append(param)
        elif any(k in name for k in ['backbone.layer3', 'backbone.layer4']):
            encoder_high.append(param)
        else:
            decoder.append(param)

    return [
        {'params': encoder_low, 'lr': lr * 0.01},
        {'params': encoder_high, 'lr': lr * 0.1},
        {'params': decoder, 'lr': lr},
    ]


def train_phase(model, train_loader, val_loader, device, criterion,
                num_epochs, lr, phase_name, save_path,
                freeze_epochs=0, patience=10):
    """Run one training phase. Returns history dict and best mIoU."""

    # --- Optimizer setup ---
    if freeze_epochs > 0:
        for param in model.backbone.parameters():
            param.requires_grad = False
        trainable = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(get_param_groups(model, lr), weight_decay=0.01)

    total_steps = num_epochs * len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps,
                           pct_start=0.1, anneal_strategy='cos')
    scaler = torch.amp.GradScaler('cuda')

    history = {
        'train_loss': [], 'val_loss': [], 'val_miou': [],
        'val_pixel_acc': [], 'val_per_class_iou': []
    }

    best_miou = 0.0
    epochs_no_improve = 0

    print(f"\n{'=' * 60}")
    print(f"  {phase_name}")
    print(f"{'=' * 60}")

    for epoch in range(num_epochs):
        # Unfreeze encoder after warmup
        if freeze_epochs > 0 and epoch == freeze_epochs:
            print(f"\n  Unfreezing encoder at epoch {epoch + 1}")
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(get_param_groups(model, lr), weight_decay=0.01)
            remaining_steps = (num_epochs - epoch) * len(train_loader)
            scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=remaining_steps,
                                   pct_start=0.1, anneal_strategy='cos')

        # ---- Training ----
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
                    leave=False, unit="batch")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                output = model(images)
                main_loss = criterion(output['out'], masks)
                aux_loss = criterion(output['aux'], masks)
                loss = main_loss + 0.4 * aux_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ---- Validation (single pass with metrics) ----
        model.eval()
        val_losses = []
        total_inter = torch.zeros(n_classes)
        total_union = torch.zeros(n_classes)
        total_correct = 0
        total_pixels = 0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]",
                                      leave=False, unit="batch"):
                images, masks = images.to(device), masks.to(device)

                with torch.amp.autocast('cuda'):
                    output = model(images)
                    main_loss = criterion(output['out'], masks)
                    aux_loss = criterion(output['aux'], masks)
                    loss = main_loss + 0.4 * aux_loss

                val_losses.append(loss.item())

                inter, union, correct, total = compute_metrics_batch(
                    output['out'], masks, n_classes)
                total_inter += inter
                total_union += union
                total_correct += correct
                total_pixels += total

        miou, per_class_iou = compute_miou_from_accumulated(total_inter, total_union)
        pixel_acc = (total_correct / total_pixels).item()

        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_miou'].append(miou)
        history['val_pixel_acc'].append(pixel_acc)
        history['val_per_class_iou'].append(per_class_iou)

        current_lr = optimizer.param_groups[-1]['lr']
        print(f"  Epoch {epoch + 1}/{num_epochs} — train_loss: {epoch_train_loss:.4f}, "
              f"val_loss: {epoch_val_loss:.4f}, val_mIoU: {miou:.4f}, "
              f"val_acc: {pixel_acc:.4f}, lr: {current_lr:.2e}")

        # Best model saving + early stopping
        if miou > best_miou:
            best_miou = miou
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"    >> New best mIoU: {miou:.4f} — saved to '{save_path}'")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"    >> Early stopping at epoch {epoch + 1} "
                      f"(no improvement for {patience} epochs)")
                break

    print(f"\n  {phase_name} complete. Best mIoU: {best_miou:.4f}")
    return history, best_miou


# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'val')
    output_dir = os.path.join(script_dir, 'train_stats')
    os.makedirs(output_dir, exist_ok=True)

    # Compute class weights (scans all masks)
    class_weights = compute_class_weights(data_dir, n_classes).to(device)

    # Create model
    print("\nCreating DeepLabV3-ResNet50 model (COCO pretrained)...")
    model = create_model(n_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # Loss
    criterion = CombinedLoss(n_classes, class_weights=class_weights).to(device)

    # ==================================================================
    # Phase 1: Low-Resolution Training (416x736)
    # ==================================================================
    H1, W1 = 416, 736
    batch_size_1 = 8

    train_transform_1 = A.Compose([
        A.Resize(H1, W1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Affine(shift_limit=0.1, scale_limit=0.2, rotate_limit=20,
                 border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25,
                             val_shift_limit=15, p=0.4),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform_1 = A.Compose([
        A.Resize(H1, W1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    trainset_1 = MaskDataset(data_dir, transform=train_transform_1)
    valset_1 = MaskDataset(val_dir, transform=val_transform_1)
    train_loader_1 = DataLoader(trainset_1, batch_size=batch_size_1, shuffle=True,
                                num_workers=0, pin_memory=True, drop_last=True)
    val_loader_1 = DataLoader(valset_1, batch_size=batch_size_1, shuffle=False,
                              num_workers=0, pin_memory=True)

    print(f"\nTraining samples: {len(trainset_1)}")
    print(f"Validation samples: {len(valset_1)}")
    print(f"Phase 1 resolution: {W1}x{H1}, batch_size: {batch_size_1}")

    phase1_path = os.path.join(script_dir, 'best_model_phase1.pth')
    history_1, best_miou_1 = train_phase(
        model, train_loader_1, val_loader_1, device, criterion,
        num_epochs=30, lr=3e-4,
        phase_name="Phase 1: Low-Res Training (416x736)",
        save_path=phase1_path, freeze_epochs=3, patience=10
    )

    # ==================================================================
    # Phase 2: High-Resolution Finetuning (512x896)
    # ==================================================================
    torch.cuda.empty_cache()

    H2, W2 = 512, 896
    batch_size_2 = 4

    train_transform_2 = A.Compose([
        A.Resize(H2, W2),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Affine(shift_limit=0.1, scale_limit=0.2, rotate_limit=20,
                 border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25,
                             val_shift_limit=15, p=0.4),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform_2 = A.Compose([
        A.Resize(H2, W2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    trainset_2 = MaskDataset(data_dir, transform=train_transform_2)
    valset_2 = MaskDataset(val_dir, transform=val_transform_2)
    train_loader_2 = DataLoader(trainset_2, batch_size=batch_size_2, shuffle=True,
                                num_workers=0, pin_memory=True, drop_last=True)
    val_loader_2 = DataLoader(valset_2, batch_size=batch_size_2, shuffle=False,
                              num_workers=0, pin_memory=True)

    # Load best Phase 1 model
    model.load_state_dict(torch.load(phase1_path, map_location=device, weights_only=True))
    print(f"\nLoaded best Phase 1 model (mIoU: {best_miou_1:.4f})")
    print(f"Phase 2 resolution: {W2}x{H2}, batch_size: {batch_size_2}")

    final_path = os.path.join(script_dir, 'segmentation_model_best.pth')
    history_2, best_miou_2 = train_phase(
        model, train_loader_2, val_loader_2, device, criterion,
        num_epochs=15, lr=5e-5,
        phase_name="Phase 2: High-Res Finetuning (512x896)",
        save_path=final_path, freeze_epochs=0, patience=10
    )

    # ==================================================================
    # Save Results
    # ==================================================================

    # Combine histories
    combined = {
        'train_loss': history_1['train_loss'] + history_2['train_loss'],
        'val_loss': history_1['val_loss'] + history_2['val_loss'],
        'val_miou': history_1['val_miou'] + history_2['val_miou'],
        'val_pixel_acc': history_1['val_pixel_acc'] + history_2['val_pixel_acc'],
        'val_per_class_iou': history_1['val_per_class_iou'] + history_2['val_per_class_iou'],
    }

    print("\nSaving training curves...")
    save_training_plots(combined, output_dir)

    # Copy best model as primary checkpoint
    model_path = os.path.join(script_dir, 'segmentation_model.pth')
    if os.path.exists(final_path):
        import shutil
        shutil.copy2(final_path, model_path)
        print(f"Copied best model (mIoU={best_miou_2:.4f}) to '{model_path}'")
    else:
        torch.save(model.state_dict(), model_path)
        print(f"Saved final model to '{model_path}'")

    # Print final per-class IoU
    if combined['val_per_class_iou']:
        last_iou = combined['val_per_class_iou'][-1]
        print(f"\nFinal evaluation results:")
        print(f"  Final Val mIoU:      {np.nanmean(last_iou):.4f}")
        print(f"  Final Val Pixel Acc: {combined['val_pixel_acc'][-1]:.4f}")
        print(f"\nPer-Class IoU:")
        print("-" * 40)
        for i, name in enumerate(class_names):
            iou = last_iou[i]
            status = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
            print(f"  {name:20s}: {status}")

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
