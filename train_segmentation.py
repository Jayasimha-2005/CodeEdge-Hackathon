"""
Segmentation Training Script
Converted from train_mask.ipynb
Trains a segmentation head on top of DINOv2 backbone
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import torchvision
from tqdm import tqdm
from collections import Counter
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')


# ============================================================================
# Utility Functions
# ============================================================================

def save_image(img, filename):
    """Save an image tensor to file after denormalizing."""
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = (img * std + mean) * 255
    cv2.imwrite(filename, img[:, :, ::-1])


# ============================================================================
# Mask Conversion
# ============================================================================

# Mapping from raw pixel values to new class IDs
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
    10000: 10    # Sky
}
n_classes = len(value_map)


def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.data_ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        # Convert raw mask values to class IDs
        new_mask = np.zeros_like(mask, dtype=np.uint8)
        for raw_value, new_value in value_map.items():
            new_mask[mask == raw_value] = new_value

        if self.transform:
            transformed = self.transform(image=image, mask=new_mask)
            image = transformed['image']
            new_mask = transformed['mask'].long()

        return image, new_mask.unsqueeze(0) if new_mask.dim() == 2 else new_mask


# ============================================================================
# Model: Segmentation Head (ConvNeXt-style)
# ============================================================================

class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block with residual connection and LayerNorm."""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)  # Equivalent to LayerNorm for conv
        self.pwconv1 = nn.Conv2d(dim, dim * 4, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * 4, dim, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + residual


class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        hidden_dim = 256

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, padding=3),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU()
        )

        self.block1 = ConvNeXtBlock(hidden_dim)
        self.block2 = ConvNeXtBlock(hidden_dim)

        self.dropout = nn.Dropout2d(0.1)
        self.classifier = nn.Conv2d(hidden_dim, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.dropout(x)
        return self.classifier(x)


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10, ignore_index=255):
    """Compute IoU for each class and return mean IoU."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue

        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())

    return np.nanmean(iou_per_class)


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    """Compute Dice coefficient (F1 Score) per class and return mean Dice Score."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        dice_score = (2. * intersection + smooth) / (pred_inds.sum().float() + target_inds.sum().float() + smooth)

        dice_per_class.append(dice_score.cpu().numpy())

    return np.mean(dice_per_class)


def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy."""
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()


def evaluate_metrics(model, backbone, data_loader, device, num_classes=10, show_progress=True):
    """Evaluate all metrics on a dataset."""
    iou_scores = []
    dice_scores = []
    pixel_accuracies = []

    model.eval()
    loader = tqdm(data_loader, desc="Evaluating", leave=False, unit="batch") if show_progress else data_loader
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            output = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = model(output.to(device))
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            labels = labels.squeeze(dim=1).long()

            iou = compute_iou(outputs, labels, num_classes=num_classes)
            dice = compute_dice(outputs, labels, num_classes=num_classes)
            pixel_acc = compute_pixel_accuracy(outputs, labels)

            iou_scores.append(iou)
            dice_scores.append(dice)
            pixel_accuracies.append(pixel_acc)

    model.train()
    return np.mean(iou_scores), np.mean(dice_scores), np.mean(pixel_accuracies)


# ============================================================================
# Plotting Functions
# ============================================================================

def save_training_plots(history, output_dir):
    """Save all training metric plots to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Loss curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_pixel_acc'], label='train')
    plt.plot(history['val_pixel_acc'], label='val')
    plt.title('Pixel Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    print(f"Saved training curves to '{output_dir}/training_curves.png'")

    # Plot 2: IoU curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_iou'], label='Train IoU')
    plt.title('Train IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_iou'], label='Val IoU')
    plt.title('Validation IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_curves.png'))
    plt.close()
    print(f"Saved IoU curves to '{output_dir}/iou_curves.png'")

    # Plot 3: Dice curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_dice'], label='Train Dice')
    plt.title('Train Dice vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_dice'], label='Val Dice')
    plt.title('Validation Dice vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_curves.png'))
    plt.close()
    print(f"Saved Dice curves to '{output_dir}/dice_curves.png'")

    # Plot 4: Combined metrics plot
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(history['train_iou'], label='train')
    plt.plot(history['val_iou'], label='val')
    plt.title('IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(history['train_dice'], label='train')
    plt.plot(history['val_dice'], label='val')
    plt.title('Dice Score vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(history['train_pixel_acc'], label='train')
    plt.plot(history['val_pixel_acc'], label='val')
    plt.title('Pixel Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Pixel Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'))
    plt.close()
    print(f"Saved combined metrics curves to '{output_dir}/all_metrics_curves.png'")


def save_history_to_file(history, output_dir):
    """Save training history to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')

    with open(filepath, 'w') as f:
        f.write("TRAINING RESULTS\n")
        f.write("=" * 50 + "\n\n")

        f.write("Final Metrics:\n")
        f.write(f"  Final Train Loss:     {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final Val Loss:       {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Final Train IoU:      {history['train_iou'][-1]:.4f}\n")
        f.write(f"  Final Val IoU:        {history['val_iou'][-1]:.4f}\n")
        f.write(f"  Final Train Dice:     {history['train_dice'][-1]:.4f}\n")
        f.write(f"  Final Val Dice:       {history['val_dice'][-1]:.4f}\n")
        f.write(f"  Final Train Accuracy: {history['train_pixel_acc'][-1]:.4f}\n")
        f.write(f"  Final Val Accuracy:   {history['val_pixel_acc'][-1]:.4f}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Best Results:\n")
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou']) + 1})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f} (Epoch {np.argmax(history['val_dice']) + 1})\n")
        f.write(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f} (Epoch {np.argmax(history['val_pixel_acc']) + 1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})\n")
        f.write("=" * 50 + "\n\n")

        f.write("Per-Epoch History:\n")
        f.write("-" * 100 + "\n")
        headers = ['Epoch', 'Train Loss', 'Val Loss', 'Train IoU', 'Val IoU',
                   'Train Dice', 'Val Dice', 'Train Acc', 'Val Acc']
        f.write("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n".format(*headers))
        f.write("-" * 100 + "\n")

        n_epochs = len(history['train_loss'])
        for i in range(n_epochs):
            f.write("{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n".format(
                i + 1,
                history['train_loss'][i],
                history['val_loss'][i],
                history['train_iou'][i],
                history['val_iou'][i],
                history['train_dice'][i],
                history['val_dice'][i],
                history['train_pixel_acc'][i],
                history['val_pixel_acc'][i]
            ))

    print(f"Saved evaluation metrics to {filepath}")


# ============================================================================
# Combined Loss: Weighted CE + Dice
# ============================================================================

class DiceLoss(nn.Module):
    """Soft Dice loss for semantic segmentation."""
    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * targets_onehot).sum(dim=dims)
        cardinality = (probs + targets_onehot).sum(dim=dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Weighted CrossEntropy + Dice loss."""
    def __init__(self, num_classes, class_weights=None, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)


# ============================================================================
# Class Weight Computation
# ============================================================================

def compute_class_weights(data_dir, num_samples=200):
    """Scan training masks to compute inverse-frequency class weights."""
    masks_dir = os.path.join(data_dir, 'Segmentation')
    mask_files = os.listdir(masks_dir)[:num_samples]

    pixel_counts = Counter()
    for fname in tqdm(mask_files, desc="Computing class weights", leave=False):
        mask = Image.open(os.path.join(masks_dir, fname))
        arr = np.array(mask)
        new_arr = np.zeros_like(arr, dtype=np.uint8)
        for raw_value, new_value in value_map.items():
            new_arr[arr == raw_value] = new_value
        unique, counts = np.unique(new_arr, return_counts=True)
        for u, c in zip(unique, counts):
            pixel_counts[int(u)] += c

    total = sum(pixel_counts.values())
    n = len(value_map)
    weights = torch.ones(n)
    for cls_id, count in pixel_counts.items():
        if cls_id < n and count > 0:
            freq = count / total
            weights[cls_id] = 1.0 / (freq * n)

    # Clamp to avoid extreme weights
    weights = weights.clamp(min=0.5, max=10.0)
    print(f"Class weights: {weights.tolist()}")
    return weights


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 2
    scale = 0.7  # Higher resolution for better detail (was 0.5)
    w = int(((960 * scale) // 14) * 14)   # 672
    h = int(((540 * scale) // 14) * 14)   # 378
    lr = 1e-3
    n_epochs = 50
    patience = 10  # Early stopping patience

    # Output directory (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats')
    os.makedirs(output_dir, exist_ok=True)

    # Transforms (albumentations)
    train_transform = A.Compose([
        A.Resize(h, w),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(p=0.3),
        A.GaussNoise(p=0.2),
        A.Affine(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Dataset paths (relative to script location)
    data_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'val')

    # Create datasets
    trainset = MaskDataset(data_dir=data_dir, transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)

    valset = MaskDataset(data_dir=val_dir, transform=val_transform)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    print(f"Training samples: {len(trainset)}")
    print(f"Validation samples: {len(valset)}")

    # Load DINOv2 backbone
    print("Loading DINOv2 backbone...")
    BACKBONE_SIZE = "small"
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14_reg",
        "large": "vitl14_reg",
        "giant": "vitg14_reg",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.to(device)
    print("Backbone loaded successfully!")

    # Freeze all backbone params, then unfreeze last N blocks
    n_unfreeze = 4
    for param in backbone_model.parameters():
        param.requires_grad = False
    for block in backbone_model.blocks[-n_unfreeze:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in backbone_model.norm.parameters():
        param.requires_grad = True
    backbone_model.train()

    n_backbone_trainable = sum(p.numel() for p in backbone_model.parameters() if p.requires_grad)
    print(f"Unfroze last {n_unfreeze} backbone blocks ({n_backbone_trainable:,} trainable params)")

    # Get embedding dimension from backbone
    imgs, _ = next(iter(train_loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
    n_embedding = output.shape[2]
    print(f"Embedding dimension: {n_embedding}")
    print(f"Patch tokens shape: {output.shape}")

    # Create segmentation head
    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding,
        out_channels=n_classes,
        tokenW=w // 14,
        tokenH=h // 14
    )
    classifier = classifier.to(device)

    # Class weights and combined loss
    print("Computing class weights...")
    class_weights = compute_class_weights(data_dir)
    loss_fct = CombinedLoss(num_classes=n_classes, class_weights=class_weights.to(device))

    # Optimizer with differential learning rates
    backbone_params = [p for p in backbone_model.parameters() if p.requires_grad]
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': classifier.parameters(), 'lr': lr},
    ], weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda')

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_iou': [],
        'val_iou': [],
        'train_dice': [],
        'val_dice': [],
        'train_pixel_acc': [],
        'val_pixel_acc': []
    }

    # Training loop
    print("\nStarting training...")
    print("=" * 80)

    best_val_iou = 0.0
    epochs_without_improvement = 0

    epoch_pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        # Training phase
        classifier.train()
        train_losses = []

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]",
                          leave=False, unit="batch")
        for imgs, labels in train_pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
                logits = classifier(output)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
                labels = labels.squeeze(dim=1).long()
                loss = loss_fct(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation phase (compute loss + metrics in one pass)
        classifier.eval()
        val_losses = []
        val_iou_scores = []
        val_dice_scores = []
        val_pixel_accs = []

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]",
                        leave=False, unit="batch")
        with torch.no_grad():
            for imgs, labels in val_pbar:
                imgs, labels = imgs.to(device), labels.to(device)

                output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
                with torch.amp.autocast('cuda'):
                    logits = classifier(output.to(device))
                    outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
                    labels = labels.squeeze(dim=1).long()
                    loss = loss_fct(outputs, labels)

                val_losses.append(loss.item())
                val_iou_scores.append(compute_iou(outputs, labels, num_classes=n_classes))
                val_dice_scores.append(compute_dice(outputs, labels, num_classes=n_classes))
                val_pixel_accs.append(compute_pixel_accuracy(outputs, labels))
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_iou = np.mean(val_iou_scores)
        val_dice = np.mean(val_dice_scores)
        val_pixel_acc = np.mean(val_pixel_accs)

        # Step scheduler
        scheduler.step()

        # Store history
        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_iou'].append(0.0)  # Skip train eval for speed
        history['val_iou'].append(val_iou)
        history['train_dice'].append(0.0)
        history['val_dice'].append(val_dice)
        history['train_pixel_acc'].append(0.0)
        history['val_pixel_acc'].append(val_pixel_acc)

        # Update epoch progress bar with metrics
        epoch_pbar.set_postfix(
            train_loss=f"{epoch_train_loss:.3f}",
            val_loss=f"{epoch_val_loss:.3f}",
            val_iou=f"{val_iou:.3f}",
            val_acc=f"{val_pixel_acc:.3f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}"
        )

        # Early stopping & best model saving
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            epochs_without_improvement = 0
            best_model_path = os.path.join(script_dir, "segmentation_head_best.pth")
            torch.save({
                'classifier': classifier.state_dict(),
                'backbone': backbone_model.state_dict(),
            }, best_model_path)
            print(f"\n  New best val IoU: {val_iou:.4f} — saved to '{best_model_path}'")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    # Save plots
    print("\nSaving training curves...")
    save_training_plots(history, output_dir)
    save_history_to_file(history, output_dir)

    # Save final model (in scripts directory) — full checkpoint
    model_path = os.path.join(script_dir, "segmentation_head.pth")
    torch.save({
        'classifier': classifier.state_dict(),
        'backbone': backbone_model.state_dict(),
    }, model_path)
    print(f"Saved final model to '{model_path}'")

    # Copy best model as the primary checkpoint
    best_model_path = os.path.join(script_dir, "segmentation_head_best.pth")
    if os.path.exists(best_model_path):
        import shutil
        shutil.copy2(best_model_path, model_path)
        print(f"Copied best model (val IoU={best_val_iou:.4f}) to '{model_path}'")

    # Final evaluation
    print("\nFinal evaluation results:")
    print(f"  Final Val Loss:     {history['val_loss'][-1]:.4f}")
    print(f"  Final Val IoU:      {history['val_iou'][-1]:.4f}")
    print(f"  Final Val Dice:     {history['val_dice'][-1]:.4f}")
    print(f"  Final Val Accuracy: {history['val_pixel_acc'][-1]:.4f}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()

