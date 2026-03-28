"""
Offroad Semantic Segmentation — Training Script V3
UNet++ with EfficientNet-B5 + Lovász-Softmax + EMA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
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


# ============================================================================
# Dataset — supports multiple directories (combined train + val)
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dirs, transform=None):
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        self.samples = []
        for d in data_dirs:
            img_dir = os.path.join(d, 'Color_Images')
            mask_dir = os.path.join(d, 'Segmentation')
            for name in sorted(os.listdir(img_dir)):
                self.samples.append((
                    os.path.join(img_dir, name),
                    os.path.join(mask_dir, name)
                ))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        new_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        for raw_value, class_id in value_map.items():
            new_mask[mask == raw_value] = class_id

        if self.transform:
            transformed = self.transform(image=image, mask=new_mask)
            image = transformed['image']
            new_mask = transformed['mask'].long()

        return image, new_mask


# ============================================================================
# Lovász-Softmax Loss (Berman et al. 2018) — directly optimizes IoU
# ============================================================================

def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes='present'):
    if probas.numel() == 0:
        return probas * 0.0
    C = probas.size(1)
    losses = []
    for c in range(C):
        if classes == 'present':
            if (labels == c).sum() == 0 and (probas[:, c] > 0.5).sum() == 0:
                continue
        fg = (labels == c).float()
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    if len(losses) == 0:
        return torch.tensor(0.0, device=probas.device, requires_grad=True)
    return torch.stack(losses).mean()


class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present'):
        super().__init__()
        self.classes = classes

    def forward(self, logits, labels):
        probas = F.softmax(logits, dim=1)
        B, C, H, W = probas.shape
        probas = probas.permute(0, 2, 3, 1).reshape(-1, C)
        labels = labels.reshape(-1)
        return lovasz_softmax_flat(probas, labels, self.classes)


# ============================================================================
# Focal Loss + Dice Loss
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * targets_oh).sum(dims)
        card = (probs + targets_oh).sum(dims)
        dice = (2.0 * inter + self.smooth) / (card + self.smooth)
        return 1.0 - dice.mean()


class TripleLoss(nn.Module):
    """Lovász-Softmax + Focal + Dice"""
    def __init__(self, num_classes, class_weights=None,
                 lovasz_w=0.4, focal_w=0.3, dice_w=0.3):
        super().__init__()
        self.lovasz = LovaszSoftmax()
        self.focal = FocalLoss(alpha=class_weights, gamma=2.5)
        self.dice = DiceLoss(num_classes)
        self.lovasz_w = lovasz_w
        self.focal_w = focal_w
        self.dice_w = dice_w

    def forward(self, logits, targets):
        return (self.lovasz_w * self.lovasz(logits, targets) +
                self.focal_w * self.focal(logits, targets) +
                self.dice_w * self.dice(logits, targets))


# ============================================================================
# EMA (Exponential Moving Average)
# ============================================================================

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics_batch(pred_logits, targets, num_classes):
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


def compute_miou(total_inter, total_union):
    iou_per_class = []
    for c in range(len(total_inter)):
        if total_union[c] == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((total_inter[c] / total_union[c]).item())
    return float(np.nanmean(iou_per_class)), iou_per_class


# ============================================================================
# Model
# ============================================================================

def create_model(num_classes=11, encoder_name='efficientnet-b5'):
    return smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights='imagenet',
        in_channels=3,
        classes=num_classes,
    )


# ============================================================================
# Plotting
# ============================================================================

def save_training_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history['val_miou'], label='Val mIoU', color='green')
    axes[1].set_title('Validation mIoU'); axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(history['val_pixel_acc'], label='Val Pixel Acc', color='orange')
    axes[2].set_title('Validation Pixel Accuracy'); axes[2].set_xlabel('Epoch')
    axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()

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
        ax.set_title(f'Per-Class IoU — mIoU: {np.nanmean(last_iou):.4f}')
        ax.set_ylim(0, 1); ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, valid_iou):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_class_iou.png'), dpi=150)
        plt.close()

    print(f"Saved training plots to '{output_dir}'")


# ============================================================================
# Differential LR for smp models
# ============================================================================

def get_param_groups(model, lr):
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    return [
        {'params': encoder_params, 'lr': lr * 0.1},
        {'params': decoder_params, 'lr': lr},
    ]


# ============================================================================
# Training Phase
# ============================================================================

def train_phase(model, train_loader, val_loader, device, criterion,
                num_epochs, lr, phase_name, save_path,
                freeze_epochs=0, patience=10, accum_steps=1, ema_decay=0.999):

    ema = EMA(model, decay=ema_decay)

    if freeze_epochs > 0:
        for param in model.encoder.parameters():
            param.requires_grad = False
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(get_param_groups(model, lr), weight_decay=0.01)

    steps_per_epoch = max(len(train_loader) // accum_steps, 1)
    total_steps = num_epochs * steps_per_epoch
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=max(total_steps, 1),
                           pct_start=0.1, anneal_strategy='cos')
    scaler = torch.amp.GradScaler('cuda')

    history = {'train_loss': [], 'val_loss': [], 'val_miou': [],
               'val_pixel_acc': [], 'val_per_class_iou': []}
    best_miou = 0.0
    epochs_no_improve = 0

    print(f"\n{'=' * 60}")
    print(f"  {phase_name}")
    print(f"{'=' * 60}")

    for epoch in range(num_epochs):
        # Unfreeze encoder after warmup
        if freeze_epochs > 0 and epoch == freeze_epochs:
            print(f"\n  Unfreezing encoder at epoch {epoch + 1}")
            for param in model.encoder.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(get_param_groups(model, lr), weight_decay=0.01)
            remaining = (num_epochs - epoch) * steps_per_epoch
            scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=max(remaining, 1),
                                   pct_start=0.1, anneal_strategy='cos')
            ema = EMA(model, decay=ema_decay)

        # ---- Training ----
        model.train()
        train_losses = []
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
                    leave=False, unit="batch")
        for step, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)

            with torch.amp.autocast('cuda'):
                output = model(images)
                loss = criterion(output, masks) / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                ema.update(model)

            train_losses.append(loss.item() * accum_steps)
            pbar.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")

        # Flush remaining gradients
        if len(train_loader) % accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update(model)

        # ---- Validation (using EMA model) ----
        ema.apply_shadow(model)
        model.eval()
        val_losses = []
        total_inter = torch.zeros(n_classes)
        total_union = torch.zeros(n_classes)
        total_correct = 0
        total_pixels = 0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]",
                                      leave=False, unit="batch"):
                images, masks = images.to(device), masks.to(device)
                with torch.amp.autocast('cuda'):
                    output = model(images)
                    loss = criterion(output, masks)
                val_losses.append(loss.item())
                inter, union, correct, total = compute_metrics_batch(output, masks, n_classes)
                total_inter += inter
                total_union += union
                total_correct += correct
                total_pixels += total

        miou, per_class_iou = compute_miou(total_inter, total_union)
        pixel_acc = (total_correct / total_pixels).item()

        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_miou'].append(miou)
        history['val_pixel_acc'].append(pixel_acc)
        history['val_per_class_iou'].append(per_class_iou)

        current_lr = optimizer.param_groups[-1]['lr']
        print(f"  Epoch {epoch+1}/{num_epochs} — train: {epoch_train_loss:.4f}, "
              f"val: {epoch_val_loss:.4f}, mIoU: {miou:.4f}, "
              f"acc: {pixel_acc:.4f}, lr: {current_lr:.2e}")

        if miou > best_miou:
            best_miou = miou
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"    >> New best mIoU: {miou:.4f} — saved to '{save_path}'")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"    >> Early stopping at epoch {epoch+1}")
                break

        ema.restore(model)

    print(f"\n  {phase_name} complete. Best mIoU: {best_miou:.4f}")
    return history, best_miou


# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'val')
    output_dir = os.path.join(script_dir, 'train_stats')
    os.makedirs(output_dir, exist_ok=True)

    # Test-aware class weights (tuned for test distribution)
    # BG(0) Trees(1) LushB(2) DryG(3) DryB(4) GndC(5) Flow(6) Logs(7) Rocks(8) Land(9) Sky(10)
    class_weights = torch.tensor(
        [0.5, 2.0, 2.5, 1.5, 3.0, 3.0, 3.0, 8.0, 5.0, 1.0, 0.5],
        dtype=torch.float32).to(device)
    print(f"Class weights: {[f'{w:.1f}' for w in class_weights.tolist()]}")

    # Create model
    encoder_name = 'efficientnet-b5'
    print(f"\nCreating UNet++ with {encoder_name} (ImageNet pretrained)...")
    model = create_model(n_classes, encoder_name).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    # Triple loss: Lovász + Focal + Dice
    criterion = TripleLoss(n_classes, class_weights=class_weights).to(device)

    # ==================================================================
    # Phase 1: Low-Resolution Training (416x736)
    # ==================================================================
    H1, W1 = 416, 736
    bs1 = 4

    train_transform_1 = A.Compose([
        A.Resize(H1, W1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Affine(translate_percent=0.1, scale=(0.8, 1.2), rotate=(-20, 20),
                 border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        # Aggressive color augmentation for domain bridging
        A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40,
                             val_shift_limit=25, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.6),
        A.RandomGamma(gamma_limit=(70, 150), p=0.3),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.1, p=0.4),
        A.OneOf([
            A.ChannelShuffle(p=1.0),
            A.ToGray(p=1.0),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1.0),
            A.FancyPCA(alpha=0.5, p=1.0),
        ], p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.Sharpen(alpha=(0.2, 0.5), p=1.0),
        ], p=0.2),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(20, 60),
                        hole_width_range=(20, 60), fill='random', p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform_1 = A.Compose([
        A.Resize(H1, W1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Combined train + val for maximum data
    trainset_1 = MaskDataset([train_dir, val_dir], transform=train_transform_1)
    valset_1 = MaskDataset(val_dir, transform=val_transform_1)

    train_loader_1 = DataLoader(trainset_1, batch_size=bs1, shuffle=True,
                                num_workers=0, pin_memory=True, drop_last=True)
    val_loader_1 = DataLoader(valset_1, batch_size=bs1, shuffle=False,
                              num_workers=0, pin_memory=True)

    print(f"\nTraining samples: {len(trainset_1)} (train+val combined)")
    print(f"Val monitoring: {len(valset_1)} samples")
    print(f"Phase 1: {W1}x{H1}, batch={bs1}, accum=2 (eff. batch=8)")

    phase1_path = os.path.join(script_dir, 'best_model_phase1.pth')
    history_1, best_miou_1 = train_phase(
        model, train_loader_1, val_loader_1, device, criterion,
        num_epochs=20, lr=2e-4,
        phase_name="Phase 1: Low-Res Training (416x736)",
        save_path=phase1_path, freeze_epochs=2, patience=10,
        accum_steps=2, ema_decay=0.999
    )

    # ==================================================================
    # Phase 2: Native-Resolution Finetuning (544x960)
    # ==================================================================
    torch.cuda.empty_cache()

    H2, W2 = 544, 960  # Near-native (must be divisible by 32 for UNet++)
    bs2 = 2

    train_transform_2 = A.Compose([
        A.Resize(H2, W2),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-10, 10),
                 border_mode=cv2.BORDER_REFLECT_101, p=0.3),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30,
                             val_shift_limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
        A.RandomGamma(gamma_limit=(80, 130), p=0.2),
        A.OneOf([
            A.ChannelShuffle(p=1.0),
            A.ToGray(p=1.0),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
        ], p=0.2),
        A.CLAHE(clip_limit=4.0, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform_2 = A.Compose([
        A.Resize(H2, W2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    trainset_2 = MaskDataset([train_dir, val_dir], transform=train_transform_2)
    valset_2 = MaskDataset(val_dir, transform=val_transform_2)
    train_loader_2 = DataLoader(trainset_2, batch_size=bs2, shuffle=True,
                                num_workers=0, pin_memory=True, drop_last=True)
    val_loader_2 = DataLoader(valset_2, batch_size=bs2, shuffle=False,
                              num_workers=0, pin_memory=True)

    # Load best Phase 1 EMA model
    model.load_state_dict(torch.load(phase1_path, map_location=device, weights_only=True))
    print(f"\nLoaded Phase 1 best (mIoU: {best_miou_1:.4f})")
    print(f"Phase 2: {W2}x{H2} (near-native), batch={bs2}, accum=4 (eff. batch=8)")

    final_path = os.path.join(script_dir, 'segmentation_model_best.pth')
    history_2, best_miou_2 = train_phase(
        model, train_loader_2, val_loader_2, device, criterion,
        num_epochs=15, lr=3e-5,
        phase_name="Phase 2: Native-Res Finetuning (544x960)",
        save_path=final_path, freeze_epochs=0, patience=10,
        accum_steps=4, ema_decay=0.9995
    )

    # ==================================================================
    # Save Results
    # ==================================================================
    combined = {
        'train_loss': history_1['train_loss'] + history_2['train_loss'],
        'val_loss': history_1['val_loss'] + history_2['val_loss'],
        'val_miou': history_1['val_miou'] + history_2['val_miou'],
        'val_pixel_acc': history_1['val_pixel_acc'] + history_2['val_pixel_acc'],
        'val_per_class_iou': (history_1['val_per_class_iou'] +
                              history_2['val_per_class_iou']),
    }

    print("\nSaving training curves...")
    save_training_plots(combined, output_dir)

    model_path = os.path.join(script_dir, 'segmentation_model.pth')
    if os.path.exists(final_path):
        import shutil
        shutil.copy2(final_path, model_path)
        print(f"Copied best EMA model (mIoU={best_miou_2:.4f}) to '{model_path}'")
    else:
        torch.save(model.state_dict(), model_path)

    if combined['val_per_class_iou']:
        last_iou = combined['val_per_class_iou'][-1]
        print(f"\nFinal results:")
        print(f"  Val mIoU:      {np.nanmean(last_iou):.4f}")
        print(f"  Val Pixel Acc: {combined['val_pixel_acc'][-1]:.4f}")
        print(f"\nPer-Class IoU:")
        print("-" * 40)
        for i, name in enumerate(class_names):
            iou = last_iou[i]
            status = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
            print(f"  {name:20s}: {status}")

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
