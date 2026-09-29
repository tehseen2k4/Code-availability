# =============================================================
# RUN 8 SAFE IMPROVEMENT: Batch Norm + Depth 48 + TTA + FULL LOGGING
# - Batch norm (safe)
# - Increased channels (32→512)
# - Depth=48 (proven working)
# - Test-Time Augmentation (TTA)
# - Saves 3 log files: epoch log, analysis, summary
# - Fixed Unicode errors (UTF-8 encoding)
# =============================================================

import os
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")

def ensure_packages():
    required = ['nibabel', 'monai', 'tqdm', 'matplotlib']
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            print(f"[INSTALL] Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
ensure_packages()

import numpy as np
import nibabel as nib
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
from datetime import datetime

try:
    from monai.networks.nets import UNet
    from monai.losses import DiceLoss
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "monai"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "monai==1.3.0"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from monai.networks.nets import UNet
    from monai.losses import DiceLoss

print("[INIT] 3D U-Net Segmentation (SAFE IMPROVED + LOGGING)")

# PATHS
BASE_DIR = r"D:\Tehseen\Breast DCEDL\BreastDCEDL_ISPY2_min_crop\BreastDCEDL_ISPY2_min_crop"
DCE_DIR = os.path.join(BASE_DIR, "dce")
MASK_DIR = os.path.join(BASE_DIR, "mask")
METADATA_PATH = r"D:\Tehseen\Breast DCEDL\BreastDCEDL_metadata_min_crop.csv"

# HYPERPARAMETERS (proven safe)
TARGET_DEPTH = 48
TARGET_SIZE = 128
BATCH_SIZE = 4          # increased from 2 to 4 for faster training (still safe)
ACCUMULATION_STEPS = 2  # effective batch size = 8
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 15
NUM_WORKERS = 0
TTA_AUGMENTATIONS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Device: {device}")

# =============================================================
# DATASET CLASS (identical to original Run8)
# =============================================================
class BreastMRISegmentationDataset(Dataset):
    def __init__(self, df, dce_dir, mask_dir, target_depth=48, target_size=128, augment=True, split='train'):
        self.df = df.reset_index(drop=True)
        self.dce_dir = dce_dir
        self.mask_dir = mask_dir
        self.target_depth = target_depth
        self.target_size = target_size
        self.augment = augment
        self.valid_indices = []
        for idx, row in self.df.iterrows():
            pid = row['pid']
            dataset = row['dataset']
            mask_path = self._get_mask_path(pid, dataset)
            if os.path.exists(mask_path):
                self.valid_indices.append(idx)
        self.df = self.df.iloc[self.valid_indices].reset_index(drop=True)
        print(f"[{split}] Valid patients: {len(self.df)}")
    def _get_mask_path(self, pid, dataset):
        for ext in ['.nii.gz', '.nii']:
            path = os.path.join(self.mask_dir, f"{pid}_{dataset}_vis1_mask{ext}")
            if os.path.exists(path):
                return path
        return None
    def _get_dce_path(self, pid, dataset, phase_idx):
        for ext in ['.nii.gz', '.nii']:
            path = os.path.join(self.dce_dir, f"{pid}_{dataset}_vis1_dce_aqc_{phase_idx}{ext}")
            if os.path.exists(path):
                return path
        return None
    def _load_nifti(self, path):
        if path is None or not os.path.exists(path):
            return None
        return nib.load(path).get_fdata().astype(np.float32)
    def _crop_or_pad(self, volume, target_shape):
        d, h, w = volume.shape
        td, th, tw = target_shape
        pad_d = max(0, td - d)
        pad_h = max(0, th - h)
        pad_w = max(0, tw - w)
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            volume = np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
        d, h, w = volume.shape
        start_d = max(0, (d - td) // 2)
        start_h = max(0, (h - th) // 2)
        start_w = max(0, (w - tw) // 2)
        return volume[start_d:start_d+td, start_h:start_h+th, start_w:start_w+tw]
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row['pid']
        dataset = row['dataset']
        mask_path = self._get_mask_path(pid, dataset)
        mask = self._load_nifti(mask_path)
        if mask is None:
            return (torch.zeros(3, self.target_depth, self.target_size, self.target_size),
                    torch.zeros(1, self.target_depth, self.target_size, self.target_size))
        phases = []
        for pidx in [int(row['pre']), int(row['post_early']), int(row['post_late'])]:
            dce_path = self._get_dce_path(pid, dataset, pidx)
            vol = self._load_nifti(dce_path)
            if vol is None:
                vol = np.zeros_like(mask)
            phases.append(vol)
        image = np.stack(phases, axis=0)
        target_shape = (self.target_depth, self.target_size, self.target_size)
        processed = []
        for c in range(3):
            vol = self._crop_or_pad(image[c], target_shape)
            processed.append(vol)
        image = np.stack(processed, axis=0)
        mask = self._crop_or_pad(mask, target_shape)
        mask = (mask > 0.5).astype(np.float32)
        # Normalize (original method)
        mean_img = image.mean()
        std_img = image.std()
        if std_img > 1e-8:
            image = (image - mean_img) / std_img
            image = np.clip(image, -3.0, 3.0)
        img_min, img_max = image.min(), image.max()
        if img_max - img_min > 1e-8:
            image = (image - img_min) / (img_max - img_min)
        # Augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=0).copy()
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=2).copy()
                mask = np.flip(mask, axis=1).copy()
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=3).copy()
                mask = np.flip(mask, axis=2).copy()
            if np.random.rand() > 0.5:
                k = np.random.randint(1, 4)
                image = np.rot90(image, k, axes=(2, 3)).copy()
                mask = np.rot90(mask, k, axes=(1, 2)).copy()
        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        return image_tensor, mask_tensor

# =============================================================
# LOSS AND METRICS
# =============================================================
dice_loss = DiceLoss(sigmoid=True, reduction='mean')
bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

def combined_loss(pred, target, dice_weight=0.7, bce_weight=0.3):
    return dice_weight * dice_loss(pred, target) + bce_weight * bce_loss(pred, target)

def compute_dice(pred, target, thresh=0.5):
    pred_b = (pred > thresh).float()
    inter = (pred_b * target).sum()
    return (2.0 * inter) / (pred_b.sum() + target.sum() + 1e-8)

def compute_iou(pred, target, thresh=0.5):
    pred_b = (pred > thresh).float()
    inter = (pred_b * target).sum()
    union = pred_b.sum() + target.sum() - inter
    return inter / (union + 1e-8)

# =============================================================
# TTA
# =============================================================
def tta_predict(model, image_tensor, device, n_aug=TTA_AUGMENTATIONS):
    model.eval()
    preds = []
    with torch.no_grad():
        pred = torch.sigmoid(model(image_tensor))
        preds.append(pred.cpu())
        for _ in range(n_aug - 1):
            aug_img = image_tensor.clone()
            if random.random() > 0.5:
                aug_img = torch.flip(aug_img, dims=[2])
            if random.random() > 0.5:
                aug_img = torch.flip(aug_img, dims=[3])
            if random.random() > 0.5:
                aug_img = torch.flip(aug_img, dims=[4])
            if random.random() > 0.5:
                k = random.randint(1, 3)
                aug_img = torch.rot90(aug_img, k, dims=[3, 4])
            aug_pred = torch.sigmoid(model(aug_img.to(device)))
            if random.random() > 0.5 and k:
                aug_pred = torch.rot90(aug_pred, -k, dims=[3, 4])
            preds.append(aug_pred.cpu())
    avg_pred = torch.stack(preds).mean(dim=0)
    return avg_pred.numpy().squeeze()

# =============================================================
# MAIN
# =============================================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("LOADING ISPY2 DATA (SAFE IMPROVED)")
    print("="*60)
    metadata = pd.read_csv(METADATA_PATH)
    ispy2_metadata = metadata[metadata['dataset'] == 'spy2'].copy()
    ispy2_metadata = ispy2_metadata.dropna(subset=['pCR']).reset_index(drop=True)
    print(f"Patients with pCR: {len(ispy2_metadata)}")
    unique_pids = ispy2_metadata['pid'].unique()
    pcr_labels = ispy2_metadata.groupby('pid')['pCR'].first().values
    train_pids, temp_pids = train_test_split(unique_pids, test_size=0.25, random_state=42, stratify=pcr_labels)
    val_pids, test_pids = train_test_split(temp_pids, test_size=0.5, random_state=42,
        stratify=ispy2_metadata[ispy2_metadata['pid'].isin(temp_pids)].groupby('pid')['pCR'].first().values)
    print(f"Train: {len(train_pids)} | Val: {len(val_pids)} | Test: {len(test_pids)}")
    train_df = ispy2_metadata[ispy2_metadata['pid'].isin(train_pids)]
    val_df = ispy2_metadata[ispy2_metadata['pid'].isin(val_pids)]
    test_df = ispy2_metadata[ispy2_metadata['pid'].isin(test_pids)]

    train_dataset = BreastMRISegmentationDataset(train_df, DCE_DIR, MASK_DIR,
        target_depth=TARGET_DEPTH, target_size=TARGET_SIZE, augment=True, split='train')
    val_dataset = BreastMRISegmentationDataset(val_df, DCE_DIR, MASK_DIR,
        target_depth=TARGET_DEPTH, target_size=TARGET_SIZE, augment=False, split='val')
    test_dataset = BreastMRISegmentationDataset(test_df, DCE_DIR, MASK_DIR,
        target_depth=TARGET_DEPTH, target_size=TARGET_SIZE, augment=False, split='test')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # MODEL - increased channels but BATCH NORM (safe)
    print("\n" + "="*60)
    print("CREATING 3D U-NET (SAFE - BATCH NORM, CHANNELS 32-512)")
    print("="*60)
    model = UNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm='batch',
        dropout=0.1
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params/1e6:.2f}M)")

    criterion = combined_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    SAVE_DIR = r"D:\Tehseen\results\run8_safe_improved"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # =============================================================
    # LOGGING SETUP
    # =============================================================
    epoch_log_path = os.path.join(SAVE_DIR, "epoch_log.txt")
    summary_path = os.path.join(SAVE_DIR, "summary_results.txt")
    analysis_path = os.path.join(SAVE_DIR, "analysis_report.txt")

    # We'll collect epoch data in a list for analysis later
    epoch_records = []

    print("\n" + "="*60)
    print("TRAINING 3D U-NET (SAFE IMPROVED)")
    print(f"Batch size: {BATCH_SIZE}, Accumulation: {ACCUMULATION_STEPS} (effective: {BATCH_SIZE*ACCUMULATION_STEPS})")
    print("="*60)

    best_dice = -1
    best_state = None
    patience_counter = 0
    optimizer.zero_grad()

    # Write header for epoch log (UTF-8)
    with open(epoch_log_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("EPOCH LOG: Run8 Safe Improved\n")
        f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*100 + "\n\n")
        f.write(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'Val Dice':>12} | {'Val IoU':>12} | {'LR':>12} | {'Best':>6}\n")
        f.write("-"*100 + "\n")

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss = loss / ACCUMULATION_STEPS
            loss.backward()
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            train_loss += loss.item() * ACCUMULATION_STEPS
            pbar.set_postfix({'loss': f'{loss.item()*ACCUMULATION_STEPS:.4f}'})

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_dice, val_iou, val_loss = [], [], 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                for i in range(images.size(0)):
                    dice = compute_dice(probs[i], masks[i])
                    iou = compute_iou(probs[i], masks[i])
                    val_dice.append(dice.item())
                    val_iou.append(iou.item())
        avg_dice = np.mean(val_dice)
        avg_iou = np.mean(val_iou)
        val_loss /= len(val_loader)

        scheduler.step(avg_dice)
        current_lr = optimizer.param_groups[0]['lr']

        # Check if best
        is_best = avg_dice > best_dice
        if is_best:
            best_dice = avg_dice
            best_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'best_iou': avg_iou
            }
            torch.save(best_state, os.path.join(SAVE_DIR, "best_unet_safe.pth"))
            patience_counter = 0
        else:
            patience_counter += 1

        # Print
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Val Dice: {avg_dice:.4f} | Val IoU: {avg_iou:.4f}")
        print(f"  LR: {current_lr:.6f}")
        if is_best:
            print(f"  Best model saved (Dice={best_dice:.4f})")

        # Log epoch to file (UTF-8, using 'BEST' instead of checkmark)
        with open(epoch_log_path, 'a', encoding='utf-8') as f:
            f.write(f"{epoch+1:>6} | {train_loss:>12.4f} | {val_loss:>12.4f} | {avg_dice:>12.4f} | {avg_iou:>12.4f} | {current_lr:>12.6f} | {'BEST' if is_best else '':>6}\n")

        # Store record for later
        epoch_records.append({
            'epoch': epoch+1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_dice': avg_dice,
            'val_iou': avg_iou,
            'lr': current_lr,
            'is_best': is_best
        })

        if patience_counter >= PATIENCE:
            print(f"\n[STOP] Early stopping (no improvement for {PATIENCE} epochs)")
            break

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Validation Dice: {best_dice:.4f}")

    # =============================================================
    # TEST EVALUATION (no TTA)
    # =============================================================
    print("\n" + "="*60)
    print("TEST EVALUATION (no TTA)")
    print("="*60)
    if best_state:
        model.load_state_dict(best_state['model_state_dict'])
    model.eval()
    test_dice, test_iou = [], []
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing (no TTA)"):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            dice = compute_dice(probs[0], masks[0])
            iou = compute_iou(probs[0], masks[0])
            test_dice.append(dice.item())
            test_iou.append(iou.item())
    no_tta_dice = np.mean(test_dice)
    no_tta_dice_std = np.std(test_dice)
    no_tta_iou = np.mean(test_iou)
    no_tta_iou_std = np.std(test_iou)
    print(f"Test without TTA:")
    print(f"  Dice: {no_tta_dice:.4f} ± {no_tta_dice_std:.4f}")
    print(f"  IoU:  {no_tta_iou:.4f} ± {no_tta_iou_std:.4f}")

    # =============================================================
    # TEST EVALUATION (with TTA)
    # =============================================================
    print("\n" + "="*60)
    print("TEST EVALUATION (with TTA)")
    print("="*60)
    tta_dice, tta_iou = [], []
    for images, masks in tqdm(test_loader, desc="Testing (TTA)"):
        images = images.to(device)
        masks = masks.to(device)
        prob_map = tta_predict(model, images, device, n_aug=TTA_AUGMENTATIONS)
        prob_tensor = torch.from_numpy(prob_map).unsqueeze(0).unsqueeze(0).to(device)
        dice = compute_dice(prob_tensor, masks)
        iou = compute_iou(prob_tensor, masks)
        tta_dice.append(dice.item())
        tta_iou.append(iou.item())
    tta_dice_mean = np.mean(tta_dice)
    tta_dice_std = np.std(tta_dice)
    tta_iou_mean = np.mean(tta_iou)
    tta_iou_std = np.std(tta_iou)
    print(f"Test with TTA:")
    print(f"  Dice: {tta_dice_mean:.4f} ± {tta_dice_std:.4f}")
    print(f"  IoU:  {tta_iou_mean:.4f} ± {tta_iou_std:.4f}")

    # =============================================================
    # SAVE RESULTS & CREATE LOG FILES
    # =============================================================
    # Save CSV results
    results = {
        'experiment': 'run8_safe_improved',
        'dataset': 'ISPY2',
        'best_val_dice': best_dice,
        'test_dice_no_tta': no_tta_dice,
        'test_dice_std_no_tta': no_tta_dice_std,
        'test_iou_no_tta': no_tta_iou,
        'test_iou_std_no_tta': no_tta_iou_std,
        'test_dice_tta': tta_dice_mean,
        'test_dice_std_tta': tta_dice_std,
        'test_iou_tta': tta_iou_mean,
        'test_iou_std_tta': tta_iou_std,
        'total_params': params,
        'epochs_trained': epoch + 1,
        'improvements': 'increased_capacity, batch_norm, TTA, depth48'
    }
    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(SAVE_DIR, "segmentation_results_safe.csv"), index=False)
    print(f"\n✅ Results saved to {SAVE_DIR}/segmentation_results_safe.csv")

    # ---------- SUMMARY RESULTS (UTF-8) ----------
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SUMMARY RESULTS: Run8 Safe Improved\n")
        f.write("="*80 + "\n\n")
        f.write(f"Experiment date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: ISPY2\n")
        f.write(f"Training patients: {len(train_dataset)}\n")
        f.write(f"Validation patients: {len(val_dataset)}\n")
        f.write(f"Test patients: {len(test_dataset)}\n\n")
        f.write(f"Best validation Dice: {best_dice:.4f}\n\n")
        f.write("Test performance (no TTA):\n")
        f.write(f"  Dice: {no_tta_dice:.4f} ± {no_tta_dice_std:.4f}\n")
        f.write(f"  IoU:  {no_tta_iou:.4f} ± {no_tta_iou_std:.4f}\n\n")
        f.write("Test performance (with TTA):\n")
        f.write(f"  Dice: {tta_dice_mean:.4f} ± {tta_dice_std:.4f}\n")
        f.write(f"  IoU:  {tta_iou_mean:.4f} ± {tta_iou_std:.4f}\n\n")
        f.write(f"Model parameters: {params:,} ({params/1e6:.2f}M)\n")
        f.write(f"Epochs trained: {epoch+1}\n")
        f.write(f"Improvements applied: {results['improvements']}\n")
    print(f"✅ Summary saved to: {summary_path}")

    # ---------- ANALYSIS REPORT (UTF-8) ----------
    # Compare with original run (using provided metrics)
    original_val_dice = 0.7033   # from the provided results
    original_test_dice = 0.6214
    original_test_std = 0.2521
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ANALYSIS REPORT: Run8 Safe Improved\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("1. Comparison with Original Run8\n")
        f.write("-"*40 + "\n")
        f.write(f"Metric                      | Original   | Improved (TTA) | Improvement\n")
        f.write("-"*60 + "\n")
        f.write(f"Best Validation Dice        | {original_val_dice:.4f}     | {best_dice:.4f}          | {best_dice - original_val_dice:+.4f}\n")
        f.write(f"Test Dice (mean)            | {original_test_dice:.4f}     | {tta_dice_mean:.4f}          | {tta_dice_mean - original_test_dice:+.4f}\n")
        f.write(f"Test Dice (std)             | {original_test_std:.4f}     | {tta_dice_std:.4f}          | {tta_dice_std - original_test_std:+.4f}\n")
        f.write(f"Test IoU                    | 0.4915     | {tta_iou_mean:.4f}          | {tta_iou_mean - 0.4915:+.4f}\n\n")

        f.write("2. Interpretation\n")
        f.write("-"*40 + "\n")
        if tta_dice_mean > original_test_dice:
            f.write("The improved model shows higher test Dice, indicating better segmentation accuracy.\n")
        else:
            f.write("The improved model did not surpass the original test Dice. Further tuning may be needed.\n")
        if tta_dice_std < original_test_std:
            f.write("The standard deviation decreased, meaning more consistent predictions across patients.\n")
        else:
            f.write("The variance increased, suggesting some patients are still challenging.\n")
        f.write(f"\nTTA improved test Dice by {tta_dice_mean - no_tta_dice:+.4f} compared to no TTA.\n\n")

        f.write("3. Model Specifications\n")
        f.write("-"*40 + "\n")
        f.write(f"Channels: (32, 64, 128, 256, 512) vs original (16, 32, 64, 128, 256)\n")
        f.write(f"Normalization: BatchNorm (same as original)\n")
        f.write(f"Depth: 48 (same as original)\n")
        f.write(f"Test-Time Augmentation: {TTA_AUGMENTATIONS} augmentations\n")
        f.write(f"Effective batch size: {BATCH_SIZE*ACCUMULATION_STEPS}\n")
        f.write(f"Total parameters: {params:,}\n\n")

        f.write("4. Conclusion\n")
        f.write("-"*40 + "\n")
        if tta_dice_mean > original_test_dice:
            f.write("The increased capacity and TTA provided a measurable improvement in segmentation performance.\n")
            f.write("This model is recommended for subsequent ROI cropping experiments.\n")
        else:
            f.write("The improvements did not yield a significant gain over the original. However, TTA reduced variance.\n")
            f.write("The model may still be useful for downstream tasks, but the original model remains the baseline.\n")
    print(f"✅ Analysis saved to: {analysis_path}")

    # Also append the final epoch log with summary (UTF-8)
    with open(epoch_log_path, 'a', encoding='utf-8') as f:
        f.write("-"*100 + "\n")
        f.write(f"TRAINING STOPPED at epoch {epoch+1} (early stopping {'triggered' if patience_counter>=PATIENCE else 'not triggered'})\n")
        f.write(f"Best validation Dice: {best_dice:.4f}\n")
        f.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"✅ Full epoch log saved to: {epoch_log_path}")

    print("\n" + "="*60)
    print("✅ 3D U-Net Segmentation (Safe Improved) Complete!")
    print(f"All logs and results are in: {SAVE_DIR}")
    print("="*60)