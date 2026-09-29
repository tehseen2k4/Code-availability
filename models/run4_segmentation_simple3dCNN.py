# =============================================================
# SIMPLE 3D CNN (MEMORY‑EFFICIENT) – FAST BASELINE
# - No checkpoint wrapper (compatible with all PyTorch versions)
# - Smaller batch size (2) to save memory
# - ~0.28M parameters
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
from datetime import datetime

try:
    from monai.losses import DiceLoss
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "monai"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "monai==1.3.0"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from monai.losses import DiceLoss

print("[INIT] Simple 3D CNN (Memory Efficient)")

# Set environment variable to reduce fragmentation (optional)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -------------------- PATHS --------------------
BASE_DIR = r"D:\Tehseen\Breast DCEDL\BreastDCEDL_ISPY2_min_crop\BreastDCEDL_ISPY2_min_crop"
DCE_DIR = os.path.join(BASE_DIR, "dce")
MASK_DIR = os.path.join(BASE_DIR, "mask")
METADATA_PATH = r"D:\Tehseen\Breast DCEDL\BreastDCEDL_metadata_min_crop.csv"

# -------------------- HYPERPARAMETERS --------------------
TARGET_DEPTH = 48
TARGET_SIZE = 128
BATCH_SIZE = 2          # Small batch to save memory
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8
NUM_WORKERS = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Device: {device}")

# Enable memory efficient settings
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

# =============================================================
# DATASET CLASS (same as before)
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
        # Normalize
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
# MEMORY‑EFFICIENT SIMPLE 3D CNN (NO CHECKPOINT WRAPPER)
# =============================================================
class Simple3DCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Encoder with very few channels
        self.enc1 = self._conv_block(in_channels, 4)
        self.enc2 = self._conv_block(4, 8)
        self.enc3 = self._conv_block(8, 16)
        self.enc4 = self._conv_block(16, 32)
        # Bottleneck
        self.bottleneck = self._conv_block(32, 64)
        # Decoder with transposed conv
        self.dec4 = self._deconv_block(64, 32)
        self.dec3 = self._deconv_block(32, 16)
        self.dec2 = self._deconv_block(16, 8)
        self.dec1 = self._deconv_block(8, 4)
        self.final = nn.Conv3d(4, out_channels, kernel_size=1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _deconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder with max pooling
        e1 = self.enc1(x)                      # 48,128,128
        e2 = self.enc2(F.max_pool3d(e1, 2))    # 24,64,64
        e3 = self.enc3(F.max_pool3d(e2, 2))    # 12,32,32
        e4 = self.enc4(F.max_pool3d(e3, 2))    # 6,16,16
        b = self.bottleneck(F.max_pool3d(e4, 2))  # 3,8,8
        # Decoder with transposed conv
        d4 = self.dec4(b)      # 6,16,16
        d3 = self.dec3(d4)     # 12,32,32
        d2 = self.dec2(d3)     # 24,64,64
        d1 = self.dec1(d2)     # 48,128,128
        out = self.final(d1)
        return out

# =============================================================
# LOSS & METRICS
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
# MAIN
# =============================================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("LOADING DATA")
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

    train_dataset = BreastMRISegmentationDataset(train_df, DCE_DIR, MASK_DIR, augment=True, split='train')
    val_dataset = BreastMRISegmentationDataset(val_df, DCE_DIR, MASK_DIR, augment=False, split='val')
    test_dataset = BreastMRISegmentationDataset(test_df, DCE_DIR, MASK_DIR, augment=False, split='test')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # =============================================================
    # MODEL
    # =============================================================
    print("\n" + "="*60)
    print("CREATING Memory‑Efficient Simple 3D CNN")
    print("="*60)
    model = Simple3DCNN(in_channels=3, out_channels=1).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params/1e6:.2f}M)")

    criterion = combined_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    SAVE_DIR = r"D:\Tehseen\results\segmentation\simple_cnn"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # =============================================================
    # TRAINING
    # =============================================================
    print("\n" + "="*60)
    print("TRAINING Simple 3D CNN")
    print(f"Batch size: {BATCH_SIZE}")
    print("="*60)

    best_dice = -1
    best_state = None
    patience_counter = 0

    with open(os.path.join(SAVE_DIR, "epoch_log.txt"), 'w', encoding='utf-8') as f:
        f.write("="*100 + "\nEPOCH LOG: Simple 3D CNN\n")
        f.write(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + "="*100 + "\n\n")
        f.write(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'Val Dice':>12} | {'Val IoU':>12} | {'LR':>12} | {'Best':>6}\n")
        f.write("-"*100 + "\n")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_dice, val_iou, val_loss = [], [], 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                for i in range(images.size(0)):
                    val_dice.append(compute_dice(probs[i], masks[i]).item())
                    val_iou.append(compute_iou(probs[i], masks[i]).item())
        avg_dice = np.mean(val_dice)
        avg_iou = np.mean(val_iou)
        val_loss /= len(val_loader)

        scheduler.step(avg_dice)
        current_lr = optimizer.param_groups[0]['lr']

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
            torch.save(best_state, os.path.join(SAVE_DIR, "best_simple_cnn.pth"))
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Val Dice: {avg_dice:.4f} | Val IoU: {avg_iou:.4f}")
        print(f"  LR: {current_lr:.6f}" + (" ✅ BEST" if is_best else ""))

        with open(os.path.join(SAVE_DIR, "epoch_log.txt"), 'a', encoding='utf-8') as f:
            f.write(f"{epoch+1:>6} | {train_loss:>12.4f} | {val_loss:>12.4f} | {avg_dice:>12.4f} | {avg_iou:>12.4f} | {current_lr:>12.6f} | {'BEST' if is_best else '':>6}\n")

        if patience_counter >= PATIENCE:
            print(f"\n[STOP] Early stopping")
            break

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print(f"Best Dice: {best_dice:.4f}")
    print("="*60)

    # =============================================================
    # TEST
    # =============================================================
    print("\nTEST EVALUATION")
    if best_state:
        model.load_state_dict(best_state['model_state_dict'])
    model.eval()
    test_dice, test_iou = [], []
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)
            probs = torch.sigmoid(model(images))
            test_dice.append(compute_dice(probs[0], masks[0]).item())
            test_iou.append(compute_iou(probs[0], masks[0]).item())
    test_dice_mean = np.mean(test_dice)
    test_dice_std = np.std(test_dice)
    test_iou_mean = np.mean(test_iou)
    test_iou_std = np.std(test_iou)
    print(f"Test Dice: {test_dice_mean:.4f} ± {test_dice_std:.4f}")
    print(f"Test IoU:  {test_iou_mean:.4f} ± {test_iou_std:.4f}")

    # =============================================================
    # SAVE RESULTS
    # =============================================================
    results = {
        'experiment': 'simple_cnn_memory_efficient',
        'dataset': 'ISPY2',
        'best_val_dice': best_dice,
        'best_val_iou': best_state['best_iou'] if best_state else None,
        'val_dice_final': avg_dice,
        'val_iou_final': avg_iou,
        'val_loss_final': val_loss,
        'train_loss_final': train_loss,
        'test_dice': test_dice_mean,
        'test_dice_std': test_dice_std,
        'test_iou': test_iou_mean,
        'test_iou_std': test_iou_std,
        'total_params': params,
        'epochs_trained': epoch + 1
    }
    pd.DataFrame([results]).to_csv(os.path.join(SAVE_DIR, "simple_cnn_results.csv"), index=False)
    print(f"\n✅ Results saved to {SAVE_DIR}/simple_cnn_results.csv")

    print("\n" + "="*60)
    print("✅ SIMPLE 3D CNN COMPLETE!")
    print(f"Best Val Dice: {best_dice:.4f}")
    print(f"Test Dice: {test_dice_mean:.4f} ± {test_dice_std:.4f}")
    print(f"Parameters: {params:,} ({params/1e6:.2f}M)")
    print(f"Results in: {SAVE_DIR}")
    print("="*60)