# =============================================================
# RUN 8: 3D U-Net Segmentation (FULLY FIXED + MAIN GUARD)
# =============================================================

import os
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")

# Auto-install missing packages
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
import matplotlib.pyplot as plt

# MONAI imports
try:
    from monai.networks.nets import UNet
    from monai.losses import DiceLoss
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "monai"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "monai==1.3.0"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from monai.networks.nets import UNet
    from monai.losses import DiceLoss

print("[INIT] 3D U-Net Segmentation for Breast Tumors")

# -------------------------------------------------------------
# PATHS
# -------------------------------------------------------------
BASE_DIR = r"D:\Tehseen\Breast DCEDL\BreastDCEDL_ISPY2_min_crop\BreastDCEDL_ISPY2_min_crop"
DCE_DIR = os.path.join(BASE_DIR, "dce")
MASK_DIR = os.path.join(BASE_DIR, "mask")
METADATA_PATH = r"D:\Tehseen\Breast DCEDL\BreastDCEDL_metadata_min_crop.csv"

# -------------------------------------------------------------
# HYPERPARAMETERS
# -------------------------------------------------------------
TARGET_DEPTH = 48
TARGET_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 10
NUM_WORKERS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Device: {device}")

if device.type == "cpu":
    print("[WARNING] Using CPU! Reducing batch size...")
    BATCH_SIZE = 1
    NUM_WORKERS = 0

# =============================================================
# DATASET CLASS - FULLY FIXED
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
        
        # Stack phases: [C, D, H, W]
        image = np.stack(phases, axis=0)
        target_shape = (self.target_depth, self.target_size, self.target_size)
        
        # Process each channel
        processed = []
        for c in range(3):
            vol = self._crop_or_pad(image[c], target_shape)
            processed.append(vol)
        image = np.stack(processed, axis=0)  # [C, D, H, W]
        
        # Process mask: [D, H, W]
        mask = self._crop_or_pad(mask, target_shape)
        mask = (mask > 0.5).astype(np.float32)  # [D, H, W]
        
        # Normalize image
        mean_img = image.mean()
        std_img = image.std()
        if std_img > 1e-8:
            image = (image - mean_img) / std_img
            image = np.clip(image, -3.0, 3.0)
        img_min, img_max = image.min(), image.max()
        if img_max - img_min > 1e-8:
            image = (image - img_min) / (img_max - img_min)
        
        # Data augmentation - FIXED AXIS
        if self.augment:
            # Flip along Depth (axis=1)
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=0).copy()
            # Flip along Height (axis=2)
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=2).copy()
                mask = np.flip(mask, axis=1).copy()
            # Flip along Width (axis=3)
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=3).copy()
                mask = np.flip(mask, axis=2).copy()
            # Rotate 90 degrees in H-W plane (axes 2,3)
            if np.random.rand() > 0.5:
                k = np.random.randint(1, 4)
                image = np.rot90(image, k, axes=(2, 3)).copy()
                mask = np.rot90(mask, k, axes=(1, 2)).copy()
        
        # Convert to tensors: both [C, D, H, W]
        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()  # [1, D, H, W]
        
        return image_tensor, mask_tensor

# =============================================================
# TRAINING FUNCTIONS
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
# MAIN EXECUTION - WITH GUARD
# =============================================================
if __name__ == '__main__':
    
    print("\n" + "="*60)
    print("LOADING ISPY2 DATA")
    print("="*60)
    
    metadata = pd.read_csv(METADATA_PATH)
    ispy2_metadata = metadata[metadata['dataset'] == 'spy2'].copy()
    ispy2_metadata = ispy2_metadata.dropna(subset=['pCR']).reset_index(drop=True)
    print(f"Patients with pCR: {len(ispy2_metadata)}")
    
    # Stratified split
    unique_pids = ispy2_metadata['pid'].unique()
    pcr_labels = ispy2_metadata.groupby('pid')['pCR'].first().values
    
    train_pids, temp_pids = train_test_split(
        unique_pids, test_size=0.25, random_state=42, stratify=pcr_labels
    )
    val_pids, test_pids = train_test_split(
        temp_pids, test_size=0.5, random_state=42,
        stratify=ispy2_metadata[ispy2_metadata['pid'].isin(temp_pids)].groupby('pid')['pCR'].first().values
    )
    
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
    print("CREATING 3D U-NET")
    print("="*60)
    
    model = UNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm='batch',
        dropout=0.1
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params/1e6:.2f}M)")
    
    # =============================================================
    # TRAINING SETUP
    # =============================================================
    criterion = combined_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    SAVE_DIR = r"D:\Tehseen\results\run8_segmentation"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # =============================================================
    # TRAINING LOOP
    # =============================================================
    print("\n" + "="*60)
    print("TRAINING 3D U-NET")
    print("="*60)
    
    best_dice = -1
    best_state = None
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
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
        val_dice = []
        val_iou = []
        val_loss = 0
        
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
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Val Dice: {avg_dice:.4f} | Val IoU: {avg_iou:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        if avg_dice > best_dice:
            best_dice = avg_dice
            best_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'best_iou': avg_iou
            }
            torch.save(best_state, os.path.join(SAVE_DIR, "best_unet.pth"))
            print(f"  ✅ Best model saved (Dice={best_dice:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"\n[STOP] Early stopping (no improvement for {PATIENCE} epochs)")
            break
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Validation Dice: {best_dice:.4f}")
    
    # =============================================================
    # TEST EVALUATION
    # =============================================================
    print("\n" + "="*60)
    print("TEST EVALUATION")
    print("="*60)
    
    if best_state is not None:
        model.load_state_dict(best_state['model_state_dict'])
    
    model.eval()
    test_dice = []
    test_iou = []
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            
            dice = compute_dice(probs[0], masks[0])
            iou = compute_iou(probs[0], masks[0])
            test_dice.append(dice.item())
            test_iou.append(iou.item())
    
    print("\n" + "-"*50)
    print("TEST RESULTS (3D U-Net)")
    print("-"*50)
    print(f"Dice:  {np.mean(test_dice):.4f} ± {np.std(test_dice):.4f}")
    print(f"IoU:   {np.mean(test_iou):.4f} ± {np.std(test_iou):.4f}")
    
    # Save results
    results = {
        'experiment': 'run8_3d_unet_segmentation',
        'dataset': 'ISPY2',
        'best_val_dice': best_dice,
        'test_dice': np.mean(test_dice),
        'test_dice_std': np.std(test_dice),
        'test_iou': np.mean(test_iou),
        'test_iou_std': np.std(test_iou),
        'total_params': params,
        'epochs_trained': epoch + 1
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(SAVE_DIR, "segmentation_results.csv"), index=False)
    print(f"\n✅ Results saved to {SAVE_DIR}/segmentation_results.csv")
    print(f"✅ Model saved to {SAVE_DIR}/best_unet.pth")
    
    print("\n" + "="*60)
    print("✅ 3D U-Net Segmentation Complete!")
    print("="*60)