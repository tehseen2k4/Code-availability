# =============================================================
# RUN 1: PURE 3D DENSENET121 - ISPY2 ONLY
# Using Original Phases (Not Subtraction)
# With Warmup + Optional Manual LR Slider (Post-Warmup)
# =============================================================

import os
import warnings
import subprocess
import sys
warnings.filterwarnings("ignore")

import numpy as np
import nibabel as nib
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, confusion_matrix,
    classification_report
)

# Install / import MONAI (bulletproof)
try:
    from monai.networks.nets import DenseNet121
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "monai"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "monai==1.3.0"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from monai.networks.nets import DenseNet121

print("[INIT] Run 1: Pure 3D DenseNet121 - ISPY2 Only")

# -------------------------------------------------------------
# WORKSTATION PATHS (UPDATED)
# -------------------------------------------------------------
BASE_DIR = r"D:\Tehseen"

# Dataset paths
DCE_DIR   = os.path.join(BASE_DIR, "Breast DCEDL", "BreastDCEDL_ISPY2_min_crop", "BreastDCEDL_ISPY2_min_crop", "dce")
MASK_DIR  = os.path.join(BASE_DIR, "Breast DCEDL", "BreastDCEDL_ISPY2_min_crop", "BreastDCEDL_ISPY2_min_crop", "mask")

# Split CSV paths (organized structure)
SPLIT_DIR = os.path.join(BASE_DIR, "dataset_splits", "ISPY2")

TRAIN_CSV = os.path.join(SPLIT_DIR, "train.csv")
VAL_CSV   = os.path.join(SPLIT_DIR, "val.csv")
TEST_CSV  = os.path.join(SPLIT_DIR, "test.csv")

# Output directory for models and results
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "run1_pure3d")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[INIT] DCE_DIR: {DCE_DIR}")
print(f"[INIT] MASK_DIR: {MASK_DIR}")
print(f"[INIT] TRAIN_CSV: {TRAIN_CSV}")
print(f"[INIT] OUTPUT_DIR: {OUTPUT_DIR}")

# -------------------------------------------------------------
# HYPERPARAMETERS
# -------------------------------------------------------------
TARGET_DEPTH    = 32
TARGET_SIZE     = 128
BATCH_SIZE      = 4
EPOCHS          = 100
LR              = 1e-4        # base learning rate (used by warmup and slider)
WEIGHT_DECAY    = 1e-4
PATIENCE        = 15
WARMUP_EPOCHS   = 3            # warmup for first 3 epochs
MARGIN          = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Device: {device}")

# =============================================================
# DATASET CLASS
# =============================================================
class ISPY2BreastMRIDataset(Dataset):
    def __init__(self, csv_path, dce_dir, mask_dir=None,
                 target_depth=32, target_size=128,
                 augment=False, margin=15):
        
        self.df = pd.read_csv(csv_path)
        self.dce_dir = dce_dir
        self.mask_dir = mask_dir
        self.target_depth = target_depth
        self.target_size = target_size
        self.augment = augment
        self.margin = margin
        self.df = self.df.dropna(subset=['pCR']).reset_index(drop=True)
        
        n_neg = (self.df['pCR'] == 0).sum()
        n_pos = (self.df['pCR'] == 1).sum()
        print(f"[Dataset] {os.path.basename(csv_path)}: {len(self.df)} patients | "
              f"pCR-: {n_neg} | pCR+: {n_pos} | Ratio: {n_neg/max(n_pos,1):.2f}:1")
    
    def __len__(self):
        return len(self.df)
    
    def _load_nifti(self, fpath):
        if not os.path.exists(fpath):
            return None
        return nib.load(fpath).get_fdata().astype(np.float32)
    
    def _get_tumor_bbox(self, mask):
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return None
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)
        z_min = max(0, z_min - self.margin)
        y_min = max(0, y_min - self.margin)
        x_min = max(0, x_min - self.margin)
        z_max = min(mask.shape[0], z_max + self.margin + 1)
        y_max = min(mask.shape[1], y_max + self.margin + 1)
        x_max = min(mask.shape[2], x_max + self.margin + 1)
        return (z_min, z_max, y_min, y_max, x_min, x_max)
    
    def _crop_resize(self, volume, bbox, target_shape):
        if bbox is None:
            dz, dy, dx = volume.shape
            cz, cy, cx = dz // 2, dy // 2, dx // 2
            z_min = max(0, cz - target_shape[0] // 2)
            y_min = max(0, cy - target_shape[1] // 2)
            x_min = max(0, cx - target_shape[2] // 2)
            cropped = volume[z_min:z_min+target_shape[0],
                           y_min:y_min+target_shape[1],
                           x_min:x_min+target_shape[2]]
        else:
            z_min, z_max, y_min, y_max, x_min, x_max = bbox
            cropped = volume[z_min:z_max, y_min:y_max, x_min:x_max]
        
        pad_z = max(0, target_shape[0] - cropped.shape[0])
        pad_y = max(0, target_shape[1] - cropped.shape[1])
        pad_x = max(0, target_shape[2] - cropped.shape[2])
        if pad_z or pad_y or pad_x:
            cropped = np.pad(cropped, ((0,pad_z),(0,pad_y),(0,pad_x)), mode='constant')
        
        cropped = cropped[:target_shape[0], :target_shape[1], :target_shape[2]]
        tensor = torch.from_numpy(cropped).unsqueeze(0).unsqueeze(0).float()
        tensor = F.interpolate(tensor, size=target_shape, mode='trilinear', align_corners=False)
        return tensor.squeeze().numpy()
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row['pid']
        dataset_tag = row['dataset']
        label = int(row['pCR'])
        
        mask = None
        if self.mask_dir and os.path.exists(self.mask_dir):
            for ext in ['.nii.gz', '.nii']:
                fpath = os.path.join(self.mask_dir, f"{pid}_{dataset_tag}_vis1_mask{ext}")
                mask = self._load_nifti(fpath)
                if mask is not None:
                    break
        
        phase_indices = [
            int(float(row['pre'])),
            int(float(row['post_early'])),
            int(float(row['post_late']))
        ]
        
        phases = []
        for pidx in phase_indices:
            vol = None
            for ext in ['.nii.gz', '.nii']:
                fpath = os.path.join(self.dce_dir, f"{pid}_{dataset_tag}_vis1_dce_aqc_{pidx}{ext}")
                vol = self._load_nifti(fpath)
                if vol is not None:
                    break
            if vol is None:
                vol = np.zeros((self.target_depth, self.target_size, self.target_size), dtype=np.float32)
            phases.append(vol)
        
        early_phase, post_early_phase, post_late_phase = phases
        
        bbox = None
        if mask is not None:
            if mask.shape != early_phase.shape:
                m_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
                m_t = F.interpolate(m_t, size=early_phase.shape, mode='nearest')
                mask = m_t.squeeze().numpy()
            bbox = self._get_tumor_bbox(mask)
        
        target_shape = (self.target_depth, self.target_size, self.target_size)
        early_c   = self._crop_resize(early_phase, bbox, target_shape)
        post_early_c = self._crop_resize(post_early_phase, bbox, target_shape)
        post_late_c  = self._crop_resize(post_late_phase, bbox, target_shape)
        
        image = np.stack([early_c, post_early_c, post_late_c], axis=0)
        
        mean = image.mean()
        std = image.std()
        if std > 1e-8:
            image = (image - mean) / std
            image = np.clip(image, -3.0, 3.0)
        img_min, img_max = image.min(), image.max()
        if img_max - img_min > 1e-8:
            image = (image - img_min) / (img_max - img_min)
        else:
            image = np.zeros_like(image)
        
        if self.augment:
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=2).copy()
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=3).copy()
            if np.random.rand() > 0.7:
                image = np.flip(image, axis=1).copy()
            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.9, 1.1)
                beta = np.random.uniform(-0.05, 0.05)
                image = np.clip(image * alpha + beta, 0, 1)
            if np.random.rand() > 0.5:
                noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
                image = np.clip(image + noise, 0, 1)
        
        return torch.from_numpy(image).float(), torch.tensor(label, dtype=torch.float32), pid

print("[INIT] Dataset class defined")

# =============================================================
# CREATE MODEL
# =============================================================
print("\n[INIT] Creating Pure 3D DenseNet121...")
model = DenseNet121(
    spatial_dims=3,
    in_channels=3,
    out_channels=1,
    dropout_prob=0.5,
    pretrained=False,
).to(device)

for m in model.modules():
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

total_params = sum(p.numel() for p in model.parameters())
print(f"[INIT] Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")

# =============================================================
# DATA LOADERS
# =============================================================
print("\n" + "="*60)
print("LOADING DATASETS")
print("="*60)

train_dataset = ISPY2BreastMRIDataset(
    TRAIN_CSV, DCE_DIR, MASK_DIR,
    target_depth=TARGET_DEPTH, target_size=TARGET_SIZE,
    augment=True, margin=MARGIN
)
val_dataset = ISPY2BreastMRIDataset(
    VAL_CSV, DCE_DIR, MASK_DIR,
    target_depth=TARGET_DEPTH, target_size=TARGET_SIZE,
    augment=False, margin=MARGIN
)
test_dataset = ISPY2BreastMRIDataset(
    TEST_CSV, DCE_DIR, MASK_DIR,
    target_depth=TARGET_DEPTH, target_size=TARGET_SIZE,
    augment=False, margin=MARGIN
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# Class weights
num_neg = (train_dataset.df['pCR'] == 0).sum()
num_pos = (train_dataset.df['pCR'] == 1).sum()
pos_weight = torch.tensor([num_neg / max(num_pos, 1)]).to(device)
print(f"[INIT] Class imbalance: {num_neg/num_pos:.2f}:1 | pos_weight: {pos_weight.item():.2f}")

# =============================================================
# TRAINING SETUP (with dynamic LR slider)
# =============================================================
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# ---------- MANUAL LR CONTROL (Slider) ----------
from ipywidgets import interact, FloatSlider

# Global variable for manual scale (applies only after warmup)
manual_lr_scale = None   # None means "use automatic (warmup or scheduler)"

def set_lr_scale(scale=1.0):
    """Callback for slider – updates the manual scale (does not touch optimizer yet)."""
    global manual_lr_scale
    manual_lr_scale = scale
    print(f"✓ Manual LR scale set to {scale:.2f} (will take effect after warmup epochs {WARMUP_EPOCHS}+)")

# Create a slider widget (0.1 to 3.0, step 0.05)
interact(set_lr_scale, scale=FloatSlider(min=0.1, max=3.0, step=0.05, value=1.0, description='LR scale'));
print("[INIT] LR slider ready – you can adjust it during training (post-warmup)")

# =============================================================
# TRAINING FUNCTIONS
# =============================================================
def get_lr(epoch, base_lr, warmup_epochs):
    """Warmup: linear increase from 0 to base_lr over warmup_epochs."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr

def train_one_epoch(epoch):
    model.train()
    total_loss = 0
    
    # Step 1: compute base LR from warmup schedule
    base_lr_for_this_epoch = get_lr(epoch, LR, WARMUP_EPOCHS)
    
    # Step 2: if after warmup AND manual scale is set, override with manual LR
    if epoch >= WARMUP_EPOCHS and manual_lr_scale is not None:
        current_lr = LR * manual_lr_scale
    else:
        current_lr = base_lr_for_this_epoch
    
    # Apply the chosen LR to optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    
    # Training loop
    for images, labels, _ in train_loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader), current_lr

def evaluate(loader):
    model.eval()
    all_labels, all_probs, total_loss = [], [], 0
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            probs = torch.sigmoid(outputs)
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    auroc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    auprc = average_precision_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.1, 0.9, 0.02):
        preds = (all_probs > thresh).astype(int)
        tp = ((preds == 1) & (all_labels == 1)).sum()
        fp = ((preds == 1) & (all_labels == 0)).sum()
        fn = ((preds == 0) & (all_labels == 1)).sum()
        if tp + fp > 0 and tp + fn > 0:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh
    
    return {
        'loss': total_loss / len(loader),
        'auroc': auroc,
        'auprc': auprc,
        'best_thresh': best_thresh,
        'best_f1': best_f1
    }

class EarlyStopping:
    def __init__(self, patience=15):
        self.patience = patience
        self.best_auprc = -1
        self.counter = 0
    def step(self, auprc):
        if auprc > self.best_auprc:
            self.best_auprc = auprc
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

early_stop = EarlyStopping(patience=PATIENCE)

# =============================================================
# TRAINING LOOP
# =============================================================
print("\n" + "="*60)
print("TRAINING STARTED")
print(f"Warmup: first {WARMUP_EPOCHS} epochs (LR {LR/3:.1e} → {LR:.1e})")
print(f"Model checkpoint will be saved to: {OUTPUT_DIR}")
print("After warmup, you can use the slider to override LR")
print("="*60)

best_auprc = -1
best_auroc = -1
best_state = None

for epoch in range(EPOCHS):
    train_loss, current_lr = train_one_epoch(epoch)
    val_res = evaluate(val_loader)
    scheduler.step(val_res['auprc'])
    
    print(f"\nEpoch {epoch+1}/{EPOCHS} | LR: {current_lr:.6f}")
    print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_res['loss']:.4f}")
    print(f"  Val AUROC: {val_res['auroc']:.4f} | Val AUPRC: {val_res['auprc']:.4f} (primary)")
    print(f"  Best F1: {val_res['best_f1']:.3f} @ thresh={val_res['best_thresh']:.2f}")
    
    if val_res['auprc'] > best_auprc:
        best_auprc = val_res['auprc']
        best_auroc = val_res['auroc']
        best_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'auprc': val_res['auprc'],
            'auroc': val_res['auroc'],
            'threshold': val_res['best_thresh']
        }
        model_path = os.path.join(OUTPUT_DIR, "best_pure3d_densenet121.pth")
        torch.save(best_state, model_path)
        print(f"  ✅ [CHECKPOINT] Best model saved (AUPRC={best_auprc:.4f})")
    
    if early_stop.step(val_res['auprc']):
        print(f"\n[STOP] Early stopping triggered (no improvement for {PATIENCE} epochs)")
        break

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Best Validation AUROC: {best_auroc:.4f}")
print(f"Best Validation AUPRC: {best_auprc:.4f}")
print(f"Best Threshold: {best_state['threshold']:.2f}")

# =============================================================
# TEST EVALUATION
# =============================================================
print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

model.load_state_dict(best_state['model_state_dict'])
model.eval()

test_labels, test_probs = [], []
with torch.no_grad():
    for images, labels, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        test_labels.extend(labels.numpy().flatten())
        test_probs.extend(probs.cpu().numpy().flatten())

test_labels = np.array(test_labels)
test_probs = np.array(test_probs)
thresh = best_state['threshold']
test_preds = (test_probs > thresh).astype(int)

test_auroc = roc_auc_score(test_labels, test_probs)
test_auprc = average_precision_score(test_labels, test_probs)
prec, rec, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='binary', zero_division=0)
acc = (test_preds == test_labels).mean()
tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()

print("\n" + "-"*50)
print("TEST RESULTS")
print("-"*50)
print(f"AUROC:     {test_auroc:.4f}")
print(f"AUPRC:     {test_auprc:.4f}")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Threshold: {thresh:.2f}")
print("\nConfusion Matrix:")
print(f"            Predicted Negative   Predicted Positive")
print(f"Actual Neg:        {tn:5d}                {fp:5d}")
print(f"Actual Pos:        {fn:5d}                {tp:5d}")

print("\nClassification Report:")
print(classification_report(test_labels, test_preds, target_names=['non-pCR', 'pCR'], zero_division=0))

# Save results
results = pd.DataFrame([{
    'experiment': 'run1_pure3d_densenet121',
    'input_type': 'original_phases_early_postearly_postlate',
    'architecture': 'DenseNet121_3D',
    'pretrained': 'None (trained from scratch)',
    'dataset': 'ISPY2',
    'test_auroc': test_auroc,
    'test_auprc': test_auprc,
    'test_accuracy': acc,
    'test_precision': prec,
    'test_recall': rec,
    'test_f1': f1,
    'threshold': thresh,
    'best_val_auprc': best_auprc,
    'best_val_auroc': best_auroc,
    'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
}])
results_path = os.path.join(OUTPUT_DIR, "test_results_pure3d_densenet121.csv")
results.to_csv(results_path, index=False)
print(f"\n[SAVED] Results to {results_path}")
print("\n" + "="*60)
print("✅ Pure 3D DenseNet121 Experiment Complete")
print("="*60)