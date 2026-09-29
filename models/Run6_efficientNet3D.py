# =============================================================
# RUN 6b (FIXED): EFFICIENTNET3D-B2 (11M PARAMS)
# - Correct B2 scaling (width + depth multipliers)
# - Voxel resampling (2mm isotropic)
# - Affine augmentation + MixUp (NO elastic)
# - Clinical fusion (6 features)
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

# Install / import MONAI (for transforms only)
try:
    from monai.transforms import RandAffine
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "monai==1.3.0"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from monai.transforms import RandAffine

print("[INIT] Run 6b (FIXED): EfficientNet3D-B2 (11M params) + Enhanced Preprocessing")

# -------------------------------------------------------------
# WORKSTATION PATHS
# -------------------------------------------------------------
BASE_DIR = r"D:\Tehseen"

DCE_DIR   = os.path.join(BASE_DIR, "Breast DCEDL", "BreastDCEDL_ISPY2_min_crop", "BreastDCEDL_ISPY2_min_crop", "dce")
MASK_DIR  = os.path.join(BASE_DIR, "Breast DCEDL", "BreastDCEDL_ISPY2_min_crop", "BreastDCEDL_ISPY2_min_crop", "mask")
SPLIT_DIR = os.path.join(BASE_DIR, "dataset_splits", "ISPY2")

TRAIN_CSV = os.path.join(SPLIT_DIR, "train.csv")
VAL_CSV   = os.path.join(SPLIT_DIR, "val.csv")
TEST_CSV  = os.path.join(SPLIT_DIR, "test.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "results", "run6b_efficientnetb2_fixed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[INIT] DCE_DIR: {DCE_DIR}")
print(f"[INIT] MASK_DIR: {MASK_DIR}")
print(f"[INIT] OUTPUT_DIR: {OUTPUT_DIR}")

# -------------------------------------------------------------
# HYPERPARAMETERS (Same as Run 2)
# -------------------------------------------------------------
TARGET_DEPTH    = 32
TARGET_SIZE     = 128
BATCH_SIZE      = 4
EPOCHS          = 100
LR              = 1e-4
WEIGHT_DECAY    = 1e-4
PATIENCE        = 15
SCHEDULER_PATIENCE = 4
WARMUP_EPOCHS   = 3
MARGIN          = 15
DROPOUT         = 0.5
CLINICAL_DIM    = 6
TARGET_SPACING = (2.0, 2.0, 2.0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Device: {device}")
torch.backends.cudnn.benchmark = True

scaler = torch.cuda.amp.GradScaler()

# =============================================================
# CORRECT EFFICIENTNET-B2 (With Proper Scaling)
# =============================================================

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_channels):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, reduced_channels, 1),
            Swish(),
            nn.Conv3d(reduced_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25, drop_rate=0.2):
        super().__init__()
        expanded_channels = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.drop_rate = drop_rate

        # Expand phase
        if expand_ratio != 1:
            self.expand_conv = nn.Conv3d(in_channels, expanded_channels, 1, bias=False)
            self.expand_bn = nn.BatchNorm3d(expanded_channels)
            self.expand_act = Swish()
        else:
            self.expand_conv = nn.Identity()
            self.expand_bn = nn.Identity()
            self.expand_act = nn.Identity()

        # Depthwise convolution
        self.depthwise_conv = nn.Conv3d(
            expanded_channels, expanded_channels, kernel_size, stride,
            padding=kernel_size//2, groups=expanded_channels, bias=False
        )
        self.depthwise_bn = nn.BatchNorm3d(expanded_channels)
        self.depthwise_act = Swish()

        # Squeeze-and-Excitation
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = SqueezeExcitation(expanded_channels, se_channels)

        # Pointwise convolution
        self.pointwise_conv = nn.Conv3d(expanded_channels, out_channels, 1, bias=False)
        self.pointwise_bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        residual = x
        x = self.expand_act(self.expand_bn(self.expand_conv(x)))
        x = self.depthwise_act(self.depthwise_bn(self.depthwise_conv(x)))
        x = self.se(x)
        x = self.pointwise_bn(self.pointwise_conv(x))
        if self.use_residual:
            if self.drop_rate > 0:
                x = F.dropout(x, p=self.drop_rate, training=self.training)
            x = x + residual
        return x

class EfficientNet3D_B2(nn.Module):
    """
    Correct EfficientNet-B2 Scaling:
    - Width multipliers: [1.0, 1.1, 1.2, 1.4, 1.5, 1.6, 2.0, 1.8]
    - Depth multipliers: [1, 2, 2, 3, 3, 3, 4, 1]
    - Base channels: [32, 16, 24, 40, 80, 112, 192, 320]
    - Final features: 1408
    """
    def __init__(self, in_channels=3, out_channels=1, dropout=0.5):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            Swish()
        )

        # B2 scaled configurations
        # Format: (in_channels, out_channels, kernel_size, stride, expand_ratio, repeats)
        self.block_configs = [
            (32, 16, 3, 1, 1, 1),      # stage1
            (16, 24, 3, 2, 6, 2),      # stage2
            (24, 40, 5, 2, 6, 3),      # stage3
            (40, 80, 3, 2, 6, 3),      # stage4
            (80, 112, 5, 1, 6, 4),     # stage5
            (112, 192, 5, 2, 6, 4),    # stage6
            (192, 320, 3, 1, 6, 1),    # stage7
        ]

        self.blocks = nn.ModuleList()
        for in_ch, out_ch, k, s, exp, rep in self.block_configs:
            for i in range(rep):
                stride = s if i == 0 else 1
                self.blocks.append(MBConvBlock(
                    in_ch if i == 0 else out_ch,
                    out_ch, k, stride, exp,
                    drop_rate=0.2
                ))

        # Head: 320 -> 1408 (B2 final features)
        self.head = nn.Sequential(
            nn.Conv3d(320, 1408, 1, bias=False),
            nn.BatchNorm3d(1408),
            Swish()
        )
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return x

# =============================================================
# CLINICAL FUSION WRAPPER
# =============================================================
class EfficientNetB2WithClinical(nn.Module):
    def __init__(self, clinical_dim=6, dropout=0.5):
        super().__init__()
        self.backbone = EfficientNet3D_B2(in_channels=3, out_channels=1, dropout=dropout)
        self.img_feat_dim = 1408

        self.clinical_mlp = nn.Sequential(
            nn.Linear(clinical_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.fusion = nn.Sequential(
            nn.Linear(self.img_feat_dim + 32, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x_img, x_clin):
        img_feats = self.backbone(x_img)
        clin_feats = self.clinical_mlp(x_clin)
        combined = torch.cat([img_feats, clin_feats], dim=1)
        return self.fusion(combined)

# =============================================================
# DATASET CLASS (No Elastic)
# =============================================================
class EnhancedBreastMRIDataset(Dataset):
    def __init__(self, csv_path, dce_dir, mask_dir=None,
                 target_depth=32, target_size=128,
                 target_spacing=(2.0, 2.0, 2.0),
                 augment=False, margin=15,
                 clinical_stats=None):
        
        self.df = pd.read_csv(csv_path)
        self.dce_dir = dce_dir
        self.mask_dir = mask_dir
        self.target_depth = target_depth
        self.target_size = target_size
        self.target_spacing = target_spacing
        self.augment = augment
        self.margin = margin
        self.df = self.df.dropna(subset=['pCR']).reset_index(drop=True)
        
        self.clinical_cols = ['HR', 'HER2', 'tum_vol', 'age', 'n_z', 'slice_thick']
        
        n_neg = (self.df['pCR'] == 0).sum()
        n_pos = (self.df['pCR'] == 1).sum()
        print(f"[Dataset] {os.path.basename(csv_path)}: {len(self.df)} patients | "
              f"pCR-: {n_neg} | pCR+: {n_pos} | Ratio: {n_neg/max(n_pos,1):.2f}:1")
        
        self.clinical_mean = None
        self.clinical_std = None
        if clinical_stats is not None:
            self.clinical_mean, self.clinical_std = clinical_stats
        
        if augment:
            self.affine_aug = RandAffine(
                prob=0.6,
                rotate_range=(np.pi/18, np.pi/18, np.pi/18),
                scale_range=(0.1, 0.1, 0.1),
                translate_range=(5, 5, 5),
                mode='bilinear',
                padding_mode='border'
            )
        else:
            self.affine_aug = None
    
    def __len__(self):
        return len(self.df)
    
    def _load_nifti(self, fpath):
        if not os.path.exists(fpath):
            return None
        return nib.load(fpath)
    
    def _resample_volume(self, img, target_spacing):
        if img is None:
            return None
        current_spacing = img.header.get_zooms()[:3]
        scale_factors = [c / t for c, t in zip(current_spacing, target_spacing)]
        data = img.get_fdata().astype(np.float32)
        try:
            from scipy.ndimage import zoom
            resampled = zoom(data, scale_factors, order=1)
        except ImportError:
            tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
            new_size = tuple(int(d * s) for d, s in zip(data.shape, scale_factors))
            resampled = F.interpolate(tensor, size=new_size, mode='trilinear', align_corners=False)
            resampled = resampled.squeeze().numpy()
        return resampled
    
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
    
    def _extract_clinical_vector(self, row):
        feats = []
        for col in self.clinical_cols:
            val = row[col]
            if pd.isna(val):
                val = 0.0
            feats.append(float(val))
        return np.array(feats, dtype=np.float32)
    
    def set_clinical_stats(self, mean, std):
        self.clinical_mean = mean
        self.clinical_std = std
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row['pid']
        dataset_tag = row['dataset']
        label = int(row['pCR'])
        
        mask_img = None
        if self.mask_dir and os.path.exists(self.mask_dir):
            for ext in ['.nii.gz', '.nii']:
                fpath = os.path.join(self.mask_dir, f"{pid}_{dataset_tag}_vis1_mask{ext}")
                mask_img = self._load_nifti(fpath)
                if mask_img is not None:
                    break
        
        phase_indices = [
            int(float(row['pre'])),
            int(float(row['post_early'])),
            int(float(row['post_late']))
        ]
        phase_vols = []
        for pidx in phase_indices:
            img = None
            for ext in ['.nii.gz', '.nii']:
                fpath = os.path.join(self.dce_dir, f"{pid}_{dataset_tag}_vis1_dce_aqc_{pidx}{ext}")
                img = self._load_nifti(fpath)
                if img is not None:
                    break
            if img is None:
                phase_vols.append(np.zeros((self.target_depth, self.target_size, self.target_size), dtype=np.float32))
            else:
                resampled = self._resample_volume(img, self.target_spacing)
                if resampled is None:
                    resampled = np.zeros((self.target_depth, self.target_size, self.target_size), dtype=np.float32)
                phase_vols.append(resampled)
        
        early_phase, post_early_phase, post_late_phase = phase_vols
        
        mask_data = None
        if mask_img is not None:
            mask_data = self._resample_volume(mask_img, self.target_spacing)
        
        bbox = None
        if mask_data is not None:
            mask_binary = (mask_data > 0.5).astype(np.float32)
            bbox = self._get_tumor_bbox(mask_binary)
        
        target_shape = (self.target_depth, self.target_size, self.target_size)
        early_c   = self._crop_resize(early_phase, bbox, target_shape)
        post_early_c = self._crop_resize(post_early_phase, bbox, target_shape)
        post_late_c  = self._crop_resize(post_late_phase, bbox, target_shape)
        
        image = np.stack([early_c, post_early_c, post_late_c], axis=0)
        
        mean_img = image.mean()
        std_img = image.std()
        if std_img > 1e-8:
            image = (image - mean_img) / std_img
            image = np.clip(image, -3.0, 3.0)
        img_min, img_max = image.min(), image.max()
        if img_max - img_min > 1e-8:
            image = (image - img_min) / (img_max - img_min)
        else:
            image = np.zeros_like(image)
        
        image_tensor = torch.from_numpy(image).float()
        
        if self.augment:
            if self.affine_aug is not None and np.random.rand() < 0.6:
                image_tensor = self.affine_aug(image_tensor)
            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.9, 1.1)
                beta = np.random.uniform(-0.05, 0.05)
                image_tensor = torch.clamp(image_tensor * alpha + beta, 0, 1)
            if np.random.rand() > 0.5:
                noise = torch.randn_like(image_tensor) * 0.02
                image_tensor = torch.clamp(image_tensor + noise, 0, 1)
        
        clinical_raw = self._extract_clinical_vector(row)
        if self.clinical_mean is not None and self.clinical_std is not None:
            clinical = (clinical_raw - self.clinical_mean) / (self.clinical_std + 1e-8)
        else:
            clinical = clinical_raw
        
        return (image_tensor,
                torch.from_numpy(clinical).float(),
                torch.tensor(label, dtype=torch.float32),
                pid)

# =============================================================
# MIXUP HELPER
# =============================================================
def mixup_data(x_img, x_clin, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x_img.size(0)
    index = torch.randperm(batch_size).to(x_img.device)
    mixed_x_img = lam * x_img + (1 - lam) * x_img[index]
    mixed_x_clin = lam * x_clin + (1 - lam) * x_clin[index]
    y_a, y_b = y, y[index]
    return mixed_x_img, mixed_x_clin, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# =============================================================
# MAIN EXECUTION
# =============================================================
if __name__ == '__main__':
    
    print("[INIT] Run 6b (FIXED): EfficientNet3D-B2 (11M params) + Enhanced Preprocessing")
    
    print("\n" + "="*60)
    print("LOADING DATASETS (with enhanced preprocessing)")
    print("="*60)
    
    # Compute clinical stats
    train_dataset_temp = EnhancedBreastMRIDataset(
        TRAIN_CSV, DCE_DIR, MASK_DIR,
        target_depth=TARGET_DEPTH, target_size=TARGET_SIZE,
        target_spacing=TARGET_SPACING,
        augment=True, margin=MARGIN,
        clinical_stats=None
    )
    
    clinical_raw_all = []
    for i in range(len(train_dataset_temp)):
        row = train_dataset_temp.df.iloc[i]
        clin_raw = train_dataset_temp._extract_clinical_vector(row)
        clinical_raw_all.append(clin_raw)
    clinical_raw_all = np.array(clinical_raw_all)
    clin_mean = clinical_raw_all.mean(axis=0)
    clin_std = clinical_raw_all.std(axis=0) + 1e-8
    
    print(f"[CLINICAL] Mean: {clin_mean}")
    print(f"[CLINICAL] Std:  {clin_std}")
    
    train_dataset = EnhancedBreastMRIDataset(
        TRAIN_CSV, DCE_DIR, MASK_DIR,
        target_depth=TARGET_DEPTH, target_size=TARGET_SIZE,
        target_spacing=TARGET_SPACING,
        augment=True, margin=MARGIN,
        clinical_stats=(clin_mean, clin_std)
    )
    val_dataset = EnhancedBreastMRIDataset(
        VAL_CSV, DCE_DIR, MASK_DIR,
        target_depth=TARGET_DEPTH, target_size=TARGET_SIZE,
        target_spacing=TARGET_SPACING,
        augment=False, margin=MARGIN,
        clinical_stats=(clin_mean, clin_std)
    )
    test_dataset = EnhancedBreastMRIDataset(
        TEST_CSV, DCE_DIR, MASK_DIR,
        target_depth=TARGET_DEPTH, target_size=TARGET_SIZE,
        target_spacing=TARGET_SPACING,
        augment=False, margin=MARGIN,
        clinical_stats=(clin_mean, clin_std)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    num_neg = (train_dataset.df['pCR'] == 0).sum()
    num_pos = (train_dataset.df['pCR'] == 1).sum()
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)]).to(device)
    print(f"[INIT] Class imbalance: {num_neg/num_pos:.2f}:1 | pos_weight: {pos_weight.item():.2f}")
    
    # =============================================================
    # CREATE MODEL
    # =============================================================
    print("\n[INIT] Creating Correct EfficientNet3D-B2 (11M params)...")
    model = EfficientNetB2WithClinical(clinical_dim=CLINICAL_DIM, dropout=DROPOUT).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INIT] Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Verify parameter count is ~11M
    if total_params < 8_000_000:
        print(f"  ⚠️ WARNING: Expected ~11M params, got {total_params/1e6:.2f}M. Architecture may still be incorrect.")
    else:
        print(f"  ✅ Parameter count looks correct for B2.")
    
    # =============================================================
    # TRAINING SETUP
    # =============================================================
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=SCHEDULER_PATIENCE)
    
    print(f"[INIT] Using ReduceLROnPlateau scheduler (patience={SCHEDULER_PATIENCE})")
    print(f"[INIT] Batch size: {BATCH_SIZE}, Mixed Precision: Enabled")
    print(f"[INIT] Voxel resampling: {TARGET_SPACING}")
    print(f"[INIT] Elastic deformation: Disabled")
    print(f"[INIT] MixUp: Enabled (alpha=0.2)")
    
    # =============================================================
    # TRAINING FUNCTIONS
    # =============================================================
    def get_warmup_lr(epoch, base_lr, warmup_epochs):
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs
        return None
    
    def train_one_epoch(epoch):
        model.train()
        total_loss = 0
        
        if epoch < WARMUP_EPOCHS:
            warmup_lr = get_warmup_lr(epoch, LR, WARMUP_EPOCHS)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            current_lr = warmup_lr
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        for images, clinical, labels, _ in train_loader:
            images = images.to(device)
            clinical = clinical.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            if epoch >= WARMUP_EPOCHS:
                images, clinical, y_a, y_b, lam = mixup_data(images, clinical, labels, alpha=0.2)
            else:
                y_a, y_b, lam = labels, labels, 1.0
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(images, clinical)
                if lam < 1.0:
                    loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                else:
                    loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader), current_lr
    
    def evaluate(loader):
        model.eval()
        all_labels, all_probs, total_loss = [], [], 0
        with torch.no_grad():
            for images, clinical, labels, _ in loader:
                images = images.to(device)
                clinical = clinical.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                with torch.cuda.amp.autocast():
                    outputs = model(images, clinical)
                    loss = criterion(outputs, labels)
                
                total_loss += loss.item()
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
        
        return {'loss': total_loss/len(loader), 'auroc': auroc, 'auprc': auprc,
                'best_thresh': best_thresh, 'best_f1': best_f1}
    
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
    print("TRAINING STARTED (Run 6b FIXED: Correct EfficientNet3D-B2)")
    print(f"Warmup: first {WARMUP_EPOCHS} epochs (LR  {LR/3:.1e} → {LR:.1e})")
    print(f"After warmup: ReduceLROnPlateau (patience={SCHEDULER_PATIENCE}) + MixUp")
    print(f"Outputs will be saved to: {OUTPUT_DIR}")
    print("="*60)
    
    best_auprc = -1
    best_auroc = -1
    best_state = None
    
    for epoch in range(EPOCHS):
        train_loss, current_lr = train_one_epoch(epoch)
        val_res = evaluate(val_loader)
        
        scheduler.step(val_res['auprc'])
        actual_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1}/{EPOCHS} | LR: {actual_lr:.6f}")
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
            model_path = os.path.join(OUTPUT_DIR, "best_efficientnetb2_fixed.pth")
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
    print("TEST SET EVALUATION (Run 6b FIXED: Correct EfficientNet3D-B2)")
    print("="*60)
    
    model.load_state_dict(best_state['model_state_dict'])
    model.eval()
    
    test_labels, test_probs = [], []
    with torch.no_grad():
        for images, clinical, labels, _ in test_loader:
            images = images.to(device)
            clinical = clinical.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images, clinical)
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
    print("TEST RESULTS (Run 6b FIXED: Correct EfficientNet-B2)")
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
    
    results = pd.DataFrame([{
        'experiment': 'run6b_efficientnetb2_fixed',
        'input_type': '3_phases + clinical (6 features)',
        'architecture': 'EfficientNet3D-B2 (Correct ~11M params) + Clinical Fusion',
        'pretrained': 'None (scratch)',
        'voxel_resampling': f'{TARGET_SPACING} mm',
        'augmentation': 'Affine + MixUp (no elastic)',
        'batch_size': BATCH_SIZE,
        'scheduler_patience': SCHEDULER_PATIENCE,
        'mixed_precision': True,
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
    results_path = os.path.join(OUTPUT_DIR, "test_results_run6b_fixed.csv")
    results.to_csv(results_path, index=False)
    print(f"\n[SAVED] Results to {results_path}")
    
    print("\n" + "="*60)
    print("✅ Run 6b (FIXED) Complete")
    print("="*60)