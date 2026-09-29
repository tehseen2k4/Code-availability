# =============================================================
# EVALUATE MODEL ON FULL ISPY1 DATASET (ALL 172 PATIENTS)
# =============================================================

import os
import warnings
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

try:
    from monai.transforms import RandAffine
    from monai.networks.nets import DenseNet121
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "monai==1.3.0"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from monai.transforms import RandAffine
    from monai.networks.nets import DenseNet121

print("[INIT] Evaluating on full ISPY1 dataset (all 172 patients)")

# -------------------------------------------------------------
# PATHS
# -------------------------------------------------------------
BASE_DIR = r"D:\Tehseen"

# ISPY1 paths
DCE_DIR   = os.path.join(BASE_DIR, "Breast DCEDL", "BreastDCEDL_ISPY1_min_crop", "BreastDCEDL_ISPY1_min_crop", "dce")
MASK_DIR  = os.path.join(BASE_DIR, "Breast DCEDL", "BreastDCEDL_ISPY1_min_crop", "BreastDCEDL_ISPY1_min_crop", "mask")
SPLIT_DIR = os.path.join(BASE_DIR, "dataset_splits", "ISPY1")

# Load all three splits
TRAIN_CSV = os.path.join(SPLIT_DIR, "train.csv")
VAL_CSV   = os.path.join(SPLIT_DIR, "val.csv")
TEST_CSV  = os.path.join(SPLIT_DIR, "test.csv")

# Model checkpoint (Run 2)
CHECKPOINT_PATH = os.path.join(BASE_DIR, "Previous Results", "pure 3d Densenet121", "Run 2", "best_multimodal_densenet121.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "ispy1_validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[INIT] Model checkpoint: {CHECKPOINT_PATH}")
print(f"[INIT] Output directory: {OUTPUT_DIR}")

# -------------------------------------------------------------
# HYPERPARAMETERS
# -------------------------------------------------------------
TARGET_DEPTH    = 32
TARGET_SIZE     = 128
BATCH_SIZE      = 4
MARGIN          = 15
DROPOUT         = 0.5
CLINICAL_DIM    = 6
TARGET_SPACING = (2.0, 2.0, 2.0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Device: {device}")

# =============================================================
# MODEL DEFINITION (Run 2 – DenseNet121)
# =============================================================
class MultiModalDenseNet121(nn.Module):
    def __init__(self, clinical_dim=6, dropout=0.5):
        super().__init__()
        self.imaging = DenseNet121(
            spatial_dims=3,
            in_channels=3,
            out_channels=1,
            dropout_prob=dropout,
            pretrained=False
        )
        self.num_img_features = 1024
        self.clinical_mlp = nn.Sequential(
            nn.Linear(clinical_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.num_img_features + 32, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    def forward(self, x_img, x_clin):
        img_feats = self.imaging.features(x_img)
        img_feats = F.relu(img_feats)
        img_feats = F.adaptive_avg_pool3d(img_feats, (1, 1, 1))
        img_feats = torch.flatten(img_feats, 1)
        clin_feats = self.clinical_mlp(x_clin)
        combined = torch.cat([img_feats, clin_feats], dim=1)
        return self.fusion(combined)

# =============================================================
# DATASET CLASS (same as before)
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
            pass

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
# MAIN
# =============================================================
if __name__ == '__main__':

    print("\n" + "="*60)
    print("LOADING FULL ISPY1 DATASET (ALL 172 PATIENTS)")
    print("="*60)

    # Compute clinical stats from ISPY1 training set (to normalise)
    train_dataset_temp = EnhancedBreastMRIDataset(
        TRAIN_CSV, DCE_DIR, MASK_DIR,
        target_depth=TARGET_DEPTH, target_size=TARGET_SIZE,
        target_spacing=TARGET_SPACING,
        augment=False, margin=MARGIN,
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

    print(f"[CLINICAL] Mean (ISPY1 train): {clin_mean}")
    print(f"[CLINICAL] Std (ISPY1 train):  {clin_std}")

    # Combine all splits into one DataFrame
    df_train = pd.read_csv(TRAIN_CSV)
    df_val   = pd.read_csv(VAL_CSV)
    df_test  = pd.read_csv(TEST_CSV)
    df_all   = pd.concat([df_train, df_val, df_test], ignore_index=True)

    # Save combined CSV for reproducibility
    combined_csv = os.path.join(SPLIT_DIR, "ispy1_all.csv")
    df_all.to_csv(combined_csv, index=False)
    print(f"Combined ISPY1 patients: {len(df_all)}")

    # Create dataset with combined CSV
    all_dataset = EnhancedBreastMRIDataset(
        combined_csv, DCE_DIR, MASK_DIR,
        target_depth=TARGET_DEPTH, target_size=TARGET_SIZE,
        target_spacing=TARGET_SPACING,
        augment=False, margin=MARGIN,
        clinical_stats=(clin_mean, clin_std)
    )
    all_loader = DataLoader(all_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # =============================================================
    # LOAD MODEL
    # =============================================================
    print("\n[INIT] Loading model...")
    model = MultiModalDenseNet121(clinical_dim=CLINICAL_DIM, dropout=DROPOUT).to(device)

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("  Loaded checkpoint from Run 2")
        threshold = checkpoint.get('threshold', 0.5)
        print(f"  Original threshold: {threshold:.2f}")
    else:
        print(f"  ❌ Checkpoint not found: {CHECKPOINT_PATH}")
        exit()

    # Use the tuned threshold from earlier (0.58) – or you can re‑tune on the val set
    # We'll use 0.58 as it was optimal on the validation set.
    best_thresh = 0.58
    print(f"  Using tuned threshold: {best_thresh:.2f}")

    model.eval()

    # =============================================================
    # EVALUATE ON ALL 172 PATIENTS
    # =============================================================
    print("\n[INIT] Evaluating on all ISPY1 patients...")
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, clinical, labels, _ in all_loader:
            images, clinical = images.to(device), clinical.to(device)
            outputs = model(images, clinical)
            probs = torch.sigmoid(outputs)
            all_labels.extend(labels.numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)
    preds = (all_probs > best_thresh).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, preds, average='binary', zero_division=0)
    acc = (preds == all_labels).mean()
    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()

    print("\n" + "="*60)
    print("FULL ISPY1 EXTERNAL VALIDATION RESULTS (n=172)")
    print("="*60)
    print(f"Threshold used: {best_thresh:.2f}")
    print(f"AUROC:     {auroc:.4f}")
    print(f"AUPRC:     {auprc:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"            Predicted Negative   Predicted Positive")
    print(f"Actual Neg:        {tn:5d}                {fp:5d}")
    print(f"Actual Pos:        {fn:5d}                {tp:5d}")

    print("\nClassification Report:")
    print(classification_report(all_labels, preds, target_names=['non-pCR', 'pCR'], zero_division=0))

    # Save results
    results = pd.DataFrame([{
        'dataset': 'ISPY1_full',
        'n_patients': len(all_labels),
        'threshold': best_thresh,
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }])
    csv_path = os.path.join(OUTPUT_DIR, "ispy1_full_evaluation.csv")
    results.to_csv(csv_path, index=False)
    print(f"\n[SAVED] Results to {csv_path}")

    print("\n" + "="*60)
    print("✅ Full ISPY1 Evaluation Complete")
    print("="*60)