# =============================================================
# ROI CROPPING USING BEST ATTENTION U‑NET
# - Generates predicted masks for ALL ISPY2 patients
# - Crops volumes around predicted tumors with 30-voxel margin
# - Enhanced visualizations: bounding boxes, mask overlays
# - Saves comprehensive metadata in Excel for re-training
# =============================================================

import os
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")

def ensure_packages():
    required = ['nibabel', 'monai', 'tqdm', 'matplotlib', 'openpyxl', 'xlsxwriter']
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
import random

# MONAI imports
try:
    from monai.networks.nets import AttentionUnet
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "monai"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "monai==1.3.0"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from monai.networks.nets import AttentionUnet

print("[INIT] ROI Cropping using Attention U‑Net")

# -------------------------------------------------------------
# PATHS
# -------------------------------------------------------------
BASE_DIR = r"D:\Tehseen\Breast DCEDL\BreastDCEDL_ISPY2_min_crop\BreastDCEDL_ISPY2_min_crop"
DCE_DIR = os.path.join(BASE_DIR, "dce")
MASK_DIR = os.path.join(BASE_DIR, "mask")  # Ground truth masks (for visualisation)
METADATA_PATH = r"D:\Tehseen\Breast DCEDL\BreastDCEDL_metadata_min_crop.csv"
ATTENTION_MODEL_PATH = r"D:\Tehseen\results\segmentation\attention_unet\best_attention_unet.pth"

# Output directories
OUTPUT_DIR = r"D:\Tehseen\results\ROI CROP USING SEGMENTATION\attention_unet_roi_crop"
PRED_MASK_DIR = os.path.join(OUTPUT_DIR, "predicted_masks")
CROPPED_VOL_DIR = os.path.join(OUTPUT_DIR, "cropped_volumes")
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")
METADATA_DIR = os.path.join(OUTPUT_DIR, "metadata")

os.makedirs(PRED_MASK_DIR, exist_ok=True)
os.makedirs(CROPPED_VOL_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(os.path.join(VISUALIZATION_DIR, "debug"), exist_ok=True)

# -------------------------------------------------------------
# HYPERPARAMETERS (same as Attention U‑Net training)
# -------------------------------------------------------------
TARGET_DEPTH = 48          # Attention U‑Net trained with depth 48
TARGET_SIZE = 128
MARGIN = 30                # Padding around tumor
BATCH_SIZE = 4
NUM_WORKERS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Device: {device}")

# =============================================================
# PART 1: LOAD TRAINED ATTENTION U‑NET
# =============================================================
print("\n" + "="*60)
print("LOADING TRAINED ATTENTION U‑NET")
print("="*60)

model = AttentionUnet(
    spatial_dims=3,
    in_channels=3,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),  # Must match training
    strides=(2, 2, 2, 2),
    kernel_size=3,
    up_kernel_size=3,
    dropout=0.1
).to(device)

checkpoint = torch.load(ATTENTION_MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✅ Attention U‑Net loaded from: {ATTENTION_MODEL_PATH}")
print(f"   Best validation Dice: {checkpoint.get('best_dice', 'N/A')}")

# =============================================================
# PART 2: GENERATE PREDICTED MASKS FOR ALL PATIENTS
# =============================================================
print("\n" + "="*60)
print("GENERATING PREDICTED MASKS FOR ALL ISPY2 PATIENTS")
print("="*60)

metadata = pd.read_csv(METADATA_PATH)
ispy2_metadata = metadata[metadata['dataset'] == 'spy2'].copy()
ispy2_metadata = ispy2_metadata.dropna(subset=['pCR']).reset_index(drop=True)
print(f"Total ISPY2 patients: {len(ispy2_metadata)}")

def load_dce_phases(pid, dataset, dce_dir, pre_idx, early_idx, late_idx):
    """Load all 3 DCE phases for a patient"""
    phases = []
    for idx in [pre_idx, early_idx, late_idx]:
        vol = None
        for ext in ['.nii.gz', '.nii']:
            path = os.path.join(dce_dir, f"{pid}_{dataset}_vis1_dce_aqc_{idx}{ext}")
            if os.path.exists(path):
                vol = nib.load(path).get_fdata().astype(np.float32)
                break
        if vol is None:
            return None
        phases.append(vol)
    return np.stack(phases, axis=0)  # [C, D, H, W]

def crop_or_pad_volume(volume, target_shape):
    """Crop or pad volume to target shape with center alignment"""
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

def normalize_volume(volume):
    """Normalize volume using robust standardization, clipping, and scaling to [0, 1]"""
    mean_img = volume.mean()
    std_img = volume.std()
    if std_img > 1e-8:
        volume = (volume - mean_img) / std_img
        volume = np.clip(volume, -3.0, 3.0)
    img_min, img_max = volume.min(), volume.max()
    if img_max - img_min > 1e-8:
        return (volume - img_min) / (img_max - img_min)
    return np.zeros_like(volume)

# Force regeneration to avoid stale files
FORCE_REGENERATE = True

# Generate masks for all patients
print("\nGenerating predicted masks...")
mask_generation_records = []

pbar = tqdm(ispy2_metadata.iterrows(), total=len(ispy2_metadata), desc="Generating masks")
for idx, row in pbar:
    pid = row['pid']
    dataset = row['dataset']
    
    record = {
        'pid': pid,
        'dataset': dataset,
        'pCR': row['pCR'],
        'HR': row.get('HR', None),
        'HER2': row.get('HER2', None),
        'tum_vol': row.get('tum_vol', None),
        'age': row.get('age', None),
        'n_z': row.get('n_z', None),
        'slice_thick': row.get('slice_thick', None),
        'mask_generated': False,
        'mask_path': None,
        'mask_nonzero': 0,
        'status': 'pending'
    }
    
    mask_path = os.path.join(PRED_MASK_DIR, f"{pid}_{dataset}_pred_mask.nii.gz")
    if os.path.exists(mask_path) and not FORCE_REGENERATE:
        record['mask_generated'] = True
        record['mask_path'] = mask_path
        record['status'] = 'already_exists'
        mask_generation_records.append(record)
        continue
    
    try:
        image = load_dce_phases(
            pid, dataset, DCE_DIR, 
            int(row['pre']), int(row['post_early']), int(row['post_late'])
        )
        
        if image is None:
            record['status'] = 'failed_missing_dce'
            mask_generation_records.append(record)
            continue
        
        target_shape = (TARGET_DEPTH, TARGET_SIZE, TARGET_SIZE)
        processed_channels = []
        for c in range(3):
            vol = crop_or_pad_volume(image[c], target_shape)
            vol = normalize_volume(vol)
            processed_channels.append(vol)
        image = np.stack(processed_channels, axis=0)
        
        with torch.no_grad():
            image_t = torch.from_numpy(image).unsqueeze(0).to(device)
            output = model(image_t)
            prob = torch.sigmoid(output).squeeze().cpu().numpy()
            mask = (prob > 0.5).astype(np.float32)
        
        mask_nifti = nib.Nifti1Image(mask, np.eye(4))
        nib.save(mask_nifti, mask_path)
        
        record['mask_generated'] = True
        record['mask_path'] = mask_path
        record['mask_nonzero'] = int((mask > 0).sum())
        record['status'] = 'success'
        
    except Exception as e:
        record['status'] = f'error_{str(e)[:50]}'
    
    mask_generation_records.append(record)

mask_df = pd.DataFrame(mask_generation_records)
mask_df.to_excel(os.path.join(METADATA_DIR, '01_mask_generation_summary.xlsx'), index=False)
print(f"\n✅ Mask generation summary saved to: {METADATA_DIR}/01_mask_generation_summary.xlsx")

# =============================================================
# PART 3: CROP VOLUMES USING PREDICTED MASKS
# =============================================================
print("\n" + "="*60)
print("CROPPING VOLUMES USING PREDICTED MASKS")
print("="*60)

def get_tumor_bbox(mask, margin=MARGIN):
    """Get bounding box around tumor with margin"""
    coords = np.argwhere(mask > 0.5)
    if len(coords) == 0:
        return None
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)
    z_min = max(0, z_min - margin)
    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)
    z_max = min(mask.shape[0], z_max + margin + 1)
    y_max = min(mask.shape[1], y_max + margin + 1)
    x_max = min(mask.shape[2], x_max + margin + 1)
    return (z_min, z_max, y_min, y_max, x_min, x_max)

def crop_and_resize_volume(volume, bbox, target_shape):
    """Crop volume around bbox and resize using trilinear interpolation"""
    if bbox is None:
        d, h, w = volume.shape
        start_d = max(0, (d - target_shape[0]) // 2)
        start_h = max(0, (h - target_shape[1]) // 2)
        start_w = max(0, (w - target_shape[2]) // 2)
        cropped = volume[start_d:start_d+target_shape[0],
                        start_h:start_h+target_shape[1],
                        start_w:start_w+target_shape[2]]
    else:
        z_min, z_max, y_min, y_max, x_min, x_max = bbox
        cropped = volume[z_min:z_max, y_min:y_max, x_min:x_max]
    
    pad_z = max(0, target_shape[0] - cropped.shape[0])
    pad_y = max(0, target_shape[1] - cropped.shape[1])
    pad_x = max(0, target_shape[2] - cropped.shape[2])
    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        cropped = np.pad(cropped, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant')
    cropped = cropped[:target_shape[0], :target_shape[1], :target_shape[2]]
    
    tensor = torch.from_numpy(cropped).float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=target_shape, mode='trilinear', align_corners=False)
    return resized.squeeze().numpy()

def crop_and_save_patient(row, dce_dir, pred_mask_dir, cropped_dir):
    """Load, crop, and save all phases for a patient"""
    pid = row['pid']
    dataset = row['dataset']
    
    mask_path = os.path.join(pred_mask_dir, f"{pid}_{dataset}_pred_mask.nii.gz")
    if not os.path.exists(mask_path):
        return None
    
    mask = nib.load(mask_path).get_fdata().astype(np.float32)
    bbox = get_tumor_bbox(mask, margin=MARGIN)
    
    phases = []
    for pidx in [int(row['pre']), int(row['post_early']), int(row['post_late'])]:
        vol = None
        for ext in ['.nii.gz', '.nii']:
            path = os.path.join(dce_dir, f"{pid}_{dataset}_vis1_dce_aqc_{pidx}{ext}")
            if os.path.exists(path):
                vol = nib.load(path).get_fdata().astype(np.float32)
                break
        if vol is None:
            return None
        phases.append(vol)
    
    target_shape = (TARGET_DEPTH, TARGET_SIZE, TARGET_SIZE)
    cropped_phases = []
    for phase in phases:
        cropped = crop_and_resize_volume(phase, bbox, target_shape)
        cropped_phases.append(cropped)
    
    image = np.stack(cropped_phases, axis=0)
    for c in range(3):
        image[c] = normalize_volume(image[c])
    
    output_path = os.path.join(cropped_dir, f"{pid}_{dataset}_cropped.npy")
    np.save(output_path, image)
    
    return {
        'pid': pid,
        'dataset': dataset,
        'bbox': bbox,
        'bbox_coords': str(bbox) if bbox else None,
        'mask_nonzero': int((mask > 0.5).sum()),
        'cropped_shape': image.shape,
        'output_path': output_path
    }

print("\nCropping volumes...")
crop_records = []

pbar = tqdm(ispy2_metadata.iterrows(), total=len(ispy2_metadata), desc="Cropping volumes")
for idx, row in pbar:
    pid = row['pid']
    dataset = row['dataset']
    
    record = {
        'pid': pid,
        'dataset': dataset,
        'pCR': row['pCR'],
        'HR': row.get('HR', None),
        'HER2': row.get('HER2', None),
        'tum_vol': row.get('tum_vol', None),
        'age': row.get('age', None),
        'n_z': row.get('n_z', None),
        'slice_thick': row.get('slice_thick', None),
        'pre_phase': row['pre'],
        'post_early_phase': row['post_early'],
        'post_late_phase': row['post_late'],
        'cropped_path': None,
        'bbox_exists': False,
        'mask_nonzero': 0,
        'status': 'pending'
    }
    
    try:
        result = crop_and_save_patient(row, DCE_DIR, PRED_MASK_DIR, CROPPED_VOL_DIR)
        if result is not None:
            record['cropped_path'] = result['output_path']
            record['bbox_exists'] = result['bbox'] is not None
            record['mask_nonzero'] = result['mask_nonzero']
            record['status'] = 'success'
        else:
            record['status'] = 'failed_no_mask'
    except Exception as e:
        record['status'] = f'error_{str(e)[:50]}'
    
    crop_records.append(record)

crop_df = pd.DataFrame(crop_records)
crop_df.to_excel(os.path.join(METADATA_DIR, '02_cropping_summary.xlsx'), index=False)
print(f"\n✅ Cropping summary saved to: {METADATA_DIR}/02_cropping_summary.xlsx")

# =============================================================
# PART 4: CREATE TRAIN/VAL/TEST SPLITS
# =============================================================
print("\n" + "="*60)
print("CREATING TRAIN/VAL/TEST SPLITS")
print("="*60)

valid_df = crop_df[crop_df['status'] == 'success'].copy()
print(f"Successfully cropped: {len(valid_df)} patients")

train_val_df, test_df = train_test_split(
    valid_df, test_size=0.15, random_state=42, stratify=valid_df['pCR']
)
train_df, val_df = train_test_split(
    train_val_df, test_size=0.15, random_state=42, stratify=train_val_df['pCR']
)
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

split_columns = [
    'pid', 'dataset', 'pCR', 'HR', 'HER2', 'tum_vol', 'age', 'n_z', 'slice_thick',
    'pre_phase', 'post_early_phase', 'post_late_phase',
    'cropped_path', 'bbox_exists', 'mask_nonzero'
]

train_df[split_columns].to_excel(os.path.join(METADATA_DIR, '03_train_set.xlsx'), index=False)
val_df[split_columns].to_excel(os.path.join(METADATA_DIR, '04_val_set.xlsx'), index=False)
test_df[split_columns].to_excel(os.path.join(METADATA_DIR, '05_test_set.xlsx'), index=False)

print(f"\n✅ Train/Val/Test splits saved to: {METADATA_DIR}")

# =============================================================
# PART 5: MASTER METADATA
# =============================================================
print("\n" + "="*60)
print("CREATING MASTER METADATA")
print("="*60)

clinical_cols = ['HR', 'HER2', 'tum_vol', 'age', 'n_z', 'slice_thick']
clinical_stats = {}
for col in clinical_cols:
    values = train_df[col].dropna().values
    clinical_stats[col] = {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values))
    }

master_metadata = {
    'experiment': 'attention_unet_roi_crop',
    'segmentation_model': 'Attention_U_Net',
    'model_checkpoint': ATTENTION_MODEL_PATH,
    'best_val_dice': checkpoint.get('best_dice', None),
    'created': datetime.now().isoformat(),
    'parameters': {
        'target_depth': TARGET_DEPTH,
        'target_size': TARGET_SIZE,
        'margin': MARGIN,
        'resize_method': 'trilinear_interpolation',
    },
    'dataset_info': {
        'total_patients': len(valid_df),
        'train': len(train_df),
        'val': len(val_df),
        'test': len(test_df)
    },
    'pCR_distribution': {
        'train': {'pCR': int(train_df['pCR'].sum()), 'non_pCR': int(len(train_df) - train_df['pCR'].sum())},
        'val': {'pCR': int(val_df['pCR'].sum()), 'non_pCR': int(len(val_df) - val_df['pCR'].sum())},
        'test': {'pCR': int(test_df['pCR'].sum()), 'non_pCR': int(len(test_df) - test_df['pCR'].sum())}
    },
    'clinical_stats': clinical_stats,
    'clinical_cols': clinical_cols
}

with open(os.path.join(METADATA_DIR, 'master_metadata.json'), 'w') as f:
    json.dump(master_metadata, f, indent=2)

master_df = pd.DataFrame([{
    'experiment': master_metadata['experiment'],
    'description': 'ROI-cropped volumes using Attention U‑Net',
    'created': master_metadata['created'],
    'target_depth': master_metadata['parameters']['target_depth'],
    'target_size': master_metadata['parameters']['target_size'],
    'margin': master_metadata['parameters']['margin'],
    'total_patients': master_metadata['dataset_info']['total_patients'],
    'train': master_metadata['dataset_info']['train'],
    'val': master_metadata['dataset_info']['val'],
    'test': master_metadata['dataset_info']['test'],
    'train_pCR': master_metadata['pCR_distribution']['train']['pCR'],
    'train_non_pCR': master_metadata['pCR_distribution']['train']['non_pCR'],
    'val_pCR': master_metadata['pCR_distribution']['val']['pCR'],
    'val_non_pCR': master_metadata['pCR_distribution']['val']['non_pCR'],
    'test_pCR': master_metadata['pCR_distribution']['test']['pCR'],
    'test_non_pCR': master_metadata['pCR_distribution']['test']['non_pCR']
}])
master_df.to_excel(os.path.join(METADATA_DIR, '06_master_metadata.xlsx'), index=False)
print(f"\n✅ Master metadata saved to: {METADATA_DIR}")

# =============================================================
# PART 6: ENHANCED VISUALIZATIONS
# =============================================================
print("\n" + "="*60)
print("GENERATING ENHANCED VISUALIZATIONS")
print("="*60)

def compute_dice(pred_mask, gt_mask):
    """Compute Dice score between predicted and ground truth masks"""
    pred_b = (pred_mask > 0.5).astype(np.float32)
    gt_b = (gt_mask > 0.5).astype(np.float32)
    inter = np.sum(pred_b * gt_b)
    return (2.0 * inter) / (np.sum(pred_b) + np.sum(gt_b) + 1e-8)

def visualize_with_bbox(row, pred_mask_dir, gt_mask_dir, dce_dir, save_dir):
    """Create enhanced visualization with bounding box and mask overlay"""
    pid = row['pid']
    dataset = row['dataset']
    pCR = row['pCR']
    
    # Load predicted mask
    pred_mask_path = os.path.join(pred_mask_dir, f"{pid}_{dataset}_pred_mask.nii.gz")
    if not os.path.exists(pred_mask_path):
        return
    
    pred_mask = nib.load(pred_mask_path).get_fdata().astype(np.float32)
    
    # Load ground truth mask (if exists)
    gt_mask = None
    gt_mask_path = os.path.join(gt_mask_dir, f"{pid}_{dataset}_vis1_mask.nii.gz")
    if os.path.exists(gt_mask_path):
        gt_mask = nib.load(gt_mask_path).get_fdata().astype(np.float32)
        gt_mask = crop_or_pad_volume(gt_mask, pred_mask.shape)
    
    # Load pre‑contrast phase for background
    pre_idx = int(row['pre'])
    dce_path = os.path.join(dce_dir, f"{pid}_{dataset}_vis1_dce_aqc_{pre_idx}.nii.gz")
    if not os.path.exists(dce_path):
        # Try .nii extension
        dce_path = os.path.join(dce_dir, f"{pid}_{dataset}_vis1_dce_aqc_{pre_idx}.nii")
    if not os.path.exists(dce_path):
        return
    
    image = nib.load(dce_path).get_fdata().astype(np.float32)
    image = crop_or_pad_volume(image, pred_mask.shape)
    image = normalize_volume(image)
    
    # Compute bbox
    bbox = get_tumor_bbox(pred_mask, margin=5)  # small margin for visualization
    has_bbox = bbox is not None
    
    # Get middle slice
    d = pred_mask.shape[0] // 2
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Image slices
    ax1 = axes[0, 0]
    ax1.imshow(image[d, :, :], cmap='gray')
    if has_bbox:
        z_min, z_max, y_min, y_max, x_min, x_max = bbox
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         linewidth=2, edgecolor='cyan', facecolor='none')
        ax1.add_patch(rect)
    ax1.set_title(f'Pre‑contrast + BBox\nPatient: {pid} (pCR={pCR})')
    ax1.axis('off')
    
    # Row 1: Predicted mask overlay
    ax2 = axes[0, 1]
    ax2.imshow(image[d, :, :], cmap='gray')
    ax2.imshow(pred_mask[d, :, :], alpha=0.5, cmap='Reds')
    if has_bbox:
        z_min, z_max, y_min, y_max, x_min, x_max = bbox
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         linewidth=2, edgecolor='cyan', facecolor='none')
        ax2.add_patch(rect)
    ax2.set_title('Predicted Mask (Red Overlay)')
    ax2.axis('off')
    
    # Row 1: Predicted mask only
    ax3 = axes[0, 2]
    ax3.imshow(pred_mask[d, :, :], cmap='Reds')
    ax3.set_title('Predicted Mask Only')
    ax3.axis('off')
    
    # Row 2: Ground truth overlay (if available)
    ax4 = axes[1, 0]
    ax4.imshow(image[d, :, :], cmap='gray')
    if gt_mask is not None:
        ax4.imshow(gt_mask[d, :, :], alpha=0.5, cmap='Greens')
        ax4.set_title('Ground Truth Mask (Green Overlay)')
    else:
        ax4.set_title('No Ground Truth Available')
    ax4.axis('off')
    
    # Row 2: Predicted vs GT overlay
    ax5 = axes[1, 1]
    ax5.imshow(image[d, :, :], cmap='gray')
    if gt_mask is not None:
        # Show GT in green, Pred in red, overlap in yellow
        gt_slice = (gt_mask[d, :, :] > 0.5).astype(np.float32)
        pred_slice = (pred_mask[d, :, :] > 0.5).astype(np.float32)
        overlay = np.zeros((*gt_slice.shape, 3))
        # Green channel for GT
        overlay[:, :, 1] = gt_slice * 0.7
        # Red channel for Pred
        overlay[:, :, 0] = pred_slice * 0.7
        # Yellow for overlap (both green and red)
        overlap = (gt_slice + pred_slice) > 1.5
        overlay[overlap, 0] = 1.0
        overlay[overlap, 1] = 1.0
        overlay[overlap, 2] = 0.0
        ax5.imshow(overlay)
        dice = compute_dice(pred_mask, gt_mask)
        ax5.set_title(f'Overlay: GT(Green) | Pred(Red) | Overlap(Yellow)\nDice: {dice:.3f}')
    else:
        ax5.imshow(pred_mask[d, :, :], cmap='Reds')
        ax5.set_title('No Ground Truth Available')
    ax5.axis('off')
    
    # Row 2: Mask comparison (side by side)
    ax6 = axes[1, 2]
    if gt_mask is not None:
        # Show GT and Pred side by side
        combined = np.zeros((pred_mask.shape[1], pred_mask.shape[2]*2 + 10))
        combined[:, :pred_mask.shape[2]] = pred_mask[d, :, :]
        combined[:, pred_mask.shape[2]+10:] = gt_mask[d, :, :]
        ax6.imshow(combined, cmap='gray')
        ax6.axvline(x=pred_mask.shape[2]+5, color='red', linestyle='--', linewidth=2)
        ax6.text(20, 20, 'Pred', color='white', fontsize=12, fontweight='bold')
        ax6.text(pred_mask.shape[2]+30, 20, 'GT', color='white', fontsize=12, fontweight='bold')
        ax6.set_title('Pred (left) vs GT (right)')
    else:
        ax6.imshow(pred_mask[d, :, :], cmap='Reds')
        ax6.set_title('Predicted Mask')
    ax6.axis('off')
    
    plt.suptitle(f'ROI Cropping Visualization – Patient: {pid} (pCR={pCR})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(save_dir, f"{pid}_{dataset}_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'pid': pid,
        'dataset': dataset,
        'has_gt': gt_mask is not None,
        'has_bbox': has_bbox,
        'dice': compute_dice(pred_mask, gt_mask) if gt_mask is not None else None
    }

# Generate visualizations for a subset of patients from train, val, test
print("Generating visualizations with bounding boxes and overlay comparisons...")
vis_results = []

for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
    split_save_dir = os.path.join(VISUALIZATION_DIR, split_name)
    os.makedirs(split_save_dir, exist_ok=True)
    
    # Select up to 10 patients per split (or all if fewer)
    n_samples = min(10, len(split_df))
    sample_df = split_df.sample(n_samples, random_state=42)
    
    for _, row in sample_df.iterrows():
        result = visualize_with_bbox(row, PRED_MASK_DIR, MASK_DIR, DCE_DIR, split_save_dir)
        if result:
            vis_results.append(result)

print(f"✅ Visualizations saved to: {VISUALIZATION_DIR}")

# =============================================================
# PART 7: DEBUG VISUALIZATIONS (compare GT vs Pred for ALL patients with GT)
# =============================================================
print("\n" + "="*60)
print("GENERATING DEBUG VISUALIZATIONS")
print("="*60)

# Find patients that have ground truth masks
debug_dir = os.path.join(VISUALIZATION_DIR, "debug")
os.makedirs(debug_dir, exist_ok=True)

debug_results = []
for _, row in ispy2_metadata.iterrows():
    pid = row['pid']
    dataset = row['dataset']
    
    # Check if GT exists
    gt_path = os.path.join(MASK_DIR, f"{pid}_{dataset}_vis1_mask.nii.gz")
    if not os.path.exists(gt_path):
        continue
    
    # Check if predicted mask exists
    pred_path = os.path.join(PRED_MASK_DIR, f"{pid}_{dataset}_pred_mask.nii.gz")
    if not os.path.exists(pred_path):
        continue
    
    # Load masks
    gt = nib.load(gt_path).get_fdata().astype(np.float32)
    pred = nib.load(pred_path).get_fdata().astype(np.float32)
    
    # Align shapes
    if gt.shape != pred.shape:
        gt = crop_or_pad_volume(gt, pred.shape)
    
    dice = compute_dice(pred, gt)
    iou = np.sum((pred > 0.5) & (gt > 0.5)) / (np.sum((pred > 0.5) | (gt > 0.5)) + 1e-8)
    
    debug_results.append({
        'pid': pid,
        'dataset': dataset,
        'dice': dice,
        'iou': iou,
        'gt_nonzero': int((gt > 0.5).sum()),
        'pred_nonzero': int((pred > 0.5).sum())
    })

debug_df = pd.DataFrame(debug_results)
debug_df.to_excel(os.path.join(debug_dir, 'debug_metrics.xlsx'), index=False)

# Create a summary plot of Dice scores
if len(debug_results) > 0:
    plt.figure(figsize=(12, 6))
    dices = [r['dice'] for r in debug_results]
    plt.hist(dices, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(np.mean(dices), color='red', linestyle='--', label=f'Mean: {np.mean(dices):.3f}')
    plt.axvline(np.median(dices), color='green', linestyle='--', label=f'Median: {np.median(dices):.3f}')
    plt.xlabel('Dice Score')
    plt.ylabel('Number of Patients')
    plt.title('Distribution of Dice Scores (Ground Truth vs Predicted)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(debug_dir, 'dice_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Debug metrics saved to: {debug_dir}")
    print(f"  Patients with GT: {len(debug_results)}")
    print(f"  Mean Dice: {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(f"  Median Dice: {np.median(dices):.4f}")
    print(f"  Min Dice: {np.min(dices):.4f}")
    print(f"  Max Dice: {np.max(dices):.4f}")

print(f"✅ Debug visualizations saved to: {debug_dir}")

# =============================================================
# PART 8: FINAL SUMMARY REPORT
# =============================================================
print("\n" + "="*60)
print("GENERATING FINAL SUMMARY REPORT")
print("="*60)

report_path = os.path.join(OUTPUT_DIR, 'FINAL_ATTENTION_UNET_ROI_SUMMARY.txt')
with open(report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("ROI CROPPING WITH ATTENTION U‑NET – FINAL SUMMARY\n")
    f.write("=" * 80 + "\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("1. MODEL INFORMATION\n")
    f.write("-" * 50 + "\n")
    f.write(f"   Model: Attention U‑Net\n")
    f.write(f"   Checkpoint: {ATTENTION_MODEL_PATH}\n")
    f.write(f"   Best Validation Dice: {checkpoint.get('best_dice', 'N/A')}\n\n")
    
    f.write("2. INPUT PARAMETERS\n")
    f.write("-" * 50 + "\n")
    f.write(f"   Target Shape: ({TARGET_DEPTH}, {TARGET_SIZE}, {TARGET_SIZE})\n")
    f.write(f"   Margin: {MARGIN} voxels\n")
    f.write(f"   Resize Method: Trilinear Interpolation\n\n")
    
    f.write("3. MASK GENERATION\n")
    f.write("-" * 50 + "\n")
    total_masks = len(mask_df)
    success_masks = len(mask_df[mask_df['status'] == 'success'])
    f.write(f"   Total Patients: {total_masks}\n")
    f.write(f"   Masks Generated: {success_masks}\n")
    f.write(f"   Failed: {total_masks - success_masks}\n\n")
    
    f.write("4. VOLUME CROPPING\n")
    f.write("-" * 50 + "\n")
    total_crops = len(crop_df)
    success_crops = len(crop_df[crop_df['status'] == 'success'])
    f.write(f"   Total Patients: {total_crops}\n")
    f.write(f"   Crops Successful: {success_crops}\n")
    f.write(f"   Failed: {total_crops - success_crops}\n\n")
    
    f.write("5. DATASET SPLITS\n")
    f.write("-" * 50 + "\n")
    f.write(f"   Train: {len(train_df)} patients\n")
    f.write(f"   Val:   {len(val_df)} patients\n")
    f.write(f"   Test:  {len(test_df)} patients\n\n")
    
    f.write("6. GROUND TRUTH COMPARISON (Debug)\n")
    f.write("-" * 50 + "\n")
    if len(debug_results) > 0:
        f.write(f"   Patients with GT: {len(debug_results)}\n")
        f.write(f"   Mean Dice: {np.mean([r['dice'] for r in debug_results]):.4f}\n")
        f.write(f"   Median Dice: {np.median([r['dice'] for r in debug_results]):.4f}\n")
        f.write(f"   Std Dice: {np.std([r['dice'] for r in debug_results]):.4f}\n")
    else:
        f.write("   No ground truth masks available for comparison.\n")
    f.write("\n")
    
    f.write("7. OUTPUT FILES\n")
    f.write("-" * 50 + "\n")
    f.write(f"   Predicted Masks: {PRED_MASK_DIR}\n")
    f.write(f"   Cropped Volumes: {CROPPED_VOL_DIR}\n")
    f.write(f"   Metadata: {METADATA_DIR}\n")
    f.write(f"   Visualizations: {VISUALIZATION_DIR}\n")
    f.write(f"   Debug: {debug_dir}\n\n")
    
    f.write("8. NEXT STEPS\n")
    f.write("-" * 50 + "\n")
    f.write("   1. Use cropped volumes for classification re-training\n")
    f.write("   2. Compare with baseline (full volume) results\n")
    f.write("   3. Update thesis with ROI cropping experiment\n")

print(f"✅ Final summary saved to: {report_path}")

# =============================================================
# FINAL SUMMARY
# =============================================================
print("\n" + "=" * 80)
print("✅ ATTENTION U‑NET ROI CROPPING COMPLETE!")
print("=" * 80)

print(f"\n📁 Output Directory: {OUTPUT_DIR}")
print(f"\n📊 Statistics:")
print(f"   Masks Generated: {success_masks}/{total_masks}")
print(f"   Volumes Cropped: {success_crops}/{total_crops}")
print(f"   Training Samples: {len(train_df)}")
print(f"   Validation Samples: {len(val_df)}")
print(f"   Test Samples: {len(test_df)}")

if len(debug_results) > 0:
    print(f"\n📊 Segmentation Performance on GT Patients:")
    print(f"   Mean Dice: {np.mean([r['dice'] for r in debug_results]):.4f} ± {np.std([r['dice'] for r in debug_results]):.4f}")

print(f"\n📁 Key Directories:")
print(f"   Predicted Masks: {PRED_MASK_DIR}")
print(f"   Cropped Volumes: {CROPPED_VOL_DIR}")
print(f"   Metadata: {METADATA_DIR}")
print(f"   Visualizations: {VISUALIZATION_DIR}")
print(f"   Debug: {debug_dir}")

print("\n" + "=" * 80)
print("🎯 NEXT: Re-train Classifier with ROI-Cropped Volumes")
print("   Use the metadata Excel files from: " + METADATA_DIR)
print("=" * 80)