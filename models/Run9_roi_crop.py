# =============================================================
# RUN 9: ROI Cropping Using Trained U-Net
# - Generates predicted masks for ALL ISPY2 patients
# - Crops volumes around predicted tumors with 15-voxel margin
# - Saves comprehensive metadata in Excel for Run2 re-training
# =============================================================

import os
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")

# Auto-install missing packages
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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

# MONAI imports
try:
    from monai.networks.nets import UNet
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "monai"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "monai==1.3.0"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from monai.networks.nets import UNet

print("[INIT] ROI Cropping Experiment (Run9)")

# -------------------------------------------------------------
# PATHS
# -------------------------------------------------------------
BASE_DIR = r"D:\Tehseen\Breast DCEDL\BreastDCEDL_ISPY2_min_crop\BreastDCEDL_ISPY2_min_crop"
DCE_DIR = os.path.join(BASE_DIR, "dce")
MASK_DIR = os.path.join(BASE_DIR, "mask")
METADATA_PATH = r"D:\Tehseen\Breast DCEDL\BreastDCEDL_metadata_min_crop.csv"
UNET_MODEL_PATH = r"D:\Tehseen\results\run8_segmentation\best_unet.pth"

# Output directories
OUTPUT_DIR = r"D:\Tehseen\results\run9_roi_crop"
PRED_MASK_DIR = os.path.join(OUTPUT_DIR, "predicted_masks")
CROPPED_VOL_DIR = os.path.join(OUTPUT_DIR, "cropped_volumes")
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")
METADATA_DIR = os.path.join(OUTPUT_DIR, "metadata")

os.makedirs(PRED_MASK_DIR, exist_ok=True)
os.makedirs(CROPPED_VOL_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

# -------------------------------------------------------------
# HYPERPARAMETERS
# -------------------------------------------------------------
TARGET_DEPTH = 32
TARGET_SIZE = 128
MARGIN = 30  # Padding around tumor
BATCH_SIZE = 4
NUM_WORKERS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Device: {device}")

# =============================================================
# PART 1: LOAD TRAINED U-NET
# =============================================================
print("\n" + "="*60)
print("LOADING TRAINED U-NET")
print("="*60)

unet = UNet(
    spatial_dims=3,
    in_channels=3,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm='batch',
    dropout=0.1
).to(device)

checkpoint = torch.load(UNET_MODEL_PATH, map_location=device, weights_only=False)
unet.load_state_dict(checkpoint['model_state_dict'])
unet.eval()
print(f"✅ U-Net loaded from: {UNET_MODEL_PATH}")

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
    
    # Pad if smaller
    pad_d = max(0, td - d)
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        volume = np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
    
    # Crop if larger
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

# Force regeneration of masks/crops to avoid using stale, poorly-normalized runs
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
    
    # Check if mask already exists
    mask_path = os.path.join(PRED_MASK_DIR, f"{pid}_{dataset}_pred_mask.nii.gz")
    if os.path.exists(mask_path) and not FORCE_REGENERATE:
        record['mask_generated'] = True
        record['mask_path'] = mask_path
        record['status'] = 'already_exists'
        mask_generation_records.append(record)
        continue
    
    try:
        # Load DCE phases
        image = load_dce_phases(
            pid, dataset, DCE_DIR, 
            int(row['pre']), int(row['post_early']), int(row['post_late'])
        )
        
        if image is None:
            record['status'] = 'failed_missing_dce'
            mask_generation_records.append(record)
            continue
        
        # Process each channel
        target_shape = (TARGET_DEPTH, TARGET_SIZE, TARGET_SIZE)
        processed_channels = []
        for c in range(3):
            vol = crop_or_pad_volume(image[c], target_shape)
            vol = normalize_volume(vol)
            processed_channels.append(vol)
        image = np.stack(processed_channels, axis=0)  # [C, D, H, W]
        
        # Predict mask
        with torch.no_grad():
            image_t = torch.from_numpy(image).unsqueeze(0).to(device)
            output = unet(image_t)
            prob = torch.sigmoid(output).squeeze().cpu().numpy()
            mask = (prob > 0.5).astype(np.float32)
        
        # Save mask
        mask_nifti = nib.Nifti1Image(mask, np.eye(4))
        nib.save(mask_nifti, mask_path)
        
        record['mask_generated'] = True
        record['mask_path'] = mask_path
        record['mask_nonzero'] = int((mask > 0).sum())
        record['status'] = 'success'
        
    except Exception as e:
        record['status'] = f'error_{str(e)[:50]}'
    
    mask_generation_records.append(record)

# Create DataFrame for mask generation
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
    
    # Add margin
    z_min = max(0, z_min - margin)
    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)
    z_max = min(mask.shape[0], z_max + margin + 1)
    y_max = min(mask.shape[1], y_max + margin + 1)
    x_max = min(mask.shape[2], x_max + margin + 1)
    
    return (z_min, z_max, y_min, y_max, x_min, x_max)

def crop_and_resize_volume(volume, bbox, target_shape):
    """Crop volume around bbox and resize to target shape using trilinear interpolation"""
    if bbox is None:
        # Center crop if no tumor found
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
    
    # Pad if needed
    pad_z = max(0, target_shape[0] - cropped.shape[0])
    pad_y = max(0, target_shape[1] - cropped.shape[1])
    pad_x = max(0, target_shape[2] - cropped.shape[2])
    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        cropped = np.pad(cropped, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant')
    
    # Crop to exact target shape
    cropped = cropped[:target_shape[0], :target_shape[1], :target_shape[2]]
    
    # Resize using trilinear interpolation
    tensor = torch.from_numpy(cropped).float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=target_shape, mode='trilinear', align_corners=False)
    
    return resized.squeeze().numpy()

def crop_and_save_patient(row, dce_dir, pred_mask_dir, cropped_dir):
    """Load, crop, and save all phases for a patient"""
    pid = row['pid']
    dataset = row['dataset']
    
    # Try to load predicted mask
    mask_path = os.path.join(pred_mask_dir, f"{pid}_{dataset}_pred_mask.nii.gz")
    if not os.path.exists(mask_path):
        return None
    
    mask = nib.load(mask_path).get_fdata().astype(np.float32)
    bbox = get_tumor_bbox(mask, margin=MARGIN)
    
    # Load DCE phases
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
    
    # Stack and normalize
    image = np.stack(cropped_phases, axis=0)  # [C, D, H, W]
    for c in range(3):
        image[c] = normalize_volume(image[c])
    
    # Save cropped volume
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

# Crop all patients
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

# Create DataFrame for cropping
crop_df = pd.DataFrame(crop_records)
crop_df.to_excel(os.path.join(METADATA_DIR, '02_cropping_summary.xlsx'), index=False)
print(f"\n✅ Cropping summary saved to: {METADATA_DIR}/02_cropping_summary.xlsx")

# =============================================================
# PART 4: CREATE TRAIN/VAL/TEST SPLITS WITH METADATA
# =============================================================
print("\n" + "="*60)
print("CREATING TRAIN/VAL/TEST SPLITS")
print("="*60)

# Filter to successfully cropped patients
valid_df = crop_df[crop_df['status'] == 'success'].copy()
print(f"Successfully cropped: {len(valid_df)} patients")

# Stratified split
from sklearn.model_selection import train_test_split

train_val_df, test_df = train_test_split(
    valid_df, test_size=0.15, random_state=42,
    stratify=valid_df['pCR']
)
train_df, val_df = train_test_split(
    train_val_df, test_size=0.15, random_state=42,
    stratify=train_val_df['pCR']
)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# Create split DataFrames with all relevant columns
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
# PART 5: CREATE MASTER METADATA FOR RUN2 RE-TRAINING
# =============================================================
print("\n" + "="*60)
print("CREATING MASTER METADATA FOR RUN2 RE-TRAINING")
print("="*60)

# Compute clinical stats on training set
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

# Create master metadata
master_metadata = {
    'experiment': 'run9_roi_crop',
    'description': 'ROI-cropped volumes for Run2 classifier re-training',
    'created': datetime.now().isoformat(),
    'parameters': {
        'target_depth': TARGET_DEPTH,
        'target_size': TARGET_SIZE,
        'margin': MARGIN,
        'resize_method': 'trilinear_interpolation',
        'unet_model': UNET_MODEL_PATH
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

# Save as JSON
with open(os.path.join(METADATA_DIR, 'master_metadata.json'), 'w') as f:
    json.dump(master_metadata, f, indent=2)

# Save as Excel
master_df = pd.DataFrame([{
    'experiment': master_metadata['experiment'],
    'description': master_metadata['description'],
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
# PART 6: CREATE RUN2-READY CONFIGURATION FILE
# =============================================================
print("\n" + "="*60)
print("CREATING RUN2-READY CONFIGURATION")
print("="*60)

# Generate a Python config file for Run2
config_content = f'''# =============================================================
# RUN2 CONFIGURATION FOR ROI-CROPPED DATA
# Generated by Run9 on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# =============================================================

# Dataset paths
DCE_DIR = r"{CROPPED_VOL_DIR}"  # .npy files for cropped volumes
METADATA_DIR = r"{METADATA_DIR}"

# Split files (Excel format)
TRAIN_CSV = os.path.join(METADATA_DIR, "03_train_set.xlsx")
VAL_CSV = os.path.join(METADATA_DIR, "04_val_set.xlsx")
TEST_CSV = os.path.join(METADATA_DIR, "05_test_set.xlsx")

# Clinical columns
CLINICAL_COLS = {clinical_cols}

# Clinical stats (for normalization)
CLINICAL_MEAN = {[clinical_stats[col]['mean'] for col in clinical_cols]}
CLINICAL_STD = {[clinical_stats[col]['std'] for col in clinical_cols]}

# Hyperparameters (same as Run2)
TARGET_DEPTH = {TARGET_DEPTH}
TARGET_SIZE = {TARGET_SIZE}
BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 15
POS_WEIGHT = 2.11  # Based on class imbalance

# Expected improvement
BASELINE_AUROC = 0.791
EXPECTED_AUROC = 0.80-0.82

print("[CONFIG] Run2 configuration loaded for ROI-cropped data")
print(f"[CONFIG] Training samples: {len(train_df)}")
print(f"[CONFIG] Validation samples: {len(val_df)}")
print(f"[CONFIG] Test samples: {len(test_df)}")
print(f"[CONFIG] pCR ratio: {train_df['pCR'].sum()}/{len(train_df)-train_df['pCR'].sum()}")
'''

config_path = os.path.join(METADATA_DIR, 'run2_roi_config.py')
with open(config_path, 'w') as f:
    f.write(config_content)

print(f"✅ Run2 config saved to: {config_path}")

# =============================================================
# PART 7: VISUALIZATION
# =============================================================
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

def visualize_sample(row, cropped_dir, save_dir):
    """Visualize cropped volume for a sample patient"""
    pid = row['pid']
    dataset = row['dataset']
    
    # Load cropped volume
    cropped_path = os.path.join(cropped_dir, f"{pid}_{dataset}_cropped.npy")
    if not os.path.exists(cropped_path):
        return
    
    image = np.load(cropped_path)  # [C, D, H, W]
    
    # Get middle slice
    d = image.shape[1] // 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    phase_names = ['Pre', 'Post-Early', 'Post-Late']
    for c, (ax, phase_name) in enumerate(zip(axes, phase_names)):
        ax.imshow(image[c, d, :, :], cmap='gray')
        ax.set_title(f"{phase_name} Phase")
        ax.axis('off')
    
    plt.suptitle(f"Patient: {pid} (ROI-Cropped, Margin={MARGIN})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{pid}_{dataset}_cropped.png"), dpi=150)
    plt.close()

# Visualize samples from train, val, test
for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
    sample_rows = split_df.sample(min(5, len(split_df)), random_state=42)
    split_save_dir = os.path.join(VISUALIZATION_DIR, split_name)
    os.makedirs(split_save_dir, exist_ok=True)
    
    for _, row in sample_rows.iterrows():
        visualize_sample(row, CROPPED_VOL_DIR, split_save_dir)

print(f"✅ Visualizations saved to: {VISUALIZATION_DIR}")

# =============================================================
# PART 8: FINAL SUMMARY REPORT
# =============================================================
print("\n" + "="*60)
print("GENERATING FINAL SUMMARY REPORT")
print("="*60)

report_path = os.path.join(OUTPUT_DIR, 'FINAL_RUN9_SUMMARY.txt')
with open(report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("RUN 9: ROI CROPPING - FINAL SUMMARY REPORT\n")
    f.write("=" * 80 + "\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("1. INPUT PARAMETERS\n")
    f.write("-" * 50 + "\n")
    f.write(f"   U-Net Model: {UNET_MODEL_PATH}\n")
    f.write(f"   Target Shape: ({TARGET_DEPTH}, {TARGET_SIZE}, {TARGET_SIZE})\n")
    f.write(f"   Margin: {MARGIN} voxels\n")
    f.write(f"   Resize Method: Trilinear Interpolation\n\n")
    
    f.write("2. MASK GENERATION\n")
    f.write("-" * 50 + "\n")
    total_masks = len(mask_df)
    success_masks = len(mask_df[mask_df['status'] == 'success'])
    f.write(f"   Total Patients: {total_masks}\n")
    f.write(f"   Masks Generated: {success_masks}\n")
    f.write(f"   Failed: {total_masks - success_masks}\n\n")
    
    f.write("3. VOLUME CROPPING\n")
    f.write("-" * 50 + "\n")
    total_crops = len(crop_df)
    success_crops = len(crop_df[crop_df['status'] == 'success'])
    f.write(f"   Total Patients: {total_crops}\n")
    f.write(f"   Crops Successful: {success_crops}\n")
    f.write(f"   Failed: {total_crops - success_crops}\n\n")
    
    f.write("4. DATASET SPLITS\n")
    f.write("-" * 50 + "\n")
    f.write(f"   Train: {len(train_df)} patients\n")
    f.write(f"   Val:   {len(val_df)} patients\n")
    f.write(f"   Test:  {len(test_df)} patients\n\n")
    
    f.write("5. pCR DISTRIBUTION\n")
    f.write("-" * 50 + "\n")
    f.write(f"   Train - pCR: {int(train_df['pCR'].sum())}, non-pCR: {len(train_df) - int(train_df['pCR'].sum())}\n")
    f.write(f"   Val   - pCR: {int(val_df['pCR'].sum())}, non-pCR: {len(val_df) - int(val_df['pCR'].sum())}\n")
    f.write(f"   Test  - pCR: {int(test_df['pCR'].sum())}, non-pCR: {len(test_df) - int(test_df['pCR'].sum())}\n\n")
    
    f.write("6. OUTPUT FILES\n")
    f.write("-" * 50 + "\n")
    f.write(f"   Predicted Masks: {PRED_MASK_DIR}\n")
    f.write(f"   Cropped Volumes: {CROPPED_VOL_DIR}\n")
    f.write(f"   Metadata: {METADATA_DIR}\n")
    f.write(f"   Visualizations: {VISUALIZATION_DIR}\n\n")
    
    f.write("7. METADATA FILES (Excel Format)\n")
    f.write("-" * 50 + "\n")
    f.write(f"   01_mask_generation_summary.xlsx\n")
    f.write(f"   02_cropping_summary.xlsx\n")
    f.write(f"   03_train_set.xlsx\n")
    f.write(f"   04_val_set.xlsx\n")
    f.write(f"   05_test_set.xlsx\n")
    f.write(f"   06_master_metadata.xlsx\n")
    f.write(f"   run2_roi_config.py\n")
    f.write(f"   master_metadata.json\n\n")
    
    f.write("8. NEXT STEPS\n")
    f.write("-" * 50 + "\n")
    f.write("   1. Run: python Run2_roi.py\n")
    f.write("   2. Expected AUROC Improvement: 0.791 -> 0.80-0.82\n")
    f.write("   3. Compare with baseline Run2 results\n")
    f.write("   4. Update thesis with ROI cropping results\n")

print(f"✅ Final summary saved to: {report_path}")

# =============================================================
# FINAL SUMMARY
# =============================================================
print("\n" + "=" * 80)
print("✅ RUN 9: ROI CROPPING COMPLETE!")
print("=" * 80)

print(f"\n📁 Output Directory: {OUTPUT_DIR}")
print(f"\n📊 Statistics:")
print(f"   Masks Generated: {success_masks}/{total_masks}")
print(f"   Volumes Cropped: {success_crops}/{total_crops}")
print(f"   Training Samples: {len(train_df)}")
print(f"   Validation Samples: {len(val_df)}")
print(f"   Test Samples: {len(test_df)}")

print(f"\n📁 Key Directories:")
print(f"   Predicted Masks: {PRED_MASK_DIR}")
print(f"   Cropped Volumes: {CROPPED_VOL_DIR}")
print(f"   Metadata: {METADATA_DIR}")
print(f"   Visualizations: {VISUALIZATION_DIR}")

print(f"\n📄 Excel Metadata Files (for Run2 re-training):")
print(f"   ✅ 01_mask_generation_summary.xlsx")
print(f"   ✅ 02_cropping_summary.xlsx")
print(f"   ✅ 03_train_set.xlsx")
print(f"   ✅ 04_val_set.xlsx")
print(f"   ✅ 05_test_set.xlsx")
print(f"   ✅ 06_master_metadata.xlsx")

print(f"\n⚙️ Run2 Configuration:")
print(f"   ✅ run2_roi_config.py")

print("\n" + "=" * 80)
print("🎯 NEXT: Re-train Run2 Classifier with ROI-Cropped Volumes")
print("   Expected AUROC: 0.791 → 0.80-0.82")
print("=" * 80)