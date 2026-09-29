# =============================================================
# VISUALIZE ROI: OVERLAY PREDICTED VS GROUND TRUTH MASKS & BBOX
# =============================================================

import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Paths
BASE_DIR = r"D:\Tehseen"
DCE_DIR = os.path.join(BASE_DIR, "Breast DCEDL", "BreastDCEDL_ISPY2_min_crop", "BreastDCEDL_ISPY2_min_crop", "dce")
MASK_DIR = os.path.join(BASE_DIR, "Breast DCEDL", "BreastDCEDL_ISPY2_min_crop", "BreastDCEDL_ISPY2_min_crop", "mask")
METADATA_PATH = os.path.join(BASE_DIR, "Breast DCEDL", "BreastDCEDL_metadata_min_crop.csv")
PRED_MASK_DIR = r"D:\Tehseen\results\run9_roi_crop\predicted_masks"
SAVE_DIR = r"D:\Tehseen\results\run9_roi_crop\visualizations\overlay_comparison"

os.makedirs(SAVE_DIR, exist_ok=True)

# Hyperparameters
MARGIN = 30
TARGET_DEPTH = 32
TARGET_SIZE = 128

def load_nifti(path):
    if not os.path.exists(path):
        return None
    return nib.load(path).get_fdata().astype(np.float32)

def crop_or_pad_volume(volume, target_shape):
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

def get_tumor_bbox(mask, margin=MARGIN):
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

def visualize_patient(pid, dataset, row):
    # 1. Load DCE phases
    phases = []
    for idx in [int(row['pre']), int(row['post_early']), int(row['post_late'])]:
        vol = None
        for ext in ['.nii.gz', '.nii']:
            path = os.path.join(DCE_DIR, f"{pid}_{dataset}_vis1_dce_aqc_{idx}{ext}")
            if os.path.exists(path):
                vol = load_nifti(path)
                break
        if vol is None:
            print(f"[-] Missing DCE file for patient {pid}")
            return
        phases.append(vol)
    
    target_shape = (TARGET_DEPTH, TARGET_SIZE, TARGET_SIZE)
    phases = [crop_or_pad_volume(p, target_shape) for p in phases]
    image = np.stack(phases, axis=0)  # [3, D, H, W]
    
    # Normalize image for visualization
    for c in range(3):
        mean_img = image[c].mean()
        std_img = image[c].std()
        if std_img > 1e-8:
            image[c] = (image[c] - mean_img) / std_img
            image[c] = np.clip(image[c], -3.0, 3.0)
        img_min, img_max = image[c].min(), image[c].max()
        if img_max - img_min > 1e-8:
            image[c] = (image[c] - img_min) / (img_max - img_min)
        else:
            image[c] = 0.0

    # 2. Load Ground Truth mask (if exists)
    gt_mask = None
    for ext in ['.nii.gz', '.nii']:
        path = os.path.join(MASK_DIR, f"{pid}_{dataset}_vis1_mask{ext}")
        if os.path.exists(path):
            gt_mask = crop_or_pad_volume(load_nifti(path), target_shape)
            gt_mask = (gt_mask > 0.5).astype(np.float32)
            break
            
    # 3. Load Predicted mask
    pred_path = os.path.join(PRED_MASK_DIR, f"{pid}_{dataset}_pred_mask.nii.gz")
    pred_mask = None
    if os.path.exists(pred_path):
        pred_mask = load_nifti(pred_path)  # Already padded/cropped to target_shape
    
    # 4. Find bounding box (using predicted mask)
    bbox = get_tumor_bbox(pred_mask, margin=MARGIN) if pred_mask is not None else None
    
    # Find middle slice of the tumor
    slice_idx = TARGET_DEPTH // 2
    if pred_mask is not None and pred_mask.sum() > 0:
        coords = np.argwhere(pred_mask > 0.5)
        slice_idx = int(np.median(coords[:, 0]))
    elif gt_mask is not None and gt_mask.sum() > 0:
        coords = np.argwhere(gt_mask > 0.5)
        slice_idx = int(np.median(coords[:, 0]))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    phase_names = ['Pre-contrast', 'Post-early', 'Post-late']
    
    for c in range(3):
        ax = axes[c]
        ax.imshow(image[c, slice_idx, :, :], cmap='gray')
        ax.set_title(f"{phase_names[c]} (Slice {slice_idx})")
        ax.axis('off')
        
        # Overlay GT Mask (Green)
        if gt_mask is not None:
            mask_slice = gt_mask[slice_idx]
            overlay = np.zeros((*mask_slice.shape, 4))
            overlay[mask_slice > 0.5] = [0.0, 1.0, 0.0, 0.4] # Semi-transparent green
            ax.imshow(overlay)
            
        # Overlay Pred Mask (Red)
        if pred_mask is not None:
            mask_slice = pred_mask[slice_idx]
            overlay = np.zeros((*mask_slice.shape, 4))
            overlay[mask_slice > 0.5] = [1.0, 0.0, 0.0, 0.4] # Semi-transparent red
            ax.imshow(overlay)
            
        # Draw Bounding Box (Yellow outline)
        if bbox is not None:
            z_min, z_max, y_min, y_max, x_min, x_max = bbox
            # Bounding box coordinates in H, W plane
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                     linewidth=1.5, edgecolor='yellow', facecolor='none', linestyle='--')
            ax.add_patch(rect)

    # Info text
    pcr_status = "pCR (Responder)" if row['pCR'] == 1 else "non-pCR (Non-responder)"
    hr_status = "Positive" if row['HR'] == 1 else "Negative"
    her2_status = "Positive" if row['HER2'] == 1 else "Negative"
    
    info_str = (
        f"Patient ID: {pid} | Dataset: {dataset}\n"
        f"Label: {pcr_status}\n"
        f"Clinical: Age={row['age']:.0f} | HR={hr_status} | HER2={her2_status} | Tumor Vol={row['tum_vol']:.1f}cc\n"
        f"BBox (Z, Y, X): {bbox}\n"
        f"Color Key: Green = Ground Truth Mask | Red = U-Net Predicted Mask | Yellow Box = Cropped Region"
    )
    
    plt.suptitle(info_str, y=0.98, fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(SAVE_DIR, f"{pid}_{dataset}_visualization.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[+] Saved visualization for {pid} to {save_path}")

def main():
    metadata = pd.read_csv(METADATA_PATH)
    ispy2_metadata = metadata[metadata['dataset'] == 'spy2'].copy()
    ispy2_metadata = ispy2_metadata.dropna(subset=['pCR']).reset_index(drop=True)
    
    # We select some random patients
    # Let's find patients who have Ground Truth masks, and some who don't
    has_gt_pids = []
    no_gt_pids = []
    
    for _, row in ispy2_metadata.iterrows():
        pid = row['pid']
        dataset = row['dataset']
        gt_exists = False
        for ext in ['.nii.gz', '.nii']:
            if os.path.exists(os.path.join(MASK_DIR, f"{pid}_{dataset}_vis1_mask{ext}")):
                gt_exists = True
                break
        if gt_exists:
            has_gt_pids.append(row)
        else:
            no_gt_pids.append(row)
            
    print(f"Total with GT: {len(has_gt_pids)} | Total without GT: {len(no_gt_pids)}")
    
    # Sample and visualize
    np.random.seed(42)
    
    # Use indices to safely sample list of rows
    indices_with_gt = np.random.choice(len(has_gt_pids), min(5, len(has_gt_pids)), replace=False)
    samples_with_gt = [has_gt_pids[i] for i in indices_with_gt]
    
    if len(no_gt_pids) > 0:
        indices_no_gt = np.random.choice(len(no_gt_pids), min(5, len(no_gt_pids)), replace=False)
        samples_no_gt = [no_gt_pids[i] for i in indices_no_gt]
    else:
        samples_no_gt = []
    
    for row in samples_with_gt:
        visualize_patient(row['pid'], row['dataset'], row)
        
    for row in samples_no_gt:
        visualize_patient(row['pid'], row['dataset'], row)

if __name__ == '__main__':
    main()
