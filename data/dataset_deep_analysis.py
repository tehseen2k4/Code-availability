import os
import sys
import glob
import re
import numpy as np
import pandas as pd
import nibabel as nib
from datetime import datetime
from collections import defaultdict

# =============================================================
# DEEP MULTI-DATASET ANALYSIS SCRIPT (FINAL FIXED)
# Handles all patient ID formats correctly
# =============================================================

def print_header(text, char="="):
    print("\n" + char*70)
    print(f"  {text}")
    print(char*70)

def print_subheader(text):
    print(f"\n--- {text} ---")

def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/1024**2:.2f} MB"
    else:
        return f"{size_bytes/1024**3:.2f} GB"

def extract_patient_id(basename, dataset_type):
    """Extract patient ID from filename based on dataset type."""
    # DUKE format: Breast_MRI_XXX_duke_dce_aqc_N.nii.gz
    if dataset_type == 'duke' or 'duke' in basename.lower():
        for part in basename.split('_'):
            if part.isdigit() and len(part) >= 3:
                return part
        return None
    
    # ISPY1 format: ISPY1_1001_spy1_vis1_dce_aqc_0.nii.gz
    if 'spy1' in basename.lower() or 'ispy1' in basename.lower():
        for part in basename.split('_'):
            if part.startswith('ISPY1_'):
                patient_id = part.replace('ISPY1_', '')
                if patient_id.isdigit():
                    return patient_id
            if part.isdigit() and len(part) >= 4:
                return part
        return None
    
    # ISPY2 format: Multiple patterns
    if 'spy2' in basename.lower() or 'ispy2' in basename.lower():
        match = re.search(r'ACRIN-6698-\d{6}', basename)
        if match:
            return match.group(0)
        match = re.search(r'ISPY2-\d{6}', basename)
        if match:
            return match.group(0)
        # Also try to extract just the number
        for part in basename.split('_'):
            if part.isdigit() and len(part) == 6:
                return part
            if part.isdigit() and len(part) >= 4:
                return part
        return None
    
    return None

def normalize_patient_id(pid, dataset_type):
    """Normalize patient ID to a common format for matching."""
    if pid is None:
        return None
    
    # Remove any prefix like ISPY1_, ISPY2-, ACRIN-6698-
    # Keep only the numeric part
    match = re.search(r'\d{4,6}', str(pid))
    if match:
        return match.group(0)
    return str(pid)

def get_phase_from_filename(basename):
    """Extract phase (aqc number) from filename."""
    for part in basename.split('_'):
        if part.startswith('aqc_'):
            return part.replace('aqc_', '')
    return None

def scan_dataset_files(base_path, dataset_name):
    """Scan all DCE and mask files for a dataset."""
    results = {
        'dataset': dataset_name,
        'dce_files': [],
        'mask_files': [],
        'dce_patients': set(),
        'mask_patients': set(),
        'dce_phases': defaultdict(int),
        'mask_phases': defaultdict(int),
        'file_sizes': [],
        'missing_masks': [],
        'extra_masks': [],
        'dce_count': 0,
        'mask_count': 0,
        'total_size': 0,
    }
    
    if not os.path.exists(base_path):
        print(f"  ❌ Path not found: {base_path}")
        return results
    
    # Find DCE files
    dce_patterns = ['**/*dce*.nii.gz', '**/*DCE*.nii.gz', '**/*aqc*.nii.gz']
    for pattern in dce_patterns:
        full_pattern = os.path.join(base_path, pattern)
        found = glob.glob(full_pattern, recursive=True)
        for f in found:
            if f not in results['dce_files']:
                results['dce_files'].append(f)
    
    # Find mask files
    mask_patterns = ['**/*mask*.nii.gz', '**/*MASK*.nii.gz']
    for pattern in mask_patterns:
        full_pattern = os.path.join(base_path, pattern)
        found = glob.glob(full_pattern, recursive=True)
        for f in found:
            if f not in results['mask_files']:
                results['mask_files'].append(f)
    
    results['dce_count'] = len(results['dce_files'])
    results['mask_count'] = len(results['mask_files'])
    
    # Calculate total size
    total_size = 0
    for f in results['dce_files'] + results['mask_files']:
        if os.path.exists(f):
            total_size += os.path.getsize(f)
    results['total_size'] = total_size
    
    # Extract patient IDs and phases from DCE files
    for f in results['dce_files']:
        basename = os.path.basename(f)
        pid = extract_patient_id(basename, dataset_name)
        # Normalize the PID
        if pid:
            norm_pid = normalize_patient_id(pid, dataset_name)
            if norm_pid:
                results['dce_patients'].add(norm_pid)
        phase = get_phase_from_filename(basename)
        if phase:
            results['dce_phases'][phase] += 1
        if os.path.exists(f):
            results['file_sizes'].append(os.path.getsize(f))
    
    # Extract patient IDs from mask files
    for f in results['mask_files']:
        basename = os.path.basename(f)
        pid = extract_patient_id(basename, dataset_name)
        if pid:
            norm_pid = normalize_patient_id(pid, dataset_name)
            if norm_pid:
                results['mask_patients'].add(norm_pid)
        phase = get_phase_from_filename(basename)
        if phase:
            results['mask_phases'][phase] += 1
    
    # Find missing masks (patients with DCE but no mask)
    for pid in results['dce_patients']:
        if pid not in results['mask_patients']:
            results['missing_masks'].append(pid)
    
    # Find extra masks (patients with mask but no DCE)
    for pid in results['mask_patients']:
        if pid not in results['dce_patients']:
            results['extra_masks'].append(pid)
    
    return results

def analyze_nifti_file(file_path):
    """Analyze a NIfTI file: shape, dtype, memory usage."""
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        shape = data.shape
        dtype = str(data.dtype)
        size_mb = os.path.getsize(file_path) / (1024**2)
        
        return {
            'shape': shape,
            'dtype': dtype,
            'size_mb': size_mb,
            'ndim': len(shape),
            'voxel_count': np.prod(shape),
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'shape': None,
            'dtype': None,
            'size_mb': os.path.getsize(file_path) / (1024**2) if os.path.exists(file_path) else 0,
            'ndim': None,
            'voxel_count': None,
            'success': False,
            'error': str(e)[:100]
        }

def analyze_metadata(metadata_path):
    """Load and analyze metadata CSV."""
    if not os.path.exists(metadata_path):
        return None
    
    df = pd.read_csv(metadata_path)
    return df

# =============================================================
# MAIN ANALYSIS
# =============================================================

print_header("DEEP MULTI-DATASET ANALYSIS")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

BASE_DIR = r"D:\Tehseen"
DATASET_DIR = os.path.join(BASE_DIR, "Breast DCEDL")
METADATA_PATH = os.path.join(DATASET_DIR, "BreastDCEDL_metadata_min_crop.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_analysis_reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================
# 1. LOAD METADATA
# =============================================================
print_header("1. LOADING METADATA")

metadata = analyze_metadata(METADATA_PATH)
if metadata is None:
    print("❌ Metadata not found!")
    sys.exit(1)

print(f"  Total rows: {len(metadata)}")
print(f"  Columns: {list(metadata.columns)}")

# Filter by dataset
datasets_in_metadata = {
    'duke': metadata[metadata['dataset'] == 'duke'],
    'ispy1': metadata[metadata['dataset'] == 'spy1'],
    'ispy2': metadata[metadata['dataset'] == 'spy2'],
}

for name, df in datasets_in_metadata.items():
    print(f"  {name.upper()}: {len(df)} patients")

# =============================================================
# 2. SCAN ALL DATASETS
# =============================================================
print_header("2. SCANNING DATASET FILES")

dataset_configs = {
    'duke': {
        'paths': [
            os.path.join(DATASET_DIR, "BreastDCEDL_DUKE_min_crop"),
            os.path.join(DATASET_DIR, "BreastDCEDL_DUKE_min_crop", "BreastDCEDL_DUKE_min_crop"),
        ],
        'short_name': 'DUKE'
    },
    'ispy1': {
        'paths': [
            os.path.join(DATASET_DIR, "BreastDCEDL_ISPY1_min_crop"),
            os.path.join(DATASET_DIR, "BreastDCEDL_ISPY1_min_crop", "BreastDCEDL_ISPY1_min_crop"),
        ],
        'short_name': 'ISPY1'
    },
    'ispy2': {
        'paths': [
            os.path.join(DATASET_DIR, "BreastDCEDL_ISPY2_min_crop"),
            os.path.join(DATASET_DIR, "BreastDCEDL_ISPY2_min_crop", "BreastDCEDL_ISPY2_min_crop"),
        ],
        'short_name': 'ISPY2'
    }
}

dataset_results = {}

for key, config in dataset_configs.items():
    print_subheader(f"Scanning {config['short_name']}")
    found_path = None
    for path in config['paths']:
        if os.path.exists(path):
            found_path = path
            break
    
    if found_path:
        print(f"  Found at: {found_path}")
        results = scan_dataset_files(found_path, key)
        results['found_path'] = found_path
        dataset_results[key] = results
    else:
        print(f"  ❌ Not found")
        dataset_results[key] = None

# =============================================================
# 3. METADATA VS FILES COMPARISON
# =============================================================
print_header("3. METADATA VS FILES COMPARISON")

comparison_results = []

for key, results in dataset_results.items():
    if results is None:
        continue
    
    short_name = dataset_configs[key]['short_name']
    meta_df = datasets_in_metadata.get(key, pd.DataFrame())
    
    print_subheader(f"{short_name}")
    
    # Get patients from metadata and normalize
    meta_patients_raw = set(str(p).strip() for p in meta_df['pid'].tolist()) if len(meta_df) > 0 else set()
    meta_patients = set()
    for pid in meta_patients_raw:
        norm = normalize_patient_id(pid, key)
        if norm:
            meta_patients.add(norm)
    
    # Get patients from files (already normalized)
    file_patients = results['dce_patients']
    
    # Find matches
    matches = meta_patients.intersection(file_patients)
    
    # Patient counts
    print(f"  Patients in metadata: {len(meta_patients)}")
    print(f"  Patients in files: {len(file_patients)}")
    print(f"  Matching patients: {len(matches)}")
    
    # Patients missing in files
    missing_in_files = meta_patients - file_patients
    if missing_in_files:
        print(f"  ⚠️ Missing in files: {len(missing_in_files)} patients")
        missing_sample = list(missing_in_files)[:5]
        for norm in missing_sample:
            print(f"    - {norm}")
    
    # Extra patients in files
    extra_in_files = file_patients - meta_patients
    if extra_in_files:
        print(f"  ⚠️ Extra in files: {len(extra_in_files)} patients")
        extra_sample = list(extra_in_files)[:5]
        for norm in extra_sample:
            print(f"    - {norm}")
    
    # Store comparison
    comparison_results.append({
        'dataset': short_name,
        'metadata_patients': len(meta_patients),
        'file_patients': len(file_patients),
        'matching_patients': len(matches),
        'missing_in_files': len(missing_in_files),
        'extra_in_files': len(extra_in_files),
        'dce_files': results['dce_count'],
        'mask_files': results['mask_count'],
        'missing_masks': len(results['missing_masks']),
        'total_size_gb': results['total_size'] / (1024**3),
    })

# =============================================================
# 4. NIFTI FILE ANALYSIS (Sample)
# =============================================================
print_header("4. NIFTI FILE ANALYSIS (Sample)")

nifti_analysis = []

for key, results in dataset_results.items():
    if results is None or not results['dce_files']:
        continue
    
    short_name = dataset_configs[key]['short_name']
    print_subheader(f"{short_name} (sampling 10 files)")
    
    sample_files = results['dce_files'][:10]
    shapes = []
    sizes = []
    ndims = []
    errors = 0
    
    for f in sample_files:
        info = analyze_nifti_file(f)
        if info['success']:
            shapes.append(info['shape'])
            sizes.append(info['size_mb'])
            ndims.append(info['ndim'])
        else:
            errors += 1
            print(f"  ❌ Error: {os.path.basename(f)} - {info['error']}")
    
    if shapes:
        print(f"  Shape range: {min(shapes)} -> {max(shapes)}")
        print(f"  Size range: {min(sizes):.2f} MB -> {max(sizes):.2f} MB")
        print(f"  Dimensions: {set(ndims)}")
    
    if errors > 0:
        print(f"  ⚠️ {errors} files had errors")

# =============================================================
# 5. PHASE AVAILABILITY
# =============================================================
print_header("5. PHASE AVAILABILITY")

for key, results in dataset_results.items():
    if results is None:
        continue
    
    short_name = dataset_configs[key]['short_name']
    print_subheader(f"{short_name}")
    
    dce_phases = sorted(results['dce_phases'].keys())
    if dce_phases:
        print(f"  DCE phases found: {dce_phases}")
        print(f"  DCE phase counts: {dict(results['dce_phases'])}")
    else:
        print(f"  ⚠️ No DCE phases extracted (filename parsing may need adjustment)")
    
    if results['mask_files']:
        mask_phases = sorted(results['mask_phases'].keys())
        if mask_phases:
            print(f"  Mask phases found: {mask_phases}")

# =============================================================
# 6. MISSING FILES ANALYSIS
# =============================================================
print_header("6. MISSING FILES ANALYSIS")

for key, results in dataset_results.items():
    if results is None:
        continue
    
    short_name = dataset_configs[key]['short_name']
    print_subheader(f"{short_name}")
    
    if results['missing_masks']:
        print(f"  ⚠️ Patients with DCE but missing mask: {len(results['missing_masks'])}")
        if len(results['missing_masks']) <= 10:
            for pid in results['missing_masks'][:10]:
                print(f"    - {pid}")
    else:
        print(f"  ✅ All DCE patients have masks")
    
    if results['extra_masks']:
        print(f"  ⚠️ Patients with mask but missing DCE: {len(results['extra_masks'])}")
        if len(results['extra_masks']) <= 10:
            for pid in results['extra_masks'][:10]:
                print(f"    - {pid}")

# =============================================================
# 7. FILE SIZE STATISTICS
# =============================================================
print_header("7. FILE SIZE STATISTICS")

for key, results in dataset_results.items():
    if results is None or not results['file_sizes']:
        continue
    
    short_name = dataset_configs[key]['short_name']
    sizes = results['file_sizes']
    sizes_mb = [s / (1024**2) for s in sizes]
    
    print_subheader(f"{short_name}")
    print(f"  Files sampled: {len(sizes_mb)}")
    print(f"  Min size: {min(sizes_mb):.2f} MB")
    print(f"  Max size: {max(sizes_mb):.2f} MB")
    print(f"  Mean size: {np.mean(sizes_mb):.2f} MB")
    print(f"  Median size: {np.median(sizes_mb):.2f} MB")
    print(f"  Std dev: {np.std(sizes_mb):.2f} MB")

# =============================================================
# 8. GENERATE REPORTS
# =============================================================
print_header("8. GENERATING REPORTS")

# Report 1: Dataset overview
overview_df = pd.DataFrame(comparison_results)
overview_path = os.path.join(OUTPUT_DIR, "dataset_overview.csv")
overview_df.to_csv(overview_path, index=False)
print(f"  ✅ Saved: {overview_path}")

# Report 2: Detailed per-dataset analysis
for key, results in dataset_results.items():
    if results is None:
        continue
    
    short_name = dataset_configs[key]['short_name']
    meta_df = datasets_in_metadata.get(key, pd.DataFrame())
    
    meta_patients_raw = set(str(p).strip() for p in meta_df['pid'].tolist()) if len(meta_df) > 0 else set()
    meta_patients = set()
    meta_original = {}
    for pid in meta_patients_raw:
        norm = normalize_patient_id(pid, key)
        if norm:
            meta_patients.add(norm)
            meta_original[norm] = pid
    
    file_patients = results['dce_patients']
    
    matches = meta_patients.intersection(file_patients)
    
    patient_data = []
    for norm in sorted(matches):
        meta_pid = meta_original.get(norm, '')
        file_pid = norm
        has_mask = norm in results['mask_patients']
        
        patient_data.append({
            'patient_id': meta_pid,
            'normalized_id': norm,
            'has_dce': True,
            'has_mask': has_mask,
        })
    
    if patient_data:
        patient_df = pd.DataFrame(patient_data)
        patient_path = os.path.join(OUTPUT_DIR, f"{short_name}_patient_analysis.csv")
        patient_df.to_csv(patient_path, index=False)
        print(f"  ✅ Saved: {patient_path}")

# Report 3: Summary report (with UTF-8 encoding to handle emojis)
summary_path = os.path.join(OUTPUT_DIR, "analysis_summary.txt")
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("  DATASET ANALYSIS SUMMARY\n")
    f.write("="*70 + "\n\n")
    f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("OVERVIEW:\n")
    f.write("-"*50 + "\n")
    for _, row in overview_df.iterrows():
        f.write(f"\n{row['dataset']}:\n")
        f.write(f"  Metadata patients: {row['metadata_patients']}\n")
        f.write(f"  File patients: {row['file_patients']}\n")
        f.write(f"  Matching patients: {row['matching_patients']}\n")
        f.write(f"  DCE files: {row['dce_files']}\n")
        f.write(f"  Mask files: {row['mask_files']}\n")
        f.write(f"  Missing masks: {row['missing_masks']}\n")
        f.write(f"  Total size: {row['total_size_gb']:.2f} GB\n")
    
    f.write("\n\n" + "-"*50 + "\n")
    f.write("STATUS:\n")
    for _, row in overview_df.iterrows():
        if row['missing_masks'] == 0 and row['matching_patients'] == row['metadata_patients']:
            status = "READY"
        elif row['missing_masks'] > 0:
            status = "MISSING MASKS"
        else:
            status = "PATIENT MISMATCH"
        f.write(f"  {row['dataset']}: {status}\n")

print(f"  ✅ Saved: {summary_path}")

# =============================================================
# 9. FINAL SUMMARY
# =============================================================
print_header("9. FINAL SUMMARY")

print("\n  Dataset Status:")
for _, row in overview_df.iterrows():
    if row['missing_masks'] == 0 and row['matching_patients'] == row['metadata_patients']:
        status = "READY FOR TRAINING"
    elif row['missing_masks'] > 0:
        status = "MISSING MASKS"
    else:
        status = "PATIENT MISMATCH"
    print(f"    {row['dataset']}: {status}")

print("\n  Reports saved to:")
print(f"    {OUTPUT_DIR}")

print("\n" + "="*70)
print(f"  Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)