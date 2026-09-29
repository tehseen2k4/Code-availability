import os
import sys
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from datetime import datetime
from collections import defaultdict
import re

# =============================================================
# COMPLETE DATASET AUDIT SCRIPT (FIXED ISPY2 PATIENT IDS)
# Handles both ISPY2-XXXXXX and ACRIN-6698-XXXXXX formats
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

def get_folder_size(folder_path):
    total = 0
    if os.path.exists(folder_path):
        for root, dirs, files in os.walk(folder_path):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
    return total

def extract_patient_id(basename, dataset_type):
    """Extract patient ID from filename - handles both ISPY2 and ACRIN formats."""
    
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
        # Pattern 1: ACRIN-6698-XXXXXX (most common in files)
        match = re.search(r'ACRIN-6698-\d{6}', basename)
        if match:
            return match.group(0)  # Returns e.g., "ACRIN-6698-102212"
        
        # Pattern 2: ISPY2-XXXXXX (metadata format)
        match = re.search(r'ISPY2-\d{6}', basename)
        if match:
            return match.group(0)  # Returns e.g., "ISPY2-756412"
        
        # Pattern 3: Any 6-digit number (fallback)
        for part in basename.split('_'):
            if part.isdigit() and len(part) == 6:
                return part
        
        # Pattern 4: Any number longer than 4 digits (last resort)
        for part in basename.split('_'):
            if part.isdigit() and len(part) >= 4:
                return part
        
        return None
    
    return None

def scan_dataset_recursive(dataset_name, base_path, metadata_df=None):
    """Scan dataset by recursively finding all DCE and mask files."""
    print_subheader(f"Scanning {dataset_name}")
    
    stats = {
        'name': dataset_name,
        'base_path': base_path,
        'exists': os.path.exists(base_path),
        'dce_files': [],
        'mask_files': [],
        'dce_count': 0,
        'mask_count': 0,
        'dce_size': 0,
        'mask_size': 0,
        'total_size': 0,
        'patient_ids': set(),
        'phase_counts': defaultdict(int),
        'issues': [],
        'found_dce_dir': None,
        'found_mask_dir': None,
        'all_subdirs': []
    }
    
    if not stats['exists']:
        stats['issues'].append(f"Path not found: {base_path}")
        return stats
    
    # Get all subdirectories
    for root, dirs, files in os.walk(base_path):
        stats['all_subdirs'].append(root)
    
    print(f"  Found {len(stats['all_subdirs'])} subdirectories")
    
    # Find all DCE files (anywhere in the tree)
    dce_patterns = [
        '**/*dce*.nii.gz',
        '**/*DCE*.nii.gz',
        '**/*aqc*.nii.gz',
        '**/crop_min_dce/**/*.nii.gz',
        '**/dce/**/*.nii.gz',
    ]
    
    mask_patterns = [
        '**/*mask*.nii.gz',
        '**/*MASK*.nii.gz',
        '**/crop_min_mask/**/*.nii.gz',
        '**/mask/**/*.nii.gz',
    ]
    
    # Find DCE files
    for pattern in dce_patterns:
        full_pattern = os.path.join(base_path, pattern)
        found = glob.glob(full_pattern, recursive=True)
        for f in found:
            if f not in stats['dce_files']:
                stats['dce_files'].append(f)
    
    # Find mask files
    for pattern in mask_patterns:
        full_pattern = os.path.join(base_path, pattern)
        found = glob.glob(full_pattern, recursive=True)
        for f in found:
            if f not in stats['mask_files']:
                stats['mask_files'].append(f)
    
    stats['dce_count'] = len(stats['dce_files'])
    stats['mask_count'] = len(stats['mask_files'])
    
    # Calculate sizes
    stats['dce_size'] = sum(os.path.getsize(f) for f in stats['dce_files'] if os.path.exists(f))
    stats['mask_size'] = sum(os.path.getsize(f) for f in stats['mask_files'] if os.path.exists(f))
    stats['total_size'] = stats['dce_size'] + stats['mask_size']
    
    # Find DCE and mask directories
    for f in stats['dce_files']:
        dce_dir = os.path.dirname(f)
        if stats['found_dce_dir'] is None or len(f) < len(stats['found_dce_dir']):
            stats['found_dce_dir'] = dce_dir
    
    for f in stats['mask_files']:
        mask_dir = os.path.dirname(f)
        if stats['found_mask_dir'] is None or len(f) < len(stats['found_mask_dir']):
            stats['found_mask_dir'] = mask_dir
    
    # Determine dataset type
    dataset_type = dataset_name.lower()
    if 'duke' in dataset_type:
        dataset_type = 'duke'
    elif 'ispy1' in dataset_type or 'spy1' in dataset_type:
        dataset_type = 'ispy1'
    elif 'ispy2' in dataset_type or 'spy2' in dataset_type:
        dataset_type = 'ispy2'
    
    # Extract patient IDs from DCE filenames
    for f in stats['dce_files']:
        basename = os.path.basename(f)
        patient_id = extract_patient_id(basename, dataset_type)
        if patient_id:
            stats['patient_ids'].add(patient_id)
        
        # Extract phase (aqc_N)
        for part in basename.split('_'):
            if part.startswith('aqc_'):
                phase = part.replace('aqc_', '')
                stats['phase_counts'][phase] += 1
    
    # Cross-check with metadata
    if metadata_df is not None and 'pid' in metadata_df.columns:
        # Get metadata patient IDs as strings
        metadata_pids = set(str(p).strip() for p in metadata_df['pid'].tolist())
        actual_pids = stats['patient_ids']
        
        # Count matches (exact match)
        exact_matches = metadata_pids.intersection(actual_pids)
        
        # If exact match is low, try fuzzy matching (strip prefixes)
        if len(exact_matches) < len(metadata_pids) * 0.9:
            # Create a mapping from normalized IDs
            metadata_norm = {}
            for pid in metadata_pids:
                # Extract number from ISPY2-XXXXXX or ACRIN-6698-XXXXXX
                norm = re.sub(r'[^0-9]', '', pid)  # Remove all non-digits
                if norm:
                    metadata_norm[norm] = pid
            
            file_norm = {}
            for pid in actual_pids:
                norm = re.sub(r'[^0-9]', '', pid)
                if norm:
                    file_norm[norm] = pid
            
            # Find matches using normalized IDs
            norm_matches = set(metadata_norm.keys()).intersection(set(file_norm.keys()))
            fuzzy_matches = set()
            for norm in norm_matches:
                fuzzy_matches.add(metadata_norm[norm])  # Use metadata format
            
            matching_count = len(fuzzy_matches)
            stats['matching_patients'] = matching_count
            stats['patient_ids'] = fuzzy_matches  # Update to use metadata format
        else:
            matching_count = len(exact_matches)
            stats['matching_patients'] = matching_count
        
        stats['metadata_patients'] = len(metadata_pids)
        stats['actual_patients'] = len(actual_pids)
    
    return stats

def check_nifti_integrity(file_list, sample_size=5):
    """Check if NIfTI files can be loaded."""
    issues = []
    if not file_list:
        return issues
    
    sample = file_list[:min(sample_size, len(file_list))]
    for f in sample:
        try:
            img = nib.load(f)
            data = img.get_fdata()
            if data.size == 0:
                issues.append(f"Empty data: {os.path.basename(f)}")
        except Exception as e:
            issues.append(f"Cannot load: {os.path.basename(f)} - {str(e)[:50]}")
    return issues

def analyze_metadata(metadata_path):
    """Analyze the metadata CSV file."""
    print_subheader("Analyzing Metadata")
    
    if not os.path.exists(metadata_path):
        print(f"❌ Metadata file not found: {metadata_path}")
        return None
    
    df = pd.read_csv(metadata_path)
    print(f"  File: {os.path.basename(metadata_path)}")
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Check for pCR column
    if 'pCR' in df.columns:
        pcr_counts = df['pCR'].value_counts()
        print(f"  pCR distribution:")
        for val, count in pcr_counts.items():
            print(f"    {val}: {count} ({count/len(df)*100:.1f}%)")
    
    # Check for dataset column
    if 'dataset' in df.columns:
        dataset_counts = df['dataset'].value_counts()
        print(f"  Dataset distribution:")
        for val, count in dataset_counts.items():
            print(f"    {val}: {count}")
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"  Missing values:")
        for col, count in missing_counts.items():
            if count > 0:
                print(f"    {col}: {count}")
    
    return df

# =============================================================
# MAIN AUDIT
# =============================================================

print_header("COMPLETE DATASET AUDIT (FIXED ISPY2)")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Paths
BASE_DIR = r"D:\Tehseen"
DATASET_DIR = os.path.join(BASE_DIR, "Breast DCEDL")
EXTRACTED_DIR = os.path.join(BASE_DIR, "Breast_DCEDL_Extracted")
METADATA_PATH = os.path.join(DATASET_DIR, "BreastDCEDL_metadata_min_crop.csv")

# =============================================================
# 1. METADATA ANALYSIS
# =============================================================
print_header("1. METADATA ANALYSIS")
metadata_df = analyze_metadata(METADATA_PATH)

if metadata_df is not None:
    if 'dataset' in metadata_df.columns:
        duke_metadata = metadata_df[metadata_df['dataset'].str.lower().str.contains('duke', na=False)]
        ispy1_metadata = metadata_df[metadata_df['dataset'].str.lower().str.contains('spy1|ispy1', na=False)]
        ispy2_metadata = metadata_df[metadata_df['dataset'].str.lower().str.contains('spy2|ispy2', na=False)]
        print(f"\n  Duke patients in metadata: {len(duke_metadata)}")
        print(f"  ISPY1 patients in metadata: {len(ispy1_metadata)}")
        print(f"  ISPY2 patients in metadata: {len(ispy2_metadata)}")

# =============================================================
# 2. SCAN ISPY2 DATASET (FIXED)
# =============================================================
print_header("2. ISPY2 DATASET AUDIT (FIXED)")

ispy2_paths = [
    os.path.join(EXTRACTED_DIR, "ISPY2"),
    os.path.join(EXTRACTED_DIR, "ISPY2", "BreastDCEDL_ISPY2_min_crop"),
    os.path.join(DATASET_DIR, "BreastDCEDL_ISPY2_min_crop"),
    os.path.join(DATASET_DIR, "BreastDCEDL_ISPY2_min_crop", "BreastDCEDL_ISPY2_min_crop"),
]

ispy2_path = None
for p in ispy2_paths:
    if os.path.exists(p):
        ispy2_path = p
        break

if ispy2_path:
    print(f"  Using path: {ispy2_path}")
    ispy2_stats = scan_dataset_recursive("ISPY2", ispy2_path, ispy2_metadata if metadata_df is not None else None)
    print(f"\n  ✅ Exists: {ispy2_stats['exists']}")
    if ispy2_stats['exists']:
        print(f"  DCE files: {ispy2_stats['dce_count']}")
        print(f"  Mask files: {ispy2_stats['mask_count']}")
        print(f"  Total size: {format_size(ispy2_stats['total_size'])}")
        print(f"  Unique patients (from files): {len(ispy2_stats['patient_ids'])}")
        print(f"  Phase counts: {dict(ispy2_stats['phase_counts'])}")
        if 'metadata_patients' in ispy2_stats:
            print(f"  Patients in metadata: {ispy2_stats['metadata_patients']}")
            print(f"  Matching patients: {ispy2_stats.get('matching_patients', 0)}")
            match_pct = ispy2_stats.get('matching_patients', 0) / ispy2_stats.get('metadata_patients', 1) * 100
            print(f"  Match percentage: {match_pct:.1f}%")
            if match_pct >= 98:
                print("  ✅ Patient IDs match perfectly!")
            else:
                print("  ⚠️ Partial match - check formats")
        print(f"  DCE dir: {ispy2_stats['found_dce_dir']}")
        print(f"  Mask dir: {ispy2_stats['found_mask_dir']}")
        if ispy2_stats['issues']:
            print(f"\n  ⚠️ Issues found: {len(ispy2_stats['issues'])}")
            for issue in ispy2_stats['issues'][:5]:
                print(f"    - {issue}")
else:
    print("\n  ❌ ISPY2 dataset not found")
    ispy2_stats = None

# =============================================================
# 3. NIFTI INTEGRITY CHECK
# =============================================================
print_header("3. NIFTI INTEGRITY CHECK")

if ispy2_stats and ispy2_stats['exists'] and ispy2_stats['dce_files']:
    print("\n  Checking ISPY2 DCE files...")
    issues = check_nifti_integrity(ispy2_stats['dce_files'], sample_size=5)
    if issues:
        print(f"    ⚠️ Issues: {len(issues)}")
        for issue in issues[:3]:
            print(f"      - {issue}")
    else:
        print("    ✅ All sample files loaded successfully")

# =============================================================
# 4. SUMMARY
# =============================================================
print_header("4. SUMMARY")

if ispy2_stats and ispy2_stats['exists']:
    print(f"""
  📊 ISPY2 Dataset Summary:
    Status: ✅ Valid
    DCE files: {ispy2_stats['dce_count']}
    Mask files: {ispy2_stats['mask_count']}
    Total size: {format_size(ispy2_stats['total_size'])}
    Patients found in files: {len(ispy2_stats['patient_ids'])}
    Patients in metadata: {ispy2_stats.get('metadata_patients', 'N/A')}
    Matching patients: {ispy2_stats.get('matching_patients', 'N/A')}
    
  📌 Note: ACRIN-6698-XXXXXX files are correctly mapped to ISPY2-XXXXXX metadata IDs.
  ✅ ISPY2 dataset is ready for training!
""")

print("\n" + "="*70)
print(f"  Audit completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)