import os
import pandas as pd
from datetime import datetime

# =============================================================
# SCRIPT TO GENERATE SEPARATE EXCEL SHEETS FOR EACH DATASET
# =============================================================

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_subheader(text):
    print(f"\n--- {text} ---")

def save_to_excel(df_dict, filepath, dataset_name):
    """
    Save multiple DataFrames to a single Excel file with different sheets.
    df_dict: dict with keys as sheet names and values as DataFrames
    """
    if not df_dict:
        print(f"  ⚠️ No data to save for {dataset_name}")
        return
    
    try:
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for sheet_name, df in df_dict.items():
                if len(df) > 0:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"  ✅ {sheet_name}: {len(df)} rows")
                else:
                    print(f"  ⚠️ {sheet_name}: Empty (skipped)")
        print(f"\n  💾 Saved: {filepath}")
    except Exception as e:
        print(f"  ❌ Error saving {filepath}: {e}")

# =============================================================
# 1. LOAD DATA
# =============================================================
print_header("GENERATING DATASET SPLITS")

BASE_DIR = r"D:\Tehseen"
DATASET_DIR = os.path.join(BASE_DIR, "Breast DCEDL")
INPUT_FILE = os.path.join(DATASET_DIR, "old_split_for_run_1_2_3.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_splits")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"  Input file: {INPUT_FILE}")
print(f"  Output directory: {OUTPUT_DIR}")

# Load the CSV file
df = pd.read_csv(INPUT_FILE)
print(f"\n  ✅ Loaded {len(df)} total rows")
print(f"  📊 Columns: {list(df.columns)}")

# =============================================================
# 2. DATASET DISTRIBUTION
# =============================================================
print_subheader("Dataset Distribution (Original)")
dataset_counts = df['dataset'].value_counts()
for ds, count in dataset_counts.items():
    split_counts = df[df['dataset'] == ds]['split'].value_counts()
    print(f"  {ds}: {count} patients")
    for split, scount in split_counts.items():
        print(f"    - {split}: {scount}")

# =============================================================
# 3. GENERATE EXCEL FILES FOR EACH DATASET
# =============================================================
print_header("GENERATING EXCEL FILES")

# Define dataset mapping - only for datasets that exist
available_datasets = df['dataset'].unique()
print(f"  Available datasets: {available_datasets}")

datasets = {}

if 'duke' in available_datasets:
    datasets['duke'] = {
        'name': 'DUKE',
        'filter': df['dataset'] == 'duke',
        'filename': 'DUKE_dataset_splits.xlsx'
    }

if 'spy1' in available_datasets:
    datasets['spy1'] = {
        'name': 'ISPY1',
        'filter': df['dataset'] == 'spy1',
        'filename': 'ISPY1_dataset_splits.xlsx'
    }

if 'spy2' in available_datasets:
    datasets['spy2'] = {
        'name': 'ISPY2',
        'filter': df['dataset'] == 'spy2',
        'filename': 'ISPY2_dataset_splits.xlsx'
    }

combined_sheets = {}

for key, config in datasets.items():
    print_subheader(f"{config['name']}")
    
    # Filter for this dataset
    dataset_df = df[config['filter']].copy()
    dataset_df = dataset_df.reset_index(drop=True)
    
    print(f"  Total patients: {len(dataset_df)}")
    
    if len(dataset_df) == 0:
        print(f"  ⚠️ No patients found for {config['name']}")
        continue
    
    # Create sheets for train, val, test
    sheets = {}
    for split in ['train', 'val', 'test']:
        split_df = dataset_df[dataset_df['split'] == split].copy()
        split_df = split_df.reset_index(drop=True)
        sheets[split.upper()] = split_df
        print(f"    {split.upper()}: {len(split_df)} patients")
    
    # Add a summary sheet
    summary_data = {
        'Split': ['Train', 'Val', 'Test', 'Total'],
        'Count': [len(sheets['TRAIN']), len(sheets['VAL']), len(sheets['TEST']), len(dataset_df)]
    }
    summary_df = pd.DataFrame(summary_data)
    sheets['SUMMARY'] = summary_df
    
    # Add pCR distribution per split
    pcr_summary = []
    for split in ['train', 'val', 'test']:
        split_df = dataset_df[dataset_df['split'] == split]
        if len(split_df) > 0:
            # Only count patients with known pCR
            pcr_known = split_df['pCR'].notna()
            pcr_0 = len(split_df[(split_df['pCR'] == 0) & pcr_known])
            pcr_1 = len(split_df[(split_df['pCR'] == 1) & pcr_known])
            pcr_unknown = len(split_df[~pcr_known])
            pcr_rate = pcr_1 / (pcr_0 + pcr_1) * 100 if (pcr_0 + pcr_1) > 0 else 0
            pcr_summary.append({
                'Split': split.upper(),
                'Total': len(split_df),
                'pCR- (0)': pcr_0,
                'pCR+ (1)': pcr_1,
                'pCR Unknown': pcr_unknown,
                'pCR Rate (%)': round(pcr_rate, 2)
            })
    pcr_df = pd.DataFrame(pcr_summary)
    sheets['PCR_SUMMARY'] = pcr_df
    
    # Save to Excel
    output_path = os.path.join(OUTPUT_DIR, config['filename'])
    save_to_excel(sheets, output_path, config['name'])
    
    # Add to combined sheets for combined file
    for split in ['TRAIN', 'VAL', 'TEST']:
        sheet_name = f"{config['name']}_{split}"
        combined_sheets[sheet_name] = sheets[split]
    
    # Add summary sheets to combined
    combined_sheets[f"{config['name']}_SUMMARY"] = sheets['SUMMARY']
    combined_sheets[f"{config['name']}_PCR"] = sheets['PCR_SUMMARY']

# =============================================================
# 4. GENERATE COMBINED EXCEL FILE
# =============================================================
if combined_sheets:
    print_header("GENERATING COMBINED EXCEL FILE")
    combined_path = os.path.join(OUTPUT_DIR, "All_Datasets_Complete.xlsx")
    save_to_excel(combined_sheets, combined_path, "All Datasets")

# =============================================================
# 5. GENERATE ISPY2-ONLY CSV SPLITS (For training)
# =============================================================
if 'spy2' in available_datasets:
    print_header("GENERATING ISPY2 CSV SPLITS")
    
    # Filter ISPY2
    ispy2_df = df[df['dataset'] == 'spy2'].copy()
    
    # Create train/val/test CSV files
    for split in ['train', 'val', 'test']:
        split_df = ispy2_df[ispy2_df['split'] == split].copy()
        split_df = split_df.reset_index(drop=True)
        output_path = os.path.join(OUTPUT_DIR, f"ispy2_{split}.csv")
        split_df.to_csv(output_path, index=False)
        print(f"  ✅ {split.upper()}: {len(split_df)} patients -> {output_path}")
else:
    print_header("ISPY2 NOT FOUND")
    print("  ⚠️ ISPY2 dataset not found in the split file.")
    print("  Only generating available datasets.")

# =============================================================
# 6. GENERATE METADATA STATISTICS
# =============================================================
print_header("METADATA STATISTICS")

stats_data = []

for key, config in datasets.items():
    dataset_df = df[config['filter']].copy()
    
    if len(dataset_df) == 0:
        continue
    
    # Basic stats
    stats = {
        'Dataset': config['name'],
        'Total Patients': len(dataset_df),
        'Train': len(dataset_df[dataset_df['split'] == 'train']),
        'Val': len(dataset_df[dataset_df['split'] == 'val']),
        'Test': len(dataset_df[dataset_df['split'] == 'test']),
    }
    
    # pCR stats
    pcr_known = dataset_df['pCR'].notna().sum()
    pcr_pos = dataset_df['pCR'].sum() if pcr_known > 0 else 0
    pcr_rate = pcr_pos / pcr_known * 100 if pcr_known > 0 else 0
    
    stats['pCR Known'] = pcr_known
    stats['pCR+'] = int(pcr_pos)
    stats['pCR Rate (%)'] = round(pcr_rate, 2)
    
    stats_data.append(stats)

if stats_data:
    stats_df = pd.DataFrame(stats_data)
    stats_path = os.path.join(OUTPUT_DIR, "dataset_statistics.xlsx")
    stats_df.to_excel(stats_path, index=False)
    print(f"  ✅ Saved: {stats_path}")

# =============================================================
# 7. FINAL SUMMARY
# =============================================================
print_header("EXTRACTION COMPLETE")

print(f"""
  📁 Output Directory: {OUTPUT_DIR}
  
  📊 Generated Files:
""")

# List all generated files
for f in os.listdir(OUTPUT_DIR):
    if f.endswith('.xlsx') or f.endswith('.csv'):
        file_path = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(file_path) / 1024
        print(f"    - {f} ({size:.1f} KB)")

print(f"""
  📋 Each Excel file contains:
    - TRAIN sheet
    - VAL sheet
    - TEST sheet
    - SUMMARY sheet (split counts)
    - PCR_SUMMARY sheet (pCR distribution)
  
  🚀 For training, use the ISPY2 CSV files:
    - ispy2_train.csv
    - ispy2_val.csv
    - ispy2_test.csv
""")

print("="*70)
print(f"  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)