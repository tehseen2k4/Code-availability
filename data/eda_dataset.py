import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict

# =============================================================
# EDA SCRIPT – Breast DCEDL Dataset Analysis
# =============================================================

def print_header(text, char="="):
    print("\n" + char*70)
    print(f"  {text}")
    print(char*70)

def print_subheader(text):
    print(f"\n--- {text} ---")

def save_plot(fig, filename):
    """Save figure to plots directory."""
    plots_dir = r"D:\Tehseen\eda_plots"
    os.makedirs(plots_dir, exist_ok=True)
    filepath = os.path.join(plots_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  📊 Saved: {filepath}")
    return filepath

# =============================================================
# 1. LOAD DATA
# =============================================================
print_header("BREAST DCEDL DATASET – EXPLORATORY DATA ANALYSIS")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

BASE_DIR = r"D:\Tehseen"
DATASET_DIR = os.path.join(BASE_DIR, "Breast DCEDL")
METADATA_PATH = os.path.join(DATASET_DIR, "BreastDCEDL_metadata_min_crop.csv")

# Load metadata
print_subheader("Loading Metadata")
df = pd.read_csv(METADATA_PATH)
print(f"  Total rows: {len(df)}")
print(f"  Columns: {list(df.columns)}")

# =============================================================
# 2. CLINICAL DATA SUMMARY
# =============================================================
print_header("2. CLINICAL DATA SUMMARY")

# Dataset distribution
print_subheader("Dataset Distribution")
dataset_counts = df['dataset'].value_counts()
for ds, count in dataset_counts.items():
    print(f"  {ds}: {count} patients ({count/len(df)*100:.1f}%)")

# pCR distribution
print_subheader("pCR (Pathological Complete Response) Distribution")
if 'pCR' in df.columns:
    pcr_counts = df['pCR'].value_counts(dropna=False)
    for val, count in pcr_counts.items():
        label = "pCR+" if val == 1 else ("pCR-" if val == 0 else "NaN")
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")

# pCR by dataset
print_subheader("pCR by Dataset")
for ds in df['dataset'].unique():
    subset = df[df['dataset'] == ds]
    if 'pCR' in subset.columns:
        pcr_rate = subset['pCR'].mean() * 100
        pcr_count = subset['pCR'].sum()
        print(f"  {ds}: {pcr_count}/{len(subset)} ({pcr_rate:.1f}%) pCR+")

# =============================================================
# 3. DEMOGRAPHICS
# =============================================================
print_header("3. DEMOGRAPHICS")

# Age
if 'age' in df.columns:
    age_valid = df['age'].dropna()
    print_subheader("Age Distribution")
    print(f"  Mean: {age_valid.mean():.1f} years")
    print(f"  Median: {age_valid.median():.1f} years")
    print(f"  Std: {age_valid.std():.1f}")
    print(f"  Min: {age_valid.min():.1f}")
    print(f"  Max: {age_valid.max():.1f}")

# Menopause
if 'menopause' in df.columns:
    print_subheader("Menopause Status")
    menopause_counts = df['menopause'].value_counts(dropna=False)
    for val, count in menopause_counts.items():
        if pd.isna(val):
            label = "Unknown/Missing"
        else:
            label = str(val)
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")

# Race
if 'race_white' in df.columns and 'race_black' in df.columns:
    print_subheader("Race Distribution")
    white_count = df['race_white'].sum() if 'race_white' in df.columns else 0
    black_count = df['race_black'].sum() if 'race_black' in df.columns else 0
    other_count = len(df) - white_count - black_count
    print(f"  White: {white_count} ({white_count/len(df)*100:.1f}%)")
    print(f"  Black: {black_count} ({black_count/len(df)*100:.1f}%)")
    print(f"  Other/Unknown: {other_count} ({other_count/len(df)*100:.1f}%)")

# =============================================================
# 4. MOLECULAR SUBTYPES
# =============================================================
print_header("4. MOLECULAR SUBTYPES")

# HR/HER2 Status
if 'HR' in df.columns and 'HER2' in df.columns:
    print_subheader("HR/HER2 Status Distribution")
    hr_her2 = df.groupby(['HR', 'HER2']).size()
    for (hr, her2), count in hr_her2.items():
        label = f"HR{'+" if hr==1 else "-'}, HER2{'+" if her2==1 else "-'}"
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")

# HR_HER2_STATUS (pre-classified)
if 'HR_HER2_STATUS' in df.columns:
    print_subheader("HR_HER2_STATUS Distribution")
    status_counts = df['HR_HER2_STATUS'].value_counts(dropna=False)
    for val, count in status_counts.items():
        if pd.isna(val):
            label = "Unknown"
        else:
            label = str(val)
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")

# Triple Negative
if 'TripleNeg' in df.columns:
    print_subheader("Triple Negative Status")
    tn_count = df['TripleNeg'].sum() if 'TripleNeg' in df.columns else 0
    print(f"  Triple Negative: {tn_count} ({tn_count/len(df)*100:.1f}%)")

# HER2 Positive
if 'HER2pos' in df.columns:
    print_subheader("HER2 Positive Status")
    her2pos_count = df['HER2pos'].sum() if 'HER2pos' in df.columns else 0
    print(f"  HER2 Positive: {her2pos_count} ({her2pos_count/len(df)*100:.1f}%)")

# HR+/HER2- (Luminal)
if 'HRposHER2neg' in df.columns:
    print_subheader("HR+/HER2- (Luminal) Status")
    luminal_count = df['HRposHER2neg'].sum() if 'HRposHER2neg' in df.columns else 0
    print(f"  HR+/HER2- (Luminal): {luminal_count} ({luminal_count/len(df)*100:.1f}%)")

# =============================================================
# 5. pCR BY SUBTYPE
# =============================================================
print_header("5. pCR BY MOLECULAR SUBTYPE")

if 'pCR' in df.columns and 'HR_HER2_STATUS' in df.columns:
    print_subheader("pCR Rate by HR_HER2_STATUS")
    pcr_by_subtype = df.groupby('HR_HER2_STATUS')['pCR'].agg(['count', 'mean', 'sum'])
    pcr_by_subtype['pCR_rate'] = pcr_by_subtype['mean'] * 100
    pcr_by_subtype = pcr_by_subtype.sort_values('pCR_rate', ascending=False)
    for status, row in pcr_by_subtype.iterrows():
        if pd.isna(status):
            status = "Unknown"
        print(f"  {status}: {int(row['sum'])}/{int(row['count'])} ({row['pCR_rate']:.1f}%)")

# Triple Negative vs Others
if 'TripleNeg' in df.columns and 'pCR' in df.columns:
    print_subheader("pCR Rate: Triple Negative vs Others")
    tn_pcr = df[df['TripleNeg'] == 1]['pCR'].mean() * 100
    other_pcr = df[df['TripleNeg'] == 0]['pCR'].mean() * 100
    print(f"  Triple Negative: {tn_pcr:.1f}%")
    print(f"  Others: {other_pcr:.1f}%")

# HER2 Positive vs Others
if 'HER2pos' in df.columns and 'pCR' in df.columns:
    print_subheader("pCR Rate: HER2 Positive vs Others")
    her2pos_pcr = df[df['HER2pos'] == 1]['pCR'].mean() * 100
    other_pcr = df[df['HER2pos'] == 0]['pCR'].mean() * 100
    print(f"  HER2 Positive: {her2pos_pcr:.1f}%")
    print(f"  Others: {other_pcr:.1f}%")

# =============================================================
# 6. TUMOR VOLUME ANALYSIS
# =============================================================
print_header("6. TUMOR VOLUME ANALYSIS")

if 'tum_vol' in df.columns:
    tum_vol_valid = df['tum_vol'].dropna()
    print_subheader("Tumor Volume Distribution")
    print(f"  Mean: {tum_vol_valid.mean():.1f} mm³")
    print(f"  Median: {tum_vol_valid.median():.1f} mm³")
    print(f"  Std: {tum_vol_valid.std():.1f}")
    print(f"  Min: {tum_vol_valid.min():.1f}")
    print(f"  Max: {tum_vol_valid.max():.1f}")

    # pCR by tumor volume (quartiles)
    print_subheader("pCR Rate by Tumor Volume Quartiles")
    quartiles = tum_vol_valid.quantile([0.25, 0.5, 0.75])
    print(f"  Q1 (<{quartiles[0.25]:.1f}): {df[df['tum_vol'] < quartiles[0.25]]['pCR'].mean()*100:.1f}%")
    print(f"  Q2 ({quartiles[0.25]:.1f}-{quartiles[0.5]:.1f}): {df[(df['tum_vol'] >= quartiles[0.25]) & (df['tum_vol'] < quartiles[0.5])]['pCR'].mean()*100:.1f}%")
    print(f"  Q3 ({quartiles[0.5]:.1f}-{quartiles[0.75]:.1f}): {df[(df['tum_vol'] >= quartiles[0.5]) & (df['tum_vol'] < quartiles[0.75])]['pCR'].mean()*100:.1f}%")
    print(f"  Q4 (>{quartiles[0.75]:.1f}): {df[df['tum_vol'] >= quartiles[0.75]]['pCR'].mean()*100:.1f}%")

# =============================================================
# 7. MISSING DATA ANALYSIS
# =============================================================
print_header("7. MISSING DATA ANALYSIS")

print_subheader("Missing Values per Column")
missing_counts = df.isnull().sum()
missing_percent = (missing_counts / len(df)) * 100
missing_df = pd.DataFrame({'Count': missing_counts, 'Percent': missing_percent})
missing_df = missing_df[missing_df['Count'] > 0].sort_values('Count', ascending=False)

if len(missing_df) > 0:
    for col, row in missing_df.iterrows():
        print(f"  {col}: {int(row['Count'])} ({row['Percent']:.1f}%)")
else:
    print("  No missing values found!")

# =============================================================
# 8. IMAGING DATA SUMMARY (File Counts)
# =============================================================
print_header("8. IMAGING DATA SUMMARY")

def count_files_in_dir(base_path, pattern):
    if not os.path.exists(base_path):
        return 0, []
    files = glob.glob(os.path.join(base_path, pattern), recursive=True)
    return len(files), files

# Check each dataset
datasets = {
    'Duke': os.path.join(DATASET_DIR, "BreastDCEDL_DUKE_min_crop"),
    'ISPY1': os.path.join(DATASET_DIR, "BreastDCEDL_ISPY1_min_crop"),
    'ISPY2': os.path.join(DATASET_DIR, "BreastDCEDL_ISPY2_min_crop"),
}

print_subheader("Dataset File Counts (if extracted)")
for name, path in datasets.items():
    if os.path.exists(path):
        dce_count, _ = count_files_in_dir(path, "**/*dce*.nii.gz")
        mask_count, _ = count_files_in_dir(path, "**/*mask*.nii.gz")
        print(f"  {name}: DCE={dce_count}, Masks={mask_count}")
    else:
        print(f"  {name}: Not found (not extracted yet)")

# =============================================================
# 9. DATA SPLITS (Train/Val/Test)
# =============================================================
print_header("9. DATA SPLITS")

split_files = ['train.csv', 'val.csv', 'test.csv']
for split_file in split_files:
    split_path = os.path.join(DATASET_DIR, split_file)
    if os.path.exists(split_path):
        split_df = pd.read_csv(split_path)
        print(f"\n  {split_file.upper()}: {len(split_df)} patients")
        if 'pCR' in split_df.columns:
            pcr_count = split_df['pCR'].sum()
            pcr_rate = pcr_count / len(split_df) * 100
            print(f"    pCR+: {pcr_count} ({pcr_rate:.1f}%)")
    else:
        print(f"  {split_file}: Not found")

# =============================================================
# 10. VISUALIZATIONS
# =============================================================
print_header("10. VISUALIZATIONS")

print("  Generating plots...")

# Set style
sns.set_style("whitegrid")
sns.set_context("talk")

# Plot 1: pCR Distribution
fig1, ax1 = plt.subplots(figsize=(8, 6))
pcr_counts = df['pCR'].value_counts().sort_index()
colors = ['#e74c3c', '#2ecc71']
bars = ax1.bar(['pCR- (0)', 'pCR+ (1)'], pcr_counts.values, color=colors)
ax1.set_title('pCR Distribution in Breast DCEDL Dataset')
ax1.set_ylabel('Number of Patients')
ax1.set_xlabel('pCR Status')
for bar, count in zip(bars, pcr_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(count), ha='center', va='bottom')
save_plot(fig1, '01_pcr_distribution.png')
plt.close(fig1)

# Plot 2: pCR by Dataset
if 'dataset' in df.columns and 'pCR' in df.columns:
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ds_pcr = df.groupby('dataset')['pCR'].agg(['mean', 'count']).reset_index()
    ds_pcr.columns = ['Dataset', 'pCR_rate', 'Count']
    ds_pcr['pCR_rate'] = ds_pcr['pCR_rate'] * 100
    ds_pcr = ds_pcr.sort_values('pCR_rate', ascending=True)
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = ax2.barh(ds_pcr['Dataset'], ds_pcr['pCR_rate'], color=colors[:len(ds_pcr)])
    ax2.set_title('pCR Rate by Dataset')
    ax2.set_xlabel('pCR Rate (%)')
    ax2.set_ylabel('Dataset')
    for bar, rate in zip(bars, ds_pcr['pCR_rate']):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{rate:.1f}%', va='center')
    save_plot(fig2, '02_pcr_by_dataset.png')
    plt.close(fig2)

# Plot 3: pCR by HR_HER2_STATUS
if 'HR_HER2_STATUS' in df.columns and 'pCR' in df.columns:
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    status_pcr = df.groupby('HR_HER2_STATUS')['pCR'].agg(['mean', 'count']).reset_index()
    status_pcr = status_pcr.dropna()
    status_pcr['pCR_rate'] = status_pcr['mean'] * 100
    status_pcr = status_pcr.sort_values('pCR_rate', ascending=True)
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    bars = ax3.barh(status_pcr['HR_HER2_STATUS'], status_pcr['pCR_rate'], color=colors[:len(status_pcr)])
    ax3.set_title('pCR Rate by Molecular Subtype')
    ax3.set_xlabel('pCR Rate (%)')
    ax3.set_ylabel('Molecular Subtype')
    for bar, rate in zip(bars, status_pcr['pCR_rate']):
        ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{rate:.1f}%', va='center')
    save_plot(fig3, '03_pcr_by_subtype.png')
    plt.close(fig3)

# Plot 4: Age Distribution
if 'age' in df.columns:
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    age_valid = df['age'].dropna()
    ax4.hist(age_valid, bins=30, edgecolor='black', alpha=0.7, color='#3498db')
    ax4.axvline(age_valid.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {age_valid.mean():.1f}')
    ax4.axvline(age_valid.median(), color='green', linestyle='dashed', linewidth=2, label=f'Median: {age_valid.median():.1f}')
    ax4.set_title('Age Distribution of Patients')
    ax4.set_xlabel('Age (years)')
    ax4.set_ylabel('Number of Patients')
    ax4.legend()
    save_plot(fig4, '04_age_distribution.png')
    plt.close(fig4)

# Plot 5: Tumor Volume Distribution
if 'tum_vol' in df.columns:
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    tum_vol_valid = df['tum_vol'].dropna()
    ax5.hist(tum_vol_valid, bins=30, edgecolor='black', alpha=0.7, color='#2ecc71')
    ax5.axvline(tum_vol_valid.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {tum_vol_valid.mean():.1f}')
    ax5.axvline(tum_vol_valid.median(), color='green', linestyle='dashed', linewidth=2, label=f'Median: {tum_vol_valid.median():.1f}')
    ax5.set_title('Tumor Volume Distribution')
    ax5.set_xlabel('Tumor Volume (mm³)')
    ax5.set_ylabel('Number of Patients')
    ax5.legend()
    save_plot(fig5, '05_tumor_volume_distribution.png')
    plt.close(fig5)

# Plot 6: Missing Data Heatmap
fig6, ax6 = plt.subplots(figsize=(14, 8))
missing_plot = df.isnull().sum()
missing_plot = missing_plot[missing_plot > 0].sort_values(ascending=False)
colors = ['#e74c3c' if x > 500 else '#f39c12' if x > 100 else '#3498db' for x in missing_plot.values]
bars = ax6.barh(missing_plot.index, missing_plot.values, color=colors)
ax6.set_title('Missing Data by Column')
ax6.set_xlabel('Number of Missing Values')
ax6.set_ylabel('Column')
for bar, count in zip(bars, missing_plot.values):
    ax6.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, str(int(count)), va='center')
save_plot(fig6, '06_missing_data.png')
plt.close(fig6)

print(f"\n  ✅ All plots saved to: D:\\Tehseen\\eda_plots\\")

# =============================================================
# 11. SUMMARY
# =============================================================
print_header("11. EDA SUMMARY")

print(f"""
  📊 Dataset Overview:
    - Total patients: {len(df)}
    - pCR+ rate: {df['pCR'].mean()*100:.1f}%
    - Datasets: {', '.join(df['dataset'].unique())}

  🧬 Molecular Subtypes (pCR rates):
""")

if 'HR_HER2_STATUS' in df.columns and 'pCR' in df.columns:
    status_pcr = df.groupby('HR_HER2_STATUS')['pCR'].mean().sort_values(ascending=False)
    for status, rate in status_pcr.items():
        if pd.isna(status):
            status = "Unknown"
        print(f"    {status}: {rate*100:.1f}%")

print(f"""
  📈 Key Observations:
    - pCR rate varies significantly by molecular subtype
    - Triple Negative and HER2+ have highest pCR rates
    - HR+/HER2- (Luminal) have lowest pCR rates

  💾 Files Generated:
    - EDA plots saved to: D:\\Tehseen\\eda_plots\\
    - This report printed above

  ✅ EDA Complete!
""")

print("="*70)
print(f"  EDA completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)