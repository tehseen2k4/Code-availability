# =============================================================
# APPEND SHALLOW 3D CNN RESULTS TO MASTER CSV
# AND GENERATE THREE TEXT FILES IN SHALLOW CNN FOLDER
# =============================================================

import os
import pandas as pd
from datetime import datetime

# -------------------- PATHS --------------------
MASTER_CSV = r"D:\Tehseen\results\segmentation\master_segmentation_results_enhanced.csv"
SHALLOW_DIR = r"D:\Tehseen\results\segmentation\shallow_cnn"
os.makedirs(SHALLOW_DIR, exist_ok=True)

# -------------------- SHALLOW CNN METRICS (from run output) --------------------
SHALLOW_DATA = {
    'experiment': 'shallow3d_cnn',
    'dataset': 'ISPY2',
    'best_val_dice': 0.6858,
    'best_val_iou': None,           # not available
    'best_val_loss': None,
    'val_dice_final': 0.6637,        # from epoch 25
    'val_iou_final': 0.5163,
    'val_loss_final': 0.2778,
    'train_loss_final': 0.2519,
    'test_dice': 0.6107,
    'test_dice_std': 0.2186,
    'test_iou': 0.4710,
    'test_iou_std': 0.2031,
    'test_dice_tta': 0.6180,
    'test_dice_std_tta': 0.2210,
    'test_iou_tta': 0.4796,
    'test_iou_std_tta': 0.2058,
    'total_params': 68465,
    'epochs_trained': 25
}

# Define the standard column order (including TTA columns)
BASE_COLUMNS = [
    'experiment', 'dataset',
    'best_val_dice', 'best_val_iou', 'best_val_loss',
    'val_dice_final', 'val_iou_final', 'val_loss_final', 'train_loss_final',
    'test_dice', 'test_dice_std', 'test_iou', 'test_iou_std',
    'total_params', 'epochs_trained'
]
TTA_COLUMNS = ['test_dice_tta', 'test_dice_std_tta', 'test_iou_tta', 'test_iou_std_tta']
ALL_COLUMNS = BASE_COLUMNS + TTA_COLUMNS

def append_to_master():
    """Ensure TTA columns exist, then append/update Shallow CNN row."""
    if os.path.exists(MASTER_CSV):
        df_master = pd.read_csv(MASTER_CSV)
        print(f"Loaded master CSV with {len(df_master)} row(s)")
    else:
        df_master = pd.DataFrame(columns=ALL_COLUMNS)
        print("Master CSV not found – creating new")

    # Add TTA columns if they don't exist
    for col in TTA_COLUMNS:
        if col not in df_master.columns:
            df_master[col] = None

    # Remove existing Shallow CNN row if present
    if 'shallow3d_cnn' in df_master['experiment'].values:
        print("⚠️ shallow3d_cnn already exists – updating row")
        df_master = df_master[df_master['experiment'] != 'shallow3d_cnn']

    # Create new row
    df_new = pd.DataFrame([SHALLOW_DATA])

    # Concatenate
    df_combined = pd.concat([df_master, df_new], ignore_index=True)
    # Ensure all columns are present
    for col in ALL_COLUMNS:
        if col not in df_combined.columns:
            df_combined[col] = None
    df_combined = df_combined[ALL_COLUMNS]  # order
    df_combined.sort_values('experiment', inplace=True)
    df_combined.to_csv(MASTER_CSV, index=False)
    print(f"✅ Master CSV updated: {MASTER_CSV}")
    return df_combined

def generate_summary(df):
    """Create a concise summary of all models, using TTA for Shallow if available."""
    lines = []
    lines.append("=" * 80)
    lines.append("SUMMARY OF SEGMENTATION MODELS (including Shallow 3D CNN)")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"{'Model':<30} | {'Best Val Dice':>14} | {'Test Dice (TTA)':>18} | {'Test IoU (TTA)':>14} | {'Params (M)':>12}")
    lines.append("-" * 80)

    for _, row in df.iterrows():
        model = row['experiment'][:28]
        best_dice = row['best_val_dice'] if pd.notna(row['best_val_dice']) else float('nan')
        # Use TTA if available and not NaN, otherwise fallback to non‑TTA
        if 'test_dice_tta' in row and pd.notna(row['test_dice_tta']):
            test_dice = f"{row['test_dice_tta']:.4f} ± {row['test_dice_std_tta']:.4f}"
            test_iou = f"{row['test_iou_tta']:.4f} ± {row['test_iou_std_tta']:.4f}"
        else:
            test_dice = f"{row['test_dice']:.4f} ± {row['test_dice_std']:.4f}"
            test_iou = f"{row['test_iou']:.4f} ± {row['test_iou_std']:.4f}"
        params = f"{row['total_params']/1e6:.2f}" if pd.notna(row['total_params']) else "N/A"
        lines.append(f"{model:<30} | {best_dice:>14.4f} | {test_dice:>18} | {test_iou:>14} | {params:>12}")

    lines.append("-" * 80)
    lines.append("")
    lines.append("Best validation Dice is the primary metric for model selection.")
    lines.append("Test Dice and IoU are reported with standard deviation over the test set.")
    lines.append("For models that support TTA, the TTA values are shown; otherwise non‑TTA values are used.")
    return "\n".join(lines)

def generate_analysis(df):
    """Detailed comparison, highlighting Shallow CNN."""
    lines = []
    lines.append("=" * 80)
    lines.append("ANALYSIS: SHALLOW 3D CNN vs OTHER MODELS")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Find Shallow CNN row
    shallow_row = df[df['experiment'] == 'shallow3d_cnn']
    if shallow_row.empty:
        lines.append("Shallow 3D CNN not found in master CSV.")
        return "\n".join(lines)
    shallow_row = shallow_row.iloc[0]

    # Baseline (original U‑Net)
    baseline = df[df['experiment'].str.contains('run8_3d_unet')]
    if baseline.empty:
        lines.append("Baseline (original U‑Net) not found.")
        return "\n".join(lines)
    baseline = baseline.iloc[0]

    # Best model (by validation dice)
    best_row = df.loc[df['best_val_dice'].idxmax()]
    best_name = best_row['experiment']
    best_dice = best_row['best_val_dice']

    # Simple CNN (previous)
    simple = df[df['experiment'].str.contains('simple_cnn')]
    simple_name = simple.iloc[0]['experiment'] if not simple.empty else "N/A"

    # Attention U‑Net
    attn = df[df['experiment'] == 'attention_unet']
    attn_name = attn.iloc[0]['experiment'] if not attn.empty else "N/A"

    # Get test dice values (prefer TTA if available)
    shallow_test = shallow_row['test_dice_tta'] if pd.notna(shallow_row['test_dice_tta']) else shallow_row['test_dice']
    shallow_test_std = shallow_row['test_dice_std_tta'] if pd.notna(shallow_row['test_dice_std_tta']) else shallow_row['test_dice_std']
    baseline_test = baseline['test_dice']
    baseline_test_std = baseline['test_dice_std']

    lines.append("1. Shallow 3D CNN Performance")
    lines.append("-" * 40)
    lines.append(f"  Best Validation Dice: {shallow_row['best_val_dice']:.4f}")
    lines.append(f"  Test Dice (TTA used if available): {shallow_test:.4f} ± {shallow_test_std:.4f}")
    if pd.notna(shallow_row['test_dice_tta']):
        lines.append(f"    Non‑TTA Test Dice: {shallow_row['test_dice']:.4f} ± {shallow_row['test_dice_std']:.4f}")
        lines.append(f"    TTA improvement: {shallow_row['test_dice_tta'] - shallow_row['test_dice']:+.4f}")
    lines.append(f"  Test IoU:              {shallow_row['test_iou_tta'] if pd.notna(shallow_row['test_iou_tta']) else shallow_row['test_iou']:.4f} ± {shallow_row['test_iou_std_tta'] if pd.notna(shallow_row['test_iou_std_tta']) else shallow_row['test_iou_std']:.4f}")
    lines.append(f"  Parameters:            {shallow_row['total_params']:,} ({shallow_row['total_params']/1e6:.2f}M)")
    lines.append(f"  Epochs trained:        {shallow_row['epochs_trained']}\n")

    lines.append("2. Comparison with Baseline (Original U‑Net)")
    lines.append("-" * 40)
    val_diff = shallow_row['best_val_dice'] - baseline['best_val_dice']
    test_diff = shallow_test - baseline_test
    lines.append(f"  Best Val Dice: {shallow_row['best_val_dice']:.4f} vs {baseline['best_val_dice']:.4f} ({val_diff:+.4f})")
    lines.append(f"  Test Dice:     {shallow_test:.4f} vs {baseline_test:.4f} ({test_diff:+.4f})")
    lines.append(f"  Parameters:    {shallow_row['total_params']/1e6:.2f}M vs {baseline['total_params']/1e6:.2f}M")
    lines.append("")

    if not simple.empty:
        lines.append("3. Comparison with Previous Simple CNN (Ablation)")
        lines.append("-" * 40)
        simple_row = simple.iloc[0]
        simple_test = simple_row['test_dice_tta'] if pd.notna(simple_row.get('test_dice_tta', None)) else simple_row['test_dice']
        lines.append(f"  Simple CNN Val Dice: {simple_row['best_val_dice']:.4f}")
        lines.append(f"  Shallow CNN Val Dice: {shallow_row['best_val_dice']:.4f} (+{(shallow_row['best_val_dice'] - simple_row['best_val_dice']):+.4f})")
        lines.append(f"  Test Dice:       {shallow_test:.4f} vs {simple_test:.4f} ({shallow_test - simple_test:+.4f})")
        lines.append(f"  Parameters:      {shallow_row['total_params']/1e6:.2f}M vs {simple_row['total_params']/1e6:.2f}M")
        lines.append("")

    if not attn.empty:
        lines.append("4. Comparison with Best Model (Attention U‑Net)")
        lines.append("-" * 40)
        attn_row = attn.iloc[0]
        attn_test = attn_row['test_dice_tta'] if pd.notna(attn_row.get('test_dice_tta', None)) else attn_row['test_dice']
        lines.append(f"  Attention U‑Net Val Dice: {attn_row['best_val_dice']:.4f}")
        lines.append(f"  Shallow CNN Val Dice:     {shallow_row['best_val_dice']:.4f} (gap: {attn_row['best_val_dice'] - shallow_row['best_val_dice']:.4f})")
        lines.append(f"  Test Dice:                {shallow_test:.4f} vs {attn_test:.4f} ({shallow_test - attn_test:+.4f})")
        lines.append(f"  Parameters:               {shallow_row['total_params']/1e6:.2f}M vs {attn_row['total_params']/1e6:.2f}M")
        lines.append("")

    lines.append("5. Interpretation")
    lines.append("-" * 40)
    if shallow_row['best_val_dice'] > baseline['best_val_dice']:
        lines.append("  ⚠️ The shallow model outperforms the original U‑Net – unexpected!")
    else:
        lines.append("  ✅ As expected, the shallow model underperforms the original U‑Net.")

    if shallow_test < baseline_test:
        lines.append("  ✅ Test Dice is lower than the original U‑Net, confirming that depth matters.")
    else:
        lines.append("  ⚠️ Test Dice is surprisingly high – but still below the best model.")

    lines.append("")
    lines.append("  This model demonstrates that even a very shallow architecture can learn")
    lines.append("  meaningful features, but deeper architectures (U‑Net, Attention U‑Net)")
    lines.append("  are clearly superior for this task.")

    return "\n".join(lines)

def generate_epoch_log():
    """Create a log from the existing epoch_log.txt."""
    epoch_log_path = os.path.join(SHALLOW_DIR, "epoch_log.txt")
    lines = []
    lines.append("=" * 80)
    lines.append("EPOCH LOG SUMMARY – Shallow 3D CNN")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if os.path.exists(epoch_log_path):
        with open(epoch_log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        lines.append("Full epoch‑by‑epoch log is available in:")
        lines.append(f"  {epoch_log_path}\n")
        lines.append("Key milestones from training:\n")
        # Extract best epochs (lines with 'BEST')
        best_lines = []
        for line in content.split('\n'):
            if 'BEST' in line and '|' in line:
                parts = line.split('|')
                if len(parts) >= 7:
                    epoch = parts[0].strip()
                    val_dice = parts[4].strip()
                    best_lines.append(f"  Epoch {epoch}: Val Dice = {val_dice}")
        if best_lines:
            lines.append("Best epochs:")
            for b in best_lines[-10:]:  # show last 10 best epochs
                lines.append(b)
        else:
            lines.append("No best epochs found; see epoch_log.txt for details.")
    else:
        lines.append(f"⚠️ epoch_log.txt not found at {epoch_log_path}")
        lines.append("Using summary data from the evaluation:\n")
        lines.append(f"  Best Validation Dice: {SHALLOW_DATA['best_val_dice']:.4f}")
        lines.append(f"  Test Dice (TTA):       {SHALLOW_DATA['test_dice_tta']:.4f} ± {SHALLOW_DATA['test_dice_std_tta']:.4f}")
        lines.append(f"  Test IoU (TTA):        {SHALLOW_DATA['test_iou_tta']:.4f} ± {SHALLOW_DATA['test_iou_std_tta']:.4f}")
        lines.append(f"  Epochs trained:        {SHALLOW_DATA['epochs_trained']}")

    return "\n".join(lines)

def main():
    # 1. Append Shallow CNN to master CSV
    df = append_to_master()

    # 2. Generate three text files in shallow_cnn directory
    summary_text = generate_summary(df)
    with open(os.path.join(SHALLOW_DIR, "summary_results.txt"), 'w', encoding='utf-8') as f:
        f.write(summary_text)

    analysis_text = generate_analysis(df)
    with open(os.path.join(SHALLOW_DIR, "analysis_report.txt"), 'w', encoding='utf-8') as f:
        f.write(analysis_text)

    epoch_log_text = generate_epoch_log()
    with open(os.path.join(SHALLOW_DIR, "epoch_log.txt"), 'w', encoding='utf-8') as f:
        f.write(epoch_log_text)

    print(f"\n✅ Three text files created in: {SHALLOW_DIR}")
    print("   - summary_results.txt")
    print("   - analysis_report.txt")
    print("   - epoch_log.txt")

    # 3. Print console summary
    print("\n" + "=" * 80)
    print("📊 SHALLOW 3D CNN – FINAL SUMMARY")
    print("=" * 80)
    print(f"Best Validation Dice:   {SHALLOW_DATA['best_val_dice']:.4f}")
    print(f"Test Dice (no TTA):      {SHALLOW_DATA['test_dice']:.4f} ± {SHALLOW_DATA['test_dice_std']:.4f}")
    print(f"Test Dice (TTA):         {SHALLOW_DATA['test_dice_tta']:.4f} ± {SHALLOW_DATA['test_dice_std_tta']:.4f}")
    print(f"Test IoU (TTA):          {SHALLOW_DATA['test_iou_tta']:.4f} ± {SHALLOW_DATA['test_iou_std_tta']:.4f}")
    print(f"Parameters:              {SHALLOW_DATA['total_params']:,} ({SHALLOW_DATA['total_params']/1e6:.2f}M)")
    print(f"Epochs trained:          {SHALLOW_DATA['epochs_trained']}")
    print("=" * 80)
    print(f"✅ Master CSV updated: {MASTER_CSV}")
    print(f"✅ All reports saved in: {SHALLOW_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()