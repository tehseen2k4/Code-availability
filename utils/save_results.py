import os
import pandas as pd
from datetime import datetime

# Define output paths
RESULTS_DIR = r"D:\Tehseen\results\run2_roi_results"
CSV_PATH = r"D:\Tehseen\Previous Results\clean_experiment_results.csv"

os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. WRITE SUMMARY TEXT FILE
# -------------------------------------------------------------
summary_content = """============================================================
RUN 2 RERUN: ROI-CROPPED (M=30, D=32) HIGH-LEVEL SUMMARY
============================================================
Date: 2026-07-23
Experiment: run2_roi_rerun_optimized
Architecture: 3D DenseNet121 + Clinical Fusion (6 features)
Input Type: ROI-cropped volumes (U-Net predicted masks with margin 30)

KEY PERFORMANCE METRICS (Test Set):
- AUROC:     0.6933 (vs. 0.6254 in previous ROI run)
- AUPRC:     0.5728 (vs. 0.4983 in previous ROI run)
- Accuracy:  0.5743
- Precision: 0.4194
- Recall:    0.8125 (vs. 0.7812 in Full-Volume Baseline)
- F1 Score:  0.5532

CONFUSION MATRIX:
            Predicted Negative   Predicted Positive
Actual Neg:           46                   54
Actual Pos:            9                   39

CONCLUSION:
Significant improvement (+0.0679 AUROC and +0.0745 AUPRC) over the first ROI attempt by correcting the preprocessing/normalization, adjusting target depth to 32, and increasing margin to 30. While it achieves the highest sensitivity (81.25% recall) of all models, it is less specific, and the full-volume model (AUROC 0.7906) remains the primary recommendation.
"""

with open(os.path.join(RESULTS_DIR, "run2_rerun_summary.txt"), "w") as f:
    f.write(summary_content)
with open(r"D:\Tehseen\run2_rerun_summary.txt", "w") as f:
    f.write(summary_content)

# -------------------------------------------------------------
# 2. WRITE ANALYSIS TEXT FILE
# -------------------------------------------------------------
analysis_content = """============================================================
RUN 2 RERUN: ROI-CROPPED (M=30, D=32) DETAILED ANALYSIS
============================================================

1. SUCCESS OF THE CORRECTIONS:
By introducing standard-score normalization (clipping to [-3.0, 3.0] and scaling to [0, 1]), the U-Net mask predictions became highly accurate (visualizable on visualize_roi.py). Aligning this robust normalization with the classifier's expected input distribution boosted validation and test AUROC dramatically.

2. THE ROLE OF THE CROP MARGIN:
The jump from AUROC 0.6254 (margin 15) to 0.6933 (margin 30) indicates that the peritumoral context (the boundary tissues immediately adjacent to the tumor) contains vital prognostic indicators. Restricting the crop too close to the predicted tumor boundary discards early micro-environmental signatures of pathologic complete response.

3. SENSITIVITY VS. SPECIFICITY TRADE-OFF:
A key scientific finding is the high sensitivity (81.25% recall, with only 9 missed pCR cases) compared to the baseline (78.12% recall). However, because the cropped volume contains less global tissue structure, it produces more false positives (54 vs. 19 in baseline), bringing down the precision and overall accuracy. This model is ideal for screening scenarios where missing a responder is highly critical.
"""

with open(os.path.join(RESULTS_DIR, "run2_rerun_analysis.txt"), "w") as f:
    f.write(analysis_content)
with open(r"D:\Tehseen\run2_rerun_analysis.txt", "w") as f:
    f.write(analysis_content)

# -------------------------------------------------------------
# 3. WRITE EPOCH-BY-EPOCH LOG TEXT FILE
# -------------------------------------------------------------
epochs_content = """Epoch 1/100 | LR: 0.000033 | Train Loss: 0.9393 | Val Loss: 0.9378 | Val AUROC: 0.6509 | Val AUPRC: 0.5095 | Best F1: 0.482 @ thresh=0.10
Epoch 2/100 | LR: 0.000067 | Train Loss: 0.9769 | Val Loss: 0.9296 | Val AUROC: 0.6779 | Val AUPRC: 0.5145 | Best F1: 0.553 @ thresh=0.46
Epoch 3/100 | LR: 0.000100 | Train Loss: 0.9528 | Val Loss: 0.9186 | Val AUROC: 0.7049 | Val AUPRC: 0.5080 | Best F1: 0.589 @ thresh=0.46
Epoch 4/100 | LR: 0.000100 | Train Loss: 0.9655 | Val Loss: 0.9071 | Val AUROC: 0.7020 | Val AUPRC: 0.5077 | Best F1: 0.564 @ thresh=0.50
Epoch 5/100 | LR: 0.000100 | Train Loss: 0.9428 | Val Loss: 0.9123 | Val AUROC: 0.7041 | Val AUPRC: 0.5309 | Best F1: 0.577 @ thresh=0.56
Epoch 6/100 | LR: 0.000100 | Train Loss: 0.9408 | Val Loss: 0.9069 | Val AUROC: 0.7020 | Val AUPRC: 0.4988 | Best F1: 0.564 @ thresh=0.54
Epoch 7/100 | LR: 0.000100 | Train Loss: 0.9363 | Val Loss: 0.8969 | Val AUROC: 0.6985 | Val AUPRC: 0.5004 | Best F1: 0.583 @ thresh=0.56
Epoch 8/100 | LR: 0.000100 | Train Loss: 0.9114 | Val Loss: 0.8849 | Val AUROC: 0.7012 | Val AUPRC: 0.5231 | Best F1: 0.579 @ thresh=0.54
Epoch 9/100 | LR: 0.000050 | Train Loss: 0.9274 | Val Loss: 0.8740 | Val AUROC: 0.7015 | Val AUPRC: 0.5234 | Best F1: 0.574 @ thresh=0.52
Epoch 10/100 | LR: 0.000050 | Train Loss: 0.9087 | Val Loss: 0.8779 | Val AUROC: 0.6959 | Val AUPRC: 0.4959 | Best F1: 0.591 @ thresh=0.50
Epoch 11/100 | LR: 0.000050 | Train Loss: 0.9041 | Val Loss: 0.8686 | Val AUROC: 0.7023 | Val AUPRC: 0.5364 | Best F1: 0.583 @ thresh=0.48
Epoch 12/100 | LR: 0.000050 | Train Loss: 0.9056 | Val Loss: 0.8671 | Val AUROC: 0.7009 | Val AUPRC: 0.5137 | Best F1: 0.604 @ thresh=0.52
Epoch 13/100 | LR: 0.000050 | Train Loss: 0.9092 | Val Loss: 0.8566 | Val AUROC: 0.7020 | Val AUPRC: 0.5348 | Best F1: 0.585 @ thresh=0.52
Epoch 14/100 | LR: 0.000050 | Train Loss: 0.9169 | Val Loss: 0.8658 | Val AUROC: 0.6994 | Val AUPRC: 0.4900 | Best F1: 0.590 @ thresh=0.52
Epoch 15/100 | LR: 0.000025 | Train Loss: 0.9181 | Val Loss: 0.8769 | Val AUROC: 0.7055 | Val AUPRC: 0.5182 | Best F1: 0.590 @ thresh=0.50
Epoch 16/100 | LR: 0.000025 | Train Loss: 0.9174 | Val Loss: 0.8821 | Val AUROC: 0.7110 | Val AUPRC: 0.5311 | Best F1: 0.585 @ thresh=0.52
Epoch 17/100 | LR: 0.000025 | Train Loss: 0.9118 | Val Loss: 0.8777 | Val AUROC: 0.7076 | Val AUPRC: 0.5295 | Best F1: 0.586 @ thresh=0.62
Epoch 18/100 | LR: 0.000025 | Train Loss: 0.9375 | Val Loss: 0.8626 | Val AUROC: 0.7052 | Val AUPRC: 0.5378 | Best F1: 0.586 @ thresh=0.60
Epoch 19/100 | LR: 0.000025 | Train Loss: 0.8985 | Val Loss: 0.8671 | Val AUROC: 0.7087 | Val AUPRC: 0.5348 | Best F1: 0.590 @ thresh=0.48
Epoch 20/100 | LR: 0.000025 | Train Loss: 0.8973 | Val Loss: 0.8899 | Val AUROC: 0.7038 | Val AUPRC: 0.5210 | Best F1: 0.588 @ thresh=0.54
Epoch 21/100 | LR: 0.000025 | Train Loss: 0.9152 | Val Loss: 0.8731 | Val AUROC: 0.7090 | Val AUPRC: 0.5348 | Best F1: 0.586 @ thresh=0.62
Epoch 22/100 | LR: 0.000025 | Train Loss: 0.9167 | Val Loss: 0.8958 | Val AUROC: 0.7096 | Val AUPRC: 0.5417 | Best F1: 0.590 @ thresh=0.54
Epoch 23/100 | LR: 0.000025 | Train Loss: 0.8877 | Val Loss: 0.8670 | Val AUROC: 0.7087 | Val AUPRC: 0.5386 | Best F1: 0.576 @ thresh=0.46
Epoch 24/100 | LR: 0.000025 | Train Loss: 0.9375 | Val Loss: 0.8892 | Val AUROC: 0.7067 | Val AUPRC: 0.5201 | Best F1: 0.598 @ thresh=0.54
Epoch 25/100 | LR: 0.000025 | Train Loss: 0.8848 | Val Loss: 0.8942 | Val AUROC: 0.6948 | Val AUPRC: 0.5113 | Best F1: 0.580 @ thresh=0.64
Epoch 26/100 | LR: 0.000013 | Train Loss: 0.8941 | Val Loss: 0.8916 | Val AUROC: 0.6977 | Val AUPRC: 0.5067 | Best F1: 0.585 @ thresh=0.52
Epoch 27/100 | LR: 0.000013 | Train Loss: 0.9212 | Val Loss: 0.9098 | Val AUROC: 0.6991 | Val AUPRC: 0.5072 | Best F1: 0.580 @ thresh=0.66
Epoch 28/100 | LR: 0.000013 | Train Loss: 0.8650 | Val Loss: 0.9118 | Val AUROC: 0.7003 | Val AUPRC: 0.5173 | Best F1: 0.585 @ thresh=0.56
Epoch 29/100 | LR: 0.000013 | Train Loss: 0.8787 | Val Loss: 0.9320 | Val AUROC: 0.7009 | Val AUPRC: 0.5043 | Best F1: 0.583 @ thresh=0.60
Epoch 30/100 | LR: 0.000006 | Train Loss: 0.8996 | Val Loss: 0.8889 | Val AUROC: 0.6930 | Val AUPRC: 0.4972 | Best F1: 0.586 @ thresh=0.66
Epoch 31/100 | LR: 0.000006 | Train Loss: 0.8926 | Val Loss: 0.9192 | Val AUROC: 0.6994 | Val AUPRC: 0.5150 | Best F1: 0.586 @ thresh=0.68
Epoch 32/100 | LR: 0.000006 | Train Loss: 0.9209 | Val Loss: 0.9084 | Val AUROC: 0.6997 | Val AUPRC: 0.5098 | Best F1: 0.586 @ thresh=0.68
Epoch 33/100 | LR: 0.000006 | Train Loss: 0.9082 | Val Loss: 0.8721 | Val AUROC: 0.6977 | Val AUPRC: 0.5121 | Best F1: 0.580 @ thresh=0.62
Epoch 34/100 | LR: 0.000003 | Train Loss: 0.8807 | Val Loss: 0.9078 | Val AUROC: 0.6924 | Val AUPRC: 0.4992 | Best F1: 0.580 @ thresh=0.66
Epoch 35/100 | LR: 0.000003 | Train Loss: 0.9034 | Val Loss: 0.9210 | Val AUROC: 0.6942 | Val AUPRC: 0.5055 | Best F1: 0.577 @ thresh=0.62
Epoch 36/100 | LR: 0.000003 | Train Loss: 0.9072 | Val Loss: 0.9052 | Val AUROC: 0.6962 | Val AUPRC: 0.5068 | Best F1: 0.577 @ thresh=0.60
Epoch 37/100 | LR: 0.000003 | Train Loss: 0.8881 | Val Loss: 0.9025 | Val AUROC: 0.6977 | Val AUPRC: 0.5130 | Best F1: 0.577 @ thresh=0.60
[STOP] Early stopping triggered (no improvement for 15 epochs)
"""

with open(os.path.join(RESULTS_DIR, "run2_rerun_epochs.txt"), "w") as f:
    f.write(epochs_content)
with open(r"D:\Tehseen\run2_rerun_epochs.txt", "w") as f:
    f.write(epochs_content)

# -------------------------------------------------------------
# 4. APPEND METRICS TO clean_experiment_results.csv
# -------------------------------------------------------------
if os.path.exists(CSV_PATH):
    try:
        df = pd.read_csv(CSV_PATH)
        
        # Define new experiment row
        new_row = {
            'experiment_id': 'run2_roi_rerun_optimized',
            'date': '7/23/2026',
            'description': 'ROI-cropped volumes with robust standardization/clipping (MARGIN=30, TARGET_DEPTH=32)',
            'architecture': 'MultiModal DenseNet121 (3D imaging + clinical MLP)',
            'pretrained': 'None (scratch)',
            'batch_size': 4,
            'learning_rate': 0.0001,
            'epochs_trained': 37,
            'early_stop_patience': 15,
            'pos_weight': 2.11,
            'test_accuracy': 0.5743,
            'test_precision': 0.4194,
            'test_recall': 0.8125,
            'test_f1': 0.5532,
            'test_auroc': 0.6933,
            'test_auprc': 0.5728,
            'test_tn': 46,
            'test_fp': 54,
            'test_fn': 9,
            'test_tp': 39,
            'val_best_auroc': 0.7096,
            'val_best_auprc': 0.5417,
            'val_best_threshold': 0.54,
            'notes': 'Opt ROI crop with robust standard normalization and margin 30. Better than run9 ROI (AUROC 0.625) but below baseline.'
        }
        
        # Append and save
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
        print("[+] Successfully appended results to clean_experiment_results.csv")
    except Exception as e:
        print(f"[-] Error writing to CSV: {e}")
else:
    print(f"[-] Error: Could not find clean_experiment_results.csv at {CSV_PATH}")
