# =============================================================
# RUN 2 (ROI VERSION): 3D DENSENET121 + CLINICAL FUSION
# - Uses ROI-cropped volumes from Run9
# - Loads metadata from Excel files
# - FIXED: Windows multiprocessing issue
# =============================================================

import os
import warnings
import sys
import subprocess
warnings.filterwarnings("ignore")

import numpy as np
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

# Install / import MONAI
try:
    from monai.networks.nets import DenseNet121
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "monai"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "monai==1.3.0"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from monai.networks.nets import DenseNet121

print("[INIT] Run 2 (ROI Version): 3D DenseNet121 + Clinical Fusion on Cropped Volumes")

# -------------------------------------------------------------
# PATHS
# -------------------------------------------------------------
OUTPUT_DIR = r"D:\Tehseen\results\run9_roi_crop"
METADATA_DIR = os.path.join(OUTPUT_DIR, "metadata")
CROPPED_VOL_DIR = os.path.join(OUTPUT_DIR, "cropped_volumes")

# Excel files from Run9
TRAIN_EXCEL = os.path.join(METADATA_DIR, "03_train_set.xlsx")
VAL_EXCEL = os.path.join(METADATA_DIR, "04_val_set.xlsx")
TEST_EXCEL = os.path.join(METADATA_DIR, "05_test_set.xlsx")

# For saving results
RESULTS_DIR = r"D:\Tehseen\results\run2_roi_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------
# HYPERPARAMETERS
# -------------------------------------------------------------
TARGET_DEPTH = 32
TARGET_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 15
SCHEDULER_PATIENCE = 3
WARMUP_EPOCHS = 3
NUM_WORKERS = 0  # ← FIXED: Windows doesn't like multiprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Device: {device}")

# =============================================================
# DATASET CLASS FOR ROI-CROPPED VOLUMES
# =============================================================
class ROICroppedMRIDataset(Dataset):
    def __init__(self, df, cropped_dir, clinical_cols=None, augment=False, clinical_stats=None):
        self.df = df.reset_index(drop=True)
        self.cropped_dir = cropped_dir
        self.augment = augment
        self.clinical_cols = clinical_cols or ['HR', 'HER2', 'tum_vol', 'age', 'n_z', 'slice_thick']
        
        # Filter to patients with cropped volumes
        self.valid_indices = []
        for idx, row in self.df.iterrows():
            pid = row['pid']
            dataset = row['dataset']
            path = os.path.join(cropped_dir, f"{pid}_{dataset}_cropped.npy")
            if os.path.exists(path):
                self.valid_indices.append(idx)
        
        self.df = self.df.iloc[self.valid_indices].reset_index(drop=True)
        print(f"[Dataset] {len(self.df)} valid patients")
        
        # Clinical stats
        self.clinical_mean = None
        self.clinical_std = None
        if clinical_stats is not None:
            self.clinical_mean, self.clinical_std = clinical_stats
    
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
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row['pid']
        dataset = row['dataset']
        label = int(row['pCR'])
        
        # Load cropped volume: [C, D, H, W]
        path = os.path.join(self.cropped_dir, f"{pid}_{dataset}_cropped.npy")
        try:
            image = np.load(path).astype(np.float32)
        except:
            image = np.zeros((3, TARGET_DEPTH, TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
        
        # Ensure correct shape
        if image.shape != (3, TARGET_DEPTH, TARGET_SIZE, TARGET_SIZE):
            tensor = torch.from_numpy(image).float().unsqueeze(0)
            tensor = F.interpolate(tensor, size=(TARGET_DEPTH, TARGET_SIZE, TARGET_SIZE), 
                                  mode='trilinear', align_corners=False)
            image = tensor.squeeze().numpy()
        
        # Data augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=2).copy()
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=3).copy()
            if np.random.rand() > 0.7:
                image = np.flip(image, axis=1).copy()
            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.9, 1.1)
                beta = np.random.uniform(-0.05, 0.05)
                image = np.clip(image * alpha + beta, 0, 1)
            if np.random.rand() > 0.5:
                noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
                image = np.clip(image + noise, 0, 1)
        # Clinical features
        clinical_raw = self._extract_clinical_vector(row)
        if self.clinical_mean is not None and self.clinical_std is not None:
            clinical = (clinical_raw - self.clinical_mean) / (self.clinical_std + 1e-8)
        else:
            clinical = clinical_raw
        
        return (torch.from_numpy(image).float(),
                torch.from_numpy(clinical).float(),
                torch.tensor(label, dtype=torch.float32),
                pid)

# =============================================================
# MULTI-MODAL MODEL
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
        out = self.fusion(combined)
        return out

# =============================================================
# TRAINING FUNCTIONS
# =============================================================
def get_warmup_lr(epoch, base_lr, warmup_epochs):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return None

def train_one_epoch(epoch, model, optimizer, criterion, train_loader):
    model.train()
    total_loss = 0
    
    if epoch < WARMUP_EPOCHS:
        warmup_lr = get_warmup_lr(epoch, LR, WARMUP_EPOCHS)
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr
        current_lr = warmup_lr
    else:
        current_lr = optimizer.param_groups[0]['lr']
    
    for images, clinical, labels, _ in train_loader:
        images = images.to(device)
        clinical = clinical.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images, clinical)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader), current_lr

def evaluate(loader, model, criterion, device):
    model.eval()
    all_labels, all_probs, total_loss = [], [], 0
    with torch.no_grad():
        for images, clinical, labels, _ in loader:
            images = images.to(device)
            clinical = clinical.to(device)
            labels = labels.to(device).unsqueeze(1)
            outputs = model(images, clinical)
            total_loss += criterion(outputs, labels).item()
            probs = torch.sigmoid(outputs)
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    auroc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    auprc = average_precision_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.1, 0.9, 0.02):
        preds = (all_probs > thresh).astype(int)
        tp = ((preds == 1) & (all_labels == 1)).sum()
        fp = ((preds == 1) & (all_labels == 0)).sum()
        fn = ((preds == 0) & (all_labels == 1)).sum()
        if tp + fp > 0 and tp + fn > 0:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh
    
    return {'loss': total_loss/len(loader), 'auroc': auroc, 'auprc': auprc, 
            'best_thresh': best_thresh, 'best_f1': best_f1}

class EarlyStopping:
    def __init__(self, patience=15):
        self.patience = patience
        self.best_auprc = -1
        self.counter = 0
    def step(self, auprc):
        if auprc > self.best_auprc:
            self.best_auprc = auprc
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# =============================================================
# MAIN EXECUTION - WITH GUARD
# =============================================================
if __name__ == '__main__':
    
    print("\n" + "="*60)
    print("LOADING ROI-CROPPED DATA FROM EXCEL METADATA")
    print("="*60)
    
    # Load Excel files
    train_df = pd.read_excel(TRAIN_EXCEL)
    val_df = pd.read_excel(VAL_EXCEL)
    test_df = pd.read_excel(TEST_EXCEL)
    
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # Clinical columns
    clinical_cols = ['HR', 'HER2', 'tum_vol', 'age', 'n_z', 'slice_thick']
    
    # Compute clinical stats on training set
    clinical_raw_all = []
    for _, row in train_df.iterrows():
        feats = []
        for col in clinical_cols:
            val = row[col]
            if pd.isna(val):
                val = 0.0
            feats.append(float(val))
        clinical_raw_all.append(feats)
    clinical_raw_all = np.array(clinical_raw_all)
    clin_mean = clinical_raw_all.mean(axis=0)
    clin_std = clinical_raw_all.std(axis=0) + 1e-8
    
    print(f"[CLINICAL] Mean: {clin_mean}")
    print(f"[CLINICAL] Std:  {clin_std}")
    
    # Create datasets
    train_dataset = ROICroppedMRIDataset(
        train_df, CROPPED_VOL_DIR, 
        clinical_cols=clinical_cols, augment=True,
        clinical_stats=(clin_mean, clin_std)
    )
    val_dataset = ROICroppedMRIDataset(
        val_df, CROPPED_VOL_DIR,
        clinical_cols=clinical_cols, augment=False,
        clinical_stats=(clin_mean, clin_std)
    )
    test_dataset = ROICroppedMRIDataset(
        test_df, CROPPED_VOL_DIR,
        clinical_cols=clinical_cols, augment=False,
        clinical_stats=(clin_mean, clin_std)
    )
    
    # DataLoaders with num_workers=0 for Windows
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=NUM_WORKERS, pin_memory=True)
    
    # Class weights
    num_neg = (train_dataset.df['pCR'] == 0).sum()
    num_pos = (train_dataset.df['pCR'] == 1).sum()
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)]).to(device)
    print(f"[INIT] pos_weight: {pos_weight.item():.2f}")
    
    # =============================================================
    # CREATE MODEL
    # =============================================================
    print("\n[INIT] Creating MultiModal DenseNet121 (ROI-cropped)...")
    model = MultiModalDenseNet121(clinical_dim=len(clinical_cols), dropout=0.5).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INIT] Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # =============================================================
    # TRAINING SETUP
    # =============================================================
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                           factor=0.5, patience=SCHEDULER_PATIENCE)
    early_stop = EarlyStopping(patience=PATIENCE)
    
    # =============================================================
    # TRAINING LOOP
    # =============================================================
    print("\n" + "="*60)
    print("TRAINING STARTED (ROI-Cropped Volumes + Clinical Fusion)")
    print("="*60)
    
    best_auprc = -1
    best_auroc = -1
    best_state = None
    SAVE_DIR = RESULTS_DIR
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    for epoch in range(EPOCHS):
        train_loss, current_lr = train_one_epoch(epoch, model, optimizer, criterion, train_loader)
        val_res = evaluate(val_loader, model, criterion, device)
        
        scheduler.step(val_res['auprc'])
        actual_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1}/{EPOCHS} | LR: {actual_lr:.6f}")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_res['loss']:.4f}")
        print(f"  Val AUROC: {val_res['auroc']:.4f} | Val AUPRC: {val_res['auprc']:.4f}")
        print(f"  Best F1: {val_res['best_f1']:.3f} @ thresh={val_res['best_thresh']:.2f}")
        
        if val_res['auprc'] > best_auprc:
            best_auprc = val_res['auprc']
            best_auroc = val_res['auroc']
            best_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auprc': val_res['auprc'],
                'auroc': val_res['auroc'],
                'threshold': val_res['best_thresh']
            }
            torch.save(best_state, os.path.join(SAVE_DIR, "best_multimodal_roi.pth"))
            print(f"  ✅ [CHECKPOINT] Best model saved (AUPRC={best_auprc:.4f})")
        
        if early_stop.step(val_res['auprc']):
            print(f"\n[STOP] Early stopping triggered (no improvement for {PATIENCE} epochs)")
            break
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Validation AUROC: {best_auroc:.4f}")
    print(f"Best Validation AUPRC: {best_auprc:.4f}")
    
    # =============================================================
    # TEST EVALUATION
    # =============================================================
    print("\n" + "="*60)
    print("TEST EVALUATION (ROI-Cropped)")
    print("="*60)
    
    model.load_state_dict(best_state['model_state_dict'])
    model.eval()
    
    test_labels, test_probs = [], []
    with torch.no_grad():
        for images, clinical, labels, _ in test_loader:
            images = images.to(device)
            clinical = clinical.to(device)
            outputs = model(images, clinical)
            probs = torch.sigmoid(outputs)
            test_labels.extend(labels.numpy().flatten())
            test_probs.extend(probs.cpu().numpy().flatten())
    
    test_labels = np.array(test_labels)
    test_probs = np.array(test_probs)
    thresh = best_state['threshold']
    test_preds = (test_probs > thresh).astype(int)
    
    test_auroc = roc_auc_score(test_labels, test_probs)
    test_auprc = average_precision_score(test_labels, test_probs)
    prec, rec, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='binary', zero_division=0)
    acc = (test_preds == test_labels).mean()
    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()
    
    print("\n" + "-"*50)
    print("TEST RESULTS (ROI-Cropped + Clinical Fusion)")
    print("-"*50)
    print(f"AUROC:     {test_auroc:.4f}")
    print(f"AUPRC:     {test_auprc:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Threshold: {thresh:.2f}")
    print("\nConfusion Matrix:")
    print(f"            Predicted Negative   Predicted Positive")
    print(f"Actual Neg:        {tn:5d}                {fp:5d}")
    print(f"Actual Pos:        {fn:5d}                {tp:5d}")
    
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=['non-pCR', 'pCR'], zero_division=0))
    
    # =============================================================
    # SAVE RESULTS
    # =============================================================
    results = pd.DataFrame([{
        'experiment': 'run2_roi_cropped',
        'input_type': 'ROI-cropped volumes (3 phases + clinical)',
        'architecture': 'MultiModal DenseNet121',
        'pretrained': 'None (scratch)',
        'test_auroc': test_auroc,
        'test_auprc': test_auprc,
        'test_accuracy': acc,
        'test_precision': prec,
        'test_recall': rec,
        'test_f1': f1,
        'threshold': thresh,
        'best_val_auprc': best_auprc,
        'best_val_auroc': best_auroc,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'patients': len(test_labels)
    }])
    csv_path = os.path.join(SAVE_DIR, "test_results_roi.csv")
    results.to_csv(csv_path, index=False)
    print(f"\n[SAVED] Results to {csv_path}")
    
    # =============================================================
    # COMPARE WITH BASELINE
    # =============================================================
    print("\n" + "="*60)
    print("COMPARISON WITH BASELINE")
    print("="*60)
    
    # Baseline results from original Run2
    baseline = {
        'auroc': 0.7906,
        'auprc': 0.6145,
        'accuracy': 0.7374,
        'precision': 0.5682,
        'recall': 0.7812,
        'f1': 0.6579
    }
    
    print("\n" + "-"*50)
    print("BASELINE (Original Run2) vs ROI-CROPPED")
    print("-"*50)
    print(f"Metric          | Baseline    | ROI-Cropped | Improvement")
    print("-"*50)
    print(f"AUROC           | {baseline['auroc']:.4f}    | {test_auroc:.4f}     | {(test_auroc - baseline['auroc']):+.4f}")
    print(f"AUPRC           | {baseline['auprc']:.4f}    | {test_auprc:.4f}     | {(test_auprc - baseline['auprc']):+.4f}")
    print(f"Accuracy        | {baseline['accuracy']:.4f}    | {acc:.4f}     | {(acc - baseline['accuracy']):+.4f}")
    print(f"Precision       | {baseline['precision']:.4f}    | {prec:.4f}     | {(prec - baseline['precision']):+.4f}")
    print(f"Recall          | {baseline['recall']:.4f}    | {rec:.4f}     | {(rec - baseline['recall']):+.4f}")
    print(f"F1 Score        | {baseline['f1']:.4f}    | {f1:.4f}     | {(f1 - baseline['f1']):+.4f}")
    print("-"*50)
    
    # Save comparison
    comparison_df = pd.DataFrame([
        {'Metric': 'AUROC', 'Baseline': baseline['auroc'], 'ROI_Cropped': test_auroc, 'Improvement': test_auroc - baseline['auroc']},
        {'Metric': 'AUPRC', 'Baseline': baseline['auprc'], 'ROI_Cropped': test_auprc, 'Improvement': test_auprc - baseline['auprc']},
        {'Metric': 'Accuracy', 'Baseline': baseline['accuracy'], 'ROI_Cropped': acc, 'Improvement': acc - baseline['accuracy']},
        {'Metric': 'Precision', 'Baseline': baseline['precision'], 'ROI_Cropped': prec, 'Improvement': prec - baseline['precision']},
        {'Metric': 'Recall', 'Baseline': baseline['recall'], 'ROI_Cropped': rec, 'Improvement': rec - baseline['recall']},
        {'Metric': 'F1 Score', 'Baseline': baseline['f1'], 'ROI_Cropped': f1, 'Improvement': f1 - baseline['f1']},
    ])
    comparison_df.to_csv(os.path.join(SAVE_DIR, "comparison_baseline_vs_roi.csv"), index=False)
    
    print("\n" + "="*60)
    print("✅ RUN 2 (ROI Version) COMPLETE!")
    print("="*60)
    print(f"Results saved to: {SAVE_DIR}")
    print(f"Model saved to: {SAVE_DIR}/best_multimodal_roi.pth")
