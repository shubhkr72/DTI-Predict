# 🛠️ DTI-Predict: Environment Setup & Execution Guide

This document provides complete, step-by-step instructions for creating the virtual environment, installing all dependencies, running the Drug-Target Interaction (DTI) prediction pipeline, and viewing the generated evaluation metrics and visualization charts.

---

## 📋 System Prerequisites

- **Python**: Version 3.8 to 3.12 (Python 3.12.x tested and verified)
- **OS**: Windows, macOS, or Linux
- **RAM**: Minimum 8 GB recommended
- **Internet Connection**: Required on the first run to automatically fetch dataset files via Therapeutics Data Commons (TDC).

---

## ⚡ Quick Start (Windows PowerShell)

Open PowerShell in the project root directory (`c:\Users\Shubham\Desktop\DTI-Predict`):

```powershell
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 3. Install tested dependencies
pip install -r requirements.txt

# 4. Run the training & evaluation pipeline
python train_dti.py
```

> **Note on PowerShell Execution Policy**: If you receive a script execution error when activating the venv, run:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\venv\Scripts\Activate.ps1
> ```

---

## 🐧 Quick Start (Linux / macOS / Bash)

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate virtual environment
source venv/bin/activate

# 3. Install tested dependencies
pip install -r requirements.txt

# 4. Run the training & evaluation pipeline
python train_dti.py
```

---

## 📦 Verified Dependency Specifications

The `requirements.txt` is configured and tested for Python 3.12 compatibility:

```text
PyTDC==0.4.1
rdkit
scikit-learn>=1.0
imbalanced-learn
pandas
numpy
matplotlib
seaborn
requests
tqdm
fuzzywuzzy
setuptools<70
```

### Manual Installation (Alternative)

If you prefer installing individual packages manually:

```bash
pip install numpy pandas scikit-learn imbalanced-learn rdkit matplotlib seaborn requests tqdm fuzzywuzzy "setuptools<70" "PyTDC==0.4.1"
```

---

## 🚀 Running the Project

Run the main pipeline:

```bash
python train_dti.py
```

### Pipeline Execution Steps:
1. **Data Acquisition**: Automatically downloads the `BindingDB_Kd` benchmark dataset via Therapeutics Data Commons (TDC).
2. **Label Binarization**: Converts continuous affinity values ($K_d \le 30\text{ nM}$) into active/binding (Class 1) and inactive/non-binding (Class 0).
3. **Feature Engineering**:
   - **Drugs**: Computes 167-bit MACCS Keys fingerprints via RDKit.
   - **Protein Targets**: Computes Amino Acid Composition (AAC: 20 features) and Dipeptide Composition (DC: 400 features).
   - **Concatenated Feature Vector**: $167 + 20 + 400 = 587$ features.
4. **Data Preprocessing**: Stratified 80/20 train/test split followed by `StandardScaler` feature normalization.
5. **Class Imbalance Mitigation**: Generates synthetic training instances using **SMOTE** (Synthetic Minority Over-sampling Technique).
6. **Model Benchmarking**: Trains and evaluates:
   - Random Forest Classifier (`n_estimators=100`)
   - Decision Tree Classifier
   - Multi-Layer Perceptron (`hidden_layer_sizes=(128, 64)`)
7. **Artifact Export**: Automatically saves full evaluation metrics to text, CSV, and high-resolution image files.

---

## 📊 Evaluation Outputs & Artifacts

All evaluation results are automatically created and stored in the `results/` folder:

| File | Type | Description |
|------|------|-------------|
| `results/evaluation_metrics.txt` | 📄 Text Report | Full textual evaluation report containing dataset distribution, summary table, per-model classification reports, confusion matrices, and training times. |
| `results/evaluation_metrics.csv` | 📊 Spreadsheet | Tabular metrics (Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC, TP, FP, TN, FN) for programmatic analysis. |
| `results/metrics_comparison.png` | 🖼️ Visualization | Multi-metric comparison bar chart (Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC) across all evaluated models. |
| `results/roc_curves.png` | 🖼️ Visualization | Receiver Operating Characteristic (ROC) curves with AUC scores for RFC, DTC, and MLP classifiers. |
| `results/precision_recall_curves.png` | 🖼️ Visualization | Precision-Recall (PR) curves showing model precision across all recall levels. |
| `results/confusion_matrices.png` | 🖼️ Visualization | Side-by-side annotated confusion matrix heatmaps showing true/false positive and negative distributions. |

---

## ⚙️ Customization & Hyperparameter Tuning

Inside `train_dti.py`, you can easily customize settings:

### 1. Change Benchmark Dataset
```python
# Select between Kd, Ki, or IC50 subsets:
DATASET_NAME = "BindingDB_Kd"    # Dissociation Constant (Recommended, 52k pairs)
# DATASET_NAME = "BindingDB_Ki"    # Inhibition Constant (375k pairs)
# DATASET_NAME = "BindingDB_IC50"  # Half-Maximal Inhibitory Concentration (991k pairs)
```

### 2. Adjust Binding Threshold Cutoff
```python
# Threshold in nanomolar (nM)
THRESHOLD_NM = 30.0  # Options: 10.0, 20.0, 30.0 nM
```

### 3. Add or Modify ML Models
```python
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Multi-Layer Perceptron": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42),
}
```

---

## 🔧 Troubleshooting Guide

| Issue | Cause | Solution |
|---|---|---|
| `ModuleNotFoundError: No module named 'rdkit'` | RDKit not installed in current environment | Run `pip install rdkit` |
| `ModuleNotFoundError: No module named 'pkg_resources'` | Newer `setuptools` (>=70) removed `pkg_resources` | Run `pip install "setuptools<70"` |
| `ModuleNotFoundError: No module named 'tiledbsoma'` | PyTDC >=1.1 has genomic single-cell dependencies | Install tested `PyTDC==0.4.1` with `pip install --no-deps PyTDC==0.4.1` |
| `Permission denied / script execution disabled` | Windows PowerShell execution restriction | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |

---

## 🎓 Citation & Research Attribution

This pipeline implements the methodology described in:
> Talukder, M.A., Kazi, M. & Alazab, A. *Predicting drug-target interactions using machine learning with improved data balancing and feature engineering.* **Scientific Reports 15**, 18862 (2025). https://doi.org/10.1038/s41598-025-03932-6
