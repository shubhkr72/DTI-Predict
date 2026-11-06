# 💊 DTI-Predict: Drug-Target Interaction Prediction Using Machine Learning

A complete machine learning pipeline for predicting **Drug-Target Interactions (DTIs)** using BindingDB datasets. The project implements feature engineering from molecular structures and protein sequences, handles class imbalance via SMOTE, and benchmarks multiple ML classifiers — all built with **scikit-learn**.

---

## 📖 Table of Contents

- [Setup Guide (Quickstart)](SETUP_README.md)
- [Presentation & Defense Guide](PRESENTATION_README.md)
- [Project Overview](#-project-overview)
- [Datasets](#-datasets)
- [Pipeline Architecture](#-pipeline-architecture)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
  - [Phase 1: Data Acquisition & Preprocessing](#phase-1-data-acquisition--preprocessing)
  - [Phase 2: Feature Extraction](#phase-2-feature-extraction)
  - [Phase 3: Class Imbalance Mitigation](#phase-3-class-imbalance-mitigation)
  - [Phase 4: Model Training & Comparison](#phase-4-model-training--comparison)
  - [Phase 5: Evaluation Metrics](#phase-5-evaluation-metrics)
- [Results](#-results)
- [References & Resources](#-references--resources)
- [License](#-license)

---

## 🧬 Project Overview

Drug-Target Interaction (DTI) prediction is a critical step in computational drug discovery. This project builds a binary classification pipeline that predicts whether a given drug molecule **binds** to a target protein, using:

- **MACCS Key fingerprints** (167 bits) for drug molecules
- **Amino Acid Composition (AAC)** + **Dipeptide Composition (DC)** for protein targets
- **SMOTE** for handling class imbalance
- **Random Forest, Decision Tree, and MLP** classifiers for prediction

---

## 📦 Datasets

The project uses the **BindingDB** datasets obtained via the **[Therapeutics Data Commons (TDC)](https://tdcommons.ai/)** library.

### Primary Benchmark Access

| Dataset | Link | Description |
|---------|------|-------------|
| **BindingDB_Kd** | [TDC DTI Benchmarks](https://tdcommons.ai/multi_pred_tasks/dti/) | Dissociation Constant — direct thermodynamic measure of binding affinity |
| **BindingDB_Ki** | [TDC DTI Benchmarks](https://tdcommons.ai/multi_pred_tasks/dti/) | Inhibition Constant — absolute value for enzyme inhibitors |
| **BindingDB_IC50** | [TDC DTI Benchmarks](https://tdcommons.ai/multi_pred_tasks/dti/) | Half-Maximal Inhibitory Concentration — largest data volume available |

### Raw Database Source

- [BindingDB Download Portal](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp)

### Which Dataset to Use?

| Goal | Recommended Dataset | Reason |
|------|---------------------|--------|
| **Highest predictive accuracy** | `BindingDB_Kd` | Best overall metrics (97.46% accuracy, 99.42% ROC-AUC in reference paper) |
| **Enzyme inhibitor prediction** | `BindingDB_Ki` | Absolute value independent of experimental setup |
| **Largest data volume** | `BindingDB_IC50` | Most available samples, but values are assay-dependent |
| **Model robustness testing** | All three (Kd, Ki, IC50) | Cross-dataset benchmarking as done in the reference paper |

> **Recommendation:** Start with **`BindingDB_Kd`** for best results, then validate across all three datasets.

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA ACQUISITION (TDC)                       │
│         BindingDB_Kd / BindingDB_Ki / BindingDB_IC50            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LABEL BINARIZATION                            │
│         Binding (Y ≤ threshold) → 1 | Non-binding → 0          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌──────────────────────┐   ┌──────────────────────────┐
│   DRUG FEATURES      │   │   TARGET FEATURES        │
│   MACCS Keys (167)   │   │   AAC (20) + DC (400)    │
│   via RDKit          │   │   from protein sequence   │
└──────────┬───────────┘   └──────────┬───────────────┘
           │                          │
           └────────────┬─────────────┘
                        ▼
          ┌──────────────────────────┐
          │  CONCATENATED FEATURE    │
          │  VECTOR (587 dims)       │
          └────────────┬─────────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │  STANDARD SCALING        │
          │  + SMOTE OVERSAMPLING    │
          └────────────┬─────────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │  MODEL TRAINING          │
          │  RFC / DTC / MLP         │
          └────────────┬─────────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │  EVALUATION              │
          │  Acc, Prec, Rec, F1, AUC │
          └──────────────────────────┘
```

---

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **OS**: Windows / Linux / macOS
- **RAM**: 8 GB minimum (16 GB recommended for large datasets)

### Python Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| [PyTDC](https://pypi.org/project/PyTDC/) | latest | Dataset loading from Therapeutics Data Commons |
| [RDKit](https://www.rdkit.org/) | latest | MACCS fingerprint generation from SMILES |
| [scikit-learn](https://scikit-learn.org/) | ≥ 1.0 | ML models, train/test split, scaling, metrics |
| [imbalanced-learn](https://imbalanced-learn.org/) | latest | SMOTE oversampling for class imbalance |
| [pandas](https://pandas.pydata.org/) | latest | Data manipulation and analysis |
| [numpy](https://numpy.org/) | latest | Numerical computation |

---

## ⚙️ Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/DTI-Predict.git
cd DTI-Predict
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install PyTDC rdkit scikit-learn imbalanced-learn pandas numpy
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
```

### Troubleshooting RDKit Installation

If `pip install rdkit` fails, try:

```bash
# Option 1: Install via conda (recommended for RDKit)
conda install -c conda-forge rdkit

# Option 2: Install rdkit-pypi
pip install rdkit-pypi
```

---

## 🚀 Usage

### Run the Full Pipeline

```bash
python train_dti.py
```

This will:
1. Download `BindingDB_Kd` dataset via TDC (automatic, first-run download)
2. Extract MACCS + AAC + DC features
3. Apply SMOTE oversampling
4. Train Random Forest, Decision Tree, and MLP models
5. Print evaluation metrics for all models

### Customize the Dataset

To use a different dataset, modify the dataset name in `train_dti.py`:

```python
# Choose one:
tdc_data = DTI(name='BindingDB_Kd')    # Dissociation Constant (recommended)
tdc_data = DTI(name='BindingDB_Ki')    # Inhibition Constant
tdc_data = DTI(name='BindingDB_IC50')  # Half-Maximal Inhibitory Concentration
```

### Customize the Binding Threshold

```python
# Adjust threshold for label binarization (in nanomolar)
THRESHOLD_NM = 30.0  # Options: 10, 20, or 30 nM
```

---

## 📂 Project Structure

```
DTI-Predict/
│
├── README.md              # This file — project documentation
├── requirements.txt       # Python dependencies
├── train_dti.py           # Main pipeline script
│
└── data/                  # Auto-created by TDC on first run
    └── ...                # Downloaded dataset files
```

---

## 🔬 Methodology

### Phase 1: Data Acquisition & Preprocessing

1. **Download Dataset** via [Therapeutics Data Commons (TDC)](https://tdcommons.ai/multi_pred_tasks/dti/):
   ```python
   from tdc.multi_pred import DTI
   data = DTI(name='BindingDB_Kd')
   df = data.get_data()
   ```

2. **Label Binarization** — Convert continuous binding affinity values (Kd/Ki/IC50) into binary labels:
   - `1` = Binding (affinity ≤ threshold)
   - `0` = Non-binding (affinity > threshold)
   - Threshold options: 10 nM, 20 nM, or 30 nM

### Phase 2: Feature Extraction

| Component | Method | Dimensionality | Library |
|-----------|--------|---------------|---------|
| **Drug (Small Molecule)** | MACCS Keys from SMILES string | 167 bits | [RDKit](https://www.rdkit.org/) |
| **Target (Protein)** — AAC | Normalized frequency of 20 standard amino acids | 20 features | Custom Python |
| **Target (Protein)** — DC | Frequency of all 400 possible amino acid pairs | 400 features | Custom Python |
| **Combined Vector** | Concatenation of MACCS + AAC + DC | **587 features** | NumPy |

Feature vectors are normalized using `StandardScaler` before model training.

### Phase 3: Class Imbalance Mitigation

Non-binding pairs significantly outnumber active drug-target pairs. This is handled using:

- **SMOTE** (Synthetic Minority Over-sampling Technique) via [imbalanced-learn](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- Alternative: **ADASYN** (Adaptive Synthetic Sampling) — available in the same library
- Advanced alternative: **GAN-based oversampling** (as used in the reference paper for best results)

### Phase 4: Model Training & Comparison

Train on **80%** of data, evaluate on **20%** test split (stratified):

| Model | Library Class | Key Parameters |
|-------|--------------|----------------|
| **Random Forest** | `RandomForestClassifier` | `n_estimators=100` |
| **Decision Tree** | `DecisionTreeClassifier` | Default settings |
| **Multi-Layer Perceptron** | `MLPClassifier` | `hidden_layers=(128, 64)`, `max_iter=200` |

### Phase 5: Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | Proportion of true positives among predicted positives |
| **Recall / Sensitivity** | Proportion of actual positives correctly identified |
| **F1-Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Area Under the Receiver Operating Characteristic Curve |

---

## 📊 Results

Expected benchmark results (from reference paper on BindingDB_Kd):

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Random Forest (with GAN balancing) | **97.46%** | **99.42%** |
| Decision Tree | Baseline | Baseline |
| MLP | Competitive | Competitive |

> **Note:** Your results may vary depending on threshold, random seed, and dataset version.

---

## 📚 References & Resources

### Paper
- **Reference Paper:** [Predicting drug-target interactions using machine learning with improved data balancing and feature engineering — *Scientific Reports*](https://www.nature.com/articles/s41598-025-03932-6)
- **Detailed Paper Summary:** See [RESEARCH_PAPER_SUMMARY.md](RESEARCH_PAPER_SUMMARY.md) for a comprehensive breakdown of the methodology, results, and how our implementation relates to the original study.

### Datasets
- **Therapeutics Data Commons (TDC):** [https://tdcommons.ai/multi_pred_tasks/dti/](https://tdcommons.ai/multi_pred_tasks/dti/)
- **BindingDB Raw Download:** [https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp)
- **PyTDC GitHub:** [https://github.com/mims-harvard/TDC](https://github.com/mims-harvard/TDC)

### Libraries & Tools
- **RDKit Documentation:** [https://www.rdkit.org/docs/](https://www.rdkit.org/docs/)
- **scikit-learn Documentation:** [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- **imbalanced-learn Documentation:** [https://imbalanced-learn.org/stable/](https://imbalanced-learn.org/stable/)
- **SMOTE Paper:** Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique" (2002)

---

## 📝 License

This project is for academic purposes (7th Semester Project). Feel free to use and modify for educational use.

---

## 🤝 Acknowledgments

- [Therapeutics Data Commons](https://tdcommons.ai/) for providing standardized benchmark datasets
- [BindingDB](https://www.bindingdb.org/) for the comprehensive drug-target binding data
- The authors of the reference paper for the methodology and pipeline design
