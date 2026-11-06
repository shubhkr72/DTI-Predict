"""
Drug-Target Interaction (DTI) Prediction Pipeline
==================================================
Predicts drug-target binding using BindingDB datasets from Therapeutics Data Commons.
Uses MACCS fingerprints + Amino Acid/Dipeptide Composition features with scikit-learn classifiers.
Evaluates models using standard metrics and saves evaluation reports and publication-quality plots.

Usage:
    python train_dti.py
"""

import os
import sys
import time

# Ensure UTF-8 output encoding across all terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script plotting
import matplotlib.pyplot as plt
import seaborn as sns
from tdc.multi_pred import DTI

# RDKit for MACCS Fingerprints
from rdkit import Chem
from rdkit.Chem import MACCSkeys

# Scikit-learn utilities and models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

# Imbalanced-learn for resampling (SMOTE)
from imblearn.over_sampling import SMOTE


# ---------------------------------------------------------
# Feature Extraction Functions
# ---------------------------------------------------------

# Precomputed amino acids and dipeptide indices for high-performance feature extraction
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DIPEPTIDES = [a1 + a2 for a1 in AMINO_ACIDS for a2 in AMINO_ACIDS]
DP_TO_IDX = {dp: i for i, dp in enumerate(DIPEPTIDES)}


def get_maccs_fingerprint(smiles):
    """Generates 167-bit MACCS Keys structural fingerprint for drugs."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array([int(b) for b in fp.ToBitString()], dtype=np.float32)
    except Exception:
        return None


def get_aac_and_dc(sequence):
    """Calculates Amino Acid Composition (AAC) and Dipeptide Composition (DC).

    AAC: Normalized frequency of 20 standard amino acids (20 features).
    DC:  Frequency of all 400 possible amino acid pairs  (400 features).
    Total: 420 features per protein sequence.
    """
    if not isinstance(sequence, str):
        return None
    seq_len = len(sequence)
    if seq_len < 2:
        return None

    # 1. Amino Acid Composition (20 features)
    aac = np.array([sequence.count(aa) / seq_len for aa in AMINO_ACIDS], dtype=np.float32)

    # 2. Dipeptide Composition (400 features)
    dc = np.zeros(400, dtype=np.float32)
    denom = seq_len - 1
    for i in range(denom):
        dp = sequence[i : i + 2]
        idx = DP_TO_IDX.get(dp)
        if idx is not None:
            dc[idx] += 1
    dc /= denom

    return np.concatenate([aac, dc])


# ---------------------------------------------------------
# Visualization & Report Generation Functions
# ---------------------------------------------------------

def save_visualizations(results, y_test, results_dir="results"):
    """Generates and saves publication-quality evaluation plots."""
    os.makedirs(results_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Metrics Comparison Bar Chart
    metrics_names = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "ROC-AUC", "PR-AUC"]

    plot_data = []
    for model_name, res in results.items():
        for m in metrics_names:
            plot_data.append({
                "Model": model_name,
                "Metric": m,
                "Score": res[m]
            })
    df_metrics = pd.DataFrame(plot_data)

    plt.figure(figsize=(12, 6))
    palette = sns.color_palette("Set2", len(metrics_names))
    ax = sns.barplot(data=df_metrics, x="Model", y="Score", hue="Metric", palette=palette)
    plt.title("Model Performance Comparison on Test Set", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Classifier", fontsize=12, fontweight="bold")
    plt.ylabel("Score (0.0 - 1.0)", fontsize=12, fontweight="bold")
    plt.ylim(0.0, 1.08)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.3f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=8, rotation=90, xytext=(0, 3),
                        textcoords='offset points')

    plt.tight_layout()
    metrics_chart_path = os.path.join(results_dir, "metrics_comparison.png")
    plt.savefig(metrics_chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved performance comparison chart: {metrics_chart_path}")

    # 2. ROC Curves Plot
    plt.figure(figsize=(8, 7))
    for model_name, res in results.items():
        if res["y_prob"] is not None:
            fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
            plt.plot(fpr, tpr, lw=2.5, label=f"{model_name} (AUC = {res['ROC-AUC']:.4f})")

    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=11, frameon=True)
    plt.tight_layout()
    roc_chart_path = os.path.join(results_dir, "roc_curves.png")
    plt.savefig(roc_chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved ROC curves plot: {roc_chart_path}")

    # 3. Precision-Recall Curves Plot
    plt.figure(figsize=(8, 7))
    for model_name, res in results.items():
        if res["y_prob"] is not None:
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, res["y_prob"])
            plt.plot(recall_curve, precision_curve, lw=2.5, label=f"{model_name} (PR-AUC = {res['PR-AUC']:.4f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=12, fontweight="bold")
    plt.ylabel("Precision", fontsize=12, fontweight="bold")
    plt.title("Precision-Recall Curves", fontsize=14, fontweight="bold", pad=15)
    plt.legend(loc="lower left", fontsize=11, frameon=True)
    plt.tight_layout()
    pr_chart_path = os.path.join(results_dir, "precision_recall_curves.png")
    plt.savefig(pr_chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved Precision-Recall curves plot: {pr_chart_path}")

    # 4. Confusion Matrices Heatmap
    num_models = len(results)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    if num_models == 1:
        axes = [axes]

    for ax, (model_name, res) in zip(axes, results.items()):
        cm = res["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    annot_kws={"size": 14, "weight": "bold"},
                    xticklabels=["Non-Binding (0)", "Binding (1)"],
                    yticklabels=["Non-Binding (0)", "Binding (1)"])
        ax.set_title(f"{model_name}\nAcc: {res['Accuracy']:.4f} | F1: {res['F1-Score']:.4f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=11, fontweight="bold")

    plt.tight_layout()
    cm_chart_path = os.path.join(results_dir, "confusion_matrices.png")
    plt.savefig(cm_chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved Confusion Matrices plot: {cm_chart_path}\n")


def save_text_reports(results, dataset_info, results_dir="results"):
    """Saves comprehensive evaluation metrics to plain text and CSV files."""
    os.makedirs(results_dir, exist_ok=True)
    txt_path = os.path.join(results_dir, "evaluation_metrics.txt")
    csv_path = os.path.join(results_dir, "evaluation_metrics.csv")

    # 1. Generate text report
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("      DRUG-TARGET INTERACTION (DTI) PREDICTION EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. DATASET & EXPERIMENTAL SETUP\n")
        f.write("-" * 70 + "\n")
        f.write(f"Dataset Name:         {dataset_info['dataset_name']}\n")
        f.write(f"Binding Threshold:    {dataset_info['threshold_nm']} nM\n")
        f.write(f"Total Raw Pairs:      {dataset_info['total_raw']}\n")
        f.write(f"Valid Processed Pairs:{dataset_info['total_valid']}\n")
        f.write(f"Binding Pairs (1):    {dataset_info['binding_count']} ({dataset_info['binding_pct']:.2f}%)\n")
        f.write(f"Non-Binding Pairs (0):{dataset_info['non_binding_count']} ({dataset_info['non_binding_pct']:.2f}%)\n")
        f.write(f"Feature Dimension:    587 (167 MACCS Keys + 20 AAC + 400 DC)\n")
        f.write(f"Train/Test Split:     80% Train ({dataset_info['train_count']}) / 20% Test ({dataset_info['test_count']})\n")
        f.write(f"Class Balancing:      SMOTE oversampling (Balanced train samples: {dataset_info['resampled_count']})\n\n")

        f.write("2. SUMMARY METRICS COMPARISON TABLE\n")
        f.write("-" * 70 + "\n")
        header = f"{'Model':<25} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'Spec':<8} | {'F1':<8} | {'ROC-AUC':<8} | {'PR-AUC':<8}\n"
        f.write(header)
        f.write("-" * len(header.strip()) + "\n")

        for name, res in results.items():
            row = (f"{name:<25} | {res['Accuracy']:<8.4f} | {res['Precision']:<9.4f} | "
                   f"{res['Recall']:<8.4f} | {res['Specificity']:<8.4f} | {res['F1-Score']:<8.4f} | "
                   f"{res['ROC-AUC']:<8.4f} | {res['PR-AUC']:<8.4f}\n")
            f.write(row)
        f.write("\n")

        f.write("3. DETAILED MODEL EVALUATIONS\n")
        f.write("=" * 70 + "\n\n")

        for name, res in results.items():
            f.write(f"Model: {name}\n")
            f.write("-" * 50 + "\n")
            f.write(f"  Training Time:        {res['train_time']:.2f} seconds\n")
            f.write(f"  Accuracy:             {res['Accuracy']:.4f}\n")
            f.write(f"  Precision:            {res['Precision']:.4f}\n")
            f.write(f"  Recall (Sensitivity): {res['Recall']:.4f}\n")
            f.write(f"  Specificity:          {res['Specificity']:.4f}\n")
            f.write(f"  F1-Score:             {res['F1-Score']:.4f}\n")
            f.write(f"  ROC-AUC:              {res['ROC-AUC']:.4f}\n")
            f.write(f"  PR-AUC:               {res['PR-AUC']:.4f}\n\n")

            f.write("  Confusion Matrix:\n")
            cm = res["confusion_matrix"]
            tn, fp, fn, tp = cm.ravel()
            f.write(f"    [[TN={tn:5d}, FP={fp:5d}],\n")
            f.write(f"     [FN={fn:5d}, TP={tp:5d}]]\n\n")

            f.write("  Classification Report:\n")
            for line in res["classification_report"].split("\n"):
                f.write(f"    {line}\n")
            f.write("\n" + "-" * 50 + "\n\n")

    print(f"  [+] Saved comprehensive text report: {txt_path}")

    # 2. Generate CSV summary table
    summary_rows = []
    for name, res in results.items():
        cm = res["confusion_matrix"]
        tn, fp, fn, tp = cm.ravel()
        summary_rows.append({
            "Model": name,
            "Accuracy": round(res["Accuracy"], 4),
            "Precision": round(res["Precision"], 4),
            "Recall_Sensitivity": round(res["Recall"], 4),
            "Specificity": round(res["Specificity"], 4),
            "F1_Score": round(res["F1-Score"], 4),
            "ROC_AUC": round(res["ROC-AUC"], 4),
            "PR_AUC": round(res["PR-AUC"], 4),
            "True_Negatives": tn,
            "False_Positives": fp,
            "False_Negatives": fn,
            "True_Positives": tp,
            "Training_Time_Sec": round(res["train_time"], 2)
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(csv_path, index=False)
    print(f"  [+] Saved tabular CSV report: {csv_path}")


# ---------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------

def main():
    start_total_time = time.time()

    # =====================================================
    # Step 1: Data Acquisition (BindingDB via TDC)
    # =====================================================
    DATASET_NAME = "BindingDB_Kd"

    print("=" * 65)
    print("      DRUG-TARGET INTERACTION (DTI) PREDICTION PIPELINE")
    print("=" * 65)
    print(f"\n[1/6] Loading {DATASET_NAME} dataset via Therapeutics Data Commons...")
    tdc_data = DTI(name=DATASET_NAME)
    df = tdc_data.get_data()
    total_raw_pairs = len(df)
    print(f"      -> Loaded {total_raw_pairs} drug-target pairs.")

    # =====================================================
    # Step 2: Label Binarization
    # =====================================================
    THRESHOLD_NM = 30.0

    print(f"\n[2/6] Binarizing interaction labels (threshold = {THRESHOLD_NM} nM)...")
    df["Label"] = (df["Y"] <= THRESHOLD_NM).astype(int)
    binding_count = int(df["Label"].sum())
    non_binding_count = int(len(df) - binding_count)
    binding_pct = (binding_count / len(df)) * 100
    non_binding_pct = (non_binding_count / len(df)) * 100
    print(f"      -> Binding (Class 1):     {binding_count:6d} ({binding_pct:.2f}%)")
    print(f"      -> Non-binding (Class 0): {non_binding_count:6d} ({non_binding_pct:.2f}%)")

    # =====================================================
    # Step 3: Feature Extraction (Drugs & Targets)
    # =====================================================
    print("\n[3/6] Extracting features (MACCS Keys + AAC + DC)...")
    X_list, y_list = [], []
    skipped = 0

    for idx, row in df.iterrows():
        drug_fp = get_maccs_fingerprint(row["Drug"])
        protein_feat = get_aac_and_dc(row["Target"])

        if drug_fp is not None and protein_feat is not None:
            combined_feat = np.concatenate([drug_fp, protein_feat])
            X_list.append(combined_feat)
            y_list.append(row["Label"])
        else:
            skipped += 1

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    total_valid_pairs = len(X)
    print(f"      -> Feature matrix shape: {X.shape} (587 dims: 167 MACCS + 20 AAC + 400 DC)")
    print(f"      -> Valid pairs: {total_valid_pairs} | Skipped invalid: {skipped}")

    # =====================================================
    # Step 4: Train/Test Split & Feature Scaling
    # =====================================================
    print("\n[4/6] Splitting train/test data & applying StandardScaler...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      -> Train split: {len(X_train)} samples | Test split: {len(X_test)} samples (80/20)")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =====================================================
    # Step 5: Handling Class Imbalance (SMOTE)
    # =====================================================
    print("\n[5/6] Applying SMOTE oversampling to training partition...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    resampled_count = len(X_train_res)
    print(f"      -> Balanced training samples after SMOTE: {resampled_count}")

    # =====================================================
    # Step 6: Model Training & Comprehensive Evaluation
    # =====================================================
    print("\n[6/6] Benchmarking ML Classifiers...")
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Multi-Layer Perceptron": MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=200, random_state=42
        ),
    }

    results = {}

    print("\n" + "=" * 65)
    print(f"{'Model':<25} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1':<8} | {'ROC-AUC':<8}")
    print("=" * 65)

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train_res, y_train_res)
        train_time = time.time() - t0

        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = (
            model.predict_proba(X_test_scaled)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        # Calculate performance metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
        pr_auc = average_precision_score(y_test, y_prob) if y_prob is not None else 0.0
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        clf_rep = classification_report(y_test, y_pred, target_names=["Non-Binding (0)", "Binding (1)"], digits=4)

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "Specificity": spec,
            "F1-Score": f1,
            "ROC-AUC": auc,
            "PR-AUC": pr_auc,
            "confusion_matrix": cm,
            "classification_report": clf_rep,
            "train_time": train_time,
            "y_prob": y_prob,
            "y_pred": y_pred
        }

        print(f"{name:<25} | {acc:<8.4f} | {prec:<9.4f} | {rec:<8.4f} | {f1:<8.4f} | {auc:<8.4f}")

    print("=" * 65)

    # Save metrics and plots
    print("\nSaving evaluation artifacts to 'results/' directory...")
    dataset_info = {
        "dataset_name": DATASET_NAME,
        "threshold_nm": THRESHOLD_NM,
        "total_raw": total_raw_pairs,
        "total_valid": total_valid_pairs,
        "binding_count": binding_count,
        "binding_pct": binding_pct,
        "non_binding_count": non_binding_count,
        "non_binding_pct": non_binding_pct,
        "train_count": len(X_train),
        "test_count": len(X_test),
        "resampled_count": resampled_count
    }

    save_text_reports(results, dataset_info, results_dir="results")
    save_visualizations(results, y_test, results_dir="results")

    elapsed_total = time.time() - start_total_time
    print(f"\nPipeline executed successfully in {elapsed_total:.2f} seconds.")


if __name__ == "__main__":
    main()
