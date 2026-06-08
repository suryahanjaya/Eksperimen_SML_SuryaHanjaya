"""
modelling.py
Training baseline Random Forest model using MLflow autolog().
Author: Surya Hanjaya
"""

import sys
import types
from unittest.mock import MagicMock

# Reconfigure stdout to use UTF-8 to prevent UnicodeEncodeError from MLflow emojis on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Mock torch to prevent DLL loading errors on Windows
dummy_torch = types.ModuleType("torch")
class DummyTensor:
    pass
dummy_torch.Tensor = DummyTensor
sys.modules["torch"] = dummy_torch

# Mock sklearn.frozen to prevent import errors in corrupted sklearn installations
sys.modules['sklearn.frozen'] = MagicMock()
sys.modules['sklearn.frozen._frozen'] = MagicMock()

import os
import tempfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION — Local MLflow
# ============================================================
# Set tracking URI to local MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("adult-income-baseline")

# Enable MLflow Autologging
mlflow.sklearn.autolog()

# ============================================================
# DATA
# ============================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "adult_income_preprocessed.csv")


def load_data(path: str):
    print(f"[INFO] Loading data from: {path}")
    df = pd.read_csv(path)
    X = df.drop(columns=["income"])
    y = df["income"]
    print(f"[INFO] Features: {X.shape[1]} | Samples: {X.shape[0]}")
    print(f"[INFO] Target distribution:\n{y.value_counts()}")
    return X, y


def plot_confusion_matrix(y_true, y_pred, save_dir: str) -> str:
    """Save confusion matrix as PNG and return file path."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["<=50K (0)", ">50K (1)"],
        yticklabels=["<=50K (0)", ">50K (1)"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix — Random Forest (Baseline)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Confusion matrix saved: {path}")
    return path


def save_classification_report(y_true, y_pred, save_dir: str) -> str:
    """Save classification report as .txt and return file path."""
    report = classification_report(
        y_true, y_pred,
        target_names=["<=50K (0)", ">50K (1)"],
        digits=4,
    )
    path = os.path.join(save_dir, "classification_report.txt")
    with open(path, "w") as f:
        f.write("Classification Report — Random Forest (Baseline)\n")
        f.write("=" * 50 + "\n")
        f.write(report)
    print(f"[INFO] Classification report saved: {path}")
    return path


def plot_feature_importance(model, feature_names, save_dir: str, top_n: int = 20) -> str:
    """Save feature importance bar chart as PNG and return file path."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, top_n))
    ax.barh(range(top_n), top_importances[::-1], color=colors[::-1], edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (Mean Decrease Impurity)", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances — Random Forest (Baseline)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Feature importance plot saved: {path}")
    return path


def main():
    X, y = load_data(DATA_PATH)
    feature_names = list(X.columns)

    # Train/test split — stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # ── Hyperparameters ──────────────────────────────────────────
    params = {
        "n_estimators"    : 100,
        "max_depth"       : None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features"    : "sqrt",
        "random_state"    : 42,
        "n_jobs"          : -1,
    }

    # ── MLflow Run ───────────────────────────────────────────────
    with mlflow.start_run(run_name="RandomForest-Baseline"):

        # --- Train ---
        print("[INFO] Training RandomForestClassifier (baseline)...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # --- Predict ---
        y_pred = model.predict(X_test)

        # --- Metrics ---
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="binary")
        rec  = recall_score(y_test, y_pred, average="binary")
        f1   = f1_score(y_test, y_pred, average="binary")

        print(f"[INFO] Accuracy  : {acc:.4f}")
        print(f"[INFO] Precision : {prec:.4f}")
        print(f"[INFO] Recall    : {rec:.4f}")
        print(f"[INFO] F1-Score  : {f1:.4f}")

        # --- Log artifacts (in a temp directory) ---
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Confusion matrix image
            cm_path = plot_confusion_matrix(y_test, y_pred, tmp_dir)
            mlflow.log_artifact(cm_path, artifact_path="plots")

            # 2. Classification report txt
            report_path = save_classification_report(y_test, y_pred, tmp_dir)
            mlflow.log_artifact(report_path, artifact_path="reports")

            # 3. Feature importance image
            fi_path = plot_feature_importance(model, feature_names, tmp_dir, top_n=20)
            mlflow.log_artifact(fi_path, artifact_path="plots")

        # --- Register Model in MLflow Registry ---
        run_id = mlflow.active_run().info.run_id
        print(f"[INFO] Registering model for run {run_id}...")
        mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name="adult-income-baseline"
        )

        print(f"[SUCCESS] Run logged and model registered to local MLflow!")
        print(f"  Tracking URI : {mlflow.get_tracking_uri()}")
        print(f"  Run ID       : {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
