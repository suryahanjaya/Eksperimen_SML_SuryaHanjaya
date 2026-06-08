"""
modelling.py
Training baseline Random Forest model with manual MLflow logging to DagsHub.
Author: Surya Hanjaya
"""

import sys
import types
from unittest.mock import MagicMock

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
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
import dagshub

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
# CONFIGURATION — DagsHub + MLflow
# ============================================================
DAGSHUB_USERNAME = "suryahanjaya"
DAGSHUB_REPO     = "adult-income-classifier"

dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)

mlflow.set_tracking_uri(
    f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
)
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


def main():
    X, y = load_data(DATA_PATH)

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

        # ── MLflow logging (autolog is active, so we log test metrics and metadata manually) ──
        mlflow.log_param("test_size",  0.2)
        mlflow.log_param("stratify",   True)
        mlflow.log_param("train_rows", X_train.shape[0])
        mlflow.log_param("test_rows",  X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])

        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall",    rec)
        mlflow.log_metric("f1_score",  f1)

        print(f"[SUCCESS] Run logged to DagsHub MLflow!")
        print(f"  Tracking URI : {mlflow.get_tracking_uri()}")
        print(f"  Run ID       : {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
