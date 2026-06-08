"""
modelling_tuning.py
GridSearchCV hyperparameter tuning for Random Forest with full manual MLflow logging.
Logs: best params, all metrics, confusion matrix image, classification report txt,
      feature importance image, and trained model to DagsHub MLflow.
Author: Surya Hanjaya
"""

import os
import json
import tempfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
import dagshub

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
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
mlflow.set_experiment("adult-income-tuning")

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
    return X, y


# ────────────────────────────────────────────────────────────
# Artifact helpers
# ────────────────────────────────────────────────────────────

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
    ax.set_title("Confusion Matrix — Random Forest (Tuned)", fontsize=13, fontweight="bold")
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
        f.write("Classification Report — Random Forest (Tuned)\n")
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
    ax.set_title(f"Top {top_n} Feature Importances — Random Forest (Tuned)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Feature importance plot saved: {path}")
    return path


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────

def main():
    X, y = load_data(DATA_PATH)
    feature_names = list(X.columns)

    # Train/test split — stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # ── Hyperparameter grid ──────────────────────────────────────
    param_grid = {
        "n_estimators"     : [100, 200],
        "max_depth"        : [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf" : [1, 2],
        "max_features"     : ["sqrt"],
    }

    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("[INFO] Starting GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)

    best_model  = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_f1  = grid_search.best_score_

    print(f"[INFO] Best params : {best_params}")
    print(f"[INFO] Best CV F1  : {best_cv_f1:.4f}")

    # ── Evaluate on test set ─────────────────────────────────────
    y_pred = best_model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="binary")
    rec  = recall_score(y_test, y_pred, average="binary")
    f1   = f1_score(y_test, y_pred, average="binary")

    print(f"\n[RESULTS] Test Set Performance:")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  CV F1     : {best_cv_f1:.4f}")

    # ── MLflow Run — fully manual logging ───────────────────────
    with mlflow.start_run(run_name="RandomForest-GridSearchCV-Tuned"):

        # --- Log all best hyperparameters ---
        mlflow.log_param("n_estimators",      best_params["n_estimators"])
        mlflow.log_param("max_depth",         str(best_params["max_depth"]))
        mlflow.log_param("min_samples_split", best_params["min_samples_split"])
        mlflow.log_param("min_samples_leaf",  best_params["min_samples_leaf"])
        mlflow.log_param("max_features",      best_params["max_features"])
        mlflow.log_param("random_state",      42)
        mlflow.log_param("n_jobs",            -1)
        mlflow.log_param("cv_folds",          5)
        mlflow.log_param("cv_scoring",        "f1")
        mlflow.log_param("test_size",         0.2)
        mlflow.log_param("stratify",          True)
        mlflow.log_param("train_rows",        X_train.shape[0])
        mlflow.log_param("test_rows",         X_test.shape[0])
        mlflow.log_param("n_features",        X_train.shape[1])
        mlflow.log_param("param_grid",        json.dumps(param_grid))

        # --- Log metrics ---
        mlflow.log_metric("accuracy",     acc)
        mlflow.log_metric("precision",    prec)
        mlflow.log_metric("recall",       rec)
        mlflow.log_metric("f1_score",     f1)
        mlflow.log_metric("cv_best_f1",   best_cv_f1)

        # --- Log artifacts (in a temp directory) ---
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Confusion matrix image
            cm_path = plot_confusion_matrix(y_test, y_pred, tmp_dir)
            mlflow.log_artifact(cm_path, artifact_path="plots")

            # 2. Classification report txt
            report_path = save_classification_report(y_test, y_pred, tmp_dir)
            mlflow.log_artifact(report_path, artifact_path="reports")

            # 3. Feature importance image
            fi_path = plot_feature_importance(best_model, feature_names, tmp_dir, top_n=20)
            mlflow.log_artifact(fi_path, artifact_path="plots")

        # --- Log model ---
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="random_forest_tuned",
            registered_model_name="AdultIncome-RandomForest-Tuned",
            pip_requirements=[
                "mlflow==2.19.0",
                "scikit-learn",
                "pandas",
                "numpy"
            ]
        )

        print(f"\n[SUCCESS] Tuned run logged to DagsHub MLflow!")
        print(f"  Tracking URI : {mlflow.get_tracking_uri()}")
        print(f"  Run ID       : {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
