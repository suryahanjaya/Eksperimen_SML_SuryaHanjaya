"""
automate_SuryaHanjaya.py
Automated preprocessing pipeline for Adult Income Dataset
Author: Surya Hanjaya
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset_raw", "adult.data")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "adult_income_preprocessed.csv")
OUTPUT_SCALER = os.path.join(OUTPUT_DIR, "scaler.pkl")

COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


def load_dataset(path: str) -> pd.DataFrame:
    """Load raw Adult Income dataset from .data file."""
    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_csv(
        path,
        names=COLUMN_NAMES,
        header=None,
        skipinitialspace=True,
    )
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace '?' with NaN then drop rows with missing values."""
    print("[INFO] Handling missing values...")
    initial_count = len(df)

    # Replace '?' with NaN
    df.replace("?", np.nan, inplace=True)

    # Log missing value counts
    missing = df.isnull().sum()
    print(f"[INFO] Missing values per column:\n{missing[missing > 0]}")

    # Drop rows with any missing value
    df.dropna(inplace=True)
    final_count = len(df)
    print(f"[INFO] Rows removed (missing values): {initial_count - final_count}")
    print(f"[INFO] Rows remaining: {final_count}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the dataset."""
    print("[INFO] Removing duplicate rows...")
    initial_count = len(df)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    final_count = len(df)
    print(f"[INFO] Duplicate rows removed: {initial_count - final_count}")
    print(f"[INFO] Rows remaining: {final_count}")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode target column 'income' to binary (0/1)."""
    print("[INFO] Encoding target variable 'income'...")
    # Strip whitespace if any
    df["income"] = df["income"].str.strip()
    df["income"] = df["income"].apply(lambda x: 1 if ">50K" in x else 0)
    print(f"[INFO] Target distribution:\n{df['income'].value_counts()}")
    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Apply one-hot encoding to categorical features."""
    print("[INFO] Applying one-hot encoding to categorical features...")
    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=False)
    print(f"[INFO] Shape after one-hot encoding: {df.shape}")
    return df


def standard_scale(df: pd.DataFrame) -> tuple:
    """
    Apply StandardScaler to numeric features.
    Returns the transformed DataFrame and fitted scaler.
    """
    print("[INFO] Applying StandardScaler to numeric features...")
    scaler = StandardScaler()

    # Only scale numeric features that exist in the DataFrame
    existing_numeric = [col for col in NUMERIC_FEATURES if col in df.columns]
    df[existing_numeric] = scaler.fit_transform(df[existing_numeric])
    print(f"[INFO] Scaled features: {existing_numeric}")
    return df, scaler


def save_outputs(df: pd.DataFrame, scaler: StandardScaler) -> None:
    """Save preprocessed CSV and scaler pickle file."""
    print(f"[INFO] Saving preprocessed dataset to: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"[INFO] Saving scaler to: {OUTPUT_SCALER}")
    joblib.dump(scaler, OUTPUT_SCALER)
    print("[INFO] Outputs saved successfully.")


def main():
    print("=" * 60)
    print("  Adult Income Dataset - Automated Preprocessing Pipeline")
    print("=" * 60)

    # Step 1: Load dataset
    df = load_dataset(DATASET_PATH)

    # Step 2: Handle missing values
    df = handle_missing_values(df)

    # Step 3: Remove duplicates
    df = remove_duplicates(df)

    # Step 4: Encode target variable
    df = encode_target(df)

    # Step 5: One-hot encoding
    df = one_hot_encode(df)

    # Step 6: Standard scaling
    df, scaler = standard_scale(df)

    # Step 7: Save outputs
    save_outputs(df, scaler)

    print("=" * 60)
    print(f"[SUCCESS] Preprocessing completed!")
    print(f"  - Final shape : {df.shape}")
    print(f"  - Output CSV  : {OUTPUT_CSV}")
    print(f"  - Scaler PKL  : {OUTPUT_SCALER}")
    print("=" * 60)


if __name__ == "__main__":
    main()
