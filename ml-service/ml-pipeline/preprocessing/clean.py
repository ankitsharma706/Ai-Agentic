"""
preprocessing/clean.py  —  Raw data cleaning
=============================================
Steps:
    1. Drop duplicate customer IDs
    2. Fix dtypes (dates, booleans, numerics)
    3. Handle missing values (median impute numerics, mode for categoricals)
    4. Remove obvious outliers (IQR fence on MonthlyCharges / TotalCharges)
    5. Encode binary Yes/No columns
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Columns that are binary text → 1/0
_BINARY_MAP = {"Yes": 1, "No": 0}

# Categorical columns that need label encoding later (kept as-is here)
CATEGORICAL_COLS = ["gender", "education", "marital_status",
                    "contract", "payment_method", "paperless_billing"]

# Numerical columns expected in the dataset
NUMERIC_COLS = [
    "age", "annual_income", "tenure", "monthlycharges", "totalcharges",
    "num_services", "customer_satisfaction", "num_complaints",
    "num_service_calls", "late_payments", "avg_monthly_gb",
    "days_since_last_interaction", "credit_score", "dependents",
    "senior_citizen",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a cleaned copy of *df*.

    Args:
        df: Raw DataFrame loaded from the CSV / MongoDB.

    Returns:
        Cleaned DataFrame with consistent dtypes and no nulls.
    """
    df = df.copy()

    print(f"[clean] Raw shape: {df.shape}")

    # ── 1. Deduplicate ────────────────────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(subset=["customer_id"], keep="first", inplace=True)
    print(f"[clean] Dropped {before - len(df)} duplicate customer rows.")

    # ── 2. Parse dates ────────────────────────────────────────────────────────
    if "signup_date" in df.columns:
        df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")

    # ── 3. Coerce numeric columns ─────────────────────────────────────────────
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 4. Impute missing values ──────────────────────────────────────────────
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"[clean] Imputed '{col}' with median={median_val:.2f}")

    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)

    # ── 5. Encode Yes/No binary columns ──────────────────────────────────────
    yes_no_cols = [c for c in df.columns if df[c].dtype == object
                   and set(df[c].dropna().unique()).issubset({"Yes", "No"})]
    for col in yes_no_cols:
        df[col] = df[col].map(_BINARY_MAP).fillna(0).astype(int)

    # ── 6. Remove extreme outliers (IQR ×3) ──────────────────────────────────
    for col in ["monthlycharges", "totalcharges", "annual_income"]:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr     = q3 - q1
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        mask = df[col].between(lower, upper)
        removed = (~mask).sum()
        if removed:
            df = df[mask].copy()
            print(f"[clean] Removed {removed} outliers in '{col}'")

    # ── 7. Ensure churn label is integer ─────────────────────────────────────
    if "churn" in df.columns:
        df["churn"] = pd.to_numeric(df["churn"], errors="coerce").fillna(0).astype(int)

    print(f"[clean] Clean shape: {df.shape}  |  Churn rate: "
          f"{df['churn'].mean()*100:.1f}%" if "churn" in df.columns
          else f"[clean] Clean shape: {df.shape}")

    return df
