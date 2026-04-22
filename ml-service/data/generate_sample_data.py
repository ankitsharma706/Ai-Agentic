"""Sample data generator for local development and testing.

Generates a synthetic users.csv with realistic churn-indicative signals.

Usage:
    python data/generate_sample_data.py --n 5000 --output data/users.csv
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Simulate two cohorts: active users and dormant users
    active_mask = rng.random(n) > 0.3   # ~70% active

    recency   = np.where(active_mask, rng.integers(1, 25, n), rng.integers(20, 180, n))
    txn_7d    = np.where(active_mask, rng.integers(1, 15, n), rng.integers(0, 3, n))
    txn_30d   = txn_7d + rng.integers(0, 20, n)
    txn_90d   = txn_30d + rng.integers(0, 40, n)
    frequency = txn_90d + rng.integers(0, 100, n)
    monetary  = np.where(
        active_mask,
        rng.uniform(200, 5000, n),
        rng.uniform(0, 500, n),
    ).round(2)
    account_age = rng.integers(30, 2000, n)

    plan_tiers = rng.choice(["free", "basic", "pro"], n, p=[0.5, 0.3, 0.2])

    df = pd.DataFrame({
        "user_id":          [f"usr_{i:06d}" for i in range(n)],
        "txn_7d":           txn_7d,
        "txn_30d":          txn_30d,
        "txn_90d":          txn_90d,
        "recency_days":     recency,
        "frequency":        frequency,
        "monetary":         monetary,
        "account_age_days": account_age,
        "plan_tier":        plan_tiers,
        "event_date":       pd.date_range("2024-01-01", periods=n, freq="1h"),
    })

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sample churn dataset")
    parser.add_argument("--n", type=int, default=5000, help="Number of records")
    parser.add_argument("--output", type=str, default="data/users.csv")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate(n=args.n)
    df.to_csv(out_path, index=False)

    print(f"✅ Generated {len(df)} records → {out_path}")
    print(f"   Columns: {list(df.columns)}")
