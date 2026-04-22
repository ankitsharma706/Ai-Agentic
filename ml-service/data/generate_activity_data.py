"""
Time-series sample data generator for multi-agent pipeline testing.

Generates a realistic monthly activity CSV with the shape:
    user_id | month | year | txn_count | spend

Usage:
    python data/generate_activity_data.py --users 500 --months 18
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n_users: int = 500, n_months: int = 18, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic monthly activity data.

    Simulates three user archetypes:
        * Active (60%) — consistent high engagement
        * Declining (25%) — activity tapers off toward recent months
        * Dormant (15%) — minimal/no recent activity (churned)
    """
    rng    = np.random.default_rng(seed)
    rows   = []

    # Build ordered month/year sequence
    base_year, base_month = 2023, 1
    periods: list[tuple[int, int]] = []
    y, m = base_year, base_month
    for _ in range(n_months):
        periods.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1

    for uid in range(n_users):
        archetype = rng.choice(["active", "declining", "dormant"],
                               p=[0.60, 0.25, 0.15])

        for i, (yr, mo) in enumerate(periods):
            t = i / max(n_months - 1, 1)   # 0.0 → 1.0 over time

            if archetype == "active":
                txn   = int(rng.integers(5, 20))
                spend = round(float(rng.uniform(200, 2000)), 2)
            elif archetype == "declining":
                # Activity declines over the last third of periods
                decay = max(0.0, 1.0 - (t - 0.6) / 0.4) if t > 0.6 else 1.0
                txn   = int(rng.integers(1, 15) * decay)
                spend = round(float(rng.uniform(50, 1500) * decay), 2)
            else:  # dormant
                # Only active in first half
                active = t < 0.5
                txn   = int(rng.integers(1, 8)) if active else 0
                spend = round(float(rng.uniform(10, 500)), 2) if active else 0.0

            # Skip rows with zero activity to simulate sparse data
            if txn == 0 and rng.random() > 0.2:
                continue

            rows.append({
                "user_id":   f"usr_{uid:05d}",
                "year":      yr,
                "month":     mo,
                "txn_count": txn,
                "spend":     spend,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate monthly activity CSV")
    parser.add_argument("--users",  type=int, default=500)
    parser.add_argument("--months", type=int, default=18)
    parser.add_argument("--output", type=str, default="data/activity.csv")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = generate(n_users=args.users, n_months=args.months)
    df.to_csv(out, index=False)

    print(f"✅ Generated {len(df)} rows for {args.users} users over {args.months} months")
    print(f"   Output → {out}")
    print(f"   Columns: {list(df.columns)}")
