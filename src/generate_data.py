"""Synthetic loan-applicant data generator for credit risk experiments.

Two profiles generate the data: repaying borrowers and a default minority
(~10%). Defaulters are drawn from shifted distributions on observable
features (higher debt-to-income and utilization, more delinquencies,
shorter credit history, larger loans relative to income), so default is
learnable from the features with realistic overlap.

``applicant_group`` is a synthetic, randomly assigned grouping that is
independent of default by construction. It exists only to demonstrate the
fairness audit in ``fairness.py`` -- it is not real demographic data, and
no real borrower attributes are used anywhere here.

All data is synthetic. No real borrower, bureau, or employer data is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PURPOSES = ["home", "auto", "personal", "education", "business"]
HOME_OWNERSHIP = ["rent", "own", "mortgage"]
GROUPS = ["A", "B"]


def make_dataset(
    n: int = 30_000, default_rate: float = 0.10, seed: int = 42
) -> pd.DataFrame:
    """Return a DataFrame of synthetic applicants with a ``default`` label."""
    rng = np.random.default_rng(seed)
    is_default = (rng.random(n) < default_rate).astype(int)
    default = is_default == 1

    # Baseline (repaying) distributions for every row.
    annual_income = rng.lognormal(mean=11.0, sigma=0.6, size=n)
    credit_history_years = np.clip(rng.normal(loc=8.0, scale=5.0, size=n), 0, 40)
    dti = np.clip(rng.normal(loc=0.30, scale=0.15, size=n), 0, 1)
    utilization = np.clip(rng.normal(loc=0.35, scale=0.20, size=n), 0, 1)
    delinquencies = rng.poisson(lam=0.3, size=n)
    loan_amount = rng.lognormal(mean=10.0, sigma=0.8, size=n)
    employment_years = np.clip(rng.normal(loc=6.0, scale=5.0, size=n), 0, 40)
    age = rng.integers(21, 76, size=n)
    purpose = rng.choice(PURPOSES, size=n, p=[0.25, 0.25, 0.20, 0.15, 0.15])
    home_ownership = rng.choice(HOME_OWNERSHIP, size=n, p=[0.35, 0.30, 0.35])
    applicant_group = rng.choice(GROUPS, size=n, p=[0.5, 0.5])

    # Default profile: shift the observable risk distributions.
    m = int(default.sum())
    annual_income[default] = rng.lognormal(mean=10.8, sigma=0.7, size=m)
    credit_history_years[default] = np.clip(
        rng.normal(loc=6.0, scale=4.0, size=m), 0, 40
    )
    dti[default] = np.clip(rng.normal(loc=0.45, scale=0.20, size=m), 0, 1)
    utilization[default] = np.clip(rng.normal(loc=0.50, scale=0.22, size=m), 0, 1)
    delinquencies[default] = rng.poisson(lam=0.9, size=m)
    loan_amount[default] = rng.lognormal(mean=10.15, sigma=0.8, size=m)
    employment_years[default] = np.clip(
        rng.normal(loc=4.5, scale=4.0, size=m), 0, 40
    )

    return pd.DataFrame(
        {
            "annual_income": np.round(annual_income, 2),
            "credit_history_years": np.round(credit_history_years, 1),
            "dti": np.round(dti, 3),
            "utilization": np.round(utilization, 3),
            "delinquencies": delinquencies,
            "loan_amount": np.round(loan_amount, 2),
            "loan_to_income": np.round(loan_amount / annual_income, 3),
            "employment_years": np.round(employment_years, 1),
            "age": age,
            "purpose": purpose,
            "home_ownership": home_ownership,
            "applicant_group": applicant_group,
            "default": is_default,
        }
    )


if __name__ == "__main__":
    df = make_dataset()
    out = "data/applicants.csv"
    import os

    os.makedirs("data", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows to {out} "
          f"({df['default'].mean():.1%} default)")
