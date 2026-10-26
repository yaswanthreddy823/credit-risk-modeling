"""Synthetic loan-applicant data generator for credit risk experiments.

Default is driven by interpretable risk signals (high debt-to-income,
high utilization, delinquencies, short credit history, large loan relative
to income) plus noise. All data is synthetic; no real borrower data is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PURPOSES = ["home", "auto", "personal", "education", "business"]


def make_dataset(
    n: int = 30_000, default_rate: float = 0.10, seed: int = 42
) -> pd.DataFrame:
    """Return a DataFrame of synthetic applicants with a ``default`` label."""
    rng = np.random.default_rng(seed)

    annual_income = rng.lognormal(mean=11.0, sigma=0.6, size=n)  # ~$60k median
    credit_history_years = np.clip(
        rng.normal(loc=8.0, scale=5.0, size=n), 0, 40
    )
    dti = np.clip(rng.normal(loc=0.30, scale=0.15, size=n), 0.0, 0.9)
    employment_years = np.clip(rng.normal(loc=6.0, scale=5.0, size=n), 0, 40)
    loan_amount = rng.lognormal(mean=9.5, sigma=0.8, size=n)  # ~$13k median
    purpose = rng.choice(
        PURPOSES, size=n, p=[0.30, 0.25, 0.20, 0.15, 0.10]
    )
    num_delinquencies = rng.poisson(lam=0.4, size=n)
    utilization = np.clip(rng.beta(a=2.0, b=5.0, size=n), 0.0, 1.0)
    age = rng.integers(21, 70, size=n)

    loan_to_income = loan_amount / annual_income
    logit = (
        -3.0
        + 1.2 * (dti > 0.45)
        + 1.0 * (utilization > 0.7)
        + 0.9 * (num_delinquencies >= 2)
        + 0.6 * (credit_history_years < 3)
        + 0.6 * (loan_to_income > 0.5)
        + 0.3 * (employment_years < 2)
        + rng.normal(0, 1.0, size=n)
    )
    prob = 1 / (1 + np.exp(-logit))
    threshold = np.quantile(prob, 1 - default_rate)
    default = (prob >= threshold).astype(int)

    return pd.DataFrame(
        {
            "annual_income": np.round(annual_income, 0),
            "credit_history_years": np.round(credit_history_years, 1),
            "dti": np.round(dti, 3),
            "employment_years": np.round(employment_years, 1),
            "loan_amount": np.round(loan_amount, 0),
            "loan_purpose": purpose,
            "num_delinquencies": num_delinquencies,
            "utilization": np.round(utilization, 3),
            "age": age,
            "loan_to_income": np.round(loan_to_income, 3),
            "default": default,
        }
    )


if __name__ == "__main__":
    df = make_dataset()
    out = "data/applicants.csv"
    import os

    os.makedirs("data", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows to {out} "
          f"({df['default'].mean():.1%} defaults)")
