"""Smoke tests for the credit module: data and fairness helpers.

Run:  python -m pytest tests/ -q
No model training here, so the suite stays fast.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fairness import approval_rate_by_group, disparate_impact_ratio
from generate_data import make_dataset


def test_dataset_shape_and_default_rate():
    df = make_dataset(n=5000, seed=7)
    assert len(df) == 5000
    assert "default" in df.columns
    assert "applicant_group" in df.columns
    assert 0.08 <= df["default"].mean() <= 0.12


def test_group_is_independent_of_default():
    # Groups are assigned randomly, so default rates should be close.
    df = make_dataset(n=20000, seed=3)
    rates = df.groupby("applicant_group")["default"].mean()
    assert abs(rates.iloc[0] - rates.iloc[1]) < 0.02


def test_approval_rates_by_group():
    approved = np.array([True, True, False, True])
    groups = np.array(["A", "A", "B", "B"])
    rates = approval_rate_by_group(approved, groups)
    assert rates == {"A": 1.0, "B": 0.5}


def test_disparate_impact_ratio():
    assert disparate_impact_ratio({"A": 0.5, "B": 0.5}) == 1.0
    assert disparate_impact_ratio({"A": 0.4, "B": 0.5}) == 0.8
