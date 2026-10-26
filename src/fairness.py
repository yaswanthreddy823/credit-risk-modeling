"""Fairness audit helpers for lending decisions.

The production credit module aimed for faster, bias-free lending
decisions, so every model run included an approval-parity check: do
applicants in different groups get approved at similar rates at the
operating threshold? These helpers compute that check from predicted
approvals and a grouping column.

The grouping here is synthetic (see ``generate_data.py``); the technique
-- approval rates by group plus the disparate-impact ratio -- is the
standard audit shape used in production model governance.
"""

from __future__ import annotations

import numpy as np


def approval_rate_by_group(approved: np.ndarray,
                           groups: np.ndarray) -> dict[str, float]:
    """Approval rate per group, where ``approved`` is a boolean array."""
    rates = {}
    for group in np.unique(groups):
        mask = groups == group
        rates[str(group)] = float(approved[mask].mean())
    return rates


def disparate_impact_ratio(rates: dict[str, float]) -> float:
    """Ratio of the lowest to the highest group approval rate.

    A ratio of 1.0 means perfect parity. The commonly cited four-fifths
    guideline flags ratios below 0.8 for review -- a flag for
    investigation, not a verdict on its own.
    """
    values = list(rates.values())
    if not values or max(values) == 0:
        return float("nan")
    return min(values) / max(values)
