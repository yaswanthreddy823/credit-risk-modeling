"""Train and evaluate credit default models on synthetic applicants.

Models: logistic regression baseline vs random forest -- the pair used in
the production credit module. Reports ROC-AUC, PR-AUC, KS statistic, and
the lending decision table across operating thresholds, since a lender
picks a threshold, not just a model. A fairness audit (approval parity
by group) runs at the operating threshold, and SHAP explanation runs when
``shap`` is installed, skipped otherwise.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fairness import approval_rate_by_group, disparate_impact_ratio
from generate_data import make_dataset

NUMERIC = [
    "annual_income",
    "credit_history_years",
    "dti",
    "utilization",
    "delinquencies",
    "loan_amount",
    "loan_to_income",
    "employment_years",
    "age",
]
CATEGORICAL = ["purpose", "home_ownership"]


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def evaluate(name: str, y_true: np.ndarray, y_score: np.ndarray) -> None:
    print(f"\n===== {name} =====")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_score):.4f}")
    print(f"PR-AUC (avg precision): {average_precision_score(y_true, y_score):.4f}")
    print(f"KS statistic: {ks_statistic(y_true, y_score):.4f}")


def main() -> None:
    df = make_dataset()
    X = df[NUMERIC + CATEGORICAL]
    y = df["default"].to_numpy()
    groups = df["applicant_group"].to_numpy()
    print(f"Dataset: {len(df):,} applicants, {y.mean():.1%} default")

    X_train, X_test, y_train, y_test, _, groups_test = train_test_split(
        X, y, groups, test_size=0.25, random_state=42, stratify=y
    )

    cat = ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)
    lr = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        cat,
                        (
                            "num",
                            StandardScaler(),
                            NUMERIC,
                        ),
                    ]
                ),
            ),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    rf = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        cat,
                        ("num", "passthrough", NUMERIC),
                    ]
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    lr_score = lr.predict_proba(X_test)[:, 1]
    rf_score = rf.predict_proba(X_test)[:, 1]
    evaluate("Logistic Regression (baseline)", y_test, lr_score)
    evaluate("Random Forest", y_test, rf_score)

    # --- Lending decision table (random forest, approve if P(default) < t) --
    print("\nOperating thresholds (random forest):")
    print("  t     approve%  bad rate among approved  defaults caught")
    for t in (0.3, 0.5, 0.7):
        approved = rf_score < t
        bad_rate = float(y_test[approved].mean()) if approved.any() else 0.0
        caught = float(((rf_score >= t) & (y_test == 1)).sum() / max((y_test == 1).sum(), 1))
        print(f"  {t:.1f}   {approved.mean():.1%}      {bad_rate:.3f}                  {caught:.3f}")

    # --- Fairness audit at the operating threshold ------------------------
    approved = rf_score < 0.5
    rates = approval_rate_by_group(approved, groups_test)
    print("\nFairness audit at threshold 0.5 (approval rate by group):")
    for group, rate in rates.items():
        print(f"  group {group}: {rate:.3f}")
    di = disparate_impact_ratio(rates)
    print(f"Disparate-impact ratio: {di:.3f} "
          f"({'within' if di >= 0.8 else 'below'} the 0.8 four-fifths guideline)")

    # --- SHAP ----------------------------------------------------------------
    try:
        import shap

        prep = rf.named_steps["prep"]
        model = rf.named_steps["model"]
        X_test_t = prep.transform(X_test[:2000])
        feature_names = prep.get_feature_names_out()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_t)[1]
        importance = np.abs(shap_values).mean(axis=0)
        order = np.argsort(importance)[::-1][:10]
        print("\nTop 10 features by mean |SHAP value|:")
        for i in order:
            print(f"  {feature_names[i]:<35} {importance[i]:.4f}")
    except ImportError:
        print("\nSHAP not installed; skipping explanation.")
    except Exception as exc:
        print(f"\nSHAP explanation skipped ({exc}).")


if __name__ == "__main__":
    main()
