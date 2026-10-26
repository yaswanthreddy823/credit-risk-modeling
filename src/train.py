"""Train and evaluate credit default models on synthetic applicants.

Models: logistic regression baseline vs class-weighted XGBoost.
Reports ROC-AUC, PR-AUC, KS statistic, and precision/recall across
operating thresholds, since a lender picks a threshold, not just a model.
SHAP explanation runs when ``shap`` is installed, skipped otherwise.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from generate_data import make_dataset

NUMERIC = [
    "annual_income",
    "credit_history_years",
    "dti",
    "employment_years",
    "loan_amount",
    "num_delinquencies",
    "utilization",
    "age",
    "loan_to_income",
]
CATEGORICAL = ["loan_purpose"]


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray,
           y_score: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ks = float(np.max(tpr - fpr))
    print(f"\n===== {name} =====")
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=3))
    print(f"ROC-AUC: {roc_auc_score(y_true, y_score):.4f}")
    print(f"PR-AUC (avg precision): {average_precision_score(y_true, y_score):.4f}")
    print(f"KS statistic: {ks:.4f}")
    print("Threshold analysis (precision / recall):")
    for t in (0.3, 0.5, 0.7):
        pred_t = (y_score >= t).astype(int)
        tp = int(((pred_t == 1) & (y_true == 1)).sum())
        fp = int(((pred_t == 1) & (y_true == 0)).sum())
        fn = int(((pred_t == 0) & (y_true == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        print(f"  threshold {t:.1f}: precision {prec:.3f}, recall {rec:.3f}")


def main() -> None:
    df = make_dataset()
    X = df[NUMERIC + CATEGORICAL]
    y = df["default"].to_numpy()
    print(f"Dataset: {len(df):,} rows, {y.mean():.1%} defaults")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # --- Baseline: logistic regression (scaled numerics) ----------------
    lr_prep = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )
    lr = Pipeline(
        [
            ("prep", lr_prep),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    lr.fit(X_train, y_train)
    report("Logistic Regression", y_test, lr.predict(X_test),
           lr.predict_proba(X_test)[:, 1])

    # --- XGBoost ---------------------------------------------------------
    xgb_prep = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    xgb = Pipeline(
        [
            ("prep", xgb_prep),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    scale_pos_weight=neg / max(pos, 1),
                    eval_metric="logloss",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    xgb.fit(X_train, y_train)
    xgb_score = xgb.predict_proba(X_test)[:, 1]
    report("XGBoost", y_test, xgb.predict(X_test), xgb_score)

    # --- Explainability: SHAP -------------------------------------------
    try:
        import shap

        model = xgb.named_steps["model"]
        prep = xgb.named_steps["prep"]
        X_test_t = prep.transform(X_test)
        feature_names = prep.get_feature_names_out()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_t)
        importance = np.abs(shap_values).mean(axis=0)
        order = np.argsort(importance)[::-1][:10]
        print("\nTop 10 features by mean |SHAP value|:")
        for i in order:
            print(f"  {feature_names[i]:<35} {importance[i]:.4f}")
    except ImportError:
        print("\nSHAP not installed; skipping explanation "
              "(pip install shap to enable it).")
    except Exception as exc:
        print(f"\nSHAP explanation skipped ({exc}).")


if __name__ == "__main__":
    main()
