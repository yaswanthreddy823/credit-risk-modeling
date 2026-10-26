# Credit Risk Modeling — FinAI 360 Credit Module

A clean-room demonstration of the credit scoring module from **FinAI 360**,
the intelligent banking automation suite I built at Edward Jones: default
prediction with logistic regression and random forest, lending-threshold
analysis, SHAP explainability, and a fairness audit on approval parity.

> **What this repo is:** a from-scratch re-implementation of the module's
> modeling approach, running on synthetic data. It contains no employer
> code, no borrower data, and no production artifacts. Production outcomes
> below describe the production system; demo results describe this repo's
> synthetic-data runs.

## Production context

In production, advanced credit scoring models (random forest and logistic
regression) improved loan approval accuracy by 18%, enabling faster and
bias-free lending decisions — with explainability and governance checks
around every model version.

## Demo results (synthetic data, measured)

30,000 synthetic applicants, 9.8% default, 75/25 train/test split:

| Model | ROC-AUC | PR-AUC | KS |
| --- | --- | --- | --- |
| Logistic Regression (baseline) | 0.863 | 0.536 | 0.584 |
| Random Forest | 0.869 | 0.542 | 0.598 |

Lending decisions (random forest; approve when P(default) is below the
threshold):

| Threshold | Approval rate | Bad rate among approved | Defaults caught |
| --- | --- | --- | --- |
| 0.3 | 81.0% | 0.035 | 0.709 |
| 0.5 | 91.5% | 0.055 | 0.482 |
| 0.7 | 96.7% | 0.075 | 0.260 |

The table is the decision: tightening the threshold catches more defaults
but turns away more good borrowers. A lender picks the operating point
against risk appetite, not the model with the highest AUC.

Fairness audit at threshold 0.5: approval rates 0.919 (group A) vs 0.912
(group B), disparate-impact ratio **0.992** — within the 0.8 four-fifths
guideline. The grouping is synthetic and independent of default by
construction; the audit demonstrates the governance check, not a claim
about any real population.

Reproduce with `python src/train.py` (SHAP explanation runs when the
`shap` package is installed, and is skipped gracefully otherwise).

## Run it

```bash
pip install -r requirements.txt
python src/train.py
python -m pytest tests/ -q
```

## Project structure

```
src/
  generate_data.py    synthetic applicants: repaying vs default profiles
  train.py            logistic regression vs random forest, thresholds, SHAP
  fairness.py         approval-rate parity + disparate-impact ratio
tests/
  test_smoke.py       data and fairness-helper checks
```

## Tech stack

Python, scikit-learn, pandas, SHAP — demonstrating the production random
forest + logistic regression credit module in runnable form.

## License

MIT
