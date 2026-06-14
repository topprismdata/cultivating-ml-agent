---
name: kaggle-submission-format-by-metric
description: |
  Match the submission file format to the competition's evaluation metric BEFORE
  submitting. For ranking-based metrics (AUC, log_loss, MAP, NDCG, RMSLE) you MUST
  submit continuous probability/score values, not 0/1 class labels or rounded integers.
  Use when: (1) Preparing the final submission for any Kaggle competition, (2) AutoGluon's
  default `predict()` returns class labels for classification (must use `predict_proba()`),
  (3) Unsure whether to threshold predictions, (4) CV score is excellent but LB score
  is near-random. Real failure case: S6E2 Heart Disease rerun 2026-06-14 — OOF AUC
  0.95554 dropped to Public LB 0.88403 with 0/1 submission, recovered to 0.95357 with
  probability submission.
---

# Match Submission Format to Evaluation Metric

## Problem

A perfect model can score near-random on the public leaderboard if the submission file
format doesn't match the metric's expectations. The most common mistake: submitting
**0/1 thresholded labels** for a metric that needs **continuous probabilities**.

**Real failure case (2026-06-14, S6E2 Heart Disease)**:
- Model: AutoGluon best_quality ensemble, OOF AUC **0.95554**
- Submission A: `submission_autogluon.csv` (0/1 thresholded via `predictor.predict()`)
  - Public LB: **0.88403** ❌ (looked like complete model failure)
  - Private LB: 0.88643
- Submission B: `submission_autogluon_proba.csv` (continuous via `predictor.predict_proba()`)
  - Public LB: **0.95357** ✅
  - Private LB: **0.95510**
- Same model, same OOF — only the submission format differed. **0.07 LB drop from thresholding alone**.

## Context / Trigger Conditions

**Use this skill when**:
- About to submit the final (or any) prediction to a Kaggle competition
- AutoGluon / sklearn / xgboost default `predict()` returns hard labels for classification
- Competition metric is ranking-based (see list below)
- CV score is great but LB score is suspiciously low
- You see `sample_submission.csv` with 0.0/1.0 values (those are the *target* format, NOT necessarily the submission format)

**DO NOT threshold for ranking-based metrics**:
- `roc_auc`, `auc` — area under ROC; needs continuous scores
- `auc_mu` — multi-class AUC; needs full probability matrix
- `log_loss` — penalizes confident wrong answers; needs probabilities
- `MAP`, `NDCG` — ranking metrics; need scores
- `brier_score` — squared error on probabilities
- `mean_columnwise_auc` — column-wise AUC
- `rmse` on log-target (RMSLE) — for log-transformed regression, submit log predictions directly

**DO threshold (or round) for these metrics**:
- `accuracy` — predicted class label
- `f1`, `precision`, `recall` — predicted class label
- `quadratic_kappa` — rounded integer (for ordinal)
- `mae`, `rmse` on raw target — continuous regression predictions are fine, but rounding is harmless

## Solution: Metric → Format Decision Tree

```
Is metric ranking-based (AUC, log_loss, MAP, NDCG, etc.)?
├── YES → Submit continuous probabilities / scores
│         AutoGluon: predictor.predict_proba(test)
│         sklearn:   model.predict_proba(test)
│         xgboost:   model.predict_proba(test)
│
└── NO (accuracy, f1, kappa, etc.)
    └── Submit class labels (or rounded regression values)
        AutoGluon: predictor.predict(test)
        sklearn:   model.predict(test)
```

### Quick reference per framework

**AutoGluon**:
```python
# WRONG for AUC:
preds = predictor.predict(test)            # hard labels
# CORRECT for AUC:
preds = predictor.predict_proba(test)     # probability DataFrame
# For multi-class AUC, submit the full matrix; column order matters
# and must match sample_submission.csv column order.
```

**scikit-learn**:
```python
# WRONG for AUC:
preds = model.predict(test)                # hard labels
# CORRECT for AUC:
preds = model.predict_proba(test)[:, 1]    # positive class probability
```

**XGBoost / LightGBM / CatBoost**:
```python
# For binary AUC:
preds = model.predict_proba(test)[:, 1]
# For multi-class AUC (sklearn API):
preds = model.predict_proba(test)          # full matrix
# For multi-class AUC (learning API):
preds = model.predict(test)                # this IS probabilities in raw API
```

### Check sample_submission.csv carefully

```bash
head -3 sample_submission.csv
```

- Values are all `0` and `1` → metric is `accuracy`/`f1` etc. → submit class labels
- Values are `0.0` and `1.0` with target column → **likely a target template** (binary outcome), but metric is probably AUC. Submit probabilities anyway.
- Values are floats in [0, 1] → metric is `log_loss`/probability calibration → submit probabilities
- Values are integers in [0, 9] → `quadratic_kappa`/ordinal → submit integers
- Values are floats with no obvious range → regression, submit raw predictions

**Always cross-check the metric on the competition's Overview page, not the sample file.**

## Verification

**Before submitting**:
- ✅ Read the metric name in the competition's "Evaluation" tab
- ✅ Check whether the metric is in the "DO NOT threshold" list above
- ✅ Use `predict_proba()` (not `predict()`) when in doubt for classification
- ✅ For multi-class AUC: column order in submission = column order in `sample_submission.csv`

**Sanity check after first submission**:
- Public LB score is within 0.01-0.02 of OOF (typical for probability submissions)
- Public LB score is suspiciously low (<0.55 for binary, <1/n_classes for multi-class) → re-check format

## Example: S6E2 Heart Disease, what went wrong

```python
# Training (correct)
predictor = TabularPredictor(label='Heart Disease', eval_metric='roc_auc').fit(
    train_data, presets='best_quality', time_limit=900
)
# Note: predictor automatically uses AUC internally, so OOF is reliable

# First submission attempt (WRONG)
preds = predictor.predict(test)            # returns "Presence" / " Absence" labels
# Thresholded to 0/1
submission = pd.DataFrame({'id': test['id'], 'Heart Disease': (preds == 'Presence').astype(int)})
submission.to_csv('submission.csv', index=False)
# → Public LB 0.88403 (looked like total failure)

# Second attempt (CORRECT)
preds_proba = predictor.predict_proba(test)
# preds_proba columns: ['Absence', 'Presence'] (alphabetical)
submission = pd.DataFrame({'id': test['id'], 'Heart Disease': preds_proba['Presence']})
submission.to_csv('submission_proba.csv', index=False)
# → Public LB 0.95357 (matches OOF)
```

## Notes

**Why this is a sneaky bug**:
- OOF score looks great, code "works" locally
- Local validation passes
- The error only surfaces on the public leaderboard
- A 0/1 submission's AUC is exactly 0.5 (random) when the model is balanced — easy to mistake for a real overfitting problem and start "fixing" the model

**Why this happens more with AutoGluon**:
- `predictor.predict()` defaults to class labels
- Other frameworks (sklearn, xgboost) often default to probabilities depending on the API
- Easy to forget the `.predict_proba()` distinction

**Time investment**:
- Fix: 30 seconds (change `predict` → `predict_proba`, save, resubmit)
- Cost of skipping: 1+ day debugging "why is my model broken"

## Related Skills

- `kaggle-data-format-first` — input data format (2D vs 3D), complementary concern
- `kaggle-competition-best-practices` — overall submission workflow
- `kaggle-optimal-blending` — blend probability outputs across models
- `cv-lb-gap-acknowledgment` — distinguishes submission-format gaps from real CV-LB gaps
- `autogluon-first` — uses `predict_proba()` correctly for AUC

## References

- S6E2 Heart Disease rerun artifacts: `~/projects/s6e2-autogluon-rerun/`
  - `submission_autogluon.csv` (bad, 0/1) → LB 0.88403
  - `submission_autogluon_proba.csv` (good, proba) → LB 0.95357
- [Kaggle Metrics Documentation](https://www.kaggle.com/docs/competitions#metrics)
- [AutoGluon `predict_proba` API](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html)
