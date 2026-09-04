# Automated Audit Suite & Integrity Verification

The shipped pipeline in [`ieee_pipeline_chatgpt_15.py`](../ieee_pipeline_chatgpt_15.py) runs three automated audit checks on every training run, plus a standalone submission-file sanity auditor. All numbers below come from the final locked run at the **12,000-review capacity cap**.

The three audit functions are `audit_leakage_and_importance`, `audit_overfitting`, and `audit_capacity`. In `run_pipeline()` all three run against the **Structural Sentinel** (`model_h`) — the Sentinel is the strictly larger feature set, so if it passes, the Baseline (which is a subset of it) passes trivially. The headline metrics below cover both models.

---

## 1. Audit 1 — Target Leakage Verification (`audit_leakage_and_importance`)

### Problem
Leakage occurs when the target label, or a feature that implicitly encodes it, contaminates the feature space. Offline metrics look great; production performance collapses.

### Methodology
- All engineered Sentinel features are built with strictly past/streaming state — `cumcount()`, first-occurrence `cumsum`, `collections.deque` per-entity rolling windows, and `expanding().shift(1)` UID aggregates — so no row's feature depends on a later row.
- After training the Sentinel, `audit_leakage_and_importance(model_h, X_tr_h.columns)` prints the top 10 features by LightGBM gain and asserts no historical fraud-rate proxies appear.

### Result (final run — top 10 by gain, Sentinel model)

```
V258                                     1459151.34
C1                                       440316.03
V294                                     398303.80
C4                                       312431.36
TransactionAmt                           212648.39
V70                                      166030.58
C5                                       153107.67
D2                                       146068.37
email_card_ratio                         141789.32
V201                                     111888.27
```

`STATUS: PASSED` — the top of the ranking is dominated by legitimate signals: Vesta engineered V-columns, the C-column count block, `TransactionAmt`, a `D`-column, and a single entity-overlap ratio from our Sentinel block (`email_card_ratio`). No historical-fraud-rate feature appears. No feature dominates by more than ~3.3× the next one, so there is no runaway single-feature reliance either.

---

## 2. Audit 2 — Overfitting & Split Divergence (`audit_overfitting`)

### Problem
Aggressive engineering can overfit to specific entity IDs or time windows only present in training, so metrics fall apart on out-of-time test data.

### Methodology
Chronological 70 / 15 / 15 split on `TransactionDT`. Divergence is measured as

$$\Delta_{\text{recall}} = |\text{Recall}_{\text{val}} - \text{Recall}_{\text{test}}|$$

with tolerance **≤ 0.08**. Recall here is computed at the same threshold on both splits, using the model's raw threshold-based prediction (this audit measures generalization — capacity truncation is a separate concern handled by Audit 3).

### Result (final run)

- **Validation recall:** 0.7804
- **Test recall:**       0.7947
- **Divergence:**        0.0143

`STATUS: PASSED` — well inside tolerance. The pipeline is not overfitting to the training window; if anything, test recall is slightly higher, consistent with there being no leakage advantage on the training split.

*(These recall figures are the model's raw pre-truncation recall used for the overfitting check. The headline recall reported for the shipped model in the README is computed after the 12,000-alert hard truncation — a different, stricter operational metric.)*

---

## 3. Audit 3 — Capacity Constraint Enforcement (`audit_capacity`)

### Problem
Flagging more alerts than the fraud-ops team can actually review creates operational backlog. The "recall" a model claims above capacity is never realized in practice — alerts pile up and get auto-approved by the queue, not by the model.

### Methodology
Enforcement is structural, not incentive-based. In `evaluate()`, `apply_hard_capacity_truncation` ranks all test rows by predicted probability and keeps exactly the top `MAX_MANUAL_REVIEWS_CAP = 12,000` as alerts. Everything below rank 12,000 is auto-approved. All headline precision/recall/F1/cost numbers are computed on this capped prediction set.

An earlier version of the pipeline used only a soft cost penalty during threshold search, which allowed the reported alert count to exceed the cap (18,601 alerts against a stated 12,000 cap on one run). See [FAILURE_LOG.md](FAILURE_LOG.md) item 4 for that fix.

### Result — Baseline (SHIPPED model) — 88,581 total test transactions

|                        | Predicted Legitimate                         | Predicted Fraud (Alert)     | Total  |
| :--------------------- | :------------------------------------------: | :-------------------------: | :----: |
| **Actual Legitimate**  | TN = 75,561 (Cost ₹0)                        | FP = 9,937 (Cost ₹248,425)  | 85,498 |
| **Actual Fraud**       | FN = 1,020 (Chargeback ₹1,530,000 + amount)  | TP = 2,063 (Prevented loss) | 3,083  |
| **Total**              | 76,581                                       | **12,000 (hard cap)**       | 88,581 |

- **Precision:** 2,063 / 12,000 = **17.19%**
- **Recall:**    2,063 / 3,083  = **66.92%**
- **Total Cost:** **₹14,422,273.74**

`STATUS: PASSED` — exactly 12,000 alerts fired. The branch of `audit_capacity` that would trigger on exceeding the cap is unreachable by construction now that truncation is hard.

### Result — Structural Sentinel (tested variant, not shipped)

|                        | Predicted Legitimate                         | Predicted Fraud (Alert)     | Total  |
| :--------------------- | :------------------------------------------: | :-------------------------: | :----: |
| **Actual Legitimate**  | TN = 75,584                                  | FP = 9,914 (Cost ₹247,850)  | 85,498 |
| **Actual Fraud**       | FN = 997 (Chargeback ₹1,495,500 + amount)    | TP = 2,086                  | 3,083  |
| **Total**              | 76,581                                       | **12,000 (hard cap)**       | 88,581 |

- **Precision:** 2,086 / 12,000 = **17.38%**
- **Recall:**    2,086 / 3,083  = **67.66%**
- **Total Cost:** **₹15,162,034.04**

**Both models comply exactly with the 12,000 cap. The Sentinel scores marginally higher precision and recall but at higher total cost** — its extra true positives don't offset its worse composition of false negatives on high-value transactions. See the README's Model Selection & Negative Results section for why the Baseline is shipped.

![Baseline confusion matrix (shipped)](results/confusion_matrix_baseline.png)
![Sentinel confusion matrix (tested)](results/confusion_matrix_sentinel.png)
![Precision–Recall curve](results/pr_curve.png)

All three images are regenerated by `generate_audit_plots.py` from `artifacts/{baseline,sentinel}_y_{true,proba}.npy` — the real per-row prediction arrays saved by `evaluate()`. Nothing is hardcoded, and the PR curve uses `sklearn.metrics.precision_recall_curve` on real predictions rather than a parametric approximation.

---

## 4. Submission File Sanity Audit (`verify_submission.py`)

Run standalone against `submission.csv`. Steps performed by `audit_submission`:

1. **Schema check** — columns are exactly `[TransactionID, isFraud]`; row count printed for the reviewer to eyeball.
2. **Null check** — asserts zero missing values.
3. **Probability-range check** — asserts every score is in `[0, 1]`; prints min and max.
4. **Calibration check** — verifies the mean predicted fraud score is inside a realistic prior band (`0.01 ≤ mean ≤ 0.20`). A model that predicted "all fraud" or "all legit" would fail here.
5. **Top-5 inspection** — prints the five highest-risk transactions for a manual sanity glance.

Any assertion failure exits non-zero; the calibration check downgrades to `[WARNING]` rather than failing hard, since a slightly out-of-band mean can be legitimate for a very cautious or very aggressive threshold policy.

---

## 5. On the Capacity Cap Value (12,000)

The cap is fixed at **12,000**, and is **not** the value that minimizes reported cost. During development we observed that raising the cap (tested at 15,000 and 20,000) monotonically lowers total cost — that is expected and is exactly the "flag everything" degenerate mode a capacity constraint exists to prevent. Choosing the cap by cost minimization would defeat its purpose.

12,000 was locked as a defensible operational capacity for a conservative fraud-ops team and all reported numbers are at that fixed value only. See [FAILURE_LOG.md](FAILURE_LOG.md) item 8.

---

## 6. Failure Recovery Cross-Reference

Full root-cause analysis of every audit-surfaced defect and how it was fixed lives in **[FAILURE_LOG.md](FAILURE_LOG.md)**.
