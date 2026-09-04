# Automated Audit Suite & Integrity Verification

**Abuse-Ring Sentinel** includes three automated audit checks integrated directly into the training pipeline, plus a submission-file sanity auditor. All results below are from the final locked run at the **12,000-review capacity cap**, computed on the Baseline model (the shipped model) unless otherwise noted.

---

## 1. Audit Check 1: Target Leakage Verification

### Problem
Data leakage occurs when target labels, or features that implicitly encode them, contaminate the feature space — inflating offline metrics in ways that don't hold in production.

### Methodology
- All engineered features are computed using strictly past/historical cumulative state relative to each transaction's timestamp.
- Feature-importance gain is inspected for any target-derived or historical-fraud-rate proxies.

### Result (final run)
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
`STATUS: PASSED` — top feature gains belong to legitimate transaction/entity signals (Vesta engineered V-columns, C-columns, TransactionAmt, and an entity-overlap ratio). Zero historical-fraud-rate features present.

---

## 2. Audit Check 2: Overfitting & Split Divergence

### Problem
Complex feature engineering can overfit to specific entity IDs or time windows seen only in training, degrading performance on genuinely out-of-time data.

### Methodology
Chronological split — Train (70%) / Validation (15%) / Test (15%). Divergence is measured as:

$$\Delta_{\text{recall}} = |\text{Recall}_{\text{validation}} - \text{Recall}_{\text{test}}|$$

Tolerance: ≤ 0.08.

### Result (final run)
- **Validation Recall:** 0.7804
- **Test Recall:** 0.7947
- **Divergence:** 0.0143
`STATUS: PASSED` — well within tolerance.

*(Note: these recall figures are the model's raw/unconstrained recall used for the overfitting check itself, prior to the 12,000-alert capacity truncation applied for the headline results below.)*

---

## 3. Audit Check 3: Capacity Constraint Enforcement (Hard Truncation)

### Problem
Flagging more alerts than a fraud-ops team can review creates operational backlogs, meaning the "recall" a model claims is never actually realized in practice.

### Methodology
Predictions are ranked by probability and **hard-truncated** to exactly `MAX_MANUAL_REVIEWS_CAP = 12,000` — this is a real truncation of the prediction set, not a soft cost penalty. (An earlier version of this pipeline used only a soft penalty, which allowed the alert count to exceed the cap; see Failure Recovery item 4 for the fix.)

### Result — Baseline (SHIPPED model), 88,581 total test transactions

| | Predicted Legitimate | Predicted Fraud (Alert) | Total |
| :--- | :---: | :---: | :---: |
| **Actual Legitimate** | TN = 75,561 (Cost ₹0) | FP = 9,937 (Cost ₹248,425) | 85,498 |
| **Actual Fraud** | FN = 1,020 (Chargeback exposure ₹1,530,000 + amount) | TP = 2,063 (Prevented loss) | 3,083 |
| **Total** | 76,581 | **12,000 (hard cap)** | 88,581 |

- **Precision:** 2,063 / 12,000 = **17.19%**
- **Recall:** 2,063 / 3,083 = **66.92%**
`STATUS: PASSED` — alerts flagged exactly equal 12,000; the branch that would fire on exceeding the cap is unreachable by construction.

### Result — Structural Sentinel (tested variant, not shipped)

| | Predicted Legitimate | Predicted Fraud (Alert) | Total |
| :--- | :---: | :---: | :---: |
| **Actual Legitimate** | TN = 75,584 | FP = 9,914 (Cost ₹247,850) | 85,498 |
| **Actual Fraud** | FN = 997 (Chargeback exposure ₹1,495,500 + amount) | TP = 2,086 | 3,083 |
| **Total** | 76,581 | **12,000 (hard cap)** | 88,581 |

- **Precision:** 2,086 / 12,000 = **17.38%**
- **Recall:** 2,086 / 3,083 = **67.66%**

**Both models comply exactly with the 12,000 cap. The Sentinel scores marginally higher precision/recall but at higher total cost — see README "Model Selection & Negative Results" for why the Baseline is shipped.**

![Baseline Confusion Matrix (Shipped)](results/confusion_matrix_baseline.png)
![Sentinel Confusion Matrix (Tested)](results/confusion_matrix_sentinel.png)
![Precision-Recall Curve](results/pr_curve.png)

---

## 4. Submission File Sanity Audit (`verify_submission.py`)

1. **Schema Check** — `[TransactionID, isFraud]`, exactly 506,691 test rows.
2. **Null Check** — zero missing/non-finite values.
3. **Probability Bounds** — all scores in [0, 1].
4. **Calibration Check** — mean predicted fraud score aligns with realistic financial fraud priors (~1–15%).

---

## 5. On the Capacity Cap Value (12,000)

The cap was locked at **12,000**, not chosen to minimize reported cost. During development, we observed that raising the cap (tested at 15,000 and 20,000) monotonically lowers total cost — this is expected and is exactly the degenerate "flag everything" failure mode a capacity constraint exists to prevent. We fixed 12,000 as the defensible capacity for a conservative fraud-ops team and report results at that fixed value only, rather than tuning it post-hoc. See [FAILURE_LOG.md](FAILURE_LOG.md), item 8.

---

## 6. Failure Recovery Cross-Reference
For full root-cause analysis of all issues surfaced while building and auditing this pipeline, see **[FAILURE_LOG.md](FAILURE_LOG.md)**.
