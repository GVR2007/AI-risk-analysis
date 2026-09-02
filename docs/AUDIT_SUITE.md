# Automated Audit Suite & Integrity Verification

To ensure model safety, financial compliance, and operational reliability in real-world payment networks, **Abuse-Ring Sentinel** includes three automated audit checks integrated into the execution pipeline, complemented by a complete [Failure Recovery & Debugging Log](FAILURE_LOG.md).

---

## 1. Audit Check 1: Target Leakage Verification

### Problem
Data leakage occurs when target labels (`isFraud`) from future or concurrent transactions contaminate feature definitions, leading to artificially inflated cross-validation performance that fails in production.

### Methodology
- Features are strictly computed using past historical cumulative state up to transaction timestamp $t_i$.
- Automated gain check evaluates feature importance for any target label proxies.
- Zero target statistics or historical mean target encoding are permitted in the feature space.

### Result
`STATUS: PASSED` — 0% Target leakage detected. Top feature importance gains belong strictly to transaction counts and entity overlap indicators (`C1`, `C4`, `C5`, `TransactionAmt`, `card_degree_centrality`, `component_size`).

---

## 2. Audit Check 2: Overfitting & Split Divergence

### Problem
Models trained on complex graph topologies can overfit to specific entity IDs or historical temporal windows, leading to severe performance degradation on out-of-time test streams.

### Methodology
- The pipeline splits the temporal transaction stream chronologically into:
  - **Train set**: First 70% of stream
  - **Validation set**: Middle 15% of stream
  - **Test set**: Final 15% of stream (88,581 transactions)
- Evaluates recall divergence between validation and test sets:
  $$\Delta_{\text{recall}} = |\text{Recall}_{\text{validation}} - \text{Recall}_{\text{test}}|$$
- Maximum allowable threshold tolerance: $\le 0.08$.

### Result
- **Validation Recall**: `0.6262` (62.62%)
- **Test Recall**: `0.6413` (64.13%)
- **Divergence**: `0.0151`
- `STATUS: PASSED` — Divergence ($0.0151$) is well within safe tolerance.

---

## 3. Audit Check 3: Capacity Constraint Enforcement

### Problem
In high-volume payment processing, issuing alert notifications faster than manual review teams can process creates operational backlogs and uninspected transactions.

### Methodology
- The optimization engine monitors total flagged alert volume against a hard operational capacity budget ($N_{\text{cap}} = 12,000$ reviews).
- Candidate threshold search applies an exponential penalty ($\text{excess\_alerts} \times \text{₹}400.00$) during optimization to guide threshold selection towards compliant regions.
- Post-truncation hard capping enforces that exactly the top 12,000 highest-exposure flagged transactions are assigned to review analysts.

### Result & Visual Artifacts
![Audited Confusion Matrix](results/confusion_matrix.png)

#### Reconciled Post-Truncation Confusion Matrix (88,581 Test Transactions)

| | Predicted Legitimate ($Y=0$) | Predicted Fraud Alert ($Y=1$) | Total |
| :--- | :---: | :---: | :---: |
| **Actual Legitimate ($Y=0$)** | **TN = 75,474** (Cleared Normal)<br>Cost: ₹0.00 | **FP = 10,021** (Flagged Review)<br>Review Labor Cost: ₹250,525.00 | **85,495** |
| **Actual Fraud ($Y=1$)** | **FN = 1,107** (Missed Fraud)<br>Financial Exposure: ₹15.41M | **TP = 1,979** (Prevented Loss)<br>Fraud Prevented: ₹2.97M | **3,086** |
| **Total** | **76,581** | **12,000 (Hard Capacity Cap)** | **88,581** |

- **Pre-Truncation Alerts Flagged**: `15,542`
- **Operational Capacity Cap**: `12,000`
- **Reconciled Audited Precision**: **`16.49%`** ($\frac{\text{TP}}{\text{Total Capacity Budget}} = \frac{1,979}{12,000} = 16.49\%$)
- **Penalty Clarification**: *Because alerts are hard-truncated to the cap before cost evaluation, the exponential penalty term is structurally always zero in the final reported metric; it exists to guide the pre-truncation threshold search.*
- `STATUS: CAPACITY CAPPED & AUDITED` — Hard-truncated to 12,000 max reviews with reconciled 16.49% precision (+62.0% relative gain over baseline).

![Precision Recall Curve](results/pr_curve.png)

---

## 4. Submission File Sanity Audit (`verify_submission.py`)

Executing `verify_submission.py` performs key automated checks on `submission.csv`:

1. **Schema Check**: Confirms standard `[TransactionID, isFraud]` columns and exactly 506,691 test rows.
2. **Null & NaN Check**: Guarantees zero missing or non-finite values.
3. **Probability Bounds**: Verifies all scores lie in $[0.0, 1.0]$ (`Min: 0.0020`, `Max: 0.9783`).
4. **Calibration Check**: Mean predicted fraud score is `8.56%`, aligning with realistic financial fraud priors ($\approx 1-15\%$).

---

## 5. System Debugging & Failure Recovery
For detailed root-cause analysis on how target leakage, capacity overruns, currency misalignments, and metric reconciliation were identified and fixed, see the **[Failure Recovery & Debugging Log](FAILURE_LOG.md)**.
