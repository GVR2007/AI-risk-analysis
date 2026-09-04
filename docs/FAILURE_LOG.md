# Abuse-Ring Sentinel: Failure Recovery & Debugging Log

Nine distinct issues were identified, diagnosed, and resolved during the development of this pipeline. This log documents each one honestly, including the fact that several fixes **lowered** our reported metrics — because the pre-fix numbers were wrong, not because the model got worse.

---

## 1. Historical Fraud-Rate Target Leakage

**🔴 Issue:** Early feature engineering included historical fraud-rate features (e.g., mean historical fraud per device/card). Cross-validation looked excellent (>0.95 ROC-AUC) — a red flag in itself, since no honest result on this dataset gets close to that at useful recall.

**🛠️ Fix:** Removed all historical target-encoded features. Replaced with cumulative streaming structural features (unique counts, pair counts) computed strictly from data available at each transaction's timestamp.

**🟢 Outcome:** Audit 1 passes with 0% target leakage; top feature gains are legitimate transaction/entity signals.

---

## 2. Graph Centrality Look-Ahead Leakage

**🔴 Issue:** `device_degree_centrality`, `card_degree_centrality`, and `component_size` were computed using `df.groupby(...).transform("nunique"/"count")`, which aggregates across an **entire** entity group — including transactions that occur *after* the current row in time. A transaction on day 1 could "see" devices/cards that only appeared on day 30. This is impossible in real-time production and inflates offline metrics. It was not caught by the target-leakage audit, since it doesn't involve the label directly — only future *structure*.

**🛠️ Fix:** Rebuilt these features to derive strictly from the already time-ordered, cumulative unique-count columns (built via `cumcount()`), with a runtime guard (`RuntimeError`) if the function is called before its dependency.

**🟢 Outcome:** Centrality features are now provably past-only. Metrics changed (generally became more conservative) after this fix, as expected of a genuine leak being closed.

---

## 3. Unfair `ring_boost` Heuristic

**🔴 Issue:** The cost-evaluation function applied a manual post-hoc probability multiplier (`×1.15` or `×0.98`) based on raw structural features, applied **only** when scoring the Sentinel model — never the Baseline. This gave the Sentinel an untrained, arbitrary advantage in every reported comparison, meaning prior "Sentinel beats Baseline" results were partly the model's real skill and partly this manual thumb on the scale.

**🛠️ Fix:** Removed entirely. Both models are now scored purely on their own `predict_proba()` output, with an identical evaluation function.

**🟢 Outcome:** Fair, apples-to-apples comparison restored. This fix directly enabled the honest finding in "Model Selection & Negative Results" (see README) — without it, we could not have trusted any Baseline-vs-Sentinel comparison.

---

## 4. Soft vs. Hard Capacity Enforcement

**🔴 Issue:** The "12,000-alert capacity cap" was originally enforced only as a soft cost penalty (₹400 per excess alert added to the cost function) rather than an actual limit on the number of alerts produced. A real run flagged **18,601 alerts** — 55% over the stated cap — while the pipeline still reported `STATUS: CAPACITY CAPPED (Penalized)` as if this were acceptable. Every metric reported before this fix (precision, recall, cost) was computed on an alert set that silently violated the pipeline's own stated operational constraint.

**🛠️ Fix:** Added `apply_hard_capacity_truncation()` — ranks all transactions by probability and keeps only the top `MAX_MANUAL_REVIEWS_CAP`, regardless of threshold. `audit_capacity()` now treats any deviation from exactly the cap as a failed defensive check.

**🟢 Outcome:** Every subsequent run reports exactly 12,000 alerts — genuinely capacity-compliant, not just penalized-but-over.

---

## 5. Fabricated Precision-Recall Curve

**🔴 Issue:** An early plotting script generated the PR curve using a hand-tuned parametric quadratic formula (`precision = peak * (1 - (recall - center)^2 * width)`), anchored so it passed through exactly one real known data point. Every other point on the curve was fabricated. This directly undermined the curve's purpose: proving the operating threshold wasn't cherry-picked.

**🛠️ Fix:** `generate_audit_plots.py` now computes the PR curve via `sklearn.metrics.precision_recall_curve` on real saved model predictions (`artifacts/*.npy`), and **refuses to run** (raises `FileNotFoundError` with an explicit message) if those real predictions aren't present — eliminating the temptation to fall back to an approximation.

**🟢 Outcome:** The PR curve is now a genuine threshold sweep. It is visibly more jagged than the old fabricated curve — which is expected and correct for real data at this class imbalance, not a defect.

---

## 6. Hardcoded Confusion Matrix Risk

**🔴 Issue:** The confusion matrix plot originally had TN/FP/FN/TP values typed directly into the plotting script. This created a recurring failure mode: every time the underlying pipeline changed, the hardcoded numbers in the plot silently went stale relative to the actual model's real output — we caught this discrepancy multiple times during development.

**🛠️ Fix:** Rewrote `generate_audit_plots.py` to compute the confusion matrix live from real saved predictions using the exact same `apply_hard_capacity_truncation()` logic as the training pipeline. There is no longer a second, hardcoded copy of these numbers anywhere that can drift.

**🟢 Outcome:** Confusion matrix images for both models are regenerated from real data on every run of `generate_audit_plots.py`; they cannot go stale.

---

## 7. Stale Precision Metric Discrepancy

**🔴 Issue:** An intermediate documentation build reported a precision figure that was computed **before** the hard 12,000-alert truncation was applied, then displayed alongside a confusion matrix that **was** post-truncation — an internal inconsistency (12.73% reported vs. 14.16% implied by the shown confusion matrix at the time).

**🛠️ Fix:** Recomputed precision directly from the post-truncation confusion matrix in every subsequent report. Cross-checked TP / (TP+FP) against the printed confusion matrix on every run going forward.

**🟢 Outcome:** All final numbers in this repository are derived directly and consistently from the same confusion matrix — verified by independent recomputation multiple times during finalization.

---

## 8. Capacity Cap Ambiguity and Drift

**🔴 Issue:** During experimentation, `MAX_MANUAL_REVIEWS_CAP` was changed between runs — 12,000 → 15,000 → 20,000 — without being tracked as a deliberate decision. Because a lower-precision, higher-cap model always produces a lower total cost (a mechanical consequence of flagging more transactions, not better fraud detection), this created a real risk of unconsciously "tuning" the cap to produce a flattering cost number — exactly the degenerate "flag everything" failure mode the capacity constraint exists to prevent.

**🛠️ Fix:** Explicitly evaluated the cap as a **business defensibility question** ("what can a real fraud-ops team review?"), independent of which value produced the best cost. Locked the cap at **12,000** and confirmed every reported number in this repository was generated from a single run at that fixed value.

**🟢 Outcome:** All results are reported at one fixed, defensible capacity. We do not report the 15,000 or 20,000-cap numbers as headline results, since doing so alongside the 12,000 numbers without clear labeling would risk exactly the metric-shopping this audit exists to prevent.

---

## 9. Currency Unit Mismatch

**🔴 Issue:** `TransactionAmt` in the IEEE-CIS dataset is denominated in USD. Early cost calculations applied the ₹ symbol directly to raw USD amounts without conversion, materially understating real financial exposure and distorting cost-based threshold optimization.

**🛠️ Fix:** Introduced an explicit `USD_TO_INR = 83.00` conversion applied consistently throughout the cost function, and documented the proxy-dataset rationale (IEEE-CIS as a structural stand-in for Indian BFSI transaction patterns) in the README.

**🟢 Outcome:** All reported ₹ costs are FX-adjusted and internally consistent.

---
## 10. Model/Pipeline Version Mismatch in Production Runner
**🔴 Issue**: run_sentinel_pipeline.py's retrain fallback and MODEL_FILE constant still pointed at the superseded  `ieee_abuse_ring_sentinel_v13.pkl` and `ieee_pipeline_chatgpt_13.py` — the original leaky-centrality pipeline — rather than the actual final v15 artifacts. Had the model file ever gone missing, this would have silently retrained using old, already-fixed bugs instead of the real final pipeline.

**🛠️ Fix**: Updated both references to v15 consistently.

**🟢 Outcome**: Production runner and training pipeline now reference the same model version; no silent fallback to superseded code.

---
## 11. Shipped Model Persistence Mismatch
**🔴 Issue** : run_pipeline() called joblib.dump(model_h, MODEL_FILE) — persisting only the Sentinel (the tested-and-rejected variant) under a single generic filename. The Baseline, the model actually chosen for deployment per "Model Selection & Negative Results," was never saved to disk at all.

**🛠️ Fix** : Both models are now saved under distinct, explicit filenames — ieee_abuse_ring_sentinel_baseline_v15.pkl (shipped) and ieee_abuse_ring_sentinel_sentinel_v15.pkl (tested, not shipped).

**🟢 Outcome** : The persisted artifact set now matches the documented decision — anyone loading the "baseline" file gets the actual shipped model.

---
## Summary

| # | Bug | Effect if uncaught |
| :---: | :--- | :--- |
| 1 | Historical fraud-rate leakage | Metrics inflated to unrealistic >90% range |
| 2 | Graph centrality look-ahead | Metrics inflated via impossible future visibility |
| 3 | Unfair `ring_boost` heuristic | Sentinel falsely appeared to beat Baseline |
| 4 | Soft capacity enforcement | Reported metrics violated the stated 12,000 cap |
| 5 | Fabricated PR curve | Falsely "proved" threshold wasn't cherry-picked |
| 6 | Hardcoded confusion matrix | Silent staleness after pipeline changes |
| 7 | Stale precision metric | Internal inconsistency between text and confusion matrix |
| 8 | Capacity cap drift | Risk of unconsciously gaming the cost metric |
| 9 | Currency unit mismatch | Financial exposure understated by ~83x |
| 10 | Model/pipeline version mismatch in production runner | Silent fallback to superseded, leaky pipeline code |


Every fix in this log made our reported numbers **more conservative**, never more flattering. We consider this log, not the final headline metrics, the strongest evidence of the rigor behind this submission.
