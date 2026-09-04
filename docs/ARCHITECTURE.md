# Abuse-Ring Sentinel — System Architecture

This document describes the actual architecture of the shipped pipeline in [`ieee_pipeline_chatgpt_15.py`](../ieee_pipeline_chatgpt_15.py). Every function name is verified against the source.

---

## 1. What Is Being Built

Two models are trained and compared on the same time-split of the IEEE-CIS Fraud Detection dataset:

1. **Baseline LightGBM (SHIPPED)** — uses `BASELINE_FEATURES_EXTENDED`: 11 raw transaction columns + a curated 73-column Vesta V-subset + 13 D (timedelta) columns.
2. **Structural Sentinel (tested, not shipped)** — uses `HYBRID_FEATURES`: the same baseline set **plus** the graph/velocity/UID feature block described in §3.

Both models are fit with the **same** amount-weighted `sample_w`, so the only difference between them is the graph/entity feature block. This keeps the comparison honest.

The shipped decision is the Baseline. The Sentinel underperformed by ~5% on cost — see the [README's Model Selection section](../README.md#model-selection--negative-results) for why.

---

## 2. Data Flow — Leak-Free Time Splits

```
                      load_data()
                          │
              sort by TransactionDT, reset_index
                          │
        70% train    │    15% validation    │    15% test
       ─────────────┼────────────────────┼───────────────────
                          │
             build_train_features(train_df)      ── history = self
             build_future_features(val_df, train_df)
             build_future_features(test_df, train_df + val_df)
```

- `build_train_features(train_df)` runs the six feature stages directly on the training frame.
- `build_future_features(current_df, history_df)` concatenates history and current rows, tags each row with `__is_current`, runs the streaming/expanding stages on the combined frame, then slices out only the current rows before running the composite stage. This guarantees a validation or test row can only see transactions that happened at earlier `TransactionDT` values — a validation row can see training history; a test row can see training + validation history — but never future rows.

`build_all_features()` does not exist in this file. All external references should use `build_train_features` / `build_future_features`.

---

## 3. Feature Construction Pipeline

Order matters — later stages read columns built by earlier ones. This is the exact order used by both `build_train_features` and `build_future_features`.

### 3.1 `prepare_entities(df)`
Materializes canonical entity string columns used by every downstream stage: `_device`, `_card`, `_address`, `_email`, `_browser`, and `_time` (int64 seconds).

### 3.2 `add_structural_ring_features(df)`
Sorts by `TransactionDT` and produces **strictly time-ordered cumulative counts** — a row's feature value only reflects rows that came before it.

- `device_transaction_count`, `card_transaction_count`, `address_transaction_count` — `cumcount()` per entity.
- `device_card_pair_count`, `device_address_pair_count`, `card_address_pair_count` — `cumcount()` per entity pair.
- Nine "unique" columns (`device_unique_cards`, `device_unique_addresses`, `device_unique_emails`, `device_unique_browsers`, `card_unique_devices`, `card_unique_addresses`, `address_unique_devices`, `address_unique_cards`, `email_unique_cards`) — built from a first-occurrence indicator and `cumsum`, so each row sees the count of unique targets bound to its entity **up to and excluding itself**.

### 3.3 `add_graph_centrality_features(df)`
Depends on §3.2 (raises if the required columns are absent).

- `device_degree_centrality = device_unique_cards + device_unique_addresses`
- `card_degree_centrality = card_unique_devices`
- `component_size` — `cumcount()` per `(device, address, card)` triple.

**Historical note.** A previous version used `groupby(...).transform("nunique")`, which aggregates over the entire group — including future rows relative to the current one. That was a real look-ahead leak. It has been rebuilt to derive centrality from the already time-ordered cumulative columns above, and the leakage audit now checks that no historical fraud-rate proxy appears in the top gain-ranked features.

### 3.4 `add_multiscale_velocity_features(df)`
For each of `_device`, `_card`, `_address`, and the pairs `_device_card` / `_device_address`, counts the number of past transactions in a rolling window. Implemented in `calculate_previous_window_counts` using a `collections.deque` per entity — no pandas rolling, no leakage.

Windows:

| Window | Seconds |
| ------ | ------- |
| `1h`   | 3,600 |
| `24h`  | 86,400 |
| `7d`   | 604,800 |

Column names follow `{entity}_previous_tx_{window}`.

### 3.5 `add_uid_aggregation_features(df)`
Builds a synthetic client identifier `card + address + registration_day` (registration day is `TransactionDT_days − D1`, following the winning IEEE-CIS solution's UID convention) and computes **expanding, shifted-by-one** aggregates of `TransactionAmt` per UID:

- `uid_amt_mean_prev` — mean of past amounts, excluding the current row.
- `uid_amt_std_prev` — std of past amounts, excluding the current row.
- `uid_txn_count_prev` — count of past transactions for this UID.
- `uid_amt_deviation` — current amount minus `uid_amt_mean_prev`.

The `.shift(1).expanding()` construction is what makes this safe — the current row never sees its own value or any future value within its UID group.

### 3.6 `add_composite_ring_features(df)`
Reads columns from all earlier stages and produces the ratios, log-scaled scores, interaction terms, and boolean flags: `device_card_ratio`, `device_address_ratio`, `card_device_ratio`, `email_card_ratio`, `ring_density_score`, `card_velocity_interaction`, `entity_overlap_score`, `pair_link_score`, `ring_velocity_score`, `abuse_ring_score`, `multi_card_device_flag`, `multi_address_device_flag`, `multi_device_card_flag`, `high_velocity_device_flag`, `ring_candidate_flag`.

In `build_future_features` this stage runs **after** the current-row split (its inputs are already per-row).

---

## 4. Feature Sets Passed to the Models

```python
BASELINE_FEATURES              # 11 raw columns
  + VESTA_V_FEATURES           # 73 curated V-columns
  + VESTA_D_FEATURES           # 13 D-columns
  = BASELINE_FEATURES_EXTENDED # what the Baseline sees

BASELINE_FEATURES_EXTENDED
  + TRUE_SENTINEL_FEATURES     # graph, velocity, UID, composite
  = HYBRID_FEATURES            # what the Sentinel sees
```

The V/D columns were added to both sides on purpose. If only the Sentinel got them, any improvement would confound "more raw features" with "graph features are useful."

`make_numeric_matrix` casts every requested feature to numeric, replaces `±inf`, and fills missing values with column medians learned on the training frame (the same medians are reused for validation and test — no re-fitting on future data).

---

## 5. Model & Training Objective

### 5.1 Estimator — `create_regularized_model()`
`LGBMClassifier` with regularization tuned for class-imbalanced, moderate-depth trees:

```
n_estimators      500
learning_rate     0.02
max_depth         6
num_leaves        31
min_child_samples 50
subsample         0.8
colsample_bytree  0.8
scale_pos_weight  3.0
importance_type   'gain'
```

### 5.2 Amount-weighted training — `build_sample_weights()`
Per-row training weights scale with transaction amount so a missed ₹50,000 fraud hurts the loss more than a missed ₹500 one, aligning training with the ₹ cost objective:

```
w = 1 + log1p(amount_inr) / mean(log1p(amount_inr))
```

Weights are derived from `TransactionAmt` only — no label, no leakage. The **same** `sample_w` is passed to both `.fit()` calls, so the comparison stays fair.

---

## 6. Threshold Selection and Cost Function

Thresholds are chosen on the **validation** set (never the test set) by `find_best_threshold_grid_search`, which sweeps a two-region grid (denser between 0.05–0.15, sparser 0.15–0.85) and picks the threshold with the lowest validation cost as computed by `calculate_cost_with_capacity_constraint`:

- FP cost: `count_FP × ₹25`
- FN cost: `sum(amount_INR over FN rows) + count_FN × ₹1,500`
- Soft capacity penalty during grid search: `max(0, alerts − 12,000) × ₹400`

The soft penalty exists to nudge the grid search away from thresholds that would blow the cap. **Reported metrics do not use it** — see §7.

Both models are scored purely on their own `predict_proba()` output. A previous version applied a `ring_boost` multiplier to the Sentinel's probabilities based on raw feature values (only to the Sentinel, never the Baseline); that has been removed because it gave the Sentinel an untrained, arbitrary advantage.

---

## 7. Hard Capacity Enforcement

The manual-review team can process at most 12,000 alerts across the test window. This constraint is enforced in `evaluate()` by `apply_hard_capacity_truncation`:

1. Compute `predict_proba` for every test row.
2. Rank all rows descending by probability.
3. Keep exactly the top `MAX_MANUAL_REVIEWS_CAP = 12,000` as alerts. Everything below rank 12,000 is auto-approved.

All headline metrics (precision, recall, F1, TP/FP/FN/TN, total cost) are computed on this capped prediction set, so the reported alert count can never silently exceed 12,000. There is no soft capacity penalty in the reported cost — the constraint has already been enforced structurally.

---

## 8. Audit Suite

Three audits run every training run. See [`AUDIT_SUITE.md`](AUDIT_SUITE.md) for full details.

- **`audit_leakage_and_importance(model, feature_names)`** — prints the top 10 features by gain and asserts no historical fraud-rate proxies are present.
- **`audit_overfitting(...)`** — compares validation vs. test recall at the chosen threshold and flags a warning if the gap exceeds 0.08.
- **`audit_capacity(res)`** — confirms the reported alert count equals exactly `MAX_MANUAL_REVIEWS_CAP`.

Predictions from `evaluate()` are also saved to `artifacts/{baseline,sentinel}_y_true.npy` and `artifacts/{baseline,sentinel}_y_proba.npy`, which the downstream `generate_audit_plots.py` uses to compute the confusion matrices and PR curve from real data (nothing hardcoded).

---

## 9. Cost & FX Model

Every cost figure in the pipeline is in Indian Rupees. `TransactionAmt` is USD and is converted at a fixed `USD_TO_INR = 83.00`.

```
Total Cost = (FP × ₹25)
           + sum(TransactionAmt_INR over FN rows)
           + (FN × ₹1,500)
```

IEEE-CIS is a US-centric proxy for Indian BFSI transaction patterns. The absolute rupee costs are relative economic projections, not localized forecasts; what's meaningful is the **cost delta between the two models on the same test set**, not the absolute magnitude.

---

## 10. `SentinelGraphExplainer` — Analyst Decision Cards

Defined once, canonically, in [`sentinel_explainer.py`](../sentinel_explainer.py) and imported by [`run_sentinel_pipeline.py`](../run_sentinel_pipeline.py). This is a **template-driven layer over existing model outputs** — it renders human-readable rationale for analyst review. It does not re-score, does not modify the decision, and takes no autonomous action.

```
┌────────────────────────────────────────────────────────┐
│   🚨 ABUSE RING DETECTED                               │
├────────────────────────────────────────────────────────┤
│ Transaction ID: 4020067                                │
│ Risk Score:     97.8/100                               │
│ Accounts Bound: 2                                      │
│ Transactions:   4                                      │
│ Financial Risk: ₹170,067.00                            │
├────────────────────────────────────────────────────────┤
│ 🔗 Graph Telemetry Footprint:                          │
│    • Shared Devices:   1                               │
│    • Shared Subnets:   1                               │
│    • Shared Addresses: 2                               │
│    • Syndicate Type:   High Velocity Syndicate         │
└────────────────────────────────────────────────────────┘
```

`run_sentinel_pipeline.py` first attempts to extract a real entity subgraph from the raw test files; if those are unavailable it falls back to a clearly labeled `[SYNTHETIC DEMO DATA]` card.

---

## 11. End-to-End Pipeline (`run_pipeline()`)

```
load_data()
  │
  ▼ sort by TransactionDT, split 70 / 15 / 15
  │
  ▼ build_train_features(train_df)
  ▼ build_future_features(val_df,  train_df)
  ▼ build_future_features(test_df, train_df + val_df)
  │
  ▼ build_sample_weights(train_amt)                     # amount-weighted
  │
  ├── Baseline    : fit LGBM on BASELINE_FEATURES_EXTENDED
  │                 find_best_threshold_grid_search on validation
  │
  └── Sentinel    : fit LGBM on HYBRID_FEATURES (same sample_w)
                    find_best_threshold_grid_search on validation
  │
  ▼ audit_leakage_and_importance(sentinel, features)
  ▼ audit_overfitting(sentinel, val, test)
  │
  ▼ evaluate(baseline, test) → hard-truncate to 12,000 alerts
  ▼ evaluate(sentinel, test) → hard-truncate to 12,000 alerts
  │     └── save artifacts/{baseline,sentinel}_y_{true,proba}.npy
  │
  ▼ audit_capacity(sentinel)
  ▼ print FINAL COMPARISON (costs, precision, recall, net delta)
  ▼ joblib.dump(sentinel) → ieee_abuse_ring_sentinel_v15.pkl
```

---

## 12. Related Modules

- [`ieee_pipeline_chatgpt_15.py`](../ieee_pipeline_chatgpt_15.py) — the canonical pipeline described in this document.
- [`generate_audit_plots.py`](../generate_audit_plots.py) — regenerates the three result images live from the saved `artifacts/*.npy` prediction arrays.
- [`run_sentinel_pipeline.py`](../run_sentinel_pipeline.py) — single-batch production runner; imports the explainer.
- [`sentinel_explainer.py`](../sentinel_explainer.py) — canonical decision-card class.
- [`verify_submission.py`](../verify_submission.py) — submission integrity auditor.
- [`iterations/`](../iterations/) — archived versions (13, 13(1), 14, ensemble variant) kept for reference; not part of the shipped pipeline.
