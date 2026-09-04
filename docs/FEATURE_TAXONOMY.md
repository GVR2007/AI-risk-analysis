# Sentinel Feature Engineering Taxonomy

This document details every feature used by the shipped pipeline in [`ieee_pipeline_chatgpt_15.py`](../ieee_pipeline_chatgpt_15.py). All feature names are verified against `BASELINE_FEATURES`, `VESTA_V_FEATURES`, `VESTA_D_FEATURES`, and `TRUE_SENTINEL_FEATURES` in the source.

---

## 0. Two Feature Sets

The pipeline trains and compares two models on the same time-split. Both use the same amount-weighted `sample_w` at fit time — the only difference between them is the graph/entity feature block.

| Set                          | Model                             | Column groups                                                 | Column count |
| ---------------------------- | --------------------------------- | ------------------------------------------------------------- | -----------: |
| `BASELINE_FEATURES_EXTENDED` | Baseline LightGBM (SHIPPED)       | Raw (11) + Vesta V (73) + D (13)                              | **97**       |
| `HYBRID_FEATURES`            | Structural Sentinel (tested)      | Raw (11) + Vesta V (73) + D (13) + Sentinel graph block (48)  | **145**      |

The V/D columns were deliberately added to **both** sets. If only the Sentinel received them, any improvement would confound "more raw features" with "graph features help." Keeping the raw-column feature set identical on both sides is what makes the Baseline-vs-Sentinel comparison honest.

---

## 1. Feature Categories Overview (Sentinel block, 48 features)

| Category                                | Count | Description                                                                          |
| :-------------------------------------- | :---: | :----------------------------------------------------------------------------------- |
| **Structural counts & uniques**         | 15    | Cumulative transaction counts and unique-target counts per entity and entity pair.   |
| **Graph centrality**                    | 3     | Node degree and subgraph-component density across device/card/address links.         |
| **Multiscale velocity**                 | 11    | Sliding time-window counts (1h, 24h, 7d) across single entities and entity pairs.    |
| **UID client-aggregation** *(new)*      | 4     | Time-safe expanding aggregates over a synthetic client id (card + addr + reg-day).   |
| **Composite & interaction metrics**     | 10    | Cross-domain risk interaction scores, density ratios, and log-scaled ring metrics.   |
| **Syndicate rule flags**                | 5     | Binary indicators fired when structural abuse conditions are met.                    |
| **Total Sentinel block**                | **48** | Only these features distinguish the Sentinel from the Baseline.                     |

---

## 2. Complete Sentinel Feature Specification (48)

### A. Structural counts & uniques (15) — `add_structural_ring_features`
Time-ordered cumulative counts built via `cumcount()` and first-occurrence `cumsum` tricks. Each row sees only what came before it, never after.

1. `device_transaction_count` — cumulative transaction count for the device up to (but not including) the current row.
2. `card_transaction_count` — cumulative transaction count for the payment card.
3. `address_transaction_count` — cumulative transaction count for the billing address.
4. `device_unique_cards` — distinct cards seen on this device so far.
5. `device_unique_addresses` — distinct billing addresses seen on this device so far.
6. `device_unique_emails` — distinct email domains seen on this device so far.
7. `device_unique_browsers` — distinct browser fingerprints seen on this device so far.
8. `card_unique_devices` — distinct devices that have used this card so far.
9. `card_unique_addresses` — distinct billing addresses seen with this card so far.
10. `address_unique_devices` — distinct devices seen at this address so far.
11. `address_unique_cards` — distinct cards seen at this address so far.
12. `email_unique_cards` — distinct cards linked to this email domain so far.
13. `device_card_pair_count` — cumulative transaction count for the `(device, card)` pair.
14. `device_address_pair_count` — cumulative transaction count for the `(device, address)` pair.
15. `card_address_pair_count` — cumulative transaction count for the `(card, address)` pair.

### B. Graph centrality (3) — `add_graph_centrality_features`
Derived from the already-time-ordered §A columns. A previous version used `groupby(...).transform("nunique")` which secretly aggregated future rows — this has been rebuilt to eliminate that look-ahead leak.

16. `device_degree_centrality` — `device_unique_cards + device_unique_addresses`.
17. `card_degree_centrality` — `card_unique_devices`.
18. `component_size` — `cumcount()` per `(device, address, card)` triple; how many prior transactions this exact triple has seen.

### C. Multiscale velocity (11) — `add_multiscale_velocity_features`
Rolling-window counts of past transactions per entity or entity pair. Implemented in `calculate_previous_window_counts` with a `collections.deque` per entity — no pandas rolling, no leakage.

Windows: `1h = 3,600s`, `24h = 86,400s`, `7d = 604,800s`.

19. `device_previous_tx_1h`
20. `device_previous_tx_24h`
21. `device_previous_tx_7d`
22. `card_previous_tx_1h`
23. `card_previous_tx_24h`
24. `card_previous_tx_7d`
25. `address_previous_tx_1h`
26. `address_previous_tx_24h`
27. `address_previous_tx_7d`
28. `device_card_previous_tx_24h` — pair-level velocity for `(device, card)`.
29. `device_address_previous_tx_24h` — pair-level velocity for `(device, address)`.

### D. UID client-aggregation (4) — `add_uid_aggregation_features`  *(new)*
Modeled on the winning IEEE-CIS solution's synthetic-client convention. A client identifier is `_card + "_" + _address + "_" + _reg_day`, where `_reg_day = round(TransactionDT/86400 - D1)`. Aggregates are `expanding()` and `shift(1)`-ed within each UID, so the current row never sees its own value or any future value in its group.

30. `uid_amt_mean_prev` — mean `TransactionAmt` of past rows in this UID.
31. `uid_amt_std_prev` — std of past `TransactionAmt` in this UID.
32. `uid_txn_count_prev` — number of prior transactions in this UID (`cumcount()`).
33. `uid_amt_deviation` — current amount minus `uid_amt_mean_prev` (anomaly signal against the client's own history).

### E. Composite & interaction metrics (10) — `add_composite_ring_features`
Ratios, log-scaled scores, and interactions built from §A–§C. In `build_future_features` this stage runs **after** the current-row split — its inputs are already per-row.

34. `device_card_ratio` = `device_unique_cards / max(device_transaction_count, 1)`.
35. `device_address_ratio` = `device_unique_addresses / max(device_transaction_count, 1)`.
36. `card_device_ratio` = `card_unique_devices / max(card_transaction_count, 1)`.
37. `email_card_ratio` = `email_unique_cards / max(cumcount(email), 1)`.
38. `ring_density_score` = `device_card_ratio × device_previous_tx_24h`.
39. `card_velocity_interaction` = `card_unique_devices × card_previous_tx_24h`.
40. `entity_overlap_score` = `ln(1 + device_unique_cards) + ln(1 + device_unique_addresses)`.
41. `pair_link_score` = `ln(1 + device_card_pair_count) + ln(1 + device_address_pair_count)`.
42. `ring_velocity_score` = `ln(1 + device_previous_tx_24h) + ln(1 + card_previous_tx_24h)`.
43. `abuse_ring_score` = `0.35 × entity_overlap_score + 0.35 × pair_link_score + 0.30 × ln(1 + component_size)`.

### F. Syndicate rule flags (5) — `add_composite_ring_features`
Interpretable binary indicators. Useful for explanations and analyst-facing cards; the model does not rely on them alone.

44. `multi_card_device_flag` — `device_unique_cards ≥ 3`.
45. `multi_address_device_flag` — `device_unique_addresses ≥ 2`.
46. `multi_device_card_flag` — `card_unique_devices ≥ 2`.
47. `high_velocity_device_flag` — `device_previous_tx_24h ≥ 3`.
48. `ring_candidate_flag` — `device_unique_cards ≥ 3` AND `device_previous_tx_24h ≥ 2`.

---

## 3. Baseline Raw Feature Set (`BASELINE_FEATURES`, 11)

Given to both models as the minimal per-transaction signal:

`TransactionAmt`, `card2`, `card3`, `card5`, `addr2`, `dist1`, `C1`, `C2`, `C3`, `C4`, `C5`.

---

## 4. Vesta V-Columns (`VESTA_V_FEATURES`, 73)

A curated subset of Vesta's own 339-column `V*` engineered feature block. Using the full V1–V339 introduces heavy redundancy (many are near-duplicates); this 73-column subset is the set top IEEE-CIS solutions consistently found informative. They are **raw per-transaction columns** computed by Vesta — they do not look across our rows — so adding them introduces no look-ahead leakage.

Included: `V12, V13, V19, V20, V30, V34, V35, V36, V37, V38, V44, V45, V53, V54, V61, V62, V70, V76, V78, V82, V83, V87, V91, V94, V127, V130, V133, V136, V137, V143, V149, V160, V165, V170, V187, V189, V201, V203, V207, V208, V209, V210, V212, V218, V221, V234, V257, V258, V264, V266, V267, V271, V274, V277, V283, V285, V289, V291, V294, V307, V308, V310, V312, V313, V314, V315, V317, V320, V323, V324, V326, V329, V332`.

---

## 5. Vesta D-Columns (`VESTA_D_FEATURES`, 13)

Timedelta features — days since prior events on the account. Also raw per-transaction Vesta columns.

Included: `D1, D2, D3, D4, D5, D6, D8, D9, D10, D11, D13, D14, D15`.

*(`D7` and `D12` are deliberately excluded — their behavior across the IEEE-CIS time boundary is unstable and drifts under the overfitting audit.)*

---

## 6. Amount-Weighted Training (`build_sample_weights`)

Not a feature per se, but the training pipeline passes a per-row `sample_weight` to `LGBMClassifier.fit()` for both models:

```
w = 1 + log1p(amount_inr) / mean(log1p(amount_inr))
```

Rows with larger `TransactionAmt` carry more weight in the loss, aligning training with the ₹ cost function (a missed ₹50,000 fraud hurts the loss more than a missed ₹500 one). Weights use `TransactionAmt` only — no label information — so no leakage. The same weights are passed to both fits, keeping the comparison fair.

---

## 7. Missing-Value Handling (`make_numeric_matrix`)

- Requested features that don't exist in the frame are silently dropped.
- Every remaining column is coerced to numeric; non-parseable values become `NaN`.
- `±inf` is replaced with `NaN`.
- Missing values are filled with **column medians computed on the training frame**. Validation and test reuse those same medians — the imputer is never re-fit on future data.
