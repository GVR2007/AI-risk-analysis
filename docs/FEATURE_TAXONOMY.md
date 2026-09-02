# Sentinel Feature Engineering Taxonomy

This document details all 44 structural, graph, velocity, and composite features utilized by **Abuse-Ring Sentinel (v13)**.

---

## 1. Feature Categories Overview

| Category | Feature Count | Description |
| :--- | :---: | :--- |
| **Graph Centrality** | 3 | Node connectivity and subgraph component density across device/card/address links. |
| **Entity Overlap & Structural Counts** | 15 | Cumulative transaction counts, unique entity linkages, and entity pair bindings. |
| **Multiscale Velocity** | 11 | Sliding time-window counts (1h, 24h, 7d) across single entities and entity pairs. |
| **Composite & Interaction**| 10 | Cross-domain risk interaction scores, density ratios, and logarithmic ring metrics. |
| **Syndicate Rule Flags** | 5 | Binary indicators triggered when structural abuse conditions are met. |
| **Total Sentinel Features** | **44** | Fully audited feature matrix used in regularized Sentinel training. |

---

## 2. Complete 44-Feature Specification

### A. Graph Centrality Features (3 Features)
1. `device_degree_centrality`: Total unique cards + total unique addresses associated with the device hardware fingerprint.
2. `card_degree_centrality`: Total unique devices presenting the payment card.
3. `component_size`: Transaction count within the specific `(device, address, card)` relational subgraph.

### B. Entity Overlap & Structural Counts (15 Features)
4. `device_transaction_count`: Cumulative transaction count for the device up to current time.
5. `card_transaction_count`: Cumulative transaction count for the payment card up to current time.
6. `address_transaction_count`: Cumulative transaction count for the billing address up to current time.
7. `device_unique_cards`: Distinct credit card numbers bound to the device up to current transaction.
8. `device_unique_addresses`: Distinct billing addresses bound to the device up to current transaction.
9. `device_unique_emails`: Distinct email domains bound to the device up to current transaction.
10. `device_unique_browsers`: Distinct user-agent / browser fingerprints bound to the device.
11. `card_unique_devices`: Distinct hardware devices presenting the payment card.
12. `card_unique_addresses`: Distinct billing addresses associated with the payment card.
13. `address_unique_devices`: Distinct devices utilizing the billing address.
14. `address_unique_cards`: Distinct cards utilizing the billing address.
15. `email_unique_cards`: Distinct cards linked to the purchaser email domain.
16. `device_card_pair_count`: Cumulative transaction count for the specific `(device, card)` pair.
17. `device_address_pair_count`: Cumulative transaction count for the specific `(device, address)` pair.
18. `card_address_pair_count`: Cumulative transaction count for the specific `(card, address)` pair.

### C. Multiscale Velocity Features (11 Features)
19. `device_previous_tx_1h`: Device transactions in the past 60 minutes.
20. `device_previous_tx_24h`: Device transactions in the past 24 hours.
21. `device_previous_tx_7d`: Device transactions in the past 7 days.
22. `card_previous_tx_1h`: Card transactions in the past 60 minutes.
23. `card_previous_tx_24h`: Card transactions in the past 24 hours.
24. `card_previous_tx_7d`: Card transactions in the past 7 days.
25. `address_previous_tx_1h`: Address transactions in the past 60 minutes.
26. `address_previous_tx_24h`: Address transactions in the past 24 hours.
27. `address_previous_tx_7d`: Address transactions in the past 7 days.
28. `device_card_previous_tx_24h`: Transactions for the specific `(device, card)` pair in past 24 hours.
29. `device_address_previous_tx_24h`: Transactions for the specific `(device, address)` pair in past 24 hours.

### D. Composite & Interaction Metrics (10 Features)
30. `device_card_ratio`: Ratio of unique cards to total device transactions: $\frac{\text{device\_unique\_cards}}{\max(\text{device\_tx\_count}, 1)}$.
31. `device_address_ratio`: Ratio of unique addresses to total device transactions.
32. `card_device_ratio`: Ratio of unique devices to total card transactions.
33. `email_card_ratio`: Ratio of unique cards to total email transactions.
34. `ring_density_score`: Product of `device_card_ratio` and `device_previous_tx_24h`.
35. `card_velocity_interaction`: Product of `card_unique_devices` and `card_previous_tx_24h`.
36. `entity_overlap_score`: $\ln(1 + \text{device\_unique\_cards}) + \ln(1 + \text{device\_unique\_addresses})$.
37. `pair_link_score`: $\ln(1 + \text{device\_card\_pair\_count}) + \ln(1 + \text{device\_address\_pair\_count})$.
38. `ring_velocity_score`: $\ln(1 + \text{device\_previous\_tx\_24h}) + \ln(1 + \text{card\_previous\_tx\_24h})$.
39. `abuse_ring_score`: Composite weighted score: $0.35 \times \text{entity\_overlap\_score} + 0.35 \times \text{pair\_link\_score} + 0.30 \times \ln(1 + \text{component\_size})$.

### E. Discrete Syndicate Rule Flags (5 Features)
40. `multi_card_device_flag`: Triggered if `device_unique_cards` $\ge 3$.
41. `multi_address_device_flag`: Triggered if `device_unique_addresses` $\ge 2$.
42. `multi_device_card_flag`: Triggered if `card_unique_devices` $\ge 2$.
43. `high_velocity_device_flag`: Triggered if `device_previous_tx_24h` $\ge 3$.
44. `ring_candidate_flag`: Triggered if `device_unique_cards` $\ge 3$ AND `device_previous_tx_24h` $\ge 2$.
