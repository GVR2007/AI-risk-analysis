# Abuse-Ring Sentinel: System Architecture & Technical Specifications

## 1. System Overview

**Abuse-Ring Sentinel** is a production-grade, graph-empowered AI risk engine built to protect online payment processors and financial institutions against dynamic, multi-entity fraud rings, automated bot carding velocity bursts, and syndicate abuse.

Unlike standard transaction classifiers that process transactions independently as isolated events, **Abuse-Ring Sentinel** dynamically constructs dynamic relational entity graphs and rolling time-window velocity signals. It computes entity degree centralities, component connectivity metrics, and multi-scale temporal velocities (1-hour, 24-hour, and 7-day windows) in streaming fashion without target data leakage.

```
                                ┌─────────────────────────┐
                                │ Transaction Data Stream │
                                └────────────┬────────────┘
                                             │
               ┌────────────────────────────┴────────────────────────────┐
               │                                                         │
    ┌──────────▼──────────┐                                   ┌──────────▼──────────┐
    │  Entity Link Graph  │                                   │ Multiscale Velocity │
    │  & Centrality Engine│                                   │   Rolling Windows   │
    └──────────┬──────────┘                                   └──────────┬──────────┘
               │ (Device/Card/Address/Email overlaps)                    │ (1h / 24h / 7d windows)
               └────────────────────────────┬────────────────────────────┘
                                            │
                               ┌────────────▼────────────┐
                               │ Composite Sentinel      │
                               │ Feature Matrix (44 Feat)│
                               └────────────┬────────────┘
                                            │
                               ┌────────────▼────────────┐
                               │ Regularized LightGBM    │
                               │ Classifier Ensemble     │
                               └────────────┬────────────┘
                                            │
                               ┌────────────▼────────────┐
                               │ Capacity-Constrained    │
                               │ Threshold Grid Optimizer│
                               └────────────┬────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     │                                             │
          ┌──────────▼──────────┐                       ┌──────────▼──────────┐
          │ High-Risk Fraud Ring│                       │ Standard / Low-Risk │
          │ Flagged for Review  │                       │ Transaction Cleared │
          └─────────────────────┘                       └─────────────────────┘
```

---

## 2. Graph & Entity Topology Engine

The graph engine constructs real-time Bipartite/Multipartite entity association graphs across multiple transactional attributes:
- **Primary Entity Nodes**: Device (`_device`), Credit Card (`_card`), Billing Address (`_address`), Email Domain (`_email`), Browser Fingerprint (`_browser`).
- **Degree Centrality Computation**:
  - `device_degree_centrality`: Total count of unique cards and billing addresses associated with a single device hardware fingerprint.
  - `card_degree_centrality`: Total count of unique hardware devices presenting the same payment card.
  - `component_size`: Subgraph size representing multi-entity binding across device, address, and card combinations.
- **Dynamic Overlap Ratios**:
  - `device_card_ratio` = $\frac{\text{device\_unique\_cards}}{\max(\text{device\_transaction\_count}, 1)}$
  - `card_device_ratio` = $\frac{\text{card\_unique\_devices}}{\max(\text{card\_transaction\_count}, 1)}$

---

## 3. Multiscale Temporal Velocity Engine

Fraud syndicates execute rapid card testing and automated attacks within narrow timeframes. The Multiscale Velocity Engine uses non-leaking streaming queues (`collections.deque`) to calculate sliding window transaction counts:

1. **1-Hour Window (`1h`)**: Detects high-frequency automated bot carding and rapid submission bursts ($3,600\text{ seconds}$).
2. **24-Hour Window (`24h`)**: Tracks daily operational velocity across entity pairs ($86,400\text{ seconds}$).
3. **7-Day Window (`7d`)**: Captures medium-term syndicate persistence across weekly billing cycles ($604,800\text{ seconds}$).

---

## 4. Cost-Sensitive Capacity-Constrained Optimization & Currency Model

Traditional F1 or ROC-AUC metrics optimize model thresholds under an implicit assumption of equal error costs. In financial risk operations:
- **False Positives (FP)** incur manual investigation labor costs ($C_{\text{FP}} = \text{₹}25.00$).
- **False Negatives (FN)** incur chargeback liability penalties ($C_{\text{chargeback}} = \text{₹}1,500.00$) plus the loss of the transaction principal.
- **Manual Review Capacity Ceiling**: SOC teams operate under finite manual review capacity ($N_{\text{cap}} = 12,000\text{ reviews}$).

### FX Conversion & Proxy Dataset Caveat
- **Currency Normalization**: To evaluate financial loss in Indian Rupees ($\text{₹}$), transaction amounts are converted using a fixed FX exchange rate ($1\text{ USD} = 83.00\text{ INR}$).
- **Proxy Benchmark Rationale**: The IEEE-CIS Fraud Detection dataset serves as a standardized global benchmark representing payment gateway transaction topologies. Applying Indian BFSI economics ($\text{₹}25.00$ FP cost and $\text{₹}1,500.00$ chargeback fee) allows realistic evaluation of operational cost savings while maintaining benchmark reproducibility.

The total operational cost function evaluated during grid search threshold optimization is:

$$\text{Total Cost} = (FP \times C_{\text{FP}}) + \sum_{i \in \text{FN}} (\text{TransactionAmt}_i \times 83.00) + (FN \times C_{\text{chargeback}}) + \text{Penalty}_{\text{capacity}}$$

Where:
$$\text{Penalty}_{\text{capacity}} = \max(0, \text{Total Alerts} - N_{\text{cap}}) \times 400.00$$

---

## 5. SentinelGraph Decision Explainer (Analyst Decision-Card Renderer)

*Note: This is a template-driven explanation layer over existing model outputs — it renders human-readable rationale for analyst review and takes no autonomous action.*

The `SentinelGraphExplainer` module translates raw model risk probabilities and dynamic network topologies into human-readable investigator cards for security operations analysts:

```
┌────────────────────────────────────────────────────────┐
│   🚨 ABUSE RING DETECTED                              │
├────────────────────────────────────────────────────────┤
│ Transaction ID: 4020067                               │
│ Risk Score:     97.8/100                               │
│ Accounts Bound: 2                               │
│ Transactions:   4                               │
│ Financial Risk: ₹170,067.00                         │
├────────────────────────────────────────────────────────┤
│ 🔗 Graph Telemetry Footprint:                          │
│    • Shared Devices:   1                              │
│    • Shared Subnets:   1                              │
│    • Shared Addresses: 2                              │
│    • Syndicate Type:   High Velocity Syndicate      │
└────────────────────────────────────────────────────────┘
```
