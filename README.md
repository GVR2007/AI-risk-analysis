# Abuse-Ring Sentinel: Real-Time Graph & Multiscale Velocity AI Risk Engine

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-green.svg)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)
[![Domain](https://img.shields.io/badge/Domain-AI%20Risk%20%26%20Fraud%20Detection-orange.svg)]()

> **Abuse-Ring Sentinel** is an enterprise-grade AI risk analysis system engineered to detect coordinated fraud, multi-entity abuse rings, and high-velocity carding attacks in real-time transaction streams. Designed with a cost-sensitive and capacity-constrained optimization engine, it reduces overall financial exposure from fraud chargebacks while operating within strict manual review capacity budgets.

---

## 🎯 Key Highlights & Business Value

- **Graph & Entity Centrality Engine**: Maps dynamic relationships across devices, credit cards, billing addresses, and email domains to identify syndicate structures and shared entity clusters.
- **Multiscale Temporal Velocity Tracking**: Computes sliding-window transaction counts (1h, 24h, 7d) for devices, cards, and combined entity pairs without target data leakage.
- **Capacity-Constrained Cost Optimization**: Optimizes prediction decision thresholds directly against operational economics (e.g., ₹1,500 chargeback penalty vs. ₹25 manual review cost) subject to a hard review capacity ceiling (e.g., max 12,000 reviews).
- **Automated Audit Suite**: Built-in verification for zero target leakage, validation/test split divergence (overfitting prevention), and operational capacity compliance.
- **Memory-Efficient Inference Pipeline**: Low-memory streaming processing pipeline designed for large-scale test dataset prediction generation under strict RAM limits.

---

## 🏗️ System Architecture

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
                               │ Feature Matrix          │
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

## 📁 Repository Structure

```
├── ieee_pipeline_chatgpt_13.py  # Primary production training, auditing & evaluation pipeline
├── generate_submission.py       # Memory-optimized test set inference & submission generator
├── verify_submission.py         # Integrity and distribution validation script for predictions
├── ieee_optimized_pipeline.py   # Baseline optimized pipeline configuration
├── requirements.txt             # Project dependencies
└── README.md                    # System documentation
```

---

## 📊 Feature Engineering Taxonomy

| Feature Category | Variables / Metrics | Description |
| :--- | :--- | :--- |
| **Graph Centrality** | `device_degree_centrality`, `card_degree_centrality`, `component_size` | Measures entity connectivity and cluster density across shared infrastructure. |
| **Entity Overlap** | `device_unique_cards`, `card_unique_devices`, `email_unique_cards` | Tracks unique cross-entity linkage count to flag syndicate behavior. |
| **Multiscale Velocity**| `device_previous_tx_{1h,24h,7d}`, `card_previous_tx_{1h,24h,7d}` | Sliding window transaction velocity counters to capture burst behavior. |
| **Composite Scores** | `abuse_ring_score`, `ring_density_score`, `card_velocity_interaction` | Weighted composite signals combining degree centrality, link velocity, and component size. |
| **Syndicate Flags** | `ring_candidate_flag`, `multi_card_device_flag`, `high_velocity_device_flag` | Discrete indicators triggered when entity multiplicity thresholds are breached. |

---

## ⚙️ Financial Economics & Decision Thresholding

Traditional F1 or ROC-AUC optimization ignores operational realities. **Abuse-Ring Sentinel** optimizes decision thresholds using a custom risk objective function:

$$\text{Total Cost} = (FP \times C_{FP}) + \sum_{FN} \text{Amount}_{FN} + (FN \times C_{chargeback}) + \text{Penalty}_{\text{capacity}}$$

Where:
- $C_{FP}$: Manual investigation cost per false positive ($\approx \$25.00$).
- $C_{chargeback}$: Administrative penalty fee per chargeback ($\approx \$1,500.00$).
- $\text{Penalty}_{\text{capacity}}$: Exponential penalty applied when manual review alerts exceed the operational capacity limit (12,000 alerts).

---

## 🚀 Quick Start

### 1. Prerequisites & Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/GVR2007/AI-risk-analysis.git
cd AI-risk-analysis
pip install -r requirements.txt
```

### 2. Dataset Setup
Place the IEEE-CIS Fraud Detection dataset files in the following path:
```
test_datasets/kaggle/ieee-fraud-detection/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
└── test_identity.csv
```

### 3. Model Training & Auditing Pipeline
Run the fully audited Sentinel pipeline to train the model, execute automated checks, and calculate total cost metrics:

```bash
python ieee_pipeline_chatgpt_13.py
```

### 4. Inference & Submission Generation
Generate test predictions using memory-optimized chunk processing:

```bash
python generate_submission.py
```

### 5. Validate Output Integrity
Verify the output submission file formatting, row count, null count, and probability distribution:

```bash
python verify_submission.py
```

---

## 🛡️ Audit Suite Verification Results

| Audit Check | Methodology | Result | Status |
| :--- | :--- | :--- | :--- |
| **1. Target Leakage** | Feature importance gain check for historical target contamination | 0% Target leakage detected | `PASSED` |
| **2. Overfitting / Divergence**| Validation vs. Test recall divergence evaluation | Split divergence within tolerance ($\le 0.08$) | `PASSED` |
| **3. Capacity Constraint** | Total flagged review alert volume vs. operational limit | Operates strictly within budget ($\le 12,000$ alerts) | `PASSED` |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
