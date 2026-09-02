#!/usr/bin/env python3
"""
GENERATE AUDIT PLOTS -- CONFUSION MATRIX & REAL PRECISION-RECALL CURVE
======================================================================
Both the confusion matrix AND the precision-recall curve are computed
directly from real saved model predictions (artifacts/*.npy) -- neither
is hardcoded or fabricated. This eliminates a recurring failure mode in
this project where a hardcoded confusion matrix silently went stale after
the underlying model/pipeline changed.

Confusion matrix uses the SAME hard-capacity-truncation logic as the
training pipeline (top MAX_MANUAL_REVIEWS_CAP alerts by probability),
so the plotted matrix always matches what evaluate() actually reported
in the training run -- there is no separate hardcoded copy to go stale.

If you do not have raw probabilities saved, re-run
ieee_pipeline_chatgpt_13.py first (it saves these automatically).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix

os.makedirs("docs/results", exist_ok=True)

MAX_MANUAL_REVIEWS_CAP = 12000
FALSE_POSITIVE_REVIEW_COST = 25.0
CHARGEBACK_PENALTY_FEE = 1500.0

# =====================================================================
# LOAD REAL DATA -- must come from a real model run (ieee_pipeline_chatgpt_13.py)
# =====================================================================
BASELINE_Y_TRUE_PATH = "artifacts/baseline_y_true.npy"
BASELINE_Y_PROBA_PATH = "artifacts/baseline_y_proba.npy"
SENTINEL_Y_TRUE_PATH = "artifacts/sentinel_y_true.npy"
SENTINEL_Y_PROBA_PATH = "artifacts/sentinel_y_proba.npy"

missing = [p for p in [BASELINE_Y_TRUE_PATH, BASELINE_Y_PROBA_PATH,
                        SENTINEL_Y_TRUE_PATH, SENTINEL_Y_PROBA_PATH] if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(
        "Cannot generate REAL audit plots without real saved predictions. "
        f"Missing files: {missing}\n"
        "Run ieee_pipeline_chatgpt_13.py first -- it saves these automatically "
        "after evaluate() runs for both models."
    )

baseline_y_true = np.load(BASELINE_Y_TRUE_PATH)
baseline_y_proba = np.load(BASELINE_Y_PROBA_PATH)
sentinel_y_true = np.load(SENTINEL_Y_TRUE_PATH)
sentinel_y_proba = np.load(SENTINEL_Y_PROBA_PATH)


def apply_hard_capacity_truncation(probability, cap=MAX_MANUAL_REVIEWS_CAP):
    """Must match the exact logic used in ieee_pipeline_chatgpt_13.py's evaluate()."""
    n = len(probability)
    if n <= cap:
        return np.ones(n, dtype=int)
    order = np.argsort(-probability, kind="mergesort")
    keep_idx = order[:cap]
    y_pred = np.zeros(n, dtype=int)
    y_pred[keep_idx] = 1
    return y_pred


# =====================================================================
# REAL THRESHOLD SWEEP (unconstrained curve -- shows the full tradeoff shape)
# =====================================================================
baseline_precisions, baseline_recalls, _ = precision_recall_curve(baseline_y_true, baseline_y_proba)
sentinel_precisions, sentinel_recalls, _ = precision_recall_curve(sentinel_y_true, sentinel_y_proba)

# =====================================================================
# REAL CAPACITY-COMPLIANT CONFUSION MATRIX (matches the training pipeline exactly)
# =====================================================================
sentinel_pred_capped = apply_hard_capacity_truncation(sentinel_y_proba, MAX_MANUAL_REVIEWS_CAP)
tn, fp, fn, tp = confusion_matrix(sentinel_y_true, sentinel_pred_capped).ravel()

fp_cost = fp * FALSE_POSITIVE_REVIEW_COST
fn_penalty_only = fn * CHARGEBACK_PENALTY_FEE  # transaction-amount component not available here without amounts array

precision_capped = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall_capped = tp / (tp + fn) if (tp + fn) > 0 else 0.0

# Locate the actual sentinel operating point on the unconstrained curve nearest
# to the capacity-compliant precision/recall, purely for the annotation marker
op_idx = np.argmin(np.abs(sentinel_recalls - recall_capped))
op_recall = sentinel_recalls[op_idx]
op_precision = sentinel_precisions[op_idx]

print(f"Real capacity-compliant confusion matrix (Sentinel): TP={tp:,} FP={fp:,} FN={fn:,} TN={tn:,}")
print(f"Precision={precision_capped:.4f}  Recall={recall_capped:.4f}")

# =====================================================================
# PLOT 1: CONFUSION MATRIX (real counts, computed above -- not hardcoded)
# =====================================================================
fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')

cm_labels = [
    [f"True Negative (TN)\n{tn:,}\n(Cleared Normal)\nCost: ₹0",
     f"False Positive (FP)\n{fp:,}\n(Flagged Review)\nCost: ₹{fp_cost:,.0f}"],
    [f"False Negative (FN)\n{fn:,}\n(Missed Fraud)\nChargeback Penalty: ₹{fn_penalty_only:,.0f}+amt",
     f"True Positive (TP)\n{tp:,}\n(Prevented Loss)"]
]

colors = np.array([["#1f242d", "#2d1f2d"], ["#3b1f1f", "#1f3b2d"]])
text_colors = np.array([["#c9d1d9", "#f85149"], ["#ff7b72", "#56d364"]])

for i in range(2):
    for j in range(2):
        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=colors[i][j], edgecolor='#30363d', linewidth=2)
        ax.add_patch(rect)
        ax.text(j, i, cm_labels[i][j], ha='center', va='center', color=text_colors[i][j], fontsize=10, fontweight='bold')

ax.set_xlim(-0.5, 1.5)
ax.set_ylim(1.5, -0.5)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Predicted Legitimate', 'Predicted Fraud (Alert)'], color='#c9d1d9', fontsize=11, fontweight='bold')
ax.set_yticklabels(['Actual Legitimate', 'Actual Fraud'], color='#c9d1d9', fontsize=11, fontweight='bold')
ax.tick_params(colors='#c9d1d9', which='both', length=0)
ax.set_title("Abuse-Ring Sentinel: Audited Confusion Matrix\n(Hard-Truncated 12,000 Review Capacity Budget)",
              color='#f0f6fc', fontsize=13, fontweight='bold', pad=12)

plt.subplots_adjust(left=0.22, right=0.95, top=0.88, bottom=0.12)
plt.savefig("docs/results/confusion_matrix.png", facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

# =====================================================================
# PLOT 2: REAL PRECISION-RECALL CURVE
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')

ax.plot(baseline_recalls, baseline_precisions, label='Baseline LightGBM', color='#8b949e', linestyle='--', linewidth=2)
ax.plot(sentinel_recalls, sentinel_precisions, label='Regularized Structural Sentinel', color='#56d364', linewidth=3)

ax.scatter([op_recall], [op_precision], color='#ff7b72', s=120, zorder=5, edgecolors='#ffffff', linewidth=2)
ax.annotate(
    f'Capacity-Compliant Operating Point\n'
    f'Precision: {precision_capped*100:.2f}% | Recall: {recall_capped*100:.2f}%\n'
    f'(12,000 Review Capacity Cap)',
    xy=(op_recall, op_precision), xytext=(0.15, op_precision * 1.3 + 0.02),
    arrowprops=dict(arrowstyle="->", color="#ff7b72", lw=2),
    color='#ffffff', fontsize=9.5, fontweight='bold',
    bbox=dict(boxstyle="round,pad=0.5", fc="#21262d", ec="#30363d", lw=1.5)
)

ax.set_xlabel('Recall (True Positive Rate)', color='#c9d1d9', fontsize=11, fontweight='bold')
ax.set_ylabel('Precision (Positive Predictive Value)', color='#c9d1d9', fontsize=11, fontweight='bold')
ax.set_title('Precision-Recall Curve (Real Threshold Sweep)', color='#f0f6fc', fontsize=13, fontweight='bold', pad=12)
ax.tick_params(colors='#8b949e', which='both', labelsize=10)
ax.grid(True, linestyle=':', color='#30363d', alpha=0.7)

legend = ax.legend(facecolor='#21262d', edgecolor='#30363d', fontsize=9.5)
for text in legend.get_texts():
    text.set_color('#c9d1d9')

plt.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.12)
plt.savefig("docs/results/pr_curve.png", facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print("Successfully generated confusion matrix and PR curve -- both computed "
      "live from real saved predictions, nothing hardcoded.")