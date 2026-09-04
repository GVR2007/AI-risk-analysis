#!/usr/bin/env python3
"""
GENERATE AUDIT PLOTS -- CONFUSION MATRICES (BOTH MODELS) & REAL PR CURVE
========================================================================
Everything here is computed directly from real saved model predictions
(artifacts/*.npy) -- nothing is hardcoded or fabricated. This eliminates
a recurring failure mode where a hardcoded confusion matrix silently went
stale after the pipeline changed.

Generates THREE images:
  1. docs/results/confusion_matrix_baseline.png   (the SHIPPED model)
  2. docs/results/confusion_matrix_sentinel.png   (the tested graph variant)
  3. docs/results/pr_curve.png                    (both models, one axes)

Confusion matrices use the SAME hard-capacity-truncation logic as the
training pipeline, so they always match what evaluate() reported.

Run ieee_pipeline_chatgpt_15.py first -- it saves the .npy artifacts.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix

os.makedirs("docs/results", exist_ok=True)

MAX_MANUAL_REVIEWS_CAP = 12000
FALSE_POSITIVE_REVIEW_COST = 25.0
CHARGEBACK_PENALTY_FEE = 1500.0

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
        "Run ieee_pipeline_chatgpt_15.py first -- it saves these automatically."
    )

baseline_y_true = np.load(BASELINE_Y_TRUE_PATH)
baseline_y_proba = np.load(BASELINE_Y_PROBA_PATH)
sentinel_y_true = np.load(SENTINEL_Y_TRUE_PATH)
sentinel_y_proba = np.load(SENTINEL_Y_PROBA_PATH)


def apply_hard_capacity_truncation(probability, cap=MAX_MANUAL_REVIEWS_CAP):
    """Must match the exact logic used in the training pipeline's evaluate()."""
    n = len(probability)
    if n <= cap:
        return np.ones(n, dtype=int)
    order = np.argsort(-probability, kind="mergesort")
    keep_idx = order[:cap]
    y_pred = np.zeros(n, dtype=int)
    y_pred[keep_idx] = 1
    return y_pred


def compute_cm(y_true, y_proba):
    pred = apply_hard_capacity_truncation(y_proba, MAX_MANUAL_REVIEWS_CAP)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, precision=precision, recall=recall)


def plot_confusion_matrix(cm, title, outpath):
    fp_cost = cm["fp"] * FALSE_POSITIVE_REVIEW_COST
    fn_penalty = cm["fn"] * CHARGEBACK_PENALTY_FEE

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    cm_labels = [
        [f"True Negative (TN)\n{cm['tn']:,}\n(Cleared Normal)\nCost: ₹0",
         f"False Positive (FP)\n{cm['fp']:,}\n(Flagged Review)\nCost: ₹{fp_cost:,.0f}"],
        [f"False Negative (FN)\n{cm['fn']:,}\n(Missed Fraud)\nChargeback Penalty: ₹{fn_penalty:,.0f}+amt",
         f"True Positive (TP)\n{cm['tp']:,}\n(Prevented Loss)"]
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
    ax.set_title(title, color='#f0f6fc', fontsize=12.5, fontweight='bold', pad=12)

    plt.subplots_adjust(left=0.22, right=0.95, top=0.86, bottom=0.12)
    plt.savefig(outpath, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()


# ---- Compute both confusion matrices (real, capacity-compliant) ----
cm_base = compute_cm(baseline_y_true, baseline_y_proba)
cm_sent = compute_cm(sentinel_y_true, sentinel_y_proba)

print(f"BASELINE (SHIPPED): TP={cm_base['tp']:,} FP={cm_base['fp']:,} FN={cm_base['fn']:,} TN={cm_base['tn']:,} "
      f"| Precision={cm_base['precision']:.4f} Recall={cm_base['recall']:.4f}")
print(f"SENTINEL (tested):  TP={cm_sent['tp']:,} FP={cm_sent['fp']:,} FN={cm_sent['fn']:,} TN={cm_sent['tn']:,} "
      f"| Precision={cm_sent['precision']:.4f} Recall={cm_sent['recall']:.4f}")

plot_confusion_matrix(
    cm_base,
    "Baseline LightGBM (SHIPPED MODEL): Audited Confusion Matrix\n(Hard-Truncated 12,000 Review Capacity Budget)",
    "docs/results/confusion_matrix_baseline.png"
)
plot_confusion_matrix(
    cm_sent,
    "Structural Sentinel (Tested Variant): Audited Confusion Matrix\n(Hard-Truncated 12,000 Review Capacity Budget)",
    "docs/results/confusion_matrix_sentinel.png"
)

# ---- PR curve: both models on one axes, with BOTH operating points ----
baseline_precisions, baseline_recalls, _ = precision_recall_curve(baseline_y_true, baseline_y_proba)
sentinel_precisions, sentinel_recalls, _ = precision_recall_curve(sentinel_y_true, sentinel_y_proba)

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')

ax.plot(baseline_recalls, baseline_precisions,
        label=f"Baseline LightGBM — SHIPPED (P={cm_base['precision']*100:.1f}%, R={cm_base['recall']*100:.1f}%)",
        color='#8b949e', linestyle='--', linewidth=2)
ax.plot(sentinel_recalls, sentinel_precisions,
        label=f"Structural Sentinel — tested (P={cm_sent['precision']*100:.1f}%, R={cm_sent['recall']*100:.1f}%)",
        color='#56d364', linewidth=3)

# Mark both capacity-compliant operating points
for cm, color, tag in [(cm_base, '#8b949e', 'Baseline'), (cm_sent, '#ff7b72', 'Sentinel')]:
    ax.scatter([cm['recall']], [cm['precision']], color=color, s=110, zorder=5,
               edgecolors='#ffffff', linewidth=2)

ax.annotate(
    f"Capacity-Compliant Operating Points (12,000 cap)\n"
    f"Baseline:  P={cm_base['precision']*100:.2f}%  R={cm_base['recall']*100:.2f}%\n"
    f"Sentinel:  P={cm_sent['precision']*100:.2f}%  R={cm_sent['recall']*100:.2f}%",
    xy=(cm_base['recall'], cm_base['precision']), xytext=(0.12, 0.30),
    arrowprops=dict(arrowstyle="->", color="#c9d1d9", lw=1.5),
    color='#ffffff', fontsize=9, fontweight='bold',
    bbox=dict(boxstyle="round,pad=0.5", fc="#21262d", ec="#30363d", lw=1.5)
)

ax.set_xlabel('Recall (True Positive Rate)', color='#c9d1d9', fontsize=11, fontweight='bold')
ax.set_ylabel('Precision (Positive Predictive Value)', color='#c9d1d9', fontsize=11, fontweight='bold')
ax.set_title('Precision-Recall Curve — Baseline vs Sentinel (Real Threshold Sweep)',
             color='#f0f6fc', fontsize=12.5, fontweight='bold', pad=12)
ax.tick_params(colors='#8b949e', which='both', labelsize=10)
ax.grid(True, linestyle=':', color='#30363d', alpha=0.7)

legend = ax.legend(facecolor='#21262d', edgecolor='#30363d', fontsize=9)
for text in legend.get_texts():
    text.set_color('#c9d1d9')

plt.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.12)
plt.savefig("docs/results/pr_curve.png", facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print("Generated 3 images: confusion_matrix_baseline.png, confusion_matrix_sentinel.png, pr_curve.png")
print("All computed live from real saved predictions -- nothing hardcoded.")