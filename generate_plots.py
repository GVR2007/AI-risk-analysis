#!/usr/bin/env python3
"""
GENERATE AUDIT PLOTS — CONFUSION MATRIX & REAL PRECISION-RECALL CURVE
======================================================================
IMPORTANT: This version computes the PR curve from an ACTUAL threshold
sweep over real model probabilities and labels — it does NOT fabricate
a parametric curve. You must have y_true (real labels) and y_proba
(real predicted probabilities) available, e.g. saved from your training
pipeline as .npy files or loaded from a validation/test split.

If you do not have raw probabilities saved, you must re-run inference
on your held-out set first and save y_true / y_proba before running this.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

os.makedirs("docs/results", exist_ok=True)

# =====================================================================
# LOAD REAL DATA — replace these paths with your actual saved arrays.
# These must come from a real model run on a real held-out test set.
# DO NOT fabricate or interpolate these arrays.
# =====================================================================
BASELINE_Y_TRUE_PATH = "artifacts/baseline_y_true.npy"
BASELINE_Y_PROBA_PATH = "artifacts/baseline_y_proba.npy"
SENTINEL_Y_TRUE_PATH = "artifacts/sentinel_y_true.npy"
SENTINEL_Y_PROBA_PATH = "artifacts/sentinel_y_proba.npy"

missing = [p for p in [BASELINE_Y_TRUE_PATH, BASELINE_Y_PROBA_PATH,
                        SENTINEL_Y_TRUE_PATH, SENTINEL_Y_PROBA_PATH] if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(
        "Cannot generate a REAL precision-recall curve without real saved "
        f"predictions. Missing files: {missing}\n"
        "Save y_true and y_proba arrays from your actual model evaluation "
        "(e.g. np.save(...) right after model.predict_proba() in your "
        "training pipeline) before running this script. Do not fabricate "
        "a parametric curve as a substitute — it invalidates the plot's "
        "purpose of proving a genuine threshold sweep."
    )

baseline_y_true = np.load(BASELINE_Y_TRUE_PATH)
baseline_y_proba = np.load(BASELINE_Y_PROBA_PATH)
sentinel_y_true = np.load(SENTINEL_Y_TRUE_PATH)
sentinel_y_proba = np.load(SENTINEL_Y_PROBA_PATH)

# =====================================================================
# REAL THRESHOLD SWEEP — sklearn computes this directly from actual
# predicted probabilities vs actual labels across all distinct thresholds
# present in the data. Nothing here is invented.
# =====================================================================
baseline_precisions, baseline_recalls, baseline_thresholds = precision_recall_curve(
    baseline_y_true, baseline_y_proba
)
sentinel_precisions, sentinel_recalls, sentinel_thresholds = precision_recall_curve(
    sentinel_y_true, sentinel_y_proba
)

# Known, audited operating points (from your reconciled confusion matrix —
# these are real values, not fabricated, so hardcoding them as annotations
# is fine as long as they match your actual audited numbers)
SENTINEL_OPERATING_RECALL = 0.6413
SENTINEL_OPERATING_PRECISION = 0.1649
SENTINEL_OPERATING_THRESHOLD = 0.105172

BASELINE_OPERATING_RECALL = 0.6857
BASELINE_OPERATING_PRECISION = 0.1018

# =====================================================================
# PLOT 1: CONFUSION MATRIX WITH FINANCIAL COST ANNOTATIONS
# (Hardcoded here is acceptable ONLY because these are the real, audited
# confusion matrix counts from your actual reconciled model run.)
# =====================================================================
fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')

cm_labels = [
    [f"True Negative (TN)\n{75474:,}\n(Cleared Normal)\nCost: ₹0",
     f"False Positive (FP)\n{10021:,}\n(Flagged Review)\nCost: ₹250,525"],
    [f"False Negative (FN)\n{1107:,}\n(Missed Fraud)\nExposure: ₹15.41M",
     f"True Positive (TP)\n{1979:,}\n(Prevented Loss)\nSaved: ₹2.97M"]
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
ax.set_title("Abuse-Ring Sentinel v13: Audited Confusion Matrix\n(Post-Truncation 12,000 Review Capacity Budget)",
              color='#f0f6fc', fontsize=13, fontweight='bold', pad=12)

plt.subplots_adjust(left=0.22, right=0.95, top=0.88, bottom=0.12)
plt.savefig("docs/results/confusion_matrix.png", facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

# =====================================================================
# PLOT 2: REAL PRECISION-RECALL CURVE (computed from actual sweep data)
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')

ax.plot(baseline_recalls, baseline_precisions,
        label=f'Baseline LightGBM (Prec: {BASELINE_OPERATING_PRECISION*100:.2f}%, Rec: {BASELINE_OPERATING_RECALL*100:.2f}%)',
        color='#8b949e', linestyle='--', linewidth=2)
ax.plot(sentinel_recalls, sentinel_precisions,
        label=f'Regularized Structural Sentinel v13 (Prec: {SENTINEL_OPERATING_PRECISION*100:.2f}%, Rec: {SENTINEL_OPERATING_RECALL*100:.2f}%)',
        color='#56d364', linewidth=3)

ax.scatter([SENTINEL_OPERATING_RECALL], [SENTINEL_OPERATING_PRECISION],
           color='#ff7b72', s=120, zorder=5, edgecolors='#ffffff', linewidth=2)
ax.annotate(
    f'Operating Point (T*={SENTINEL_OPERATING_THRESHOLD})\n'
    f'Precision: {SENTINEL_OPERATING_PRECISION*100:.2f}% | Recall: {SENTINEL_OPERATING_RECALL*100:.2f}%\n'
    f'(12,000 Review Capacity Cap)',
    xy=(SENTINEL_OPERATING_RECALL, SENTINEL_OPERATING_PRECISION), xytext=(0.30, 0.25),
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

print("Successfully generated confusion matrix (audited counts) and a REAL "
      "precision-recall curve computed from actual saved model predictions.")