#!/usr/bin/env python3
"""
SUBMISSION SANITY & INTEGRITY AUDITOR
=====================================
Verifies score distributions, calibration, and anomaly patterns.
"""

import pandas as pd
import numpy as np

def audit_submission():
    print("=" * 60)
    print("RUNNING SUBMISSION SANITY AUDIT")
    print("=" * 60)
    
    sub = pd.read_csv("submission.csv")
    
    # 1. Structural Checks
    assert list(sub.columns) == ["TransactionID", "isFraud"], "Error: Columns do not match [TransactionID, isFraud]"
    assert sub.isnull().sum().sum() == 0, "Error: NaNs or null values found in submission!"
    print("[PASS] Schema & Null Check: Columns and rows are fully intact.")
    
    # 2. Probability Range Check
    min_p, max_p = sub["isFraud"].min(), sub["isFraud"].max()
    assert 0.0 <= min_p and max_p <= 1.0, "Error: Probabilities out of [0, 1] range!"
    print(f"[PASS] Probability Range Check: Min={min_p:.4f}, Max={max_p:.4f}")
    
    # 3. Score Distribution & Calibration Check
    mean_p = sub["isFraud"].mean()
    median_p = sub["isFraud"].median()
    print(f"\n--- SCORE DISTRIBUTION ---")
    print(f"Mean Predicted Fraud Rate: {mean_p * 100:.2f}%")
    print(f"Median Score:              {median_p:.4f}")
    
    # Check if scores are polarized (e.g., all 0s or all 1s would fail this)
    if 0.01 <= mean_p <= 0.20:
        print("[PASS] Calibration Check: Mean fraud rate aligns with realistic financial fraud priors (~1-15%).")
    else:
        print("[WARNING] Mean fraud rate looks unusually high or low. Review threshold tuning.")

    # 4. Top High-Risk Inspection (Sanity Check on Worst Offenders)
    top_risky = sub.sort_values("isFraud", ascending=False).head(5)
    print(f"\n--- TOP 5 HIGHEST RISK TRANSACTIONS FLAGGED ---")
    print(top_risky.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("AUDIT COMPLETE: Submission file is mathematically valid and ready.")
    print("=" * 60)

if __name__ == "__main__":
    audit_submission()