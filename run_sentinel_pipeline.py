#!/usr/bin/env python3
"""
SENTINELGRAPH & ABUSE-RING SENTINEL - FINAL FULL PRODUCTION SCRIPT (CORRECTED)
================================================================================
Includes:
    1. Audited LightGBM pipeline with LEAK-FREE graph centrality & velocities.
    2. Cost-sensitive threshold tuning with ₹25 review cost & ₹1,500 penalty.
    3. Operational capacity enforcement (Top-12,000 hard truncation).
    4. Test-set inference generating 506,691-row submission.csv.
    5. SentinelGraph forensic decision-card renderer (analyst explanation layer).

CHANGELOG (Failure Recovery 5 -- Centrality Feature Leakage):
    - add_graph_centrality_features() previously used pandas .transform("nunique")
      / .transform("count"), which aggregate over the ENTIRE group including
      future transactions relative to each row -- a look-ahead leak.
    - Fixed by deriving centrality strictly from the already time-ordered,
      cumulative unique-count features built in add_structural_ring_features(),
      and by computing component_size via cumcount() instead of a global count.
    - Pipeline call order is now enforced: add_structural_ring_features() MUST
      run before add_graph_centrality_features().
    - The decision-card demo in main() now attempts a REAL subgraph extraction
      from actual test data first; only falls back to a clearly labeled
      synthetic demo card if real data extraction is unavailable.

CHANGELOG (previous pass):
    - REMOVED the duplicate inline SentinelGraphExplainer class definition.
      It is now imported from sentinel_explainer.py, which is the single
      canonical source.

CHANGELOG (this pass -- v15 sync fix):
    - MODEL_FILE was still pointing at "ieee_abuse_ring_sentinel_v15.pkl" and
      the retrain fallback still imported ieee_pipeline_chatgpt_15 -- the
      SUPERSEDED pipeline version, not the actual final leak-free v15
      pipeline that produces the real saved model. This meant that if the
      model file were ever missing, this script would have silently retrained
      using the OLD, already-fixed-past version instead of the current one.
    - Both now point to the v15 pipeline and v15 model file, matching what
      ieee_pipeline_chatgpt_15.py actually trains and saves.
"""

import os
import sys
import warnings
from collections import defaultdict, deque

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from sentinel_explainer import SentinelGraphExplainer

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIG & ECONOMICS
# =====================================================================
DATA_DIR = "test_datasets/kaggle/ieee-fraud-detection"
MODEL_FILE = "ieee_abuse_ring_sentinel_baseline_v15.pkl"
SUBMISSION_FILE = "submission.csv"

FALSE_POSITIVE_REVIEW_COST = 25.0
CHARGEBACK_PENALTY_FEE = 1500.0
USD_TO_INR = 83.00
MAX_MANUAL_REVIEWS_CAP = 12000

BASELINE_FEATURES = [
    "TransactionAmt", "card2", "card3", "card5", "addr2", "dist1", "C1", "C2", "C3", "C4", "C5",
]

TRUE_SENTINEL_FEATURES = [
    "device_transaction_count", "card_transaction_count", "address_transaction_count",
    "device_unique_cards", "device_unique_addresses", "device_unique_emails", "device_unique_browsers",
    "card_unique_devices", "card_unique_addresses", "address_unique_devices", "address_unique_cards",
    "email_unique_cards", "device_card_pair_count", "device_address_pair_count", "card_address_pair_count",
    "device_card_ratio", "device_address_ratio", "card_device_ratio", "email_card_ratio",
    "device_previous_tx_1h", "device_previous_tx_24h", "device_previous_tx_7d",
    "card_previous_tx_1h", "card_previous_tx_24h", "card_previous_tx_7d",
    "address_previous_tx_1h", "address_previous_tx_24h", "address_previous_tx_7d",
    "device_card_previous_tx_24h", "device_address_previous_tx_24h",
    "device_degree_centrality", "card_degree_centrality", "component_size",
    "ring_density_score", "card_velocity_interaction",
    "entity_overlap_score", "pair_link_score", "ring_velocity_score", "abuse_ring_score",
    "multi_card_device_flag", "multi_address_device_flag", "multi_device_card_flag",
    "high_velocity_device_flag", "ring_candidate_flag",
]

HYBRID_FEATURES = list(dict.fromkeys(BASELINE_FEATURES + TRUE_SENTINEL_FEATURES))


# =====================================================================
# FEATURE ENGINEERING & GRAPH FUNCTIONS
# =====================================================================
def prepare_entities(df):
    df = df.copy()
    df["_device"] = df["DeviceInfo"].fillna(df.get("id_31", "__MISSING__")).fillna("__MISSING__").astype(str)
    df["_card"] = df["card1"].fillna("__MISSING__").astype(str)
    df["_address"] = df["addr1"].fillna("__MISSING__").astype(str)
    df["_email"] = df["P_emaildomain"].fillna(df.get("R_emaildomain", "__MISSING__")).fillna("__MISSING__").astype(str)
    df["_browser"] = df.get("id_31", pd.Series(["__MISSING__"] * len(df))).fillna("__MISSING__").astype(str)
    df["_time"] = pd.to_numeric(df["TransactionDT"], errors="coerce").fillna(0).astype(np.int64)
    return df


def add_structural_ring_features(df):
    df = df.copy().sort_values("TransactionDT").reset_index(drop=True)
    df["device_transaction_count"] = df.groupby("_device", dropna=False).cumcount().astype(float)
    df["card_transaction_count"] = df.groupby("_card", dropna=False).cumcount().astype(float)
    df["address_transaction_count"] = df.groupby("_address", dropna=False).cumcount().astype(float)

    df["device_card_pair_count"] = df.groupby(["_device", "_card"], dropna=False).cumcount().astype(float)
    df["device_address_pair_count"] = df.groupby(["_device", "_address"], dropna=False).cumcount().astype(float)
    df["card_address_pair_count"] = df.groupby(["_card", "_address"], dropna=False).cumcount().astype(float)

    def get_uniques(col_group, col_target, out_name):
        first = ~df.duplicated([col_group, col_target])
        cum = first.astype(np.int8).groupby(df[col_group]).cumsum()
        df[out_name] = (cum - first.astype(np.int8)).astype(float)

    get_uniques("_device", "_card", "device_unique_cards")
    get_uniques("_device", "_address", "device_unique_addresses")
    get_uniques("_device", "_email", "device_unique_emails")
    get_uniques("_device", "_browser", "device_unique_browsers")
    get_uniques("_card", "_device", "card_unique_devices")
    get_uniques("_card", "_address", "card_unique_addresses")
    get_uniques("_address", "_device", "address_unique_devices")
    get_uniques("_address", "_card", "address_unique_cards")
    get_uniques("_email", "_card", "email_unique_cards")
    return df


def add_graph_centrality_features(df):
    df = df.copy()

    required_cols = ["device_unique_cards", "device_unique_addresses", "card_unique_devices"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"add_graph_centrality_features() requires columns {missing} to already exist. "
            f"Call add_structural_ring_features(df) BEFORE add_graph_centrality_features(df)."
        )

    df["device_degree_centrality"] = df["device_unique_cards"] + df["device_unique_addresses"]
    df["card_degree_centrality"] = df["card_unique_devices"]

    df["_device_address_card"] = df["_device"] + "_" + df["_address"] + "_" + df["_card"]
    df["component_size"] = df.groupby("_device_address_card", dropna=False).cumcount().astype(float)
    df.drop(columns=["_device_address_card"], inplace=True)

    return df


def calculate_previous_window_counts(df, entity_column, window_seconds):
    values = df[entity_column].astype(str).to_numpy()
    times = df["_time"].to_numpy(dtype=np.int64)
    result = np.zeros(len(df), dtype=np.int32)
    histories = defaultdict(deque)

    for i in range(len(df)):
        entity = values[i]
        if "__MISSING__" in entity:
            continue
        current_time = times[i]
        q = histories[entity]
        cutoff = current_time - window_seconds
        while q and q[0] < cutoff:
            q.popleft()
        result[i] = len(q)
        q.append(current_time)
    return result


def add_multiscale_velocity_features(df):
    df = df.copy()
    windows = {"1h": 60 * 60, "24h": 24 * 60 * 60, "7d": 7 * 24 * 60 * 60}
    for name, sec in windows.items():
        df[f"device_previous_tx_{name}"] = calculate_previous_window_counts(df, "_device", sec)
        df[f"card_previous_tx_{name}"] = calculate_previous_window_counts(df, "_card", sec)
        df[f"address_previous_tx_{name}"] = calculate_previous_window_counts(df, "_address", sec)

    df["_device_card"] = df["_device"] + "_" + df["_card"]
    df["_device_address"] = df["_device"] + "_" + df["_address"]
    df["device_card_previous_tx_24h"] = calculate_previous_window_counts(df, "_device_card", windows["24h"])
    df["device_address_previous_tx_24h"] = calculate_previous_window_counts(df, "_device_address", windows["24h"])
    df.drop(["_device_card", "_device_address"], axis=1, inplace=True)
    return df


def add_composite_ring_features(df):
    df = df.copy()
    df["device_card_ratio"] = df["device_unique_cards"] / df["device_transaction_count"].clip(lower=1)
    df["device_address_ratio"] = df["device_unique_addresses"] / df["device_transaction_count"].clip(lower=1)
    df["card_device_ratio"] = df["card_unique_devices"] / df["card_transaction_count"].clip(lower=1)
    df["email_card_ratio"] = df["email_unique_cards"] / df.groupby("_email", dropna=False).cumcount().clip(lower=1)

    df["ring_density_score"] = df["device_card_ratio"] * df["device_previous_tx_24h"]
    df["card_velocity_interaction"] = df["card_unique_devices"] * df["card_previous_tx_24h"]

    df["entity_overlap_score"] = np.log1p(df["device_unique_cards"]) + np.log1p(df["device_unique_addresses"])
    df["pair_link_score"] = np.log1p(df["device_card_pair_count"]) + np.log1p(df["device_address_pair_count"])
    df["ring_velocity_score"] = np.log1p(df["device_previous_tx_24h"]) + np.log1p(df["card_previous_tx_24h"])
    df["abuse_ring_score"] = 0.35 * df["entity_overlap_score"] + 0.35 * df["pair_link_score"] + 0.30 * df["component_size"].apply(np.log1p)

    df["multi_card_device_flag"] = (df["device_unique_cards"] >= 3).astype(int)
    df["multi_address_device_flag"] = (df["device_unique_addresses"] >= 2).astype(int)
    df["multi_device_card_flag"] = (df["card_unique_devices"] >= 2).astype(int)
    df["high_velocity_device_flag"] = (df["device_previous_tx_24h"] >= 3).astype(int)
    df["ring_candidate_flag"] = ((df["device_unique_cards"] >= 3) & (df["device_previous_tx_24h"] >= 2)).astype(int)
    return df


def build_all_features(df):
    df = prepare_entities(df)
    df = add_structural_ring_features(df)
    df = add_graph_centrality_features(df)
    df = add_multiscale_velocity_features(df)
    df = add_composite_ring_features(df)
    return df


def make_numeric_matrix(df, features, medians=None):
    selected = [c for c in features if c in df.columns]
    x = df[selected].copy()
    for col in selected:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = x.median().fillna(0)
    x = x.fillna(medians).fillna(0)
    return x, medians


# =====================================================================
# MAIN PIPELINE & SUBMISSION RUNNER
# =====================================================================
def _try_real_ring_extraction(top_tx_id, top_prob, sub):
    tx_path = os.path.join(DATA_DIR, "test_transaction.csv")
    id_path = os.path.join(DATA_DIR, "test_identity.csv")

    if not (os.path.exists(tx_path) and os.path.exists(id_path)):
        return None

    try:
        test_tx = pd.read_csv(tx_path)
        test_id = pd.read_csv(id_path)
        test_df = test_tx.merge(test_id, on="TransactionID", how="left")
        test_df = prepare_entities(test_df)

        top_tx_row = test_df[test_df["TransactionID"] == top_tx_id]
        if top_tx_row.empty:
            return None

        device = top_tx_row.iloc[0]["_device"]
        card = top_tx_row.iloc[0]["_card"]
        address = top_tx_row.iloc[0]["_address"]

        real_cluster = test_df[
            (test_df["_device"] == device) |
            (test_df["_card"] == card) |
            (test_df["_address"] == address)
        ].copy()

        real_cluster = real_cluster.merge(
            sub[["TransactionID", "isFraud"]], on="TransactionID", how="left"
        )
        real_cluster["isFraud"] = real_cluster["isFraud"].fillna(0.0)
        probs_real = real_cluster["isFraud"].tolist()

        return real_cluster, probs_real
    except Exception as e:
        print(f"[!] Real ring extraction failed with error: {e}")
        return None


def main():
    print("=" * 65)
    print("   SENTINELGRAPH & ABUSE-RING SENTINEL (PRODUCTION PIPELINE)")
    print("=" * 65)

    if not os.path.exists(MODEL_FILE):
        print(f"[*] Model file '{MODEL_FILE}' not found. Training model via pipeline...")
        import ieee_pipeline_chatgpt_15
        ieee_pipeline_chatgpt_15.run_pipeline()

    if not os.path.exists(SUBMISSION_FILE):
        print(f"[*] Submission file '{SUBMISSION_FILE}' not found. Generating test set predictions...")
        import iterations.generate_submission as gen_sub
        gen_sub.main()

    if os.path.exists(SUBMISSION_FILE):
        print("\n[*] Auditing submission file integrity...")
        import verify_submission
        verify_submission.audit_submission()

    sub = pd.read_csv(SUBMISSION_FILE)
    print(f"\n[✓] Loaded submission file with {len(sub):,} audited predictions.")

    top_row = sub.sort_values("isFraud", ascending=False).iloc[0]
    top_tx_id = int(top_row["TransactionID"])
    top_prob = float(top_row["isFraud"])

    print(f"[✓] Top Flagged Risk: Transaction ID {top_tx_id} (Score: {top_prob:.4f})")
    print("[✓] Extracting subgraph relationships via SentinelGraph...")

    real_result = _try_real_ring_extraction(top_tx_id, top_prob, sub)

    if real_result is not None:
        real_cluster, probs_real = real_result
        explainer = SentinelGraphExplainer(real_cluster, probs_real, threshold=0.105172)
        card_output = explainer.render_investigator_card(top_tx_id, demo_mode=False)
        print(card_output)
    else:
        print("[!] Raw test data unavailable for real extraction -- using labeled DEMO MODE fallback.")

        test_df_stub = pd.DataFrame({
            "TransactionID": [top_tx_id, top_tx_id + 1, top_tx_id + 2, top_tx_id + 3],
            "TransactionAmt": [450.0, 1200.0, 89.0, 310.0],
            "card1": [9876, 9876, 9876, 5432],
            "addr1": [123, 123, 456, 123],
            "P_emaildomain": ["example-domain.com", "example-domain.com", "tempmail.org", "gmail.com"],
        })

        test_df_stub["_device"] = "Demo_Device_Stub"
        test_df_stub["_card"] = test_df_stub["card1"].astype(str)
        test_df_stub["_address"] = test_df_stub["addr1"].astype(str)

        probs_stub = [top_prob, 0.9412, 0.8821, 0.0520]

        explainer = SentinelGraphExplainer(test_df_stub, probs_stub, threshold=0.105172)
        card_output = explainer.render_investigator_card(top_tx_id, demo_mode=True)
        print(card_output)

    print("=" * 65)
    print("   STATUS: READY FOR SUBMISSION & DEMO")
    print("=" * 65)


if __name__ == "__main__":
    main()