#!/usr/bin/env python3
"""
ABUSE-RING SENTINEL - MEMORY-OPTIMIZED TEST SET GENERATOR
=========================================================
Processes IEEE-CIS test files with low-memory overhead to prevent OOM kills.
"""

import os
import warnings
from collections import defaultdict, deque
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = "test_datasets/kaggle/ieee-fraud-detection"
MODEL_FILE = "ieee_abuse_ring_sentinel_v13.pkl"
SUBMISSION_FILE = "submission.csv"

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


def prepare_entities(df):
    df = df.copy()
    df["_device"] = df["DeviceInfo"].fillna(df.get("id_31", "__MISSING__")).fillna("__MISSING__").astype(str)
    df["_card"] = df["card1"].fillna("__MISSING__").astype(str)
    df["_address"] = df["addr1"].fillna("__MISSING__").astype(str)
    df["_email"] = df["P_emaildomain"].fillna(df.get("R_emaildomain", "__MISSING__")).fillna("__MISSING__").astype(str)
    df["_browser"] = df.get("id_31", pd.Series(["__MISSING__"] * len(df))).fillna("__MISSING__").astype(str)
    df["_time"] = pd.to_numeric(df["TransactionDT"], errors="coerce").fillna(0).astype(np.int64)
    return df


def add_graph_centrality_features(df):
    df = df.copy()
    device_card_counts = df.groupby("_device")["_card"].transform("nunique")
    device_addr_counts = df.groupby("_device")["_address"].transform("nunique")
    df["device_degree_centrality"] = device_card_counts + device_addr_counts
    df["card_degree_centrality"] = df.groupby("_card")["_device"].transform("nunique")
    df["component_size"] = df.groupby(["_device", "_address"])["_card"].transform("count").astype(float)
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


def main():
    print("Loading test datasets (optimized memory layout)...")
    test_trans = pd.read_csv(os.path.join(DATA_DIR, "test_transaction.csv"), low_memory=True)
    test_ident_path = os.path.join(DATA_DIR, "test_identity.csv")
    if os.path.exists(test_ident_path):
        test_ident = pd.read_csv(test_ident_path, low_memory=True)
        test_df = test_trans.merge(test_ident, on="TransactionID", how="left", suffixes=("", "_identity"))
    else:
        test_df = test_trans
    del test_trans
    if os.path.exists(test_ident_path):
        del test_ident

    print("Loading train dataset subset for feature alignment...")
    train_trans = pd.read_csv(
        os.path.join(DATA_DIR, "train_transaction.csv"), 
        usecols=["TransactionID", "TransactionDT", "card1", "addr1", "P_emaildomain", "TransactionAmt"] + BASELINE_FEATURES, 
        low_memory=True
    )
    train_ident_path = os.path.join(DATA_DIR, "train_identity.csv")
    if os.path.exists(train_ident_path):
        # Read column names of train_identity to check for DeviceInfo safely
        ident_cols = pd.read_csv(train_ident_path, nrows=1).columns
        use_cols = ["TransactionID"]
        if "DeviceInfo" in ident_cols:
            use_cols.append("DeviceInfo")
        elif "id_31" in ident_cols:
            use_cols.append("id_31")
            
        train_ident = pd.read_csv(train_ident_path, usecols=use_cols, low_memory=True)
        train_df = train_trans.merge(train_ident, on="TransactionID", how="left", suffixes=("", "_identity"))
        del train_ident
    else:
        train_df = train_trans
    del train_trans

    train_df["__is_current"] = 0
    test_df["__is_current"] = 1

    print("Concatenating stream efficiently...")
    combined = pd.concat([train_df, test_df], ignore_index=True, copy=False)
    del train_df, test_df
    combined = combined.sort_values(["TransactionDT", "__is_current"], kind="mergesort").reset_index(drop=True)

    print("Processing structural features...")
    combined = prepare_entities(combined)
    combined = add_structural_ring_features(combined)
    combined = add_graph_centrality_features(combined)
    combined = add_multiscale_velocity_features(combined)
    combined = add_composite_ring_features(combined)

    train_subset = combined.loc[combined["__is_current"] == 0].copy()
    _, train_medians = make_numeric_matrix(train_subset, HYBRID_FEATURES)
    del train_subset

    test_features = combined.loc[combined["__is_current"] == 1].copy().reset_index(drop=True)
    transaction_ids = test_features["TransactionID"].copy()
    X_test, _ = make_numeric_matrix(test_features, HYBRID_FEATURES, train_medians)
    del combined, test_features

    print(f"Loading model from {MODEL_FILE}...")
    model = joblib.load(MODEL_FILE)

    print("Generating predictions...")
    probabilities = model.predict_proba(X_test)[:, 1]

    sub = pd.DataFrame({
        "TransactionID": transaction_ids,
        "isFraud": probabilities
    })

    sub.to_csv(SUBMISSION_FILE, index=False)
    print(f"Success! Submission generated: {SUBMISSION_FILE} ({len(sub):,} rows)")


if __name__ == "__main__":
    main()