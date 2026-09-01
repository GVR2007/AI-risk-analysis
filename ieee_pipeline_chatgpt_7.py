#!/usr/bin/env python3
"""
ABUSE-RING SENTINEL - VERSION 7 (Soft-Boost & Cost-Optimized)
=============================================================
Upgrades:
    1. Soft-Confidence Multiplier: Replaces the rigid hard filter with a probability boost for network evidence.
    2. Restored Grid Search: Uses Version 5's pure financial cost-curve minimization to protect net savings.
    3. Optimized Precision Lift: Safely elevates precision while protecting recall and merchant margin.
"""

import os
import warnings
from collections import defaultdict, deque

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIG & ECONOMICS
# =====================================================================
DATA_DIR = "test_datasets/kaggle/ieee-fraud-detection"
TRAIN_TRANSACTION_FILE = "train_transaction.csv"
TRAIN_IDENTITY_FILE = "train_identity.csv"
MODEL_FILE = "ieee_abuse_ring_sentinel_v7.pkl"

RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

FALSE_POSITIVE_REVIEW_COST = 25.0
CHARGEBACK_PENALTY_FEE = 1500.0
FRAUD_RATE_ALPHA = 20.0

BASELINE_FEATURES = [
    "TransactionAmt", "card2", "card3", "card5", "addr2", "dist1", "C1", "C2", "C3", "C4", "C5",
]

PURE_RING_FEATURES = [
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

HISTORICAL_FRAUD_FEATURES = [
    "device_historical_fraud_rate", "card_historical_fraud_rate",
    "address_historical_fraud_rate", "email_historical_fraud_rate", "historical_entity_risk",
]

HYBRID_FEATURES = list(dict.fromkeys(BASELINE_FEATURES + PURE_RING_FEATURES + HISTORICAL_FRAUD_FEATURES))


# =====================================================================
# DATA & ENTITY PREPARATION
# =====================================================================
def load_data():
    trans = pd.read_csv(os.path.join(DATA_DIR, TRAIN_TRANSACTION_FILE))
    ident_path = os.path.join(DATA_DIR, TRAIN_IDENTITY_FILE)
    if os.path.exists(ident_path):
        ident = pd.read_csv(ident_path)
        df = trans.merge(ident, on="TransactionID", how="left", suffixes=("", "_identity"))
    else:
        df = trans
    return df


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


def add_training_historical_fraud_rates(df):
    df = df.copy()
    global_rate = float(df["isFraud"].mean())
    for entity, output_name in [("_device", "device_historical_fraud_rate"), ("_card", "card_historical_fraud_rate"), ("_address", "address_historical_fraud_rate"), ("_email", "email_historical_fraud_rate")]:
        prior_count = df.groupby(entity, dropna=False).cumcount()
        prior_fraud_sum = df["isFraud"].groupby(df[entity], dropna=False).cumsum() - df["isFraud"]
        df[output_name] = ((prior_fraud_sum + FRAUD_RATE_ALPHA * global_rate) / (prior_count + FRAUD_RATE_ALPHA)).fillna(global_rate).astype(float)
    return df


def add_frozen_historical_fraud_rates(current_df, history_df):
    current = current_df.copy()
    history = history_df.copy()
    global_rate = float(history["isFraud"].mean())
    for entity, output_name in [("_device", "device_historical_fraud_rate"), ("_card", "card_historical_fraud_rate"), ("_address", "address_historical_fraud_rate"), ("_email", "email_historical_fraud_rate")]:
        hist = history[history[entity] != "__MISSING__"]
        if hist.empty:
            current[output_name] = global_rate
            continue
        grouped = hist.groupby(entity)["isFraud"].agg(fraud_sum="sum", count="count")
        grouped["rate"] = (grouped["fraud_sum"] + FRAUD_RATE_ALPHA * global_rate) / (grouped["count"] + FRAUD_RATE_ALPHA)
        current[output_name] = current[entity].map(grouped["rate"].to_dict()).fillna(global_rate).astype(float)
    return current


def build_train_features(train_df):
    x = prepare_entities(train_df)
    x = add_structural_ring_features(x)
    x = add_graph_centrality_features(x)
    x = add_multiscale_velocity_features(x)
    x = add_training_historical_fraud_rates(x)
    x = add_composite_ring_features(x)
    return x


def build_future_features(current_df, history_df):
    history = prepare_entities(history_df)
    current = prepare_entities(current_df)
    history["__is_current"] = 0
    current["__is_current"] = 1

    combined = pd.concat([history, current], ignore_index=True).sort_values(["TransactionDT", "__is_current"], kind="mergesort").reset_index(drop=True)
    combined = add_structural_ring_features(combined)
    combined = add_graph_centrality_features(combined)
    combined = add_multiscale_velocity_features(combined)

    current_features = combined.loc[combined["__is_current"] == 1].copy().reset_index(drop=True)
    current_features = add_frozen_historical_fraud_rates(current_features, history)
    current_features = add_composite_ring_features(current_features)
    return current_features


# =====================================================================
# MODELING, SOFT CONFIDENCE BOOST & GRID SEARCH
# =====================================================================
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


def create_model():
    return LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=9,
        num_leaves=128,
        scale_pos_weight=4,  # Balanced balance point for precision and recall
        random_state=RANDOM_STATE,
        n_jobs=-1,
        importance_type='gain',
        verbose=-1
    )


def calculate_cost_with_soft_boost(y_true, probability, transaction_amount, threshold, features_df=None):
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(probability, dtype=float)
    amount = np.asarray(transaction_amount, dtype=float)

    # Soft-Confidence Multiplier (NEW): Apply a probability boost if network evidence exists
    if features_df is not None and "device_unique_cards" in features_df.columns:
        dev_cards = features_df["device_unique_cards"].to_numpy()
        comp_size = features_df["component_size"].to_numpy() if "component_size" in features_df.columns else np.zeros_like(dev_cards)
        
        # Soft multiplier: 1.15x boost for coordinated nodes, 0.95x dampener for isolated single-card nodes
        ring_boost = np.where((dev_cards >= 2) | (comp_size >= 2), 1.15, 0.95)
        prob = np.clip(prob * ring_boost, 0.0, 1.0)

    y_pred = (prob >= threshold).astype(int)

    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)

    fp, fn = int(fp_mask.sum()), int(fn_mask.sum())
    fp_cost = fp * FALSE_POSITIVE_REVIEW_COST
    fn_exposure = float(amount[fn_mask].sum()) + (fn * CHARGEBACK_PENALTY_FEE)

    return {
        "fp": fp, "fn": fn, "fp_cost": fp_cost, "fn_exposure": fn_exposure,
        "total_cost": fp_cost + fn_exposure, "adjusted_preds": y_pred, "adjusted_probs": prob
    }


def find_best_threshold_grid_search(y_val, probability, transaction_amount, features_df):
    candidate_thresholds = np.linspace(0.05, 0.95, 100)
    best_cost = float("inf")
    best_thresh = 0.5

    for thresh in candidate_thresholds:
        res = calculate_cost_with_soft_boost(y_val, probability, transaction_amount, thresh, features_df)
        if res["total_cost"] < best_cost:
            best_cost = res["total_cost"]
            best_thresh = thresh

    return float(best_thresh)


def evaluate(model, X_test, y_test, amount_test, features_df, threshold, name):
    probability = model.predict_proba(X_test)[:, 1]

    cost_res = calculate_cost_with_soft_boost(y_test, probability, amount_test, threshold, features_df)
    adj_pred = cost_res["adjusted_preds"]
    adj_prob = cost_res["adjusted_probs"]

    precision = precision_score(y_test, adj_pred, zero_division=0)
    recall = recall_score(y_test, adj_pred, zero_division=0)
    f1 = f1_score(y_test, adj_pred, zero_division=0)

    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    print(f"Threshold:              {threshold:.6f}")
    print(f"Precision:              {precision:.4f}")
    print(f"Recall:                 {recall:.4f}")
    print(f"F1:                     {f1:.4f}")
    print(f"TOTAL ESTIMATED COST:   ₹{cost_res['total_cost']:,.2f}")

    return {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1, **cost_res}


def run_pipeline():
    print(f"\n{'=' * 70}\nABUSE-RING SENTINEL V7 (SOFT-BOOST)\n{'=' * 70}")
    df = load_data().sort_values("TransactionDT").reset_index(drop=True)

    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    validation_end = int(n * (TRAIN_RATIO + VALIDATION_RATIO))

    train_df, validation_df, test_df = df.iloc[:train_end], df.iloc[train_end:validation_end], df.iloc[validation_end:]

    print("Building features & graphs...")
    train_features = build_train_features(train_df)
    validation_features = build_future_features(validation_df, train_df)
    test_features = build_future_features(test_df, pd.concat([train_df, validation_df]))

    y_train = train_features["isFraud"].astype(int)
    y_val = validation_features["isFraud"].astype(int)
    y_test = test_features["isFraud"].astype(int)
    val_amt = validation_features["TransactionAmt"].astype(float)
    test_amt = test_features["TransactionAmt"].astype(float)

    # Baseline
    print("Training Baseline...")
    X_tr_b, med_b = make_numeric_matrix(train_features, BASELINE_FEATURES)
    X_va_b, _ = make_numeric_matrix(validation_features, BASELINE_FEATURES, med_b)
    X_te_b, _ = make_numeric_matrix(test_features, BASELINE_FEATURES, med_b)

    model_b = create_model().fit(X_tr_b, y_train)
    val_probs_b = model_b.predict_proba(X_va_b)[:, 1]
    thresh_b = find_best_threshold_grid_search(y_val, val_probs_b, val_amt, None)

    # Hybrid Sentinel
    print("Training Hybrid Sentinel...")
    X_tr_h, med_h = make_numeric_matrix(train_features, HYBRID_FEATURES)
    X_va_h, _ = make_numeric_matrix(validation_features, HYBRID_FEATURES, med_h)
    X_te_h, _ = make_numeric_matrix(test_features, HYBRID_FEATURES, med_h)

    model_h = create_model().fit(X_tr_h, y_train)
    val_probs_h = model_h.predict_proba(X_va_h)[:, 1]
    thresh_h = find_best_threshold_grid_search(y_val, val_probs_h, val_amt, validation_features)

    res_b = evaluate(model_b, X_te_b, y_test, test_amt, None, thresh_b, "1. BASELINE MODEL")
    res_h = evaluate(model_h, X_te_h, y_test, test_amt, test_features, thresh_h, "2. HYBRID SENTINEL")

    print(f"\n{'=' * 70}\nFINAL COMPARISON\n{'=' * 70}")
    print(f"Baseline Cost: ₹{res_b['total_cost']:,.2f} | Precision: {res_b['precision']:.2%} | Recall: {res_b['recall']:.2%}")
    print(f"Sentinel Cost: ₹{res_h['total_cost']:,.2f} | Precision: {res_h['precision']:.2%} | Recall: {res_h['recall']:.2%}")
    diff = res_b['total_cost'] - res_h['total_cost']
    print(f"Net Savings:   ₹{diff:,.2f} ({ (diff/res_b['total_cost'])*100:.2f}% improvement)")


if __name__ == "__main__":
    run_pipeline()