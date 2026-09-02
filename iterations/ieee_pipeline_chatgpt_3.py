#!/usr/bin/env python3
"""
ABUSE-RING SENTINEL - VERSION 3
===============================

IEEE-CIS Fraud Detection

Changes in this version:
    1. Trains THREE models: Baseline, Pure Ring, Hybrid Sentinel.
    2. Pure Ring model uses ZERO transaction amounts, ZERO baseline 
       features, and ZERO historical fraud rates.
    3. Added strict coordination features (ratios, pair velocities).
    
GOAL:
    Prove that hybridizing transaction-level baseline with pure 
    network topology yields the lowest overall business cost while 
    maintaining high precision.
"""

import os
import warnings
from collections import defaultdict, deque

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
# CONFIG
# =====================================================================
DATA_DIR = "test_datasets/kaggle/ieee-fraud-detection"
TRAIN_TRANSACTION_FILE = "train_transaction.csv"
TRAIN_IDENTITY_FILE = "train_identity.csv"
MODEL_FILE = "ieee_abuse_ring_sentinel_v3.pkl"

RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

VELOCITY_WINDOW_SECONDS = 24 * 60 * 60

# ---------------------------------------------------------------------
# BUSINESS ASSUMPTION
# ---------------------------------------------------------------------
FALSE_POSITIVE_REVIEW_COST = 25.0
FRAUD_RATE_ALPHA = 20.0

# =====================================================================
# FEATURE LISTS
# =====================================================================

BASELINE_FEATURES = [
    "TransactionAmt",
    "card2", "card3", "card5",
    "addr2", "dist1",
    "C1", "C2", "C3", "C4", "C5",
]

# Strictly Network & Velocity (No historical labels, no amounts)
PURE_RING_FEATURES = [
    # Entity volume
    "device_transaction_count",
    "card_transaction_count",
    "address_transaction_count",

    # Entity diversity
    "device_unique_cards",
    "device_unique_addresses",
    "device_unique_emails",
    "device_unique_browsers",
    "card_unique_devices",
    "card_unique_addresses",
    "address_unique_devices",
    "address_unique_cards",

    # Pair relationships
    "device_card_pair_count",
    "device_address_pair_count",
    "card_address_pair_count",

    # Coordination Ratios (NEW)
    "device_card_ratio",
    "device_address_ratio",
    "card_device_ratio",

    # Entity Velocity
    "device_previous_tx_24h",
    "card_previous_tx_24h",
    "address_previous_tx_24h",
    
    # Pair Velocity (NEW)
    "device_card_previous_tx_24h",
    "device_address_previous_tx_24h",

    # Composite coordination signals
    "entity_overlap_score",
    "pair_link_score",
    "ring_velocity_score",
    "abuse_ring_score",

    # Explainability flags
    "multi_card_device_flag",
    "multi_address_device_flag",
    "multi_device_card_flag",
    "high_velocity_device_flag",
    "ring_candidate_flag",
]

HISTORICAL_FRAUD_FEATURES = [
    "device_historical_fraud_rate",
    "card_historical_fraud_rate",
    "address_historical_fraud_rate",
    "email_historical_fraud_rate",
    "historical_entity_risk",
]

# Hybrid uses Baseline + Pure Ring + Fraud Rates
HYBRID_FEATURES = list(dict.fromkeys(
    BASELINE_FEATURES + PURE_RING_FEATURES + HISTORICAL_FRAUD_FEATURES
))

# =====================================================================
# LOAD DATA
# =====================================================================
def load_data():
    transaction_path = os.path.join(DATA_DIR, TRAIN_TRANSACTION_FILE)
    identity_path = os.path.join(DATA_DIR, TRAIN_IDENTITY_FILE)

    if not os.path.exists(transaction_path):
        raise FileNotFoundError(f"Transaction file not found:\n{transaction_path}")

    print(f"Loading transaction data:\n{transaction_path}")
    trans = pd.read_csv(transaction_path)
    print(f"Transactions: {len(trans):,} rows × {trans.shape[1]} columns")

    if os.path.exists(identity_path):
        print(f"Loading identity data:\n{identity_path}")
        ident = pd.read_csv(identity_path)
        print(f"Identity: {len(ident):,} rows × {ident.shape[1]} columns")
        df = trans.merge(ident, on="TransactionID", how="left", suffixes=("", "_identity"))
    else:
        print("Identity file not found.")
        df = trans

    print(f"Merged dataset: {len(df):,} rows × {df.shape[1]} columns")
    return df


# =====================================================================
# PREPARE ENTITY COLUMNS
# =====================================================================
def prepare_entities(df):
    df = df.copy()

    # Device
    if "DeviceInfo" in df.columns:
        df["_device"] = df["DeviceInfo"].fillna("__MISSING__").astype(str)
    elif "id_31" in df.columns:
        df["_device"] = df["id_31"].fillna("__MISSING__").astype(str)
    else:
        df["_device"] = "__MISSING__"

    # Card
    if "card1" in df.columns:
        df["_card"] = df["card1"].fillna("__MISSING__").astype(str)
    else:
        df["_card"] = "__MISSING__"

    # Address
    if "addr1" in df.columns:
        df["_address"] = df["addr1"].fillna("__MISSING__").astype(str)
    else:
        df["_address"] = "__MISSING__"

    # Email
    if "P_emaildomain" in df.columns:
        df["_email"] = df["P_emaildomain"].fillna("__MISSING__").astype(str)
    elif "R_emaildomain" in df.columns:
        df["_email"] = df["R_emaildomain"].fillna("__MISSING__").astype(str)
    else:
        df["_email"] = "__MISSING__"

    # Browser
    if "id_31" in df.columns:
        df["_browser"] = df["id_31"].fillna("__MISSING__").astype(str)
    else:
        df["_browser"] = "__MISSING__"

    df["_time"] = pd.to_numeric(df["TransactionDT"], errors="coerce").fillna(0).astype(np.int64)

    return df


# =====================================================================
# STRUCTURAL / RELATIONAL FEATURES
# =====================================================================
def add_structural_ring_features(df):
    df = df.copy()
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # Basic counts
    df["device_transaction_count"] = df.groupby("_device", dropna=False).cumcount().astype(float)
    df["card_transaction_count"] = df.groupby("_card", dropna=False).cumcount().astype(float)
    df["address_transaction_count"] = df.groupby("_address", dropna=False).cumcount().astype(float)

    # Pair occurrence counts
    df["device_card_pair_count"] = df.groupby(["_device", "_card"], dropna=False).cumcount().astype(float)
    df["device_address_pair_count"] = df.groupby(["_device", "_address"], dropna=False).cumcount().astype(float)
    df["card_address_pair_count"] = df.groupby(["_card", "_address"], dropna=False).cumcount().astype(float)

    # Uniques Helper
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

    # Handle Missing
    df.loc[df["_device"] == "__MISSING__", ["device_transaction_count", "device_unique_cards", "device_unique_addresses", "device_unique_emails", "device_unique_browsers"]] = 0
    df.loc[df["_card"] == "__MISSING__", ["card_transaction_count", "card_unique_devices", "card_unique_addresses"]] = 0
    df.loc[df["_address"] == "__MISSING__", ["address_transaction_count", "address_unique_devices", "address_unique_cards"]] = 0

    return df


# =====================================================================
# VELOCITY FEATURES
# =====================================================================
def calculate_previous_window_counts(df, entity_column, window_seconds=24 * 60 * 60):
    values = df[entity_column].astype(str).to_numpy()
    times = df["_time"].to_numpy(dtype=np.int64)
    result = np.zeros(len(df), dtype=np.int32)
    histories = defaultdict(deque)

    for i in range(len(df)):
        entity = values[i]
        if entity == "__MISSING__" or entity.endswith("__MISSING__") or entity.startswith("__MISSING__"):
            result[i] = 0
            continue
        current_time = times[i]
        q = histories[entity]
        cutoff = current_time - window_seconds

        while q and q[0] < cutoff:
            q.popleft()
        
        result[i] = len(q)
        q.append(current_time)
    
    return result


def add_velocity_features(df):
    df = df.copy()

    # Standard Entity Velocity
    df["device_previous_tx_24h"] = calculate_previous_window_counts(df, "_device", VELOCITY_WINDOW_SECONDS)
    df["card_previous_tx_24h"] = calculate_previous_window_counts(df, "_card", VELOCITY_WINDOW_SECONDS)
    df["address_previous_tx_24h"] = calculate_previous_window_counts(df, "_address", VELOCITY_WINDOW_SECONDS)

    # Pair Velocities (NEW)
    df["_device_card"] = df["_device"] + "_" + df["_card"]
    df["_device_address"] = df["_device"] + "_" + df["_address"]

    df["device_card_previous_tx_24h"] = calculate_previous_window_counts(df, "_device_card", VELOCITY_WINDOW_SECONDS)
    df["device_address_previous_tx_24h"] = calculate_previous_window_counts(df, "_device_address", VELOCITY_WINDOW_SECONDS)

    # Clean up temp cols
    df.drop(["_device_card", "_device_address"], axis=1, inplace=True)

    return df


# =====================================================================
# COMPOSITE RING FEATURES
# =====================================================================
def add_composite_ring_features(df):
    df = df.copy()

    # Ratios (NEW)
    df["device_card_ratio"] = df["device_unique_cards"] / df["device_transaction_count"].clip(lower=1)
    df["device_address_ratio"] = df["device_unique_addresses"] / df["device_transaction_count"].clip(lower=1)
    df["card_device_ratio"] = df["card_unique_devices"] / df["card_transaction_count"].clip(lower=1)

    df["entity_overlap_score"] = (
        np.log1p(df["device_unique_cards"]) +
        np.log1p(df["device_unique_addresses"]) +
        np.log1p(df["card_unique_devices"]) +
        np.log1p(df["address_unique_devices"])
    )

    df["pair_link_score"] = (
        np.log1p(df["device_card_pair_count"]) +
        np.log1p(df["device_address_pair_count"]) +
        np.log1p(df["card_address_pair_count"])
    )

    df["ring_velocity_score"] = (
        np.log1p(df["device_previous_tx_24h"]) +
        np.log1p(df["card_previous_tx_24h"]) +
        np.log1p(df["device_card_previous_tx_24h"])  # Added pair velocity
    )

    # Fraud risk separated out
    if "device_historical_fraud_rate" in df.columns:
        df["historical_entity_risk"] = (
            0.40 * df["device_historical_fraud_rate"] +
            0.35 * df["card_historical_fraud_rate"] +
            0.25 * df["address_historical_fraud_rate"]
        )

    # Pure Ring Score (Removed historical fraud risk)
    df["abuse_ring_score"] = (
        0.40 * df["entity_overlap_score"] +
        0.30 * df["pair_link_score"] +
        0.30 * df["ring_velocity_score"]
    )

    # Explainability flags
    df["multi_card_device_flag"] = (df["device_unique_cards"] >= 3).astype(int)
    df["multi_address_device_flag"] = (df["device_unique_addresses"] >= 2).astype(int)
    df["multi_device_card_flag"] = (df["card_unique_devices"] >= 2).astype(int)
    df["high_velocity_device_flag"] = (df["device_previous_tx_24h"] >= 3).astype(int)
    
    df["ring_candidate_flag"] = (
        (df["device_unique_cards"] >= 3) &
        (df["device_unique_addresses"] >= 2) &
        (df["device_previous_tx_24h"] >= 2)
    ).astype(int)

    return df


# =====================================================================
# HISTORICAL FRAUD RATES
# =====================================================================
def add_training_historical_fraud_rates(df):
    df = df.copy()
    global_rate = float(df["isFraud"].mean())

    for entity, output_name in [
        ("_device", "device_historical_fraud_rate"),
        ("_card", "card_historical_fraud_rate"),
        ("_address", "address_historical_fraud_rate"),
        ("_email", "email_historical_fraud_rate"),
    ]:
        prior_count = df.groupby(entity, dropna=False).cumcount()
        prior_fraud_sum = df["isFraud"].groupby(df[entity], dropna=False).cumsum() - df["isFraud"]
        
        smoothed_rate = (prior_fraud_sum + FRAUD_RATE_ALPHA * global_rate) / (prior_count + FRAUD_RATE_ALPHA)
        
        missing = df[entity] == "__MISSING__"
        smoothed_rate = smoothed_rate.fillna(global_rate).astype(float)
        smoothed_rate.loc[missing] = global_rate
        df[output_name] = smoothed_rate

    return df

def add_frozen_historical_fraud_rates(current_df, history_df):
    current = current_df.copy()
    history = history_df.copy()
    global_rate = float(history["isFraud"].mean())

    for entity, output_name in [
        ("_device", "device_historical_fraud_rate"),
        ("_card", "card_historical_fraud_rate"),
        ("_address", "address_historical_fraud_rate"),
        ("_email", "email_historical_fraud_rate"),
    ]:
        hist = history[history[entity] != "__MISSING__"]
        if hist.empty:
            current[output_name] = global_rate
            continue

        grouped = hist.groupby(entity)["isFraud"].agg(fraud_sum="sum", count="count")
        grouped["rate"] = (grouped["fraud_sum"] + FRAUD_RATE_ALPHA * global_rate) / (grouped["count"] + FRAUD_RATE_ALPHA)
        
        mapping = grouped["rate"].to_dict()
        current[output_name] = current[entity].map(mapping).fillna(global_rate).astype(float)
        current.loc[current[entity] == "__MISSING__", output_name] = global_rate

    return current


# =====================================================================
# BUILD FEATURES
# =====================================================================
def build_train_features(train_df):
    x = prepare_entities(train_df)
    x = add_structural_ring_features(x)
    x = add_velocity_features(x)
    x = add_training_historical_fraud_rates(x)
    x = add_composite_ring_features(x)
    return x

def build_future_features(current_df, history_df):
    history = prepare_entities(history_df)
    current = prepare_entities(current_df)

    history["__is_current"] = 0
    current["__is_current"] = 1

    combined = pd.concat([history, current], ignore_index=True)
    combined = combined.sort_values(["TransactionDT", "__is_current"], kind="mergesort").reset_index(drop=True)

    combined = add_structural_ring_features(combined)
    combined = add_velocity_features(combined)

    current_mask = combined["__is_current"] == 1
    current_features = combined.loc[current_mask].copy().reset_index(drop=True)

    current_features = add_frozen_historical_fraud_rates(current_features, history)
    current_features = add_composite_ring_features(current_features)

    return current_features


# =====================================================================
# MODELING UTILS
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
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# =====================================================================
# COST & THRESHOLD
# =====================================================================
def calculate_cost(y_true, y_pred, transaction_amount):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    amount = np.asarray(transaction_amount, dtype=float)

    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)

    fp, fn = int(fp_mask.sum()), int(fn_mask.sum())
    fp_cost = fp * FALSE_POSITIVE_REVIEW_COST
    fn_exposure = float(amount[fn_mask].sum())

    return {
        "fp": fp,
        "fn": fn,
        "fp_cost": fp_cost,
        "fn_exposure": fn_exposure,
        "total_cost": fp_cost + fn_exposure,
    }

def find_best_threshold(y_val, probability, transaction_amount, min_recall=0.0):
    candidate_thresholds = np.unique(np.concatenate([
        np.linspace(0.01, 0.99, 200),
        np.quantile(probability, np.linspace(0.01, 0.99, 100)),
    ]))

    rows = []
    for threshold in candidate_thresholds:
        prediction = (probability >= threshold).astype(int)
        cost = calculate_cost(y_val, prediction, transaction_amount)
        recall = recall_score(y_val, prediction, zero_division=0)
        
        rows.append({
            "threshold": float(threshold),
            "recall": float(recall),
            **cost
        })

    table = pd.DataFrame(rows)
    
    # Enforce minimum recall floor if specified
    if min_recall > 0.0:
        valid_mask = table["recall"] >= min_recall
        if valid_mask.any():
            table = table[valid_mask]
        else:
            print(f"  [Warning] Could not reach min_recall={min_recall:.4f}. Using max available.")
            max_rec = table["recall"].max()
            table = table[table["recall"] == max_rec]

    # Sort the remaining valid thresholds by total cost
    table = table.sort_values("total_cost").reset_index(drop=True)
    return float(table.iloc[0]["threshold"]), table

# =====================================================================
# EVALUATE
# =====================================================================
def evaluate(model, X_test, y_test, amount_test, threshold, name):
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= threshold).astype(int)

    precision = precision_score(y_test, prediction, zero_division=0)
    recall = recall_score(y_test, prediction, zero_division=0)
    f1 = f1_score(y_test, prediction, zero_division=0)
    pr_auc = average_precision_score(y_test, probability)
    
    cost = calculate_cost(y_test, prediction, amount_test)

    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    print(f"Threshold:              {threshold:.6f}")
    print(f"Precision:              {precision:.4f}")
    print(f"Recall:                 {recall:.4f}")
    print(f"F1:                     {f1:.4f}")
    print(f"PR-AUC:                 {pr_auc:.4f}")
    
    print("\nBUSINESS COST")
    print(f"False positives:        {cost['fp']:,}")
    print(f"False negatives:        {cost['fn']:,}")
    print(f"FP review cost:         ₹{cost['fp_cost']:,.2f}")
    print(f"Missed-fraud exposure:  ₹{cost['fn_exposure']:,.2f}")
    print(f"TOTAL ESTIMATED COST:   ₹{cost['total_cost']:,.2f}")

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        **cost,
        "probability": probability,
        "prediction": prediction,
    }


def print_multi_comparison(baseline, pure, hybrid):
    print(f"\n{'=' * 70}\nMODEL COMPARISON SUMMARY\n{'=' * 70}")
    
    models = {
        "Baseline Model": baseline,
        "Pure Ring Model": pure,
        "Hybrid Sentinel": hybrid
    }
    
    for name, res in models.items():
        print(f"{name:<20} | Cost: ₹{res['total_cost']:>10,.2f} | Precision: {res['precision']:>6.2%} | Recall: {res['recall']:>6.2%}")
    
    cost_change = baseline["total_cost"] - hybrid["total_cost"]
    
    print("\nHYBRID SENTINEL VS BASELINE")
    print(f"Cost difference:        ₹{cost_change:,.2f}")
    
    if cost_change > 0:
        improvement_pct = (cost_change / baseline["total_cost"]) * 100
        print(f"Cost improvement:       {improvement_pct:.2f}%")
        print("\nRESULT: HYBRID SENTINEL REDUCES ESTIMATED COST ✓")
    else:
        print("\nRESULT: HYBRID SENTINEL INCREASES ESTIMATED COST ✗")


# =====================================================================
# MAIN
# =====================================================================
def run_pipeline():
    print(f"\n{'=' * 70}\nABUSE-RING SENTINEL V3\n{'=' * 70}")

    df = load_data()
    df = prepare_entities(df).sort_values("TransactionDT").reset_index(drop=True)

    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    validation_end = int(n * (TRAIN_RATIO + VALIDATION_RATIO))

    train_df = df.iloc[:train_end].copy()
    validation_df = df.iloc[train_end:validation_end].copy()
    test_df = df.iloc[validation_end:].copy()

    print(f"\n{'=' * 70}\nTEMPORAL SPLIT\n{'=' * 70}")
    print(f"TRAIN:       {len(train_df):,}")
    print(f"VALIDATION:  {len(validation_df):,}")
    print(f"TEST:        {len(test_df):,}")

    print("\nBuilding TRAIN features...")
    train_features = build_train_features(train_df)

    print("\nBuilding VALIDATION features...")
    validation_features = build_future_features(validation_df, train_df)

    print("\nBuilding TEST features...")
    test_history = pd.concat([train_df, validation_df], ignore_index=True)
    test_features = build_future_features(test_df, test_history)

    y_train = train_features["isFraud"].astype(int)
    y_validation = validation_features["isFraud"].astype(int)
    y_test = test_features["isFraud"].astype(int)

    val_amt = validation_features["TransactionAmt"].astype(float)
    test_amt = test_features["TransactionAmt"].astype(float)

# -------------------------------------------------------------
    # TRAIN: BASELINE
    # -------------------------------------------------------------
    print("\nTraining BASELINE...")
    X_tr_b, med_b = make_numeric_matrix(train_features, BASELINE_FEATURES)
    X_va_b, _ = make_numeric_matrix(validation_features, BASELINE_FEATURES, med_b)
    X_te_b, _ = make_numeric_matrix(test_features, BASELINE_FEATURES, med_b)

    model_b = create_model().fit(X_tr_b, y_train)
    thresh_b, table_b = find_best_threshold(y_validation, model_b.predict_proba(X_va_b)[:, 1], val_amt)
    
    # Dynamically extract baseline's chosen recall to set a floor for the Hybrid Sentinel
    baseline_val_recall = table_b.iloc[0]["recall"]
    print(f"  -> Baseline validation recall: {baseline_val_recall:.4f}")

    # -------------------------------------------------------------
    # TRAIN: PURE RING
    # -------------------------------------------------------------
    print("\nTraining PURE RING MODEL...")
    X_tr_p, med_p = make_numeric_matrix(train_features, PURE_RING_FEATURES)
    X_va_p, _ = make_numeric_matrix(validation_features, PURE_RING_FEATURES, med_p)
    X_te_p, _ = make_numeric_matrix(test_features, PURE_RING_FEATURES, med_p)

    model_p = create_model().fit(X_tr_p, y_train)
    thresh_p, _ = find_best_threshold(y_validation, model_p.predict_proba(X_va_p)[:, 1], val_amt, min_recall=0.0)

    # -------------------------------------------------------------
    # TRAIN: HYBRID SENTINEL
    # -------------------------------------------------------------
    print("\nTraining HYBRID SENTINEL...")
    X_tr_h, med_h = make_numeric_matrix(train_features, HYBRID_FEATURES)
    X_va_h, _ = make_numeric_matrix(validation_features, HYBRID_FEATURES, med_h)
    X_te_h, _ = make_numeric_matrix(test_features, HYBRID_FEATURES, med_h)

    model_h = create_model().fit(X_tr_h, y_train)
    
    # Force Hybrid model to maintain at least 96% of the Baseline's recall
    target_recall = baseline_val_recall * 0.96
    print(f"  -> Forcing Hybrid to maintain min_recall >= {target_recall:.4f}")
    thresh_h, _ = find_best_threshold(y_validation, model_h.predict_proba(X_va_h)[:, 1], val_amt, min_recall=target_recall)

    # -------------------------------------------------------------
    # EVALUATE (No changes needed below this line)
    # -------------------------------------------------------------

    res_b = evaluate(model_b, X_te_b, y_test, test_amt, thresh_b, "1. BASELINE MODEL")
    res_p = evaluate(model_p, X_te_p, y_test, test_amt, thresh_p, "2. PURE RING MODEL")
    res_h = evaluate(model_h, X_te_h, y_test, test_amt, thresh_h, "3. HYBRID SENTINEL")

    print_multi_comparison(res_b, res_p, res_h)

    # Output Top Features for Hybrid
    print(f"\n{'=' * 70}\nTOP HYBRID SENTINEL FEATURES\n{'=' * 70}")
    importance = pd.DataFrame({"feature": HYBRID_FEATURES, "importance": model_h.feature_importances_})
    importance = importance.sort_values("importance", ascending=False).head(30)
    for _, row in importance.iterrows():
        print(f"{row['feature']:<40} {row['importance']:.6f}")


if __name__ == "__main__":
    run_pipeline()