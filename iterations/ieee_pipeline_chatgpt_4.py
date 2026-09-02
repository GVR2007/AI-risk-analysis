#!/usr/bin/env python3
"""
ABUSE-RING SENTINEL - VERSION 4 (LightGBM + Multi-Scale Velocity + Chargeback Economics)
=====================================================================================

IEEE-CIS Fraud Detection

Upgrades:
    1. Replaces Random Forest with LightGBM for gradient-boosted precision and recall.
    2. Adds multi-scale temporal velocity windows (1h, 24h, 7d) to catch burst and slow-bleed attacks.
    3. Adds email-card relationship and ratio features.
    4. Incorporates real-world chargeback penalty fees into the business cost function.
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
# CONFIG & BUSINESS ECONOMICS
# =====================================================================
DATA_DIR = "test_datasets/kaggle/ieee-fraud-detection"
TRAIN_TRANSACTION_FILE = "train_transaction.csv"
TRAIN_IDENTITY_FILE = "train_identity.csv"
MODEL_FILE = "ieee_abuse_ring_sentinel_v4.pkl"

RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

FALSE_POSITIVE_REVIEW_COST = 25.0
CHARGEBACK_PENALTY_FEE = 1500.0  # Real-world merchant fee per chargeback
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

PURE_RING_FEATURES = [
    # Volume & Uniques
    "device_transaction_count",
    "card_transaction_count",
    "address_transaction_count",
    "device_unique_cards",
    "device_unique_addresses",
    "device_unique_emails",
    "device_unique_browsers",
    "card_unique_devices",
    "card_unique_addresses",
    "address_unique_devices",
    "address_unique_cards",
    "email_unique_cards",

    # Pair relationships
    "device_card_pair_count",
    "device_address_pair_count",
    "card_address_pair_count",

    # Coordination Ratios
    "device_card_ratio",
    "device_address_ratio",
    "card_device_ratio",
    "email_card_ratio",

    # Multi-Scale Velocities
    "device_previous_tx_1h",
    "device_previous_tx_24h",
    "device_previous_tx_7d",
    "card_previous_tx_1h",
    "card_previous_tx_24h",
    "card_previous_tx_7d",
    "address_previous_tx_1h",
    "address_previous_tx_24h",
    "address_previous_tx_7d",
    "device_card_previous_tx_24h",
    "device_address_previous_tx_24h",

    # Composite Scores & Flags
    "entity_overlap_score",
    "pair_link_score",
    "ring_velocity_score",
    "abuse_ring_score",
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

HYBRID_FEATURES = list(dict.fromkeys(
    BASELINE_FEATURES + PURE_RING_FEATURES + HISTORICAL_FRAUD_FEATURES
))


# =====================================================================
# DATA LOADING & ENTITIES
# =====================================================================
def load_data():
    transaction_path = os.path.join(DATA_DIR, TRAIN_TRANSACTION_FILE)
    identity_path = os.path.join(DATA_DIR, TRAIN_IDENTITY_FILE)

    print(f"Loading transaction data:\n{transaction_path}")
    trans = pd.read_csv(transaction_path)

    if os.path.exists(identity_path):
        print(f"Loading identity data:\n{identity_path}")
        ident = pd.read_csv(identity_path)
        df = trans.merge(ident, on="TransactionID", how="left", suffixes=("", "_identity"))
    else:
        df = trans

    print(f"Merged dataset: {len(df):,} rows × {df.shape[1]} columns")
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


# =====================================================================
# FEATURE ENGINEERING (Structural, Ratios, Multi-Scale Velocity)
# =====================================================================
def add_structural_ring_features(df):
    df = df.copy()
    df = df.sort_values("TransactionDT").reset_index(drop=True)

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
    windows = {
        "1h": 60 * 60,
        "24h": 24 * 60 * 60,
        "7d": 7 * 24 * 60 * 60
    }

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
        np.log1p(df["device_card_previous_tx_24h"])
    )
    df["abuse_ring_score"] = (
        0.40 * df["entity_overlap_score"] +
        0.30 * df["pair_link_score"] +
        0.30 * df["ring_velocity_score"]
    )

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
        df[output_name] = smoothed_rate.fillna(global_rate).astype(float)
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
        current[output_name] = current[entity].map(grouped["rate"].to_dict()).fillna(global_rate).astype(float)
    return current


# =====================================================================
# PIPELINE FEATURE BUILDERS
# =====================================================================
def build_train_features(train_df):
    x = prepare_entities(train_df)
    x = add_structural_ring_features(x)
    x = add_multiscale_velocity_features(x)
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
    combined = add_multiscale_velocity_features(combined)

    current_mask = combined["__is_current"] == 1
    current_features = combined.loc[current_mask].copy().reset_index(drop=True)
    current_features = add_frozen_historical_fraud_rates(current_features, history)
    current_features = add_composite_ring_features(current_features)
    return current_features


# =====================================================================
# MODELING (LightGBM) & COST FUNCTIONS
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
        scale_pos_weight=12,  # Forces LightGBM to prioritize rare fraud cases
        random_state=RANDOM_STATE,
        n_jobs=-1,
        importance_type='gain',
        verbose=-1
    )


def calculate_cost(y_true, y_pred, transaction_amount):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    amount = np.asarray(transaction_amount, dtype=float)

    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)

    fp, fn = int(fp_mask.sum()), int(fn_mask.sum())
    fp_cost = fp * FALSE_POSITIVE_REVIEW_COST
    
    # Real-world merchant economics: Transaction amount lost + flat chargeback penalty fee
    fn_exposure = float(amount[fn_mask].sum()) + (fn * CHARGEBACK_PENALTY_FEE)

    return {
        "fp": fp, "fn": fn,
        "fp_cost": fp_cost, "fn_exposure": fn_exposure,
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
        rows.append({"threshold": float(threshold), "recall": float(recall), **cost})

    table = pd.DataFrame(rows)
    if min_recall > 0.0:
        valid = table[table["recall"] >= min_recall]
        if not valid.empty:
            table = valid

    table = table.sort_values("total_cost").reset_index(drop=True)
    return float(table.iloc[0]["threshold"]), table


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
    print(f"TOTAL ESTIMATED COST:   ₹{cost['total_cost']:,.2f}")

    return {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1, "pr_auc": pr_auc, **cost}


# =====================================================================
# MAIN EXECUTION
# =====================================================================
def run_pipeline():
    print(f"\n{'=' * 70}\nABUSE-RING SENTINEL V4 (LIGHTGBM)\n{'=' * 70}")
    df = load_data().sort_values("TransactionDT").reset_index(drop=True)

    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    validation_end = int(n * (TRAIN_RATIO + VALIDATION_RATIO))

    train_df, validation_df, test_df = df.iloc[:train_end], df.iloc[train_end:validation_end], df.iloc[validation_end:]

    print("\nBuilding features across splits...")
    train_features = build_train_features(train_df)
    validation_features = build_future_features(validation_df, train_df)
    test_features = build_future_features(test_df, pd.concat([train_df, validation_df]))

    y_train = train_features["isFraud"].astype(int)
    y_val = validation_features["isFraud"].astype(int)
    y_test = test_features["isFraud"].astype(int)
    val_amt = validation_features["TransactionAmt"].astype(float)
    test_amt = test_features["TransactionAmt"].astype(float)

    # Baseline Model
    print("\nTraining Baseline (LightGBM)...")
    X_tr_b, med_b = make_numeric_matrix(train_features, BASELINE_FEATURES)
    X_va_b, _ = make_numeric_matrix(validation_features, BASELINE_FEATURES, med_b)
    X_te_b, _ = make_numeric_matrix(test_features, BASELINE_FEATURES, med_b)

    model_b = create_model().fit(X_tr_b, y_train)
    thresh_b, table_b = find_best_threshold(y_val, model_b.predict_proba(X_va_b)[:, 1], val_amt)
    base_rec = table_b.iloc[0]["recall"]

    # Hybrid Sentinel Model
    print("Training Hybrid Sentinel (LightGBM + Pure Ring + Multi-Scale)...")
    X_tr_h, med_h = make_numeric_matrix(train_features, HYBRID_FEATURES)
    X_va_h, _ = make_numeric_matrix(validation_features, HYBRID_FEATURES, med_h)
    X_te_h, _ = make_numeric_matrix(test_features, HYBRID_FEATURES, med_h)

    model_h = create_model().fit(X_tr_h, y_train)
    thresh_h, _ = find_best_threshold(y_val, model_h.predict_proba(X_va_h)[:, 1], val_amt, min_recall=base_rec * 1.05)

    res_b = evaluate(model_b, X_te_b, y_test, test_amt, thresh_b, "1. BASELINE MODEL (LIGHTGBM)")
    res_h = evaluate(model_h, X_te_h, y_test, test_amt, thresh_h, "2. HYBRID SENTINEL (LIGHTGBM)")

    print(f"\n{'=' * 70}\nFINAL COMPARISON\n{'=' * 70}")
    print(f"Baseline Cost: ₹{res_b['total_cost']:,.2f} | Precision: {res_b['precision']:.2%} | Recall: {res_b['recall']:.2%}")
    print(f"Sentinel Cost: ₹{res_h['total_cost']:,.2f} | Precision: {res_h['precision']:.2%} | Recall: {res_h['recall']:.2%}")
    diff = res_b['total_cost'] - res_h['total_cost']
    print(f"Net Savings:   ₹{diff:,.2f} ({ (diff/res_b['total_cost'])*100:.2f}% improvement)")


if __name__ == "__main__":
    run_pipeline()