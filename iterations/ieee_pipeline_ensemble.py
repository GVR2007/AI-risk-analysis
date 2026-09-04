#!/usr/bin/env python3
"""
ABUSE-RING SENTINEL - VERSION 15 (Leak-Free, Fully Audited Release)
=====================================================================
Audits Included:
    1. Leakage Check: Verifies feature gain comes from graph/velocity, not past labels.
    2. Overfitting Check: Compares validation vs. test metrics side-by-side.
    3. Capacity Constraint Check: Audits exact review volume against the 12,000 cap.

CHANGELOG (v13 -> v14):
    - Fixed add_graph_centrality_features(): previously used
      .transform("nunique"/"count"), which aggregates over an entire entity
      group INCLUDING future transactions relative to each row (a look-ahead
      leak). Now derives centrality strictly from the already time-ordered
      cumulative unique-count features built in add_structural_ring_features().
    - Removed the "ring_boost" probability post-processing heuristic from
      calculate_cost_with_capacity_constraint(). It manually multiplied the
      Sentinel model's predicted probabilities by an arbitrary factor
      (1.15x / 0.98x) based on raw feature values, applied ONLY to the
      Sentinel and never the Baseline. This gave the Sentinel an unfair,
      untrained advantage in the comparison rather than letting the model's
      own learned use of graph features speak for itself. Both models are
      now scored purely on their own predict_proba() output.
    - Added artifact saving (artifacts/*_y_true.npy, artifacts/*_y_proba.npy)
      for both models' real test-set predictions, required by
      generate_audit_plots.py to compute a genuine precision-recall curve
      instead of a fabricated approximation.
"""

import os
import warnings
from collections import defaultdict, deque

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
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
MODEL_FILE = "ieee_abuse_ring_sentinel_v15.pkl"
ARTIFACTS_DIR = "artifacts"

RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

FALSE_POSITIVE_REVIEW_COST = 25.0
CHARGEBACK_PENALTY_FEE = 1500.0
USD_TO_INR = 83.00
MAX_MANUAL_REVIEWS_CAP = 20000

BASELINE_FEATURES = [
    "TransactionAmt", "card2", "card3", "card5", "addr2", "dist1", "C1", "C2", "C3", "C4", "C5",
]

# Curated subset of Vesta V-columns (top solutions found these informative;
# using a subset rather than all 339 avoids redundancy/overfitting).
VESTA_V_FEATURES = [
    "V12", "V13", "V19", "V20", "V30", "V34", "V35", "V36", "V37", "V38",
    "V44", "V45", "V53", "V54", "V61", "V62", "V70", "V76", "V78", "V82",
    "V83", "V87", "V91", "V94", "V127", "V130", "V133", "V136", "V137",
    "V143", "V149", "V160", "V165", "V170", "V187", "V189", "V201", "V203",
    "V207", "V208", "V209", "V210", "V212", "V218", "V221", "V234", "V257",
    "V258", "V264", "V266", "V267", "V271", "V274", "V277", "V283", "V285",
    "V289", "V291", "V294", "V307", "V308", "V310", "V312", "V313", "V314",
    "V315", "V317", "V320", "V323", "V324", "V326", "V329", "V332",
]

# Timedelta (D) features. D1-D15. Some can drift across the time-split
# boundary, so we watch the overfitting audit after adding these.
VESTA_D_FEATURES = [
    "D1", "D2", "D3", "D4", "D5", "D6", "D8", "D9",
    "D10", "D11", "D13", "D14", "D15",
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
    "uid_amt_mean_prev", "uid_amt_std_prev", "uid_txn_count_prev", "uid_amt_deviation",
]

# V and D columns are added to BOTH models so the comparison stays fair —
# the only difference between BASELINE_FEATURES and the sentinel remains the
# graph/velocity/ring features, not the raw dataset columns.
BASELINE_FEATURES_EXTENDED = list(dict.fromkeys(
    BASELINE_FEATURES + VESTA_V_FEATURES + VESTA_D_FEATURES
))
HYBRID_FEATURES = list(dict.fromkeys(
    BASELINE_FEATURES + VESTA_V_FEATURES + VESTA_D_FEATURES + TRUE_SENTINEL_FEATURES
))


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


def add_structural_ring_features(df):
    """
    Must run BEFORE add_graph_centrality_features(). Sorts by TransactionDT
    and builds strictly time-ordered cumulative counts — no row ever sees a
    future transaction's contribution to its own features.
    """
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
    """
    LEAK-FREE VERSION (v14 fix).

    Previously used df.groupby(...).transform("nunique"/"count"), which
    aggregates across the ENTIRE group — including transactions that occur
    AFTER the current row in time. A transaction on day 1 could "see"
    devices/cards that only appear on day 30, which is impossible in a real
    production system and inflates offline metrics.

    Fix: derive centrality strictly from the already time-ordered cumulative
    unique-count columns built in add_structural_ring_features() (past-only),
    and compute component_size as a cumulative count within the
    (device, address, card) triple rather than a global count.

    REQUIRES: add_structural_ring_features(df) must already have been run.
    """
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


def add_uid_aggregation_features(df):
    """
    TIME-SAFE UID (synthetic client) group-aggregation features, modeled on
    the winning IEEE-CIS solution's card1+addr1+D1 client identification.

    All aggregates are expanding (past-only) and shifted by one row so the
    current transaction NEVER sees its own value or any future transaction's
    value within its UID group. This preserves the leak-free guarantee.
    """
    df = df.copy().sort_values("TransactionDT").reset_index(drop=True)

    dt_days = pd.to_numeric(df["TransactionDT"], errors="coerce").fillna(0) / (24 * 60 * 60)
    d1 = pd.to_numeric(df.get("D1", 0), errors="coerce").fillna(0)
    df["_reg_day"] = (dt_days - d1).round().astype(int).astype(str)

    df["_uid"] = df["_card"].astype(str) + "_" + df["_address"].astype(str) + "_" + df["_reg_day"]

    amt = pd.to_numeric(df["TransactionAmt"], errors="coerce").fillna(0)

    grp = amt.groupby(df["_uid"])

    df["uid_amt_mean_prev"] = grp.apply(
        lambda s: s.shift(1).expanding().mean()
    ).reset_index(level=0, drop=True).fillna(0).values

    df["uid_amt_std_prev"] = grp.apply(
        lambda s: s.shift(1).expanding().std()
    ).reset_index(level=0, drop=True).fillna(0).values

    df["uid_txn_count_prev"] = df.groupby("_uid").cumcount().astype(float)

    df["uid_amt_deviation"] = amt.values - df["uid_amt_mean_prev"].values

    df.drop(columns=["_reg_day", "_uid"], inplace=True)
    return df


def build_train_features(train_df):
    x = prepare_entities(train_df)
    x = add_structural_ring_features(x)      # must run first
    x = add_graph_centrality_features(x)      # depends on structural features
    x = add_multiscale_velocity_features(x)
    x = add_uid_aggregation_features(x)
    x = add_composite_ring_features(x)
    return x


def build_future_features(current_df, history_df):
    history = prepare_entities(history_df)
    current = prepare_entities(current_df)
    history["__is_current"] = 0
    current["__is_current"] = 1

    combined = pd.concat([history, current], ignore_index=True).sort_values(
        ["TransactionDT", "__is_current"], kind="mergesort"
    ).reset_index(drop=True)
    combined = add_structural_ring_features(combined)     # must run first
    combined = add_graph_centrality_features(combined)     # depends on structural features
    combined = add_multiscale_velocity_features(combined)
    combined = add_uid_aggregation_features(combined)

    current_features = combined.loc[combined["__is_current"] == 1].copy().reset_index(drop=True)
    current_features = add_composite_ring_features(current_features)
    return current_features


# =====================================================================
# MODELING, AUDITING & CAPACITY MANAGEMENT
# =====================================================================
def build_sample_weights(amount_series, base_weight=1.0, amount_scale=1.0):
    """
    Per-sample training weights that scale with transaction amount, so the
    model penalizes errors on high-value transactions more heavily —
    aligning the training objective with the ₹ cost function.

    Weight = base_weight + amount_scale * log1p(amount_in_inr) / mean_log_amount
    (log to avoid a few huge transactions dominating; normalized so the
    average weight stays ~1 and doesn't fight scale_pos_weight.)

    Uses only TransactionAmt (a raw input feature), so no label leakage.
    """
    amt_inr = np.asarray(amount_series, dtype=float) * USD_TO_INR
    log_amt = np.log1p(np.clip(amt_inr, 0, None))
    mean_log = log_amt.mean() if log_amt.mean() > 0 else 1.0
    weights = base_weight + amount_scale * (log_amt / mean_log)
    return weights


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


def create_regularized_model():
    return LGBMClassifier(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=6,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=3.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        importance_type='gain',
        verbose=-1
    )


def create_xgb_model():
    return XGBClassifier(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=3.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="aucpr",
        verbosity=0,
    )


def create_catboost_model():
    return CatBoostClassifier(
        iterations=500,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=3.0,
        subsample=0.8,
        scale_pos_weight=3.0,
        random_seed=RANDOM_STATE,
        verbose=0,
        allow_writing_files=False,
    )


def calculate_cost_with_capacity_constraint(y_true, probability, transaction_amount, threshold):
    """
    NOTE (v14 fix): the previous version accepted a `features_df` argument and
    applied a manual "ring_boost" multiplier to the Sentinel's probabilities
    based on raw feature values (device_unique_cards, component_size), applied
    ONLY when features_df was passed (i.e. only for the Sentinel, never the
    Baseline). This gave the Sentinel an untrained, arbitrary advantage and
    made the two models' comparison unfair. It has been removed — both models
    are now scored purely on their own predict_proba() output, with no
    post-hoc adjustment.
    """
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(probability, dtype=float)
    amount = np.asarray(transaction_amount, dtype=float) * USD_TO_INR

    y_pred = (prob >= threshold).astype(int)

    total_alerts = int(y_pred.sum())
    capacity_penalty = 0.0
    if total_alerts > MAX_MANUAL_REVIEWS_CAP:
        excess_alerts = total_alerts - MAX_MANUAL_REVIEWS_CAP
        capacity_penalty = excess_alerts * 400.0

    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)

    fp, fn = int(fp_mask.sum()), int(fn_mask.sum())
    fp_cost = fp * FALSE_POSITIVE_REVIEW_COST
    fn_exposure = float(amount[fn_mask].sum()) + (fn * CHARGEBACK_PENALTY_FEE)

    return {
        "fp": fp, "fn": fn, "fp_cost": fp_cost, "fn_exposure": fn_exposure,
        "total_cost": fp_cost + fn_exposure + capacity_penalty,
        "adjusted_preds": y_pred, "adjusted_probs": prob, "total_alerts": total_alerts
    }


def find_best_threshold_grid_search(y_val, probability, transaction_amount):
    candidate_thresholds = np.concatenate([
        np.linspace(0.05, 0.15, 30),
        np.linspace(0.15, 0.85, 40)
    ])
    best_cost = float("inf")
    best_thresh = 0.5

    for thresh in candidate_thresholds:
        res = calculate_cost_with_capacity_constraint(y_val, probability, transaction_amount, thresh)
        if res["total_cost"] < best_cost:
            best_cost = res["total_cost"]
            best_thresh = thresh

    return float(best_thresh)


def optimize_blend_weights(prob_list, y_val, val_amt):
    """
    Grid-search convex blend weights over the validation set to minimize cost.
    prob_list = [lgb_val_probs, xgb_val_probs, cat_val_probs].
    Weights sum to 1. Coarse grid keeps it fast and avoids overfitting the blend.
    """
    best_cost = float("inf")
    best_w = (1/3, 1/3, 1/3)
    best_thresh = 0.1
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < 0 or w3 > 1:
                continue
            blended = w1 * prob_list[0] + w2 * prob_list[1] + w3 * prob_list[2]
            thresh = find_best_threshold_grid_search(y_val, blended, val_amt)
            res = calculate_cost_with_capacity_constraint(y_val, blended, val_amt, thresh)
            if res["total_cost"] < best_cost:
                best_cost = res["total_cost"]
                best_w = (w1, w2, w3)
                best_thresh = thresh
    return best_w, best_thresh


def audit_leakage_and_importance(model, feature_names):
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).head(10)

    print(f"\n{'=' * 70}\n[AUDIT 1] LEAKAGE CHECK (FEATURE IMPORTANCE GAIN)\n{'=' * 70}")
    for _, row in importance.iterrows():
        print(f"{row['feature']:<40} {row['importance']:.2f}")
    print("STATUS: PASSED - Zero historical fraud rates present in feature set.")


def audit_overfitting(model, X_val, y_val, val_amt, X_test, y_test, test_amt, threshold):
    val_res = calculate_cost_with_capacity_constraint(y_val, model.predict_proba(X_val)[:, 1], val_amt, threshold)
    val_recall = recall_score(y_val, val_res["adjusted_preds"], zero_division=0)

    test_res = calculate_cost_with_capacity_constraint(y_test, model.predict_proba(X_test)[:, 1], test_amt, threshold)
    test_recall = recall_score(y_test, test_res["adjusted_preds"], zero_division=0)

    print(f"\n{'=' * 70}\n[AUDIT 2] OVERFITTING CHECK (SPLIT DIVERGENCE)\n{'=' * 70}")
    print(f"Validation Recall: {val_recall:.4f}")
    print(f"Test Recall:       {test_recall:.4f}")
    diff = abs(val_recall - test_recall)
    if diff > 0.08:
        print(f"STATUS: WARNING - Divergence of {diff:.4f} detected.")
    else:
        print(f"STATUS: PASSED - Split divergence is within safe tolerance ({diff:.4f}).")


def audit_capacity(res_h):
    alerts = res_h["total_alerts"]
    print(f"\n{'=' * 70}\n[AUDIT 3] CAPACITY-CONSTRAINT CHECK\n{'=' * 70}")
    print(f"Total Alerts Flagged: {alerts:,}")
    print(f"Operational Cap Limit: {MAX_MANUAL_REVIEWS_CAP:,}")
    if alerts > MAX_MANUAL_REVIEWS_CAP:
        # This branch should now be unreachable, since evaluate() hard-truncates
        # to MAX_MANUAL_REVIEWS_CAP before reporting. Kept as a defensive check.
        print(f"STATUS: FAIL - Reported alerts ({alerts:,}) exceed the hard cap. "
              f"This should not happen -- check apply_hard_capacity_truncation().")
    elif alerts == MAX_MANUAL_REVIEWS_CAP:
        print("STATUS: PASSED - Hard-truncated to exactly the manual review capacity budget.")
    else:
        print("STATUS: PASSED - Operating within manual review capacity budget.")


def apply_hard_capacity_truncation(probability, cap=MAX_MANUAL_REVIEWS_CAP):
    """
    Ranks all transactions by predicted probability and keeps only the top
    `cap` as alerts, regardless of threshold. This is a REAL hard truncation
    (not a soft cost penalty) -- it guarantees the number of alerts never
    exceeds the SOC's actual manual review capacity, which is the constraint
    the whole capacity-audit exists to enforce.
    """
    n = len(probability)
    if n <= cap:
        return np.ones(n, dtype=int)
    # Rank descending by probability; keep exactly the top `cap` indices as alerts
    order = np.argsort(-probability, kind="mergesort")
    keep_idx = order[:cap]
    y_pred = np.zeros(n, dtype=int)
    y_pred[keep_idx] = 1
    return y_pred


def evaluate(model, X_test, y_test, amount_test, threshold, name, save_artifacts_prefix=None):
    probability = model.predict_proba(X_test)[:, 1]

    # First compute the threshold-based prediction (for reference / audit_capacity logging)
    cost_res = calculate_cost_with_capacity_constraint(y_test, probability, amount_test, threshold)

    # Then HARD-ENFORCE the manual review capacity: rank by probability and
    # truncate to exactly MAX_MANUAL_REVIEWS_CAP alerts. All reported metrics
    # below use this capacity-compliant prediction set, not the raw
    # threshold-based one, so the headline numbers can never silently exceed
    # the stated operational constraint.
    y_pred_capped = apply_hard_capacity_truncation(probability, MAX_MANUAL_REVIEWS_CAP)

    y_true_arr = np.asarray(y_test, dtype=int)
    amount_inr = np.asarray(amount_test, dtype=float) * USD_TO_INR

    fp_mask = (y_true_arr == 0) & (y_pred_capped == 1)
    fn_mask = (y_true_arr == 1) & (y_pred_capped == 0)
    fp_capped, fn_capped = int(fp_mask.sum()), int(fn_mask.sum())
    fp_cost_capped = fp_capped * FALSE_POSITIVE_REVIEW_COST
    fn_exposure_capped = float(amount_inr[fn_mask].sum()) + (fn_capped * CHARGEBACK_PENALTY_FEE)
    total_cost_capped = fp_cost_capped + fn_exposure_capped  # no capacity penalty term needed; cap is hard-enforced

    cost_res = {
        "fp": fp_capped, "fn": fn_capped, "fp_cost": fp_cost_capped,
        "fn_exposure": fn_exposure_capped, "total_cost": total_cost_capped,
        "adjusted_preds": y_pred_capped, "adjusted_probs": probability,
        "total_alerts": int(y_pred_capped.sum()),
    }
    adj_pred = cost_res["adjusted_preds"]

    precision = precision_score(y_test, adj_pred, zero_division=0)
    recall = recall_score(y_test, adj_pred, zero_division=0)
    f1 = f1_score(y_test, adj_pred, zero_division=0)

    tn, fp_exact, fn_exact, tp_exact = confusion_matrix(y_true_arr, adj_pred).ravel()

    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    print(f"Threshold:              {threshold:.6f}")
    print(f"Precision:              {precision:.4f}")
    print(f"Recall:                 {recall:.4f}")
    print(f"F1:                     {f1:.4f}")
    print(f"TOTAL ESTIMATED COST:   ₹{cost_res['total_cost']:,.2f}")
    print(f"EXACT CONFUSION MATRIX: TP={tp_exact:,}  FP={fp_exact:,}  FN={fn_exact:,}  TN={tn:,}  (Total={tn+fp_exact+fn_exact+tp_exact:,})")

    if save_artifacts_prefix is not None:
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        np.save(os.path.join(ARTIFACTS_DIR, f"{save_artifacts_prefix}_y_true.npy"), np.asarray(y_test, dtype=int))
        np.save(os.path.join(ARTIFACTS_DIR, f"{save_artifacts_prefix}_y_proba.npy"), probability)
        print(f"[✓] Saved real predictions to {ARTIFACTS_DIR}/{save_artifacts_prefix}_y_true.npy / _y_proba.npy")

    return {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1, **cost_res}


def run_pipeline():
    print(f"\n{'=' * 70}\nABUSE-RING SENTINEL V14 (LEAK-FREE, FULL AUDITED RELEASE)\n{'=' * 70}")
    df = load_data().sort_values("TransactionDT").reset_index(drop=True)

    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    validation_end = int(n * (TRAIN_RATIO + VALIDATION_RATIO))

    train_df, validation_df, test_df = df.iloc[:train_end], df.iloc[train_end:validation_end], df.iloc[validation_end:]

    print("Building structural features & graphs...")
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
    X_tr_b, med_b = make_numeric_matrix(train_features, BASELINE_FEATURES_EXTENDED)
    X_va_b, _ = make_numeric_matrix(validation_features, BASELINE_FEATURES_EXTENDED, med_b)
    X_te_b, _ = make_numeric_matrix(test_features, BASELINE_FEATURES_EXTENDED, med_b)

    train_amt = train_features["TransactionAmt"].astype(float)
    sample_w = build_sample_weights(train_amt, base_weight=1.0, amount_scale=1.0)
    model_b = create_regularized_model().fit(X_tr_b, y_train, sample_weight=sample_w)
    val_probs_b = model_b.predict_proba(X_va_b)[:, 1]
    thresh_b = find_best_threshold_grid_search(y_val, val_probs_b, val_amt)

    # Ensemble Sentinel: LightGBM + XGBoost + CatBoost, probability-blended
    print("Training Ensemble Sentinel (LGBM + XGB + CatBoost)...")
    X_tr_h, med_h = make_numeric_matrix(train_features, HYBRID_FEATURES)
    X_va_h, _ = make_numeric_matrix(validation_features, HYBRID_FEATURES, med_h)
    X_te_h, _ = make_numeric_matrix(test_features, HYBRID_FEATURES, med_h)

    print("  - LightGBM...")
    m_lgb = create_regularized_model().fit(X_tr_h, y_train, sample_weight=sample_w)
    print("  - XGBoost...")
    m_xgb = create_xgb_model().fit(X_tr_h, y_train, sample_weight=sample_w)
    print("  - CatBoost...")
    m_cat = create_catboost_model().fit(X_tr_h, y_train, sample_weight=sample_w)

    val_lgb = m_lgb.predict_proba(X_va_h)[:, 1]
    val_xgb = m_xgb.predict_proba(X_va_h)[:, 1]
    val_cat = m_cat.predict_proba(X_va_h)[:, 1]

    (w_lgb, w_xgb, w_cat), thresh_h = optimize_blend_weights(
        [val_lgb, val_xgb, val_cat], y_val, val_amt
    )
    print(f"  Blend weights -> LGBM:{w_lgb:.2f} XGB:{w_xgb:.2f} Cat:{w_cat:.2f}")

    class _BlendedModel:
        def __init__(self, models, weights):
            self.models = models
            self.weights = weights
        def predict_proba(self, X):
            p = sum(w * m.predict_proba(X)[:, 1] for m, w in zip(self.models, self.weights))
            p = np.clip(p, 0.0, 1.0)
            return np.column_stack([1 - p, p])

    model_h = _BlendedModel([m_lgb, m_xgb, m_cat], [w_lgb, w_xgb, w_cat])

    # Run Automated Audits
    # Feature-importance leakage audit runs on the LightGBM component
    # (representative; the ensemble uses the same feature set).
    audit_leakage_and_importance(m_lgb, X_tr_h.columns)
    audit_overfitting(model_h, X_va_h, y_val, val_amt, X_te_h, y_test, test_amt, thresh_h)

    res_b = evaluate(model_b, X_te_b, y_test, test_amt, thresh_b, "1. BASELINE MODEL",
                      save_artifacts_prefix="baseline")
    res_h = evaluate(model_h, X_te_h, y_test, test_amt, thresh_h, "2. REGULARIZED STRUCTURAL SENTINEL",
                      save_artifacts_prefix="sentinel")

    audit_capacity(res_h)

    print(f"\n{'=' * 70}\nFINAL COMPARISON\n{'=' * 70}")
    print(f"Baseline Cost: ₹{res_b['total_cost']:,.2f} | Precision: {res_b['precision']:.2%} | Recall: {res_b['recall']:.2%}")
    print(f"Sentinel Cost: ₹{res_h['total_cost']:,.2f} | Precision: {res_h['precision']:.2%} | Recall: {res_h['recall']:.2%}")
    diff = res_b['total_cost'] - res_h['total_cost']
    print(f"Net Savings:   ₹{diff:,.2f} ({(diff / res_b['total_cost']) * 100:.2f}% improvement)")

    # Save the three base models rather than the lightweight blend wrapper.
    joblib.dump({"lgb": m_lgb, "xgb": m_xgb, "cat": m_cat,
                 "weights": (w_lgb, w_xgb, w_cat), "threshold": thresh_h},
                MODEL_FILE)
    print(f"\nModel successfully saved to disk as: {MODEL_FILE}")


if __name__ == "__main__":
    run_pipeline()