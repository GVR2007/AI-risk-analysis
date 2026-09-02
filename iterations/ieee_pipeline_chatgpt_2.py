#!/usr/bin/env python3

"""
ABUSE-RING SENTINEL
===================

IEEE-CIS Fraud Detection

Goal:
    Compare a normal transaction-level fraud baseline against
    an Abuse-Ring Sentinel using coordinated entity features.

FINAL BUSINESS QUESTION:

    Does the Abuse-Ring Sentinel reduce estimated total cost?

Cost model:

    False Positive Cost
        = number of innocent alerts × manual review cost

    False Negative Exposure
        = TransactionAmt of fraud transactions that were missed

    Total Estimated Cost
        = FP Review Cost + FN Exposure

IMPORTANT:
    The ₹25 review cost is an ASSUMPTION.
    Change FALSE_POSITIVE_REVIEW_COST if you have a more
    defensible merchant-side estimate.

IEEE-CIS note:
    isFraud is transaction-level ground truth.
    The dataset does NOT explicitly label "abuse rings".

Therefore the project claim should be:

    "We construct coordinated-risk features from anonymized
     device, card, address and behavioral relationships and
     compare the resulting Sentinel against a transaction-level
     fraud baseline."
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

MODEL_FILE = "ieee_abuse_ring_sentinel.pkl"

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

VELOCITY_WINDOW_SECONDS = 24 * 60 * 60

# ---------------------------------------------------------------------
# BUSINESS ASSUMPTION
# ---------------------------------------------------------------------

FALSE_POSITIVE_REVIEW_COST = 25.0

# Historical fraud-rate smoothing.
FRAUD_RATE_ALPHA = 20.0

# =====================================================================
# BASELINE FEATURES
# =====================================================================

BASELINE_FEATURES = [
    "TransactionAmt",

    "card2",
    "card3",
    "card5",

    "addr2",
    "dist1",

    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
]

# =====================================================================
# RING FEATURES
# =====================================================================

RING_FEATURES = [
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

    # Historical entity risk
    "device_historical_fraud_rate",
    "card_historical_fraud_rate",
    "address_historical_fraud_rate",
    "email_historical_fraud_rate",

    # Velocity
    "device_previous_tx_24h",
    "card_previous_tx_24h",
    "address_previous_tx_24h",

    # Composite coordination signals
    "entity_overlap_score",
    "pair_link_score",
    "ring_velocity_score",
    "historical_entity_risk",
    "abuse_ring_score",

    # Explainability flags
    "multi_card_device_flag",
    "multi_address_device_flag",
    "multi_device_card_flag",
    "high_velocity_device_flag",
    "ring_candidate_flag",
]


# =====================================================================
# LOAD DATA
# =====================================================================

def load_data():

    transaction_path = os.path.join(
        DATA_DIR,
        TRAIN_TRANSACTION_FILE
    )

    identity_path = os.path.join(
        DATA_DIR,
        TRAIN_IDENTITY_FILE
    )

    if not os.path.exists(transaction_path):
        raise FileNotFoundError(
            f"Transaction file not found:\n{transaction_path}"
        )

    print(
        f"Loading transaction data:\n{transaction_path}"
    )

    trans = pd.read_csv(
        transaction_path
    )

    print(
        f"Transactions: "
        f"{len(trans):,} rows × {trans.shape[1]} columns"
    )

    if os.path.exists(identity_path):

        print(
            f"Loading identity data:\n{identity_path}"
        )

        ident = pd.read_csv(
            identity_path
        )

        print(
            f"Identity: "
            f"{len(ident):,} rows × {ident.shape[1]} columns"
        )

        df = trans.merge(
            ident,
            on="TransactionID",
            how="left",
            suffixes=("", "_identity")
        )

    else:

        print(
            "Identity file not found."
        )

        df = trans

    print(
        f"Merged dataset: "
        f"{len(df):,} rows × {df.shape[1]} columns"
    )

    return df


# =====================================================================
# PREPARE ENTITY COLUMNS
# =====================================================================

def prepare_entities(df):

    df = df.copy()

    # Device
    if "DeviceInfo" in df.columns:

        df["_device"] = (
            df["DeviceInfo"]
            .fillna("__MISSING__")
            .astype(str)
        )

    elif "id_31" in df.columns:

        df["_device"] = (
            df["id_31"]
            .fillna("__MISSING__")
            .astype(str)
        )

    else:

        df["_device"] = "__MISSING__"

    # Card
    if "card1" in df.columns:

        df["_card"] = (
            df["card1"]
            .fillna("__MISSING__")
            .astype(str)
        )

    else:

        df["_card"] = "__MISSING__"

    # Address
    if "addr1" in df.columns:

        df["_address"] = (
            df["addr1"]
            .fillna("__MISSING__")
            .astype(str)
        )

    else:

        df["_address"] = "__MISSING__"

    # Email
    if "P_emaildomain" in df.columns:

        df["_email"] = (
            df["P_emaildomain"]
            .fillna("__MISSING__")
            .astype(str)
        )

    elif "R_emaildomain" in df.columns:

        df["_email"] = (
            df["R_emaildomain"]
            .fillna("__MISSING__")
            .astype(str)
        )

    else:

        df["_email"] = "__MISSING__"

    # Browser
    if "id_31" in df.columns:

        df["_browser"] = (
            df["id_31"]
            .fillna("__MISSING__")
            .astype(str)
        )

    else:

        df["_browser"] = "__MISSING__"

    df["_time"] = pd.to_numeric(
        df["TransactionDT"],
        errors="coerce"
    ).fillna(0).astype(np.int64)

    return df


# =====================================================================
# STRUCTURAL / RELATIONAL FEATURES
# =====================================================================

def add_structural_ring_features(df):

    """
    Calculates historical relationship features using ONLY previous
    rows in chronological order.

    This avoids using future transactions to define the current
    transaction's relationship structure.
    """

    df = df.copy()

    df = df.sort_values(
        "TransactionDT"
    ).reset_index(
        drop=True
    )

    # -------------------------------------------------------------
    # Basic entity transaction count BEFORE current transaction
    # -------------------------------------------------------------

    df["device_transaction_count"] = (
        df.groupby("_device", dropna=False)
        .cumcount()
        .astype(float)
    )

    df["card_transaction_count"] = (
        df.groupby("_card", dropna=False)
        .cumcount()
        .astype(float)
    )

    df["address_transaction_count"] = (
        df.groupby("_address", dropna=False)
        .cumcount()
        .astype(float)
    )

    # -------------------------------------------------------------
    # Pair occurrence counts BEFORE current transaction
    # -------------------------------------------------------------

    df["device_card_pair_count"] = (
        df.groupby(
            ["_device", "_card"],
            dropna=False
        )
        .cumcount()
        .astype(float)
    )

    df["device_address_pair_count"] = (
        df.groupby(
            ["_device", "_address"],
            dropna=False
        )
        .cumcount()
        .astype(float)
    )

    df["card_address_pair_count"] = (
        df.groupby(
            ["_card", "_address"],
            dropna=False
        )
        .cumcount()
        .astype(float)
    )

    # -------------------------------------------------------------
    # DEVICE -> UNIQUE CARDS
    # -------------------------------------------------------------

    first_device_card = ~df.duplicated(
        ["_device", "_card"]
    )

    cumulative_device_card = (
        first_device_card.astype(np.int8)
        .groupby(df["_device"])
        .cumsum()
    )

    df["device_unique_cards"] = (
        cumulative_device_card
        - first_device_card.astype(np.int8)
    ).astype(float)

    # -------------------------------------------------------------
    # DEVICE -> UNIQUE ADDRESSES
    # -------------------------------------------------------------

    first_device_address = ~df.duplicated(
        ["_device", "_address"]
    )

    cumulative_device_address = (
        first_device_address.astype(np.int8)
        .groupby(df["_device"])
        .cumsum()
    )

    df["device_unique_addresses"] = (
        cumulative_device_address
        - first_device_address.astype(np.int8)
    ).astype(float)

    # -------------------------------------------------------------
    # DEVICE -> UNIQUE EMAILS
    # -------------------------------------------------------------

    first_device_email = ~df.duplicated(
        ["_device", "_email"]
    )

    cumulative_device_email = (
        first_device_email.astype(np.int8)
        .groupby(df["_device"])
        .cumsum()
    )

    df["device_unique_emails"] = (
        cumulative_device_email
        - first_device_email.astype(np.int8)
    ).astype(float)

    # -------------------------------------------------------------
    # DEVICE -> UNIQUE BROWSERS
    # -------------------------------------------------------------

    first_device_browser = ~df.duplicated(
        ["_device", "_browser"]
    )

    cumulative_device_browser = (
        first_device_browser.astype(np.int8)
        .groupby(df["_device"])
        .cumsum()
    )

    df["device_unique_browsers"] = (
        cumulative_device_browser
        - first_device_browser.astype(np.int8)
    ).astype(float)

    # -------------------------------------------------------------
    # CARD -> UNIQUE DEVICES
    # -------------------------------------------------------------

    first_card_device = ~df.duplicated(
        ["_card", "_device"]
    )

    cumulative_card_device = (
        first_card_device.astype(np.int8)
        .groupby(df["_card"])
        .cumsum()
    )

    df["card_unique_devices"] = (
        cumulative_card_device
        - first_card_device.astype(np.int8)
    ).astype(float)

    # -------------------------------------------------------------
    # CARD -> UNIQUE ADDRESSES
    # -------------------------------------------------------------

    first_card_address = ~df.duplicated(
        ["_card", "_address"]
    )

    cumulative_card_address = (
        first_card_address.astype(np.int8)
        .groupby(df["_card"])
        .cumsum()
    )

    df["card_unique_addresses"] = (
        cumulative_card_address
        - first_card_address.astype(np.int8)
    ).astype(float)

    # -------------------------------------------------------------
    # ADDRESS -> UNIQUE DEVICES
    # -------------------------------------------------------------

    first_address_device = ~df.duplicated(
        ["_address", "_device"]
    )

    cumulative_address_device = (
        first_address_device.astype(np.int8)
        .groupby(df["_address"])
        .cumsum()
    )

    df["address_unique_devices"] = (
        cumulative_address_device
        - first_address_device.astype(np.int8)
    ).astype(float)

    # -------------------------------------------------------------
    # ADDRESS -> UNIQUE CARDS
    # -------------------------------------------------------------

    first_address_card = ~df.duplicated(
        ["_address", "_card"]
    )

    cumulative_address_card = (
        first_address_card.astype(np.int8)
        .groupby(df["_address"])
        .cumsum()
    )

    df["address_unique_cards"] = (
        cumulative_address_card
        - first_address_card.astype(np.int8)
    ).astype(float)

    # -------------------------------------------------------------
    # Missing entity values should not create artificial rings
    # -------------------------------------------------------------

    missing_device = df["_device"] == "__MISSING__"
    missing_card = df["_card"] == "__MISSING__"
    missing_address = df["_address"] == "__MISSING__"

    df.loc[
        missing_device,
        [
            "device_transaction_count",
            "device_unique_cards",
            "device_unique_addresses",
            "device_unique_emails",
            "device_unique_browsers",
        ]
    ] = 0

    df.loc[
        missing_card,
        [
            "card_transaction_count",
            "card_unique_devices",
            "card_unique_addresses",
        ]
    ] = 0

    df.loc[
        missing_address,
        [
            "address_transaction_count",
            "address_unique_devices",
            "address_unique_cards",
        ]
    ] = 0

    return df


# =====================================================================
# VELOCITY FEATURES
# =====================================================================

def calculate_previous_window_counts(
    df,
    entity_column,
    window_seconds=24 * 60 * 60,
):
    """
    Count PREVIOUS transactions for each entity inside the
    previous 24-hour window.

    Current transaction is NOT included.

    Uses deques to avoid pandas rolling/index alignment problems.
    """

    values = (
        df[entity_column]
        .astype(str)
        .to_numpy()
    )

    times = (
        df["_time"]
        .to_numpy(dtype=np.int64)
    )

    result = np.zeros(
        len(df),
        dtype=np.int32,
    )

    histories = defaultdict(deque)

    for i in range(len(df)):

        entity = values[i]

        if entity == "__MISSING__":
            result[i] = 0
            continue

        current_time = times[i]

        q = histories[entity]

        cutoff = (
            current_time
            - window_seconds
        )

        while (
            q
            and q[0] < cutoff
        ):
            q.popleft()

        # IMPORTANT:
        # Count previous events BEFORE adding current event.
        result[i] = len(q)

        q.append(current_time)

    return result


def add_velocity_features(df):

    df = df.copy()

    df["device_previous_tx_24h"] = (
        calculate_previous_window_counts(
            df,
            "_device",
            VELOCITY_WINDOW_SECONDS,
        )
    )

    df["card_previous_tx_24h"] = (
        calculate_previous_window_counts(
            df,
            "_card",
            VELOCITY_WINDOW_SECONDS,
        )
    )

    df["address_previous_tx_24h"] = (
        calculate_previous_window_counts(
            df,
            "_address",
            VELOCITY_WINDOW_SECONDS,
        )
    )

    return df


# =====================================================================
# COMPOSITE RING FEATURES
# =====================================================================

def add_composite_ring_features(df):

    df = df.copy()

    # Entity overlap:
    # more cards/addresses/devices connected together.
    df["entity_overlap_score"] = (
        np.log1p(
            df["device_unique_cards"]
        )
        +
        np.log1p(
            df["device_unique_addresses"]
        )
        +
        np.log1p(
            df["card_unique_devices"]
        )
        +
        np.log1p(
            df["address_unique_devices"]
        )
    )

    # Repeated pair relationships.
    df["pair_link_score"] = (
        np.log1p(
            df["device_card_pair_count"]
        )
        +
        np.log1p(
            df["device_address_pair_count"]
        )
        +
        np.log1p(
            df["card_address_pair_count"]
        )
    )

    # High-speed activity.
    df["ring_velocity_score"] = (
        np.log1p(
            df["device_previous_tx_24h"]
        )
        +
        np.log1p(
            df["card_previous_tx_24h"]
        )
        +
        np.log1p(
            df["address_previous_tx_24h"]
        )
    )

    # Historical fraud risk.
    df["historical_entity_risk"] = (
        0.40
        * df["device_historical_fraud_rate"]
        +
        0.35
        * df["card_historical_fraud_rate"]
        +
        0.25
        * df["address_historical_fraud_rate"]
    )

    # Main interpretable ring score.
    df["abuse_ring_score"] = (
        0.30
        * df["entity_overlap_score"]
        +
        0.25
        * df["pair_link_score"]
        +
        0.20
        * df["ring_velocity_score"]
        +
        0.25
        * df["historical_entity_risk"]
    )

    # Explainability flags.
    df["multi_card_device_flag"] = (
        df["device_unique_cards"] >= 3
    ).astype(int)

    df["multi_address_device_flag"] = (
        df["device_unique_addresses"] >= 2
    ).astype(int)

    df["multi_device_card_flag"] = (
        df["card_unique_devices"] >= 2
    ).astype(int)

    df["high_velocity_device_flag"] = (
        df["device_previous_tx_24h"] >= 3
    ).astype(int)

    df["ring_candidate_flag"] = (
        (
            df["device_unique_cards"] >= 3
        )
        &
        (
            df["device_unique_addresses"] >= 2
        )
        &
        (
            df["device_previous_tx_24h"] >= 2
        )
    ).astype(int)

    return df


# =====================================================================
# HISTORICAL FRAUD RATES
# =====================================================================

def add_training_historical_fraud_rates(df):

    """
    Training data:
        use ONLY previous rows' fraud outcomes.

    This prevents the current row's target from being directly
    included in its own historical fraud-rate feature.
    """

    df = df.copy()

    global_rate = float(
        df["isFraud"].mean()
    )

    for entity, output_name in [
        (
            "_device",
            "device_historical_fraud_rate"
        ),
        (
            "_card",
            "card_historical_fraud_rate"
        ),
        (
            "_address",
            "address_historical_fraud_rate"
        ),
        (
            "_email",
            "email_historical_fraud_rate"
        ),
    ]:

        prior_count = (
            df.groupby(
                entity,
                dropna=False
            )
            .cumcount()
        )

        prior_fraud_sum = (
            df["isFraud"]
            .groupby(
                df[entity],
                dropna=False
            )
            .cumsum()
            - df["isFraud"]
        )

        smoothed_rate = (
            prior_fraud_sum
            +
            FRAUD_RATE_ALPHA
            * global_rate
        ) / (
            prior_count
            +
            FRAUD_RATE_ALPHA
        )

        # Do not assign risk to missing entity identifier.
        missing = df[entity] == "__MISSING__"

        smoothed_rate = (
            smoothed_rate
            .fillna(global_rate)
            .astype(float)
        )

        smoothed_rate.loc[missing] = global_rate

        df[output_name] = smoothed_rate

    return df


def add_frozen_historical_fraud_rates(
    current_df,
    history_df,
):
    """
    Validation/test:
        fraud-rate statistics come ONLY from historical labeled
        data.

    Example:

        validation fraud-rate maps:
            TRAIN only

        test fraud-rate maps:
            TRAIN + VALIDATION

    No test fraud labels are used to create test risk features.
    """

    current = current_df.copy()

    history = history_df.copy()

    global_rate = float(
        history["isFraud"].mean()
    )

    for entity, output_name in [
        (
            "_device",
            "device_historical_fraud_rate"
        ),
        (
            "_card",
            "card_historical_fraud_rate"
        ),
        (
            "_address",
            "address_historical_fraud_rate"
        ),
        (
            "_email",
            "email_historical_fraud_rate"
        ),
    ]:

        hist = history[
            history[entity] != "__MISSING__"
        ]

        if hist.empty:

            current[output_name] = global_rate

            continue

        grouped = (
            hist.groupby(entity)["isFraud"]
            .agg(
                fraud_sum="sum",
                count="count"
            )
        )

        grouped["rate"] = (
            grouped["fraud_sum"]
            +
            FRAUD_RATE_ALPHA
            * global_rate
        ) / (
            grouped["count"]
            +
            FRAUD_RATE_ALPHA
        )

        mapping = grouped["rate"].to_dict()

        current[output_name] = (
            current[entity]
            .map(mapping)
            .fillna(global_rate)
            .astype(float)
        )

        current.loc[
            current[entity] == "__MISSING__",
            output_name
        ] = global_rate

    return current


# =====================================================================
# BUILD TRAIN FEATURES
# =====================================================================

def build_train_features(train_df):

    x = prepare_entities(
        train_df
    )

    x = add_structural_ring_features(
        x
    )

    x = add_velocity_features(
        x
    )

    x = add_training_historical_fraud_rates(
        x
    )

    x = add_composite_ring_features(
        x
    )

    return x


# =====================================================================
# BUILD VALIDATION / TEST FEATURES
# =====================================================================

def build_future_features(
    current_df,
    history_df,
):
    """
    Current rows get:
        - relationship counts from history + current past
        - velocity from history + current past
        - fraud rates ONLY from history
    """

    history = prepare_entities(
        history_df
    )

    current = prepare_entities(
        current_df
    )

    history = history.sort_values(
        "TransactionDT"
    )

    current = current.sort_values(
        "TransactionDT"
    )

    history["__is_current"] = 0
    current["__is_current"] = 1

    combined = pd.concat(
        [
            history,
            current,
        ],
        ignore_index=True,
    )

    combined = combined.sort_values(
        [
            "TransactionDT",
            "__is_current",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    # Build structural features over all known events.
    combined = add_structural_ring_features(
        combined
    )

    # Previous activity only.
    combined = add_velocity_features(
        combined
    )

    # Split back out.
    current_mask = (
        combined["__is_current"] == 1
    )

    current_features = (
        combined.loc[current_mask]
        .copy()
        .reset_index(drop=True)
    )

    # Add fraud-risk maps from historical data only.
    current_features = (
        add_frozen_historical_fraud_rates(
            current_features,
            history
        )
    )

    current_features = (
        add_composite_ring_features(
            current_features
        )
    )

    return current_features


# =====================================================================
# PREPARE MODEL MATRIX
# =====================================================================

def make_numeric_matrix(
    df,
    features,
    medians=None,
):
    """
    Convert model features to numeric form.

    Missing values are filled using medians learned from TRAIN only.
    """

    selected = [
        c
        for c in features
        if c in df.columns
    ]

    x = df[selected].copy()

    for col in selected:

        x[col] = pd.to_numeric(
            x[col],
            errors="coerce"
        )

    x = x.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if medians is None:

        medians = (
            x.median()
            .fillna(0)
        )

    x = (
        x.fillna(medians)
        .fillna(0)
    )

    return x, medians


# =====================================================================
# TRAIN MODEL
# =====================================================================

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
# COST CALCULATION
# =====================================================================

def calculate_cost(
    y_true,
    y_pred,
    transaction_amount,
):
    """
    Business cost:

        FP cost:
            innocent alert × review cost

        FN exposure:
            amount of fraud transactions missed

        TOTAL:
            FP cost + FN exposure
    """

    y_true = np.asarray(
        y_true,
        dtype=int
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int
    )

    amount = np.asarray(
        transaction_amount,
        dtype=float
    )

    fp_mask = (
        (y_true == 0)
        &
        (y_pred == 1)
    )

    fn_mask = (
        (y_true == 1)
        &
        (y_pred == 0)
    )

    fp = int(
        fp_mask.sum()
    )

    fn = int(
        fn_mask.sum()
    )

    fp_cost = (
        fp
        * FALSE_POSITIVE_REVIEW_COST
    )

    fn_exposure = float(
        amount[fn_mask].sum()
    )

    total_cost = (
        fp_cost
        + fn_exposure
    )

    return {
        "fp": fp,
        "fn": fn,
        "fp_cost": fp_cost,
        "fn_exposure": fn_exposure,
        "total_cost": total_cost,
    }


# =====================================================================
# FIND BEST THRESHOLD
# =====================================================================

def find_best_threshold(
    y_val,
    probability,
    transaction_amount,
):
    """
    Threshold selection happens ONLY on validation.

    Final test set remains untouched.
    """

    candidate_thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(
                    0.01,
                    0.99,
                    200,
                ),
                np.quantile(
                    probability,
                    np.linspace(
                        0.01,
                        0.99,
                        100,
                    ),
                ),
            ]
        )
    )

    rows = []

    for threshold in candidate_thresholds:

        prediction = (
            probability >= threshold
        ).astype(int)

        cost = calculate_cost(
            y_true=y_val,
            y_pred=prediction,
            transaction_amount=transaction_amount,
        )

        precision = precision_score(
            y_val,
            prediction,
            zero_division=0,
        )

        recall = recall_score(
            y_val,
            prediction,
            zero_division=0,
        )

        f1 = f1_score(
            y_val,
            prediction,
            zero_division=0,
        )

        rows.append(
            {
                "threshold": float(
                    threshold
                ),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                **cost,
            }
        )

    table = pd.DataFrame(
        rows
    ).sort_values(
        "total_cost"
    ).reset_index(
        drop=True
    )

    best_threshold = float(
        table.iloc[0]["threshold"]
    )

    return (
        best_threshold,
        table
    )


# =====================================================================
# EVALUATE
# =====================================================================

def evaluate(
    model,
    X_test,
    y_test,
    amount_test,
    threshold,
    name,
):
    """

    Final held-out evaluation.
    """

    probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    prediction = (
        probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        prediction,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        prediction,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        prediction,
        zero_division=0,
    )

    pr_auc = average_precision_score(
        y_test,
        probability,
    )

    roc_auc = roc_auc_score(
        y_test,
        probability,
    )

    cost = calculate_cost(
        y_true=y_test,
        y_pred=prediction,
        transaction_amount=amount_test,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        prediction,
    ).ravel()

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Threshold:              {threshold:.6f}"
    )

    print(
        f"Precision:              {precision:.4f}"
    )

    print(
        f"Recall:                 {recall:.4f}"
    )

    print(
        f"F1:                     {f1:.4f}"
    )

    print(
        f"PR-AUC:                 {pr_auc:.4f}"
    )

    print(
        f"ROC-AUC:                {roc_auc:.4f}"
    )

    print()
    print("CONFUSION MATRIX")

    print(
        f"TN: {tn:,}"
    )

    print(
        f"FP: {fp:,}"
    )

    print(
        f"FN: {fn:,}"
    )

    print(
        f"TP: {tp:,}"
    )

    print()
    print("BUSINESS COST")

    print(
        f"False positives:        {cost['fp']:,}"
    )

    print(
        f"False negatives:        {cost['fn']:,}"
    )

    print(
        f"FP review cost:         "
        f"₹{cost['fp_cost']:,.2f}"
    )

    print(
        f"Missed-fraud exposure:  "
        f"₹{cost['fn_exposure']:,.2f}"
    )

    print(
        f"TOTAL ESTIMATED COST:   "
        f"₹{cost['total_cost']:,.2f}"
    )

    result = {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        **cost,
        "probability": probability,
        "prediction": prediction,
    }

    return result


# =====================================================================
# BASELINE VS SENTINEL COMPARISON
# =====================================================================

def print_comparison(
    baseline_result,
    sentinel_result,
):
    """
    The key hackathon comparison.
    """

    baseline_cost = (
        baseline_result["total_cost"]
    )

    sentinel_cost = (
        sentinel_result["total_cost"]
    )

    cost_change = (
        baseline_cost
        - sentinel_cost
    )

    if baseline_cost > 0:

        improvement_pct = (
            cost_change
            / baseline_cost
            * 100
        )

    else:

        improvement_pct = 0

    print()
    print("=" * 70)
    print("BASELINE VS ABUSE-RING SENTINEL")
    print("=" * 70)

    print(
        f"Baseline total cost:    "
        f"₹{baseline_cost:,.2f}"
    )

    print(
        f"Sentinel total cost:    "
        f"₹{sentinel_cost:,.2f}"
    )

    print(
        f"Cost difference:        "
        f"₹{abs(cost_change):,.2f}"
    )

    print(
        f"Cost improvement:       "
        f"{improvement_pct:.2f}%"
    )

    print()

    if sentinel_cost < baseline_cost:

        print(
            "RESULT: SENTINEL REDUCES ESTIMATED COST ✓"
        )

        print(
            "This is the result you want for the hackathon."
        )

    elif sentinel_cost > baseline_cost:

        print(
            "RESULT: SENTINEL INCREASES ESTIMATED COST ✗"
        )

        print(
            "Do NOT claim cost reduction."
        )

        print(
            "The ring features/model need improvement."
        )

    else:

        print(
            "RESULT: NO COST DIFFERENCE"
        )


# =====================================================================
# FEATURE IMPORTANCE
# =====================================================================

def print_feature_importance(
    model,
    features,
    top_n=25,
):

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance":
                model.feature_importances_,
        }
    )

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
        .head(top_n)
    )

    print()
    print("=" * 70)
    print("TOP SENTINEL FEATURES")
    print("=" * 70)

    for _, row in importance.iterrows():

        print(
            f"{row['feature']:<40}"
            f"{row['importance']:.6f}"
        )


# =====================================================================
# RING ALERT EXAMPLES
# =====================================================================

def print_ring_alerts(
    test_features,
    sentinel_result,
    n=10,
):

    df = test_features.copy()

    df["model_probability"] = (
        sentinel_result["probability"]
    )

    df["model_prediction"] = (
        sentinel_result["prediction"]
    )

    df = (
        df.sort_values(
            "model_probability",
            ascending=False,
        )
        .head(n)
    )

    columns = [
        "TransactionID",
        "TransactionAmt",
        "model_probability",

        "device_unique_cards",
        "device_unique_addresses",

        "card_unique_devices",
        "card_unique_addresses",

        "device_previous_tx_24h",
        "card_previous_tx_24h",
        "address_previous_tx_24h",

        "entity_overlap_score",
        "pair_link_score",
        "ring_velocity_score",

        "abuse_ring_score",
        "ring_candidate_flag",

        "isFraud",
    ]

    columns = [
        c
        for c in columns
        if c in df.columns
    ]

    print()
    print("=" * 70)
    print("TOP ABUSE-RING ALERTS")
    print("=" * 70)

    for _, row in df[columns].iterrows():

        print()
        print(
            f"Transaction: "
            f"{row['TransactionID']}"
        )

        print(
            f"Model probability: "
            f"{row['model_probability']:.4f}"
        )

        print(
            f"Amount: "
            f"₹{row['TransactionAmt']:,.2f}"
        )

        print(
            f"Device -> cards: "
            f"{int(row['device_unique_cards'])}"
        )

        print(
            f"Device -> addresses: "
            f"{int(row['device_unique_addresses'])}"
        )

        print(
            f"Card -> devices: "
            f"{int(row['card_unique_devices'])}"
        )

        print(
            f"Previous device transactions / 24h: "
            f"{int(row['device_previous_tx_24h'])}"
        )

        print(
            f"Previous card transactions / 24h: "
            f"{int(row['card_previous_tx_24h'])}"
        )

        print(
            f"Abuse-ring score: "
            f"{row['abuse_ring_score']:.4f}"
        )

        print(
            f"Ring candidate: "
            f"{bool(row['ring_candidate_flag'])}"
        )

        print(
            f"isFraud: "
            f"{int(row['isFraud'])}"
        )


# =====================================================================
# MAIN
# =====================================================================

def run_pipeline():

    print()
    print("=" * 70)
    print("ABUSE-RING SENTINEL")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. LOAD
    # -------------------------------------------------------------

    df = load_data()

    df = prepare_entities(
        df
    )

    df = df.sort_values(
        "TransactionDT"
    ).reset_index(
        drop=True
    )

    # -------------------------------------------------------------
    # 2. TEMPORAL SPLIT
    # -------------------------------------------------------------

    n = len(df)

    train_end = int(
        n * TRAIN_RATIO
    )

    validation_end = int(
        n * (
            TRAIN_RATIO
            + VALIDATION_RATIO
        )
    )

    train_df = (
        df.iloc[:train_end]
        .copy()
    )

    validation_df = (
        df.iloc[
            train_end:validation_end
        ]
        .copy()
    )

    test_df = (
        df.iloc[validation_end:]
        .copy()
    )

    print()
    print("=" * 70)
    print("TEMPORAL SPLIT")
    print("=" * 70)

    print(
        f"TRAIN:       {len(train_df):,}"
    )

    print(
        f"VALIDATION:  {len(validation_df):,}"
    )

    print(
        f"TEST:        {len(test_df):,}"
    )

    # -------------------------------------------------------------
    # 3. BUILD TRAIN FEATURES
    # -------------------------------------------------------------

    print()
    print(
        "Building TRAIN features..."
    )

    train_features = (
        build_train_features(
            train_df
        )
    )

    print(
        f"Train feature table: "
        f"{train_features.shape}"
    )

    # -------------------------------------------------------------
    # 4. BUILD VALIDATION FEATURES
    # -------------------------------------------------------------

    print()
    print(
        "Building VALIDATION features..."
    )

    validation_features = (
        build_future_features(
            current_df=validation_df,
            history_df=train_df,
        )
    )

    print(
        f"Validation feature table: "
        f"{validation_features.shape}"
    )

    # -------------------------------------------------------------
    # 5. BUILD TEST FEATURES
    # -------------------------------------------------------------

    print()
    print(
        "Building TEST features..."
    )

    test_history = pd.concat(
        [
            train_df,
            validation_df,
        ],
        ignore_index=True,
    )

    test_features = (
        build_future_features(
            current_df=test_df,
            history_df=test_history,
        )
    )

    print(
        f"Test feature table: "
        f"{test_features.shape}"
    )

    # -------------------------------------------------------------
    # 6. TARGETS / AMOUNTS
    # -------------------------------------------------------------

    y_train = (
        train_features["isFraud"]
        .astype(int)
    )

    y_validation = (
        validation_features["isFraud"]
        .astype(int)
    )

    y_test = (
        test_features["isFraud"]
        .astype(int)
    )

    validation_amount = (
        validation_features["TransactionAmt"]
        .astype(float)
    )

    test_amount = (
        test_features["TransactionAmt"]
        .astype(float)
    )

    # -------------------------------------------------------------
    # 7. BASELINE MODEL
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING BASELINE")
    print("=" * 70)

    available_baseline_features = [
        c
        for c in BASELINE_FEATURES
        if c in train_features.columns
    ]

    X_train_baseline, baseline_medians = (
        make_numeric_matrix(
            train_features,
            available_baseline_features,
        )
    )

    X_validation_baseline, _ = (
        make_numeric_matrix(
            validation_features,
            available_baseline_features,
            medians=baseline_medians,
        )
    )

    X_test_baseline, _ = (
        make_numeric_matrix(
            test_features,
            available_baseline_features,
            medians=baseline_medians,
        )
    )

    baseline_model = create_model()

    baseline_model.fit(
        X_train_baseline,
        y_train,
    )

    baseline_validation_probability = (
        baseline_model
        .predict_proba(
            X_validation_baseline
        )[:, 1]
    )

    # Threshold selected based on validation cost.
    baseline_threshold, baseline_table = (
        find_best_threshold(
            y_val=y_validation,
            probability=(
                baseline_validation_probability
            ),
            transaction_amount=(
                validation_amount
            ),
        )
    )

    print(
        f"Baseline validation threshold: "
        f"{baseline_threshold:.6f}"
    )

    # -------------------------------------------------------------
    # 8. SENTINEL MODEL
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING ABUSE-RING SENTINEL")
    print("=" * 70)

    available_ring_features = [
        c
        for c in RING_FEATURES
        if c in train_features.columns
    ]

    sentinel_feature_list = (
        available_baseline_features
        +
        available_ring_features
    )

    # Remove duplicate columns such as TransactionAmt.
    sentinel_feature_list = list(
        dict.fromkeys(
            sentinel_feature_list
        )
    )

    X_train_sentinel, sentinel_medians = (
        make_numeric_matrix(
            train_features,
            sentinel_feature_list,
        )
    )

    X_validation_sentinel, _ = (
        make_numeric_matrix(
            validation_features,
            sentinel_feature_list,
            medians=sentinel_medians,
        )
    )

    X_test_sentinel, _ = (
        make_numeric_matrix(
            test_features,
            sentinel_feature_list,
            medians=sentinel_medians,
        )
    )

    print(
        f"Sentinel model features: "
        f"{len(sentinel_feature_list)}"
    )

    sentinel_model = create_model()

    sentinel_model.fit(
        X_train_sentinel,
        y_train,
    )

    sentinel_validation_probability = (
        sentinel_model
        .predict_proba(
            X_validation_sentinel
        )[:, 1]
    )

    sentinel_threshold, sentinel_table = (
        find_best_threshold(
            y_val=y_validation,
            probability=(
                sentinel_validation_probability
            ),
            transaction_amount=(
                validation_amount
            ),
        )
    )

    print(
        f"Sentinel validation threshold: "
        f"{sentinel_threshold:.6f}"
    )

    # -------------------------------------------------------------
    # 9. FINAL TEST: BASELINE
    # -------------------------------------------------------------

    baseline_result = evaluate(
        model=baseline_model,
        X_test=X_test_baseline,
        y_test=y_test,
        amount_test=test_amount,
        threshold=baseline_threshold,
        name="BASELINE — HELD-OUT TEST",
    )

    # -------------------------------------------------------------
    # 10. FINAL TEST: SENTINEL
    # -------------------------------------------------------------

    sentinel_result = evaluate(
        model=sentinel_model,
        X_test=X_test_sentinel,
        y_test=y_test,
        amount_test=test_amount,
        threshold=sentinel_threshold,
        name="ABUSE-RING SENTINEL — HELD-OUT TEST",
    )

    # -------------------------------------------------------------
    # 11. COMPARE COSTS
    # -------------------------------------------------------------

    print_comparison(
        baseline_result,
        sentinel_result,
    )

    # -------------------------------------------------------------
    # 12. FEATURE IMPORTANCE
    # -------------------------------------------------------------

    print_feature_importance(
        sentinel_model,
        sentinel_feature_list,
        top_n=30,
    )

    # -------------------------------------------------------------
    # 13. EXPLAINABLE ALERTS
    # -------------------------------------------------------------

    print_ring_alerts(
        test_features,
        sentinel_result,
        n=10,
    )

    # -------------------------------------------------------------
    # 14. SAVE MODEL
    # -------------------------------------------------------------

    bundle = {
        "baseline_model": baseline_model,
        "sentinel_model": sentinel_model,

        "baseline_features":
            available_baseline_features,

        "sentinel_features":
            sentinel_feature_list,

        "baseline_medians":
            baseline_medians,

        "sentinel_medians":
            sentinel_medians,

        "baseline_threshold":
            baseline_threshold,

        "sentinel_threshold":
            sentinel_threshold,

        "false_positive_review_cost":
            FALSE_POSITIVE_REVIEW_COST,

        "velocity_window_seconds":
            VELOCITY_WINDOW_SECONDS,

        "baseline_test_metrics": {
            k: v
            for k, v in baseline_result.items()
            if k not in [
                "probability",
                "prediction",
            ]
        },

        "sentinel_test_metrics": {
            k: v
            for k, v in sentinel_result.items()
            if k not in [
                "probability",
                "prediction",
            ]
        },
    }

    joblib.dump(
        bundle,
        MODEL_FILE,
    )

    print()
    print("=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        f"{MODEL_FILE}"
    )

    # -------------------------------------------------------------
    # 15. FINAL HACKATHON SUMMARY
    # -------------------------------------------------------------

    baseline_cost = (
        baseline_result["total_cost"]
    )

    sentinel_cost = (
        sentinel_result["total_cost"]
    )

    reduction = (
        baseline_cost
        - sentinel_cost
    )

    reduction_pct = (
        reduction
        / baseline_cost
        * 100
        if baseline_cost > 0
        else 0
    )

    print()
    print("=" * 70)
    print("HACKATHON SUMMARY")
    print("=" * 70)

    print(
        f"Baseline cost: "
        f"₹{baseline_cost:,.2f}"
    )

    print(
        f"Sentinel cost: "
        f"₹{sentinel_cost:,.2f}"
    )

    if reduction > 0:

        print(
            f"Estimated cost reduction: "
            f"₹{reduction:,.2f}"
        )

        print(
            f"Estimated cost reduction %: "
            f"{reduction_pct:.2f}%"
        )

        print()
        print(
            "✓ Sentinel is cheaper than the baseline."
        )

    elif reduction < 0:

        print(
            f"Sentinel is MORE expensive by: "
            f"₹{abs(reduction):,.2f}"
        )

        print()
        print(
            "✗ Do not claim cost reduction yet."
        )

    else:

        print(
            "No estimated cost difference."
        )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    run_pipeline()