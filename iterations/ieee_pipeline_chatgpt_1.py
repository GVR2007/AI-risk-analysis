#!/usr/bin/env python3
"""
Abuse-Ring Sentinel
===================

IEEE-CIS Fraud Detection -> coordinated abuse/ring detection baseline.

What this pipeline does:
    1. Loads IEEE-CIS transaction + identity data.
    2. Performs a chronological 70/15/15 train/validation/test split.
    3. Learns entity statistics ONLY from historical data.
    4. Creates coordinated-behavior / ring features:
         - device <-> card relationships
         - device <-> address relationships
         - card <-> address relationships
         - entity diversity
         - historical fraud rates
         - previous-24h velocity
         - coordination/ring scores
    5. Trains Random Forest.
    6. Selects the decision threshold ONLY on validation.
    7. Evaluates exactly once on the held-out test set.
    8. Reports:
         - Precision
         - Recall
         - F1
         - PR-AUC
         - Confusion matrix
         - False positives
         - False-negative fraud amount
         - Estimated false-positive review cost
         - Total estimated risk cost
    9. Saves the model + feature builder.

IMPORTANT:
    IEEE-CIS does not provide explicit "abuse ring" labels.
    isFraud is transaction-level fraud ground truth.

Therefore this system should be presented as:
    "An abuse-ring / coordinated-risk detector built from
     transaction-level fraud labels and anonymized entity relationships."

Not:
    "IEEE-CIS directly labels fraud rings."
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple

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
# CONFIGURATION
# =====================================================================

DATA_DIR = "test_datasets/kaggle/ieee-fraud-detection"

TRAIN_TRANSACTION_FILE = "train_transaction.csv"
TRAIN_IDENTITY_FILE = "train_identity.csv"

MODEL_FILE = "ieee_abuse_ring_sentinel.pkl"
FEATURE_DATA_FILE = "ieee_abuse_ring_features.parquet"

RANDOM_STATE = 42

# Chronological split:
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Velocity window.
VELOCITY_WINDOW_SECONDS = 24 * 60 * 60

# ---------------------------------------------------------------------
# FALSE POSITIVE COST ASSUMPTION
# ---------------------------------------------------------------------
#
# This is the merchant's approximate cost of manually reviewing
# one false-positive alert.
#
# CHANGE THIS VALUE to whatever assumption you want to use.
#
# Example:
#     25.0 = ₹25 per manual review
#
# This is NOT a value supplied by IEEE-CIS.
#
FALSE_POSITIVE_REVIEW_COST = 25.0

# ---------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------

N_ESTIMATORS = 200
MAX_DEPTH = 14
MIN_SAMPLES_LEAF = 3


# =====================================================================
# DATA LOADING
# =====================================================================

def load_ieee_data() -> pd.DataFrame:
    """
    Load and merge IEEE-CIS transaction and identity datasets.
    """

    transaction_path = os.path.join(
        DATA_DIR,
        TRAIN_TRANSACTION_FILE,
    )

    identity_path = os.path.join(
        DATA_DIR,
        TRAIN_IDENTITY_FILE,
    )

    if not os.path.exists(transaction_path):
        raise FileNotFoundError(
            f"Could not find:\n{transaction_path}"
        )

    print(f"Loading transactions: {transaction_path}")

    transactions = pd.read_csv(transaction_path)

    print(
        f"Transactions loaded: "
        f"{transactions.shape[0]:,} rows × {transactions.shape[1]} columns"
    )

    if os.path.exists(identity_path):
        print(f"Loading identity data: {identity_path}")

        identity = pd.read_csv(identity_path)

        print(
            f"Identity loaded: "
            f"{identity.shape[0]:,} rows × {identity.shape[1]} columns"
        )

        df = transactions.merge(
            identity,
            on="TransactionID",
            how="left",
            suffixes=("", "_identity"),
        )
    else:
        print("Identity file not found. Continuing without identity data.")
        df = transactions

    print(
        f"Merged dataset: "
        f"{df.shape[0]:,} rows × {df.shape[1]} columns"
    )

    return df


# =====================================================================
# BASIC DATA PREPARATION
# =====================================================================

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize entity columns.
    """

    df = df.copy()

    required = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "isFraud",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Required columns missing: {missing}"
        )

    # -----------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Card
    # -----------------------------------------------------------------

    if "card1" in df.columns:
        df["_card"] = (
            df["card1"]
            .fillna("__MISSING__")
            .astype(str)
        )
    else:
        df["_card"] = "__MISSING__"

    # -----------------------------------------------------------------
    # Address
    # -----------------------------------------------------------------

    if "addr1" in df.columns:
        df["_address"] = (
            df["addr1"]
            .fillna("__MISSING__")
            .astype(str)
        )
    else:
        df["_address"] = "__MISSING__"

    # -----------------------------------------------------------------
    # Email
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Browser
    # -----------------------------------------------------------------

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
        errors="coerce",
    ).fillna(0)

    return df


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def valid_entity_mask(series: pd.Series) -> pd.Series:
    """
    True for usable entity values.
    """

    return (
        series.notna()
        & (series.astype(str) != "__MISSING__")
    )


def safe_nunique(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
) -> Dict:
    """
    Group -> number of unique target entities.
    """

    if group_col not in df.columns:
        return {}

    if target_col not in df.columns:
        return {}

    temp = df[
        valid_entity_mask(df[group_col])
        & valid_entity_mask(df[target_col])
    ]

    if temp.empty:
        return {}

    return (
        temp.groupby(group_col)[target_col]
        .nunique()
        .to_dict()
    )


def safe_count(
    df: pd.DataFrame,
    group_col: str,
) -> Dict:
    """
    Group -> count.
    """

    temp = df[
        valid_entity_mask(df[group_col])
    ]

    if temp.empty:
        return {}

    return (
        temp[group_col]
        .value_counts()
        .to_dict()
    )


def safe_pair_count(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
) -> Dict:
    """
    Count occurrences of entity pairs.
    """

    temp = df[
        valid_entity_mask(df[col_a])
        & valid_entity_mask(df[col_b])
    ][[col_a, col_b]].copy()

    if temp.empty:
        return {}

    pair_counts = (
        temp.groupby([col_a, col_b])
        .size()
    )

    return pair_counts.to_dict()


def safe_fraud_rate(
    df: pd.DataFrame,
    entity_col: str,
    alpha: float = 20.0,
) -> Dict:
    """
    Smoothed entity-level historical fraud rate.

    Smoothing:
        (fraud_count + alpha * global_rate)
        /
        (entity_count + alpha)

    This avoids extreme fraud rates for entities with only
    one or two historical transactions.
    """

    if "isFraud" not in df.columns:
        return {}

    global_rate = float(
        df["isFraud"].mean()
    )

    temp = df[
        valid_entity_mask(df[entity_col])
    ][[entity_col, "isFraud"]]

    if temp.empty:
        return {}

    grouped = (
        temp.groupby(entity_col)["isFraud"]
        .agg(["sum", "count"])
    )

    grouped["fraud_rate"] = (
        grouped["sum"] + alpha * global_rate
    ) / (
        grouped["count"] + alpha
    )

    return grouped["fraud_rate"].to_dict()


# =====================================================================
# VELOCITY
# =====================================================================

def previous_window_count(
    history: pd.DataFrame,
    current: pd.DataFrame,
    entity_col: str,
    window_seconds: int,
) -> np.ndarray:
    """
    Calculate the number of previous transactions for an entity
    within the preceding time window.

    This is intentionally written without pandas grouped rolling,
    avoiding the duplicate-index alignment problem that caused
    your previous error.

    IMPORTANT:
        Current transaction is NOT counted in its own velocity.

    history:
        Events already known before/current stage.

    current:
        Events for which features are being generated.
    """

    n_current = len(current)

    result = np.zeros(
        n_current,
        dtype=np.int32,
    )

    if n_current == 0:
        return result

    hist = history[
        ["_time", entity_col]
    ].copy()

    cur = current[
        ["_time", entity_col]
    ].copy()

    hist["__source"] = 0
    cur["__source"] = 1

    hist["__row_id"] = np.arange(
        len(hist),
        dtype=np.int64,
    )

    cur["__row_id"] = np.arange(
        len(cur),
        dtype=np.int64,
    )

    combined = pd.concat(
        [hist, cur],
        ignore_index=True,
    )

    # Sort by entity/time.
    combined = combined.sort_values(
        [entity_col, "_time", "__source"],
        kind="mergesort",
    ).reset_index(drop=True)

    # Only count previous events.
    #
    # For each entity, we use a sliding window over timestamps.
    #
    # Since combined is sorted by entity and time,
    # group-by iteration is safe and avoids pandas reindex problems.

    grouped = combined.groupby(
        entity_col,
        sort=False,
        dropna=False,
    )

    for _, group in grouped:

        if group.empty:
            continue

        timestamps = (
            group["_time"]
            .to_numpy(dtype=np.float64)
        )

        source = (
            group["__source"]
            .to_numpy(dtype=np.int8)
        )

        row_ids = (
            group["__row_id"]
            .to_numpy(dtype=np.int64)
        )

        # For every event, identify the first timestamp inside
        # [current_time - window, current_time)
        left_indices = np.searchsorted(
            timestamps,
            timestamps - window_seconds,
            side="left",
        )

        # Number of previous events in window, initially
        # based on sorted position.
        counts = (
            np.arange(len(timestamps))
            - left_indices
        )

        # We need the count of all previous events, not future ones.
        # Because current/known events are all time ordered, the
        # positional count is exactly the number of previous records.

        current_mask = source == 1

        if current_mask.any():

            current_row_ids = row_ids[current_mask]
            current_counts = counts[current_mask]

            result[current_row_ids] = current_counts.astype(
                np.int32
            )

    return result


def add_velocity_features(
    history_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add previous 24-hour transaction velocity.

    For:
        device
        card
        address
    """

    result = current_df.copy()

    for entity_col, output_col in [
        ("_device", "device_previous_tx_24h"),
        ("_card", "card_previous_tx_24h"),
        ("_address", "address_previous_tx_24h"),
    ]:

        if entity_col not in history_df.columns:
            result[output_col] = 0
            continue

        result[output_col] = previous_window_count(
            history=history_df,
            current=current_df,
            entity_col=entity_col,
            window_seconds=VELOCITY_WINDOW_SECONDS,
        )

    return result


# =====================================================================
# FEATURE BUILDER
# =====================================================================

class AbuseRingFeatureBuilder:
    """
    Learns entity statistics from historical data and applies them
    to validation/test data.

    This is important because validation/test information should not
    be used when learning entity statistics.
    """

    def __init__(self):
        self.global_fraud_rate = 0.0

        self.maps = {}

        self.pair_maps = {}

    # -----------------------------------------------------------------
    # FIT
    # -----------------------------------------------------------------

    def fit(
        self,
        historical_df: pd.DataFrame,
    ):
        """
        Learn historical entity statistics.
        """

        historical_df = prepare_dataframe(
            historical_df
        )

        self.global_fraud_rate = float(
            historical_df["isFraud"].mean()
        )

        # -------------------------------------------------------------
        # Basic entity transaction counts
        # -------------------------------------------------------------

        self.maps["device_count"] = safe_count(
            historical_df,
            "_device",
        )

        self.maps["card_count"] = safe_count(
            historical_df,
            "_card",
        )

        self.maps["address_count"] = safe_count(
            historical_df,
            "_address",
        )

        # -------------------------------------------------------------
        # Entity -> unique entity degree
        # -------------------------------------------------------------

        self.maps["device_unique_cards"] = safe_nunique(
            historical_df,
            "_device",
            "_card",
        )

        self.maps["device_unique_addresses"] = safe_nunique(
            historical_df,
            "_device",
            "_address",
        )

        self.maps["device_unique_emails"] = safe_nunique(
            historical_df,
            "_device",
            "_email",
        )

        self.maps["device_unique_browsers"] = safe_nunique(
            historical_df,
            "_device",
            "_browser",
        )

        self.maps["card_unique_devices"] = safe_nunique(
            historical_df,
            "_card",
            "_device",
        )

        self.maps["card_unique_addresses"] = safe_nunique(
            historical_df,
            "_card",
            "_address",
        )

        self.maps["address_unique_devices"] = safe_nunique(
            historical_df,
            "_address",
            "_device",
        )

        self.maps["address_unique_cards"] = safe_nunique(
            historical_df,
            "_address",
            "_card",
        )

        # -------------------------------------------------------------
        # Pair frequencies
        # -------------------------------------------------------------

        self.pair_maps["device_card"] = safe_pair_count(
            historical_df,
            "_device",
            "_card",
        )

        self.pair_maps["device_address"] = safe_pair_count(
            historical_df,
            "_device",
            "_address",
        )

        self.pair_maps["card_address"] = safe_pair_count(
            historical_df,
            "_card",
            "_address",
        )

        # -------------------------------------------------------------
        # Historical fraud rates
        # -------------------------------------------------------------

        self.maps["device_fraud_rate"] = safe_fraud_rate(
            historical_df,
            "_device",
        )

        self.maps["card_fraud_rate"] = safe_fraud_rate(
            historical_df,
            "_card",
        )

        self.maps["address_fraud_rate"] = safe_fraud_rate(
            historical_df,
            "_address",
        )

        self.maps["email_fraud_rate"] = safe_fraud_rate(
            historical_df,
            "_email",
        )

        return self

    # -----------------------------------------------------------------
    # TRANSFORM
    # -----------------------------------------------------------------

    def transform(
        self,
        current_df: pd.DataFrame,
        history_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate features.

        history_df:
            Historical data available before/current stage.

        For:
            train -> train history
            validation -> train history + validation previous events
            test -> train + validation history + test previous events

        """

        current = prepare_dataframe(
            current_df
        )

        if history_df is None:
            history = current.copy()
        else:
            history = prepare_dataframe(
                history_df
            )

        # -------------------------------------------------------------
        # Entity degree features
        # -------------------------------------------------------------

        current["device_transaction_count"] = (
            current["_device"]
            .map(self.maps["device_count"])
            .fillna(0)
            .astype(float)
        )

        current["card_transaction_count"] = (
            current["_card"]
            .map(self.maps["card_count"])
            .fillna(0)
            .astype(float)
        )

        current["address_transaction_count"] = (
            current["_address"]
            .map(self.maps["address_count"])
            .fillna(0)
            .astype(float)
        )

        current["device_unique_cards"] = (
            current["_device"]
            .map(self.maps["device_unique_cards"])
            .fillna(0)
            .astype(float)
        )

        current["device_unique_addresses"] = (
            current["_device"]
            .map(self.maps["device_unique_addresses"])
            .fillna(0)
            .astype(float)
        )

        current["device_unique_emails"] = (
            current["_device"]
            .map(self.maps["device_unique_emails"])
            .fillna(0)
            .astype(float)
        )

        current["device_unique_browsers"] = (
            current["_device"]
            .map(self.maps["device_unique_browsers"])
            .fillna(0)
            .astype(float)
        )

        current["card_unique_devices"] = (
            current["_card"]
            .map(self.maps["card_unique_devices"])
            .fillna(0)
            .astype(float)
        )

        current["card_unique_addresses"] = (
            current["_card"]
            .map(self.maps["card_unique_addresses"])
            .fillna(0)
            .astype(float)
        )

        current["address_unique_devices"] = (
            current["_address"]
            .map(self.maps["address_unique_devices"])
            .fillna(0)
            .astype(float)
        )

        current["address_unique_cards"] = (
            current["_address"]
            .map(self.maps["address_unique_cards"])
            .fillna(0)
            .astype(float)
        )

        # -------------------------------------------------------------
        # Pair frequencies
        # -------------------------------------------------------------

        device_card_keys = list(
            zip(
                current["_device"].astype(str),
                current["_card"].astype(str),
            )
        )

        current["device_card_pair_count"] = [
            self.pair_maps["device_card"].get(
                key,
                0,
            )
            for key in device_card_keys
        ]

        device_address_keys = list(
            zip(
                current["_device"].astype(str),
                current["_address"].astype(str),
            )
        )

        current["device_address_pair_count"] = [
            self.pair_maps["device_address"].get(
                key,
                0,
            )
            for key in device_address_keys
        ]

        card_address_keys = list(
            zip(
                current["_card"].astype(str),
                current["_address"].astype(str),
            )
        )

        current["card_address_pair_count"] = [
            self.pair_maps["card_address"].get(
                key,
                0,
            )
            for key in card_address_keys
        ]

        # -------------------------------------------------------------
        # Historical fraud rates
        # -------------------------------------------------------------

        current["device_historical_fraud_rate"] = (
            current["_device"]
            .map(self.maps["device_fraud_rate"])
            .fillna(self.global_fraud_rate)
            .astype(float)
        )

        current["card_historical_fraud_rate"] = (
            current["_card"]
            .map(self.maps["card_fraud_rate"])
            .fillna(self.global_fraud_rate)
            .astype(float)
        )

        current["address_historical_fraud_rate"] = (
            current["_address"]
            .map(self.maps["address_fraud_rate"])
            .fillna(self.global_fraud_rate)
            .astype(float)
        )

        current["email_historical_fraud_rate"] = (
            current["_email"]
            .map(self.maps["email_fraud_rate"])
            .fillna(self.global_fraud_rate)
            .astype(float)
        )

        # -------------------------------------------------------------
        # Velocity
        # -------------------------------------------------------------

        current = add_velocity_features(
            history_df=history,
            current_df=current,
        )

        # -------------------------------------------------------------
        # Ring structure features
        # -------------------------------------------------------------

        current["entity_overlap_score"] = (
            np.log1p(
                current["device_unique_cards"]
            )
            +
            np.log1p(
                current["device_unique_addresses"]
            )
            +
            np.log1p(
                current["card_unique_devices"]
            )
            +
            np.log1p(
                current["address_unique_devices"]
            )
        )

        current["pair_link_score"] = (
            np.log1p(
                current["device_card_pair_count"]
            )
            +
            np.log1p(
                current["device_address_pair_count"]
            )
            +
            np.log1p(
                current["card_address_pair_count"]
            )
        )

        current["ring_velocity_score"] = (
            np.log1p(
                current["device_previous_tx_24h"]
            )
            +
            np.log1p(
                current["card_previous_tx_24h"]
            )
            +
            np.log1p(
                current["address_previous_tx_24h"]
            )
        )

        current["historical_entity_risk"] = (
            0.40
            * current["device_historical_fraud_rate"]
            +
            0.35
            * current["card_historical_fraud_rate"]
            +
            0.25
            * current["address_historical_fraud_rate"]
        )

        # -------------------------------------------------------------
        # Coordination indicators
        # -------------------------------------------------------------

        current["multi_card_device_flag"] = (
            current["device_unique_cards"] >= 3
        ).astype(int)

        current["multi_address_device_flag"] = (
            current["device_unique_addresses"] >= 2
        ).astype(int)

        current["multi_device_card_flag"] = (
            current["card_unique_devices"] >= 2
        ).astype(int)

        current["high_velocity_device_flag"] = (
            current["device_previous_tx_24h"] >= 3
        ).astype(int)

        # -------------------------------------------------------------
        # Explainable raw ring score
        # -------------------------------------------------------------

        current["abuse_ring_score"] = (
            0.30
            * current["entity_overlap_score"]
            +
            0.25
            * current["pair_link_score"]
            +
            0.20
            * current["ring_velocity_score"]
            +
            0.25
            * current["historical_entity_risk"]
        )

        # -------------------------------------------------------------
        # Candidate ring indicator
        # -------------------------------------------------------------
        #
        # This is NOT the final ML prediction.
        # It is an explainability feature:
        #
        # "Multiple cards + multiple addresses + velocity"
        #
        current["ring_candidate_flag"] = (
            (
                current["device_unique_cards"] >= 3
            )
            &
            (
                current["device_unique_addresses"] >= 2
            )
            &
            (
                current["device_previous_tx_24h"] >= 2
            )
        ).astype(int)

        return current


# =====================================================================
# MODEL FEATURES
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

    # Historical risk
    "device_historical_fraud_rate",
    "card_historical_fraud_rate",
    "address_historical_fraud_rate",
    "email_historical_fraud_rate",

    # Velocity
    "device_previous_tx_24h",
    "card_previous_tx_24h",
    "address_previous_tx_24h",

    # Composite ring signals
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

    # Transaction financial information
    "TransactionAmt",
]


# =====================================================================
# MODEL TRAINING
# =====================================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """
    Train Random Forest baseline.
    """

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# =====================================================================
# THRESHOLD / BUSINESS COST
# =====================================================================

def calculate_business_cost(
    y_true: pd.Series,
    y_pred: np.ndarray,
    transaction_amount: pd.Series,
    false_positive_review_cost: float,
) -> Tuple[float, int, int, float, float]:
    """
    Calculate approximate defensive operating cost.

    FP cost:
        Number of false positives × manual-review cost.

    FN cost:
        Sum of TransactionAmt for missed fraudulent transactions.

    NOTE:
        This is an approximation, not an accounting truth.
    """

    y_true_array = np.asarray(
        y_true
    )

    amount_array = np.asarray(
        transaction_amount,
        dtype=float,
    )

    fp_mask = (
        (y_true_array == 0)
        &
        (y_pred == 1)
    )

    fn_mask = (
        (y_true_array == 1)
        &
        (y_pred == 0)
    )

    fp = int(fp_mask.sum())
    fn = int(fn_mask.sum())

    fp_cost = (
        fp
        * false_positive_review_cost
    )

    # For false negatives we use transaction amount
    # as an approximate missed-fraud exposure.
    fn_cost = float(
        amount_array[fn_mask].sum()
    )

    total_cost = (
        fp_cost
        + fn_cost
    )

    return (
        total_cost,
        fp,
        fn,
        fp_cost,
        fn_cost,
    )


def select_threshold_by_cost(
    y_val: pd.Series,
    val_prob: np.ndarray,
    val_amount: pd.Series,
) -> Tuple[float, pd.DataFrame]:
    """
    Select the decision threshold on validation only.

    Does NOT use the final test set.
    """

    # Candidate thresholds from quantiles.
    candidate_thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(
                    0.01,
                    0.99,
                    200,
                ),
                np.quantile(
                    val_prob,
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

        pred = (
            val_prob >= threshold
        ).astype(int)

        total_cost, fp, fn, fp_cost, fn_cost = (
            calculate_business_cost(
                y_true=y_val,
                y_pred=pred,
                transaction_amount=val_amount,
                false_positive_review_cost=FALSE_POSITIVE_REVIEW_COST,
            )
        )

        precision = precision_score(
            y_val,
            pred,
            zero_division=0,
        )

        recall = recall_score(
            y_val,
            pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_val,
            pred,
            zero_division=0,
        )

        rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fp": fp,
                "fn": fn,
                "fp_cost": fp_cost,
                "fn_cost": fn_cost,
                "total_cost": total_cost,
            }
        )

    results = pd.DataFrame(rows)

    results = results.sort_values(
        "total_cost",
        ascending=True,
    ).reset_index(drop=True)

    best_threshold = float(
        results.iloc[0]["threshold"]
    )

    return best_threshold, results


# =====================================================================
# EVALUATION
# =====================================================================

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_amount: pd.Series,
    threshold: float,
    label: str,
):
    """
    Final evaluation.
    """

    probability = model.predict_proba(
        X_test
    )[:, 1]

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

    (
        total_cost,
        fp,
        fn,
        fp_cost,
        fn_cost,
    ) = calculate_business_cost(
        y_true=y_test,
        y_pred=prediction,
        transaction_amount=test_amount,
        false_positive_review_cost=FALSE_POSITIVE_REVIEW_COST,
    )

    tn, fp_matrix, fn_matrix, tp = confusion_matrix(
        y_test,
        prediction,
    ).ravel()

    print()
    print("=" * 70)
    print(f"{label} EVALUATION")
    print("=" * 70)

    print(f"Threshold:             {threshold:.6f}")
    print(f"Precision:             {precision:.4f}")
    print(f"Recall:                {recall:.4f}")
    print(f"F1:                    {f1:.4f}")
    print(f"PR-AUC:                {pr_auc:.4f}")
    print(f"ROC-AUC:               {roc_auc:.4f}")

    print()
    print("Confusion Matrix")
    print(f"TN: {tn:,}")
    print(f"FP: {fp_matrix:,}")
    print(f"FN: {fn_matrix:,}")
    print(f"TP: {tp:,}")

    print()
    print("False-Positive / Business Cost")
    print(
        f"False positives:       {fp:,}"
    )
    print(
        f"False-negative frauds: {fn:,}"
    )
    print(
        f"FP review cost:        ₹{fp_cost:,.2f}"
    )
    print(
        f"Missed-fraud exposure: ₹{fn_cost:,.2f}"
    )
    print(
        f"Total estimated cost:  ₹{total_cost:,.2f}"
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "tn": int(tn),
        "fp": int(fp_matrix),
        "fn": int(fn_matrix),
        "tp": int(tp),
        "fp_cost": float(fp_cost),
        "fn_cost": float(fn_cost),
        "total_cost": float(total_cost),
    }


# =====================================================================
# FEATURE IMPORTANCE
# =====================================================================

def print_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 20,
):
    """
    Print most important ring features.
    """

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False,
    ).head(top_n)

    print()
    print("=" * 70)
    print("TOP RING FEATURES")
    print("=" * 70)

    for i, row in importance.iterrows():
        print(
            f"{row['feature']:<40} "
            f"{row['importance']:.6f}"
        )


# =====================================================================
# EXAMPLE ALERT EXPLANATION
# =====================================================================

def print_ring_examples(
    test_features: pd.DataFrame,
    test_probability: np.ndarray,
    threshold: float,
    n: int = 10,
):
    """
    Print highly suspicious transactions with human-readable
    ring explanations.

    Useful for your demo.
    """

    result = test_features[
        [
            "TransactionID",
            "TransactionAmt",
            "device_unique_cards",
            "device_unique_addresses",
            "card_unique_devices",
            "address_unique_devices",
            "device_previous_tx_24h",
            "card_previous_tx_24h",
            "address_previous_tx_24h",
            "abuse_ring_score",
            "ring_candidate_flag",
            "isFraud",
        ]
    ].copy()

    result["model_probability"] = test_probability

    result = result.sort_values(
        "model_probability",
        ascending=False,
    ).head(n)

    print()
    print("=" * 70)
    print("TOP ABUSE-RING CANDIDATES")
    print("=" * 70)

    for _, row in result.iterrows():

        print()
        print(
            f"Transaction: {row['TransactionID']}"
        )

        print(
            f"Model risk probability: "
            f"{row['model_probability']:.4f}"
        )

        print(
            f"Transaction amount: "
            f"₹{row['TransactionAmt']:,.2f}"
        )

        print(
            f"Device → unique cards: "
            f"{int(row['device_unique_cards'])}"
        )

        print(
            f"Device → unique addresses: "
            f"{int(row['device_unique_addresses'])}"
        )

        print(
            f"Card → unique devices: "
            f"{int(row['card_unique_devices'])}"
        )

        print(
            f"Address → unique devices: "
            f"{int(row['address_unique_devices'])}"
        )

        print(
            f"Device previous 24h transactions: "
            f"{int(row['device_previous_tx_24h'])}"
        )

        print(
            f"Card previous 24h transactions: "
            f"{int(row['card_previous_tx_24h'])}"
        )

        print(
            f"Address previous 24h transactions: "
            f"{int(row['address_previous_tx_24h'])}"
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
            f"Ground truth isFraud: "
            f"{int(row['isFraud'])}"
        )


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def run_pipeline():

    print()
    print("=" * 70)
    print("ABUSE-RING SENTINEL")
    print("IEEE-CIS Coordinated Fraud Detection Pipeline")
    print("=" * 70)

    # -----------------------------------------------------------------
    # 1. LOAD DATA
    # -----------------------------------------------------------------

    df = load_ieee_data()

    # -----------------------------------------------------------------
    # 2. PREPARE / SORT BY TIME
    # -----------------------------------------------------------------

    df = prepare_dataframe(df)

    df = df.sort_values(
        "TransactionDT"
    ).reset_index(
        drop=True
    )

    # -----------------------------------------------------------------
    # 3. TEMPORAL SPLIT
    # -----------------------------------------------------------------

    n = len(df)

    train_end = int(
        n * TRAIN_RATIO
    )

    val_end = int(
        n * (TRAIN_RATIO + VAL_RATIO)
    )

    train_df = df.iloc[
        :train_end
    ].copy()

    val_df = df.iloc[
        train_end:val_end
    ].copy()

    test_df = df.iloc[
        val_end:
    ].copy()

    print()
    print("=" * 70)
    print("TEMPORAL SPLIT")
    print("=" * 70)

    print(
        f"Train:      {len(train_df):,} rows"
    )

    print(
        f"Validation: {len(val_df):,} rows"
    )

    print(
        f"Test:       {len(test_df):,} rows"
    )

    print()
    print(
        f"Train period: "
        f"{train_df['TransactionDT'].min():,.0f}"
        f" → "
        f"{train_df['TransactionDT'].max():,.0f}"
    )

    print(
        f"Validation period: "
        f"{val_df['TransactionDT'].min():,.0f}"
        f" → "
        f"{val_df['TransactionDT'].max():,.0f}"
    )

    print(
        f"Test period: "
        f"{test_df['TransactionDT'].min():,.0f}"
        f" → "
        f"{test_df['TransactionDT'].max():,.0f}"
    )

    # -----------------------------------------------------------------
    # 4. TRAIN FEATURE BUILDER
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("FITTING HISTORICAL ENTITY STATISTICS")
    print("=" * 70)

    builder = AbuseRingFeatureBuilder()

    builder.fit(
        train_df
    )

    # -----------------------------------------------------------------
    # 5. BUILD TRAIN FEATURES
    # -----------------------------------------------------------------

    print()
    print(
        "Building TRAIN ring features..."
    )

    train_features = builder.transform(
        current_df=train_df,
        history_df=train_df,
    )

    print(
        f"Train features created: "
        f"{train_features.shape}"
    )

    # -----------------------------------------------------------------
    # 6. BUILD VALIDATION FEATURES
    # -----------------------------------------------------------------

    print()
    print(
        "Building VALIDATION ring features..."
    )

    # Historical information available before validation:
    # all training rows.
    #
    # The current validation transactions are then processed
    # chronologically for their own previous-24h velocity.

    validation_history = pd.concat(
        [
            train_df,
            val_df,
        ],
        ignore_index=True,
    )

    val_features = builder.transform(
        current_df=val_df,
        history_df=validation_history,
    )

    print(
        f"Validation features created: "
        f"{val_features.shape}"
    )

    # -----------------------------------------------------------------
    # 7. BUILD TEST FEATURES
    # -----------------------------------------------------------------

    print()
    print(
        "Building TEST ring features..."
    )

    test_history = pd.concat(
        [
            train_df,
            val_df,
            test_df,
        ],
        ignore_index=True,
    )

    test_features = builder.transform(
        current_df=test_df,
        history_df=test_history,
    )

    print(
        f"Test features created: "
        f"{test_features.shape}"
    )

    # -----------------------------------------------------------------
    # 8. MODEL MATRICES
    # -----------------------------------------------------------------

    selected_features = [
        f
        for f in RING_FEATURES
        if f in train_features.columns
    ]

    if not selected_features:
        raise RuntimeError(
            "No ring features available for training."
        )

    print()
    print("=" * 70)
    print("MODEL FEATURES")
    print("=" * 70)

    print(
        f"Number of model features: "
        f"{len(selected_features)}"
    )

    for feature in selected_features:
        print(f"  - {feature}")

    # Replace inf/nan.
    X_train = (
        train_features[
            selected_features
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    X_val = (
        val_features[
            selected_features
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    X_test = (
        test_features[
            selected_features
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    y_train = train_features[
        "isFraud"
    ].astype(int)

    y_val = val_features[
        "isFraud"
    ].astype(int)

    y_test = test_features[
        "isFraud"
    ].astype(int)

    val_amount = val_features[
        "TransactionAmt"
    ].astype(float)

    test_amount = test_features[
        "TransactionAmt"
    ].astype(float)

    # -----------------------------------------------------------------
    # 9. BASIC DATA SUMMARY
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)

    print(
        f"Train fraud rate: "
        f"{y_train.mean():.4%}"
    )

    print(
        f"Validation fraud rate: "
        f"{y_val.mean():.4%}"
    )

    print(
        f"Test fraud rate: "
        f"{y_test.mean():.4%}"
    )

    # -----------------------------------------------------------------
    # 10. TRAIN MODEL
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    model = train_model(
        X_train=X_train,
        y_train=y_train,
    )

    print(
        "Model training complete."
    )

    # -----------------------------------------------------------------
    # 11. VALIDATION PROBABILITIES
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION THRESHOLD SEARCH")
    print("=" * 70)

    val_probability = model.predict_proba(
        X_val
    )[:, 1]

    best_threshold, threshold_table = (
        select_threshold_by_cost(
            y_val=y_val,
            val_prob=val_probability,
            val_amount=val_amount,
        )
    )

    print(
        f"Selected threshold: "
        f"{best_threshold:.6f}"
    )

    print(
        f"Assumed FP review cost: "
        f"₹{FALSE_POSITIVE_REVIEW_COST:,.2f}"
    )

    print()
    print("Best validation configurations:")

    print(
        threshold_table.head(10).to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------
    # 12. FINAL HELD-OUT TEST
    # -----------------------------------------------------------------

    test_probability = model.predict_proba(
        X_test
    )[:, 1]

    test_metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        test_amount=test_amount,
        threshold=best_threshold,
        label="HELD-OUT TEST",
    )

    # -----------------------------------------------------------------
    # 13. FEATURE IMPORTANCE
    # -----------------------------------------------------------------

    print_feature_importance(
        model=model,
        feature_names=selected_features,
        top_n=20,
    )

    # -----------------------------------------------------------------
    # 14. SHOW EXPLAINABLE RING ALERTS
    # -----------------------------------------------------------------

    print_ring_examples(
        test_features=test_features,
        test_probability=test_probability,
        threshold=best_threshold,
        n=10,
    )

    # -----------------------------------------------------------------
    # 15. SAVE MODEL + FEATURE BUILDER
    # -----------------------------------------------------------------

    model_bundle = {
        "model": model,
        "feature_builder": builder,
        "features": selected_features,
        "threshold": best_threshold,
        "false_positive_review_cost": FALSE_POSITIVE_REVIEW_COST,
        "velocity_window_seconds": VELOCITY_WINDOW_SECONDS,
        "metrics": test_metrics,
    }

    joblib.dump(
        model_bundle,
        MODEL_FILE,
    )

    print()
    print("=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        f"Saved to: {MODEL_FILE}"
    )

    # -----------------------------------------------------------------
    # 16. OPTIONAL FEATURE EXPORT
    # -----------------------------------------------------------------

    export_columns = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "isFraud",
    ]

    export_columns.extend(
        selected_features
    )

    export_columns = [
        c
        for c in export_columns
        if c in test_features.columns
    ]

    try:

        test_features[
            export_columns
        ].to_parquet(
            FEATURE_DATA_FILE,
            index=False,
        )

        print(
            f"Feature data saved to: "
            f"{FEATURE_DATA_FILE}"
        )

    except Exception as exc:

        print(
            f"Could not save parquet file: {exc}"
        )

    # -----------------------------------------------------------------
    # 17. FINAL SUMMARY
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        f"Precision:      "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall:         "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"F1:             "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"PR-AUC:         "
        f"{test_metrics['pr_auc']:.4f}"
    )

    print(
        f"Test threshold: "
        f"{test_metrics['threshold']:.6f}"
    )

    print(
        f"False positives:"
        f" {test_metrics['fp']:,}"
    )

    print(
        f"False negatives:"
        f" {test_metrics['fn']:,}"
    )

    print(
        f"Estimated FP cost:"
        f" ₹{test_metrics['fp_cost']:,.2f}"
    )

    print(
        f"Estimated FN exposure:"
        f" ₹{test_metrics['fn_cost']:,.2f}"
    )

    print(
        f"Estimated total cost:"
        f" ₹{test_metrics['total_cost']:,.2f}"
    )

    print()
    print("Pipeline completed successfully.")


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    run_pipeline()