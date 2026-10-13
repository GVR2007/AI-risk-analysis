"""Model loading and inference for the AI Risk Analysis web app.

Every preprocessing step here mirrors the notebooks exactly, so a record typed into the
website goes through the same transformations as the training data did:

* Loan risk (regression)  -> notebooks/regression.ipynb      (tuned Gradient Boosting Regressor)
* Claim fraud (classif.)  -> notebooks/classification.ipynb  (tuned Bagging Classifier)
* Card segments (cluster) -> notebooks/clustering.ipynb      (K-Means, k=4, bundle with scaler + PCA)
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
STORE = APP_DIR / "model_store"
MODELS = ROOT / "models"
DATA = ROOT / "data"

LOAN_CSV = DATA / "financial_risk_loan_approval" / "Loan.csv"
CLAIMS_CSV = DATA / "insurance_claims" / "insurance_claims.csv"
CARDS_CSV = DATA / "credit_card_clustering" / "CC GENERAL.csv"

RNG = np.random.default_rng()

# Public Hugging Face model repo holding every trained .pkl from models/ (several hundred MB in
# total — too large to commit to git, including a 200MB+ Random Forest Regressor). Only the 3
# small default models are committed to app/model_store/; every other algorithm is downloaded
# from here on first use and cached in models/, so the full 22-algorithm selector works in any
# deployment without bloating the repository.
HF_REPO_ID = "fakechamp24/ai-risk-analysis-models"
HF_MODELS_URL = f"https://huggingface.co/{HF_REPO_ID}/resolve/main/models"

# Every algorithm the course guidelines require, per track, mapped to its saved pipeline.
# The *_DEFAULT entry is the one file committed to app/model_store/ (and so always available,
# including in production, with zero download); every other entry is resolved from the project's
# models/ directory locally, or downloaded from Hugging Face on first use otherwise.
REGRESSION_ALGORITHMS = {
    "Linear Regression": "reg_linear_regression.pkl",
    "Ridge Regression": "reg_ridge_regression.pkl",
    "Lasso Regression": "reg_lasso_regression.pkl",
    "ElasticNet Regression": "reg_elasticnet_regression.pkl",
    "Polynomial Regression": "reg_polynomial_regression_deg2.pkl",
    "Decision Tree Regressor": "reg_decision_tree_regressor.pkl",
    "Random Forest Regressor": "reg_random_forest_regressor.pkl",
    "Gradient Boosting Regressor": "loan_risk_gbr.pkl",
    "Support Vector Regressor (SVR)": "reg_support_vector_regressor_svr.pkl",
    "K-Nearest Neighbors Regressor": "reg_k_nearest_neighbors_regressor.pkl",
}
REGRESSION_DEFAULT = "Gradient Boosting Regressor"

CLASSIFICATION_ALGORITHMS = {
    "Logistic Regression": "clf_tuned_logistic_regression.pkl",
    "K-Nearest Neighbors": "clf_tuned_k_nearest_neighbors.pkl",
    "Naive Bayes (Gaussian)": "clf_tuned_naive_bayes_gaussian.pkl",
    "Decision Tree Classifier": "clf_tuned_decision_tree_classifier.pkl",
    "Support Vector Machine (SVC)": "clf_tuned_support_vector_machine_svc.pkl",
    "Random Forest Classifier": "clf_tuned_random_forest_classifier.pkl",
    "AdaBoost Classifier": "clf_tuned_adaboost_classifier.pkl",
    "Gradient Boosting Classifier": "clf_tuned_gradient_boosting_classifier.pkl",
    "Bagging Classifier": "claim_fraud_bagging.pkl",
    "MLP Classifier (Neural Network)": "clf_tuned_mlp_classifier_neural_network.pkl",
}
CLASSIFICATION_DEFAULT = "Bagging Classifier"

CLUSTERING_ALGORITHMS = {
    "K-Means": "card_segments_kmeans.pkl",
    "Agglomerative (Ward)": "clu_agglomerative_ward.pkl",
}
CLUSTERING_DEFAULT = "K-Means"


def _resolve_model_path(filename: str) -> Path:
    """Deployed default models live in app/model_store/ (always available, including in a fresh
    production deploy). Every other algorithm is read from the project's models/ directory when
    the notebooks have been run locally; if it isn't there, it's downloaded once from the public
    Hugging Face model repo and cached in models/, so the full algorithm selector works in
    production too without committing hundreds of MB of .pkl files to git."""
    for base in (STORE, MODELS):
        path = base / filename
        if path.exists():
            return path

    MODELS.mkdir(parents=True, exist_ok=True)
    dest = MODELS / filename
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(f"{HF_MODELS_URL}/{filename}", tmp)
        tmp.replace(dest)
    except (URLError, HTTPError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise FileNotFoundError(
            f"'{filename}' couldn't be found locally or downloaded from the Hugging Face model "
            f"repo ({HF_REPO_ID}): {exc}"
        ) from exc
    return dest


def _clean(value):
    """Make numpy / pandas scalars JSON-serialisable."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


# --------------------------------------------------------------------------------------
# Loan risk — regression
# --------------------------------------------------------------------------------------
class LoanRiskModel:
    CAP_COLS = ["AnnualIncome", "LoanAmount", "TotalAssets", "SavingsAccountBalance"]
    CATEGORICAL = ["EmploymentStatus", "EducationLevel", "MaritalStatus", "HomeOwnershipStatus", "LoanPurpose"]
    # Inputs the user types; the remaining model columns are derived exactly as the dataset generator does.
    INPUTS = [
        "Age", "EducationLevel", "EmploymentStatus", "Experience", "JobTenure", "MaritalStatus",
        "NumberOfDependents", "HomeOwnershipStatus",
        "AnnualIncome", "SavingsAccountBalance", "CheckingAccountBalance", "TotalAssets", "TotalLiabilities",
        "CreditScore", "CreditCardUtilizationRate", "NumberOfOpenCreditLines", "NumberOfCreditInquiries",
        "DebtToIncomeRatio", "MonthlyDebtPayments", "PaymentHistory", "LengthOfCreditHistory",
        "UtilityBillsPaymentHistory", "BankruptcyHistory", "PreviousLoanDefaults",
        "LoanAmount", "LoanDuration", "LoanPurpose", "InterestRate",
    ]
    DRIVERS = ["CreditScore", "AnnualIncome", "DebtToIncomeRatio", "PaymentHistory", "BankruptcyHistory",
               "PreviousLoanDefaults", "CreditCardUtilizationRate", "LengthOfCreditHistory", "TotalAssets",
               "LoanAmount", "EmploymentStatus"]

    algorithms = list(REGRESSION_ALGORITHMS)
    default_algorithm = REGRESSION_DEFAULT

    def __init__(self):
        self.model = joblib.load(STORE / "loan_risk_gbr.pkl")
        self._cache = {REGRESSION_DEFAULT: self.model}
        raw = pd.read_csv(LOAN_CSV)
        self.raw = raw
        self.fences = {}
        for col in self.CAP_COLS:
            q1, q3 = raw[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            self.fences[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        self.scores = np.sort(raw["RiskScore"].to_numpy())
        self.q25, self.q75 = np.percentile(self.scores, [25, 75])
        self.medians = raw[self.INPUTS].median(numeric_only=True).to_dict()
        self.modes = {c: raw[c].mode()[0] for c in self.CATEGORICAL}
        self.options = {c: sorted(raw[c].unique().tolist()) for c in self.CATEGORICAL}

    def _get_model(self, name: str | None):
        name = name or REGRESSION_DEFAULT
        if name not in REGRESSION_ALGORITHMS:
            raise ValueError(f"Unknown regression algorithm '{name}'.")
        if name not in self._cache:
            self._cache[name] = joblib.load(_resolve_model_path(REGRESSION_ALGORITHMS[name]))
        return name, self._cache[name]

    @staticmethod
    def derive(rec: dict) -> dict:
        """Recompute the dependent columns with the formulas from data/.../CSV Generation.py."""
        r = dict(rec)
        r["MonthlyIncome"] = r["AnnualIncome"] / 12
        r["NetWorth"] = max(r["TotalAssets"] - r["TotalLiabilities"], 1000)
        r["BaseInterestRate"] = 0.03 + (850 - r["CreditScore"]) / 2000 + r["LoanAmount"] / 1_000_000 + r["LoanDuration"] / 1200
        # The offered rate is an input; if it is left blank, use the generator's base-rate formula.
        if r.get("InterestRate") in (None, "") or not np.isfinite(float(r["InterestRate"])):
            r["InterestRate"] = r["BaseInterestRate"]
        i = r["InterestRate"] / 12
        n = r["LoanDuration"]
        r["MonthlyLoanPayment"] = r["LoanAmount"] * i / (1 - (1 + i) ** (-n))
        r["TotalDebtToIncomeRatio"] = (r["MonthlyDebtPayments"] + r["MonthlyLoanPayment"]) / r["MonthlyIncome"]
        return r

    def _frame(self, rec: dict, model=None) -> pd.DataFrame:
        model = model if model is not None else self.model
        r = self.derive(rec)
        for col, (lo, hi) in self.fences.items():
            r[col] = float(np.clip(r[col], lo, hi))
        r["LoanToIncomeRatio"] = r["LoanAmount"] / (r["AnnualIncome"] + 1)
        r["CreditHistoryPerAge"] = r["LengthOfCreditHistory"] / r["Age"]
        return pd.DataFrame([r])[list(model.feature_names_in_)]

    def parse(self, payload: dict) -> dict:
        rec = {}
        for col in self.INPUTS:
            val = payload.get(col)
            if col in self.CATEGORICAL:
                rec[col] = val if val in self.options[col] else self.modes[col]
            elif col == "InterestRate":
                rec[col] = float(val) if val not in (None, "") else None
            else:
                rec[col] = float(val) if val not in (None, "") else float(self.medians[col])
        return rec

    def predict(self, payload: dict) -> dict:
        algo_name, model = self._get_model(payload.get("algorithm"))
        rec = self.parse(payload)
        score = float(model.predict(self._frame(rec, model))[0])
        percentile = float(np.searchsorted(self.scores, score) / len(self.scores) * 100)
        band = "Low" if score < self.q25 else ("High" if score > self.q75 else "Moderate")

        # "What moved the score": swap one input at a time for the typical applicant's value.
        drivers = []
        for col in self.DRIVERS:
            typical = self.modes[col] if col in self.CATEGORICAL else self.medians[col]
            if rec[col] == typical:
                continue
            alt = dict(rec)
            alt[col] = typical
            delta = score - float(model.predict(self._frame(alt, model))[0])
            drivers.append({"feature": col, "value": rec[col], "typical": typical, "impact": round(delta, 2)})
        drivers = [d for d in drivers if abs(d["impact"]) >= 0.05]
        drivers.sort(key=lambda d: abs(d["impact"]), reverse=True)

        derived = self.derive(rec)
        return {
            "algorithm": algo_name,
            "score": round(score, 2),
            "band": band,
            "percentile": round(percentile, 1),
            "scale": {"min": float(self.scores[0]), "max": float(self.scores[-1]),
                      "q25": float(self.q25), "q75": float(self.q75)},
            "drivers": drivers[:6],
            "derived": {k: round(float(derived[k]), 4) for k in
                        ["MonthlyIncome", "NetWorth", "InterestRate", "MonthlyLoanPayment", "TotalDebtToIncomeRatio"]},
        }

    def sample(self, kind: str = "random") -> dict:
        df = self.raw
        if kind == "low":
            df = df[df["RiskScore"] < self.q25]
        elif kind == "high":
            df = df[df["RiskScore"] > self.q75]
        row = df.iloc[int(RNG.integers(len(df)))]
        return {"inputs": {c: _clean(row[c]) for c in self.INPUTS}, "actual": float(row["RiskScore"])}


# --------------------------------------------------------------------------------------
# Claim fraud — classification
# --------------------------------------------------------------------------------------
class ClaimFraudModel:
    DROP = ["_c39", "policy_number", "insured_zip", "incident_location", "fraud_reported"]
    DRIVERS = ["incident_severity", "insured_hobbies", "total_claim_amount", "vehicle_claim", "property_claim",
               "injury_claim", "policy_annual_premium", "collision_type", "authorities_contacted",
               "police_report_available", "witnesses", "number_of_vehicles_involved"]

    algorithms = list(CLASSIFICATION_ALGORITHMS)
    default_algorithm = CLASSIFICATION_DEFAULT

    def __init__(self):
        self.model = joblib.load(STORE / "claim_fraud_bagging.pkl")
        self._cache = {CLASSIFICATION_DEFAULT: self.model}
        raw = pd.read_csv(CLAIMS_CSV).replace("?", np.nan)
        self.raw = raw
        self.inputs = [c for c in raw.columns if c not in self.DROP]
        self.categorical = [c for c in self.inputs if raw[c].dtype == object and c not in ("policy_bind_date", "incident_date")]
        self.numeric = [c for c in self.inputs if c not in self.categorical and c not in ("policy_bind_date", "incident_date")]
        self.options = {c: sorted(raw[c].dropna().unique().tolist()) for c in self.categorical}
        self.medians = raw[self.numeric].median().to_dict()
        self.modes = {c: raw[c].mode()[0] for c in self.categorical}
        # auto_model options per make, so the form only offers real combinations.
        self.models_by_make = raw.groupby("auto_make")["auto_model"].apply(lambda s: sorted(s.unique().tolist())).to_dict()

    def _get_model(self, name: str | None):
        name = name or CLASSIFICATION_DEFAULT
        if name not in CLASSIFICATION_ALGORITHMS:
            raise ValueError(f"Unknown classification algorithm '{name}'.")
        if name not in self._cache:
            self._cache[name] = joblib.load(_resolve_model_path(CLASSIFICATION_ALGORITHMS[name]))
        return name, self._cache[name]

    def _frame(self, rec: dict, model=None) -> pd.DataFrame:
        model = model if model is not None else self.model
        r = dict(rec)
        r["claim_to_premium_ratio"] = r["total_claim_amount"] / (r["policy_annual_premium"] + 1)
        r["days_bind_to_incident"] = (pd.to_datetime(r["incident_date"]) - pd.to_datetime(r["policy_bind_date"])).days
        r.pop("policy_bind_date")
        r.pop("incident_date")
        return pd.DataFrame([r])[list(model.feature_names_in_)]

    def parse(self, payload: dict) -> dict:
        rec = {}
        for col in self.inputs:
            val = payload.get(col)
            if col in ("policy_bind_date", "incident_date"):
                rec[col] = val or ("2010-01-01" if col == "policy_bind_date" else "2015-01-15")
            elif col in self.categorical:
                rec[col] = val if val in self.options[col] else np.nan   # unknown -> imputed by the pipeline
            else:
                rec[col] = float(val) if val not in (None, "") else float(self.medians[col])
        # total claim is defined as the sum of its three components
        rec["total_claim_amount"] = rec["injury_claim"] + rec["property_claim"] + rec["vehicle_claim"]
        return rec

    def predict(self, payload: dict) -> dict:
        algo_name, model = self._get_model(payload.get("algorithm"))
        rec = self.parse(payload)
        p = float(model.predict_proba(self._frame(rec, model))[0, 1])
        band = "Low" if p < 0.3 else ("Elevated" if p < 0.5 else "High")

        drivers = []
        for col in self.DRIVERS:
            typical = self.modes[col] if col in self.categorical else self.medians[col]
            if rec[col] == typical or (isinstance(rec[col], float) and np.isnan(rec[col])):
                continue
            alt = dict(rec)
            alt[col] = typical
            if col in ("injury_claim", "property_claim", "vehicle_claim"):
                alt["total_claim_amount"] = alt["injury_claim"] + alt["property_claim"] + alt["vehicle_claim"]
            delta = p - float(model.predict_proba(self._frame(alt, model))[0, 1])
            drivers.append({"feature": col, "value": _clean(rec[col]), "typical": _clean(typical), "impact": round(delta, 3)})
        drivers = [d for d in drivers if abs(d["impact"]) >= 0.002]
        drivers.sort(key=lambda d: abs(d["impact"]), reverse=True)
        return {
            "algorithm": algo_name,
            "probability": round(p, 4),
            "band": band,
            "decision": "Refer to fraud investigation" if p >= 0.5 else "Process normally",
            "drivers": drivers[:6],
            "claim_to_premium_ratio": round(rec["total_claim_amount"] / (rec["policy_annual_premium"] + 1), 2),
            "days_bind_to_incident": int((pd.to_datetime(rec["incident_date"]) - pd.to_datetime(rec["policy_bind_date"])).days),
        }

    def sample(self, kind: str = "random") -> dict:
        df = self.raw
        if kind == "fraud":
            df = df[df["fraud_reported"] == "Y"]
        elif kind == "genuine":
            df = df[df["fraud_reported"] == "N"]
        row = df.iloc[int(RNG.integers(len(df)))]
        return {"inputs": {c: _clean(row[c]) for c in self.inputs}, "actual": row["fraud_reported"]}


# --------------------------------------------------------------------------------------
# Card-holder segments — clustering
# --------------------------------------------------------------------------------------
class CardSegmentModel:
    INPUTS = ["BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES",
              "CASH_ADVANCE", "PURCHASES_FREQUENCY", "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY",
              "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX", "PURCHASES_TRX", "CREDIT_LIMIT", "PAYMENTS",
              "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT", "TENURE"]
    COMPARE = ["CREDIT_UTILIZATION", "PAYMENT_TO_MIN_RATIO", "PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY",
               "PRC_FULL_PAYMENT"]

    algorithms = list(CLUSTERING_ALGORITHMS)
    default_algorithm = CLUSTERING_DEFAULT

    def __init__(self):
        self.b = joblib.load(STORE / "card_segments_kmeans.pkl")
        raw = pd.read_csv(CARDS_CSV)
        self.raw = raw
        self.medians = raw[self.INPUTS].median().to_dict()
        with open(STORE / "card_segments_points.json") as f:
            self.points = json.load(f)
        # Training-set features/projection, reused both for the K-Means segment medians below
        # and, lazily, to assign new customers to an Agglomerative (Ward) cluster — Ward has no
        # native predict() for unseen points, so new customers are matched to their nearest
        # already-labelled training point in this same PCA space (see _get_agglo_index).
        feats = self._features(raw[self.INPUTS].dropna(subset=["CREDIT_LIMIT"]).copy())
        self._z_train = self._project(feats)
        labels = self.b["kmeans"].predict(self._z_train)
        feats["cluster"] = labels
        self.segment_medians = feats.groupby("cluster")[self.COMPARE].median().to_dict(orient="index")
        self.segment_sizes = feats["cluster"].value_counts().to_dict()
        self._kmeans_labels_train = labels
        self._agglo = None  # lazy: {"labels": ..., "nn": NearestNeighbors, "ward_to_segment": {...}}

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        structural = df["MINIMUM_PAYMENTS"].isna() & (df["PAYMENTS"] == 0)
        df.loc[structural, "MINIMUM_PAYMENTS"] = 0.0
        df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(self.b["min_payments_median"])
        df["CASH_ADVANCE_FREQUENCY"] = df["CASH_ADVANCE_FREQUENCY"].clip(upper=1.0)
        df["CREDIT_UTILIZATION"] = df["BALANCE"] / df["CREDIT_LIMIT"]
        df["PAYMENT_TO_MIN_RATIO"] = df["PAYMENTS"] / (df["MINIMUM_PAYMENTS"] + 1)
        df["MONTHLY_PURCHASES"] = df["PURCHASES"] / df["TENURE"]
        df["MONTHLY_CASH_ADVANCE"] = df["CASH_ADVANCE"] / df["TENURE"]
        return df

    def _project(self, feats: pd.DataFrame):
        X = feats[self.b["feature_order"]].copy()
        X[self.b["skewed_cols"]] = np.log1p(X[self.b["skewed_cols"]])
        return self.b["pca"].transform(self.b["scaler"].transform(X))

    def _get_agglo_index(self):
        """Agglomerative clustering has no predict() for unseen points (sklearn only exposes the
        labels_ it assigned while fitting). A new customer is matched to the nearest already-labelled
        training point in the same PCA space instead (1-nearest-neighbour), and that Ward cluster is
        mapped to whichever K-Means segment most of its members overlap with, so the app can reuse the
        same named/interpreted segments the notebook already defined for K-Means."""
        if self._agglo is None:
            agglo = joblib.load(_resolve_model_path(CLUSTERING_ALGORITHMS["Agglomerative (Ward)"]))
            labels = agglo.labels_
            if len(labels) != len(self._z_train):
                raise ValueError("Saved Agglomerative labels don't match the training set; re-run clustering.ipynb.")
            centroids = np.array([self._z_train[labels == c].mean(axis=0) for c in np.unique(labels)])
            nn = NearestNeighbors(n_neighbors=1).fit(self._z_train)
            overlap = pd.crosstab(pd.Series(labels), pd.Series(self._kmeans_labels_train))
            ward_to_segment = {int(k): int(v) for k, v in overlap.idxmax(axis=1).to_dict().items()}
            self._agglo = {"labels": labels, "centroids": centroids, "nn": nn, "ward_to_segment": ward_to_segment}
        return self._agglo

    def predict(self, payload: dict) -> dict:
        algorithm = payload.get("algorithm") or CLUSTERING_DEFAULT
        if algorithm not in CLUSTERING_ALGORITHMS:
            raise ValueError(f"Unknown clustering algorithm '{algorithm}'.")

        rec = {}
        for col in self.INPUTS:
            val = payload.get(col)
            if col == "MINIMUM_PAYMENTS" and val in (None, ""):
                rec[col] = np.nan     # handled by the notebook's imputation rule
            else:
                rec[col] = float(val) if val not in (None, "") else float(self.medians[col])
        rec["CREDIT_LIMIT"] = max(rec["CREDIT_LIMIT"], 1.0)
        feats = self._features(pd.DataFrame([rec]))
        z = self._project(feats)
        X_scaled = self.b["scaler"].transform(self._log(feats))
        xy = self.b["pca_2d"].transform(X_scaled)[0]

        if algorithm == "K-Means":
            cluster = int(self.b["kmeans"].predict(z)[0])
            dists = np.linalg.norm(self.b["kmeans"].cluster_centers_ - z, axis=1)
            segment_key = cluster
            segment_source = "Direct K-Means cluster assignment (has a native predict() for new points)."
        else:
            idx = self._get_agglo_index()
            nn_row = idx["nn"].kneighbors(z, return_distance=False)[0, 0]
            cluster = int(idx["labels"][nn_row])
            dists = np.linalg.norm(idx["centroids"] - z, axis=1)
            segment_key = idx["ward_to_segment"][cluster]
            segment_source = (f"Nearest-neighbour match to Ward cluster {cluster}: Agglomerative clustering has "
                               "no native predict() for new points, so this customer is matched to its closest "
                               "already-labelled training point in the same PCA space; the segment name shown is "
                               "the K-Means segment this Ward cluster overlaps with most.")

        confidence = float(1 - dists.min() / np.sort(dists)[1]) if len(dists) > 1 else 1.0
        seg = self.b["segments"][segment_key]
        return {
            "algorithm": algorithm,
            "cluster": cluster,
            "segment": seg,
            "segment_source": segment_source,
            "confidence": round(confidence, 3),
            "position": {"x": round(float(xy[0]), 3), "y": round(float(xy[1]), 3)},
            "customer": {k: round(float(feats.iloc[0][k]), 4) for k in self.COMPARE},
            "segment_median": {k: round(float(v), 4) for k, v in self.segment_medians[segment_key].items()},
        }

    def _log(self, feats):
        X = feats[self.b["feature_order"]].copy()
        X[self.b["skewed_cols"]] = np.log1p(X[self.b["skewed_cols"]])
        return X

    def sample(self, kind: str = "random") -> dict:
        row = self.raw.iloc[int(RNG.integers(len(self.raw)))]
        return {"inputs": {c: _clean(row[c]) for c in self.INPUTS}}

    def meta(self) -> dict:
        return {
            "segments": {int(k): {**v, "size": int(self.segment_sizes.get(k, 0))} for k, v in self.b["segments"].items()},
            "points": self.points,
        }
