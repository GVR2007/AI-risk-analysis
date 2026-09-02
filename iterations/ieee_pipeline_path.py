#!/usr/bin/env python3
"""IEEE-CIS Fraud Detection Pipeline configured for the user's folder path."""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, average_precision_score, confusion_matrix
import joblib

def run_pipeline():
    # Exact target path from your screenshot
    data_dir = "test_datasets/test_datasets/kaggle/ieee-fraud-detection"
    if not os.path.exists(data_dir):
        # Fallback search if path differs slightly in root
        for root, dirs, files in os.walk('.'):
            if "train_transaction.csv" in files:
                data_dir = root
                break
                
    trans_path = os.path.join(data_dir, "train_transaction.csv")
    ident_path = os.path.join(data_dir, "train_identity.csv")
    
    print(f"Loading data from directory: {data_dir}")
    if not os.path.exists(trans_path):
        print(f"Error: Could not find train_transaction.csv at {trans_path}")
        return
        
    trans = pd.read_csv(trans_path)
    ident = pd.read_csv(ident_path) if os.path.exists(ident_path) else None
    
    if ident is not None:
        print("Merging transaction and identity tables on TransactionID...")
        df = pd.merge(trans, ident, on="TransactionID", how="left")
    else:
        df = trans
        
    print("Engineering network and graph relational features...")
    device_col = "DeviceInfo" if "DeviceInfo" in df.columns else ("id_31" if "id_31" in df.columns else None)
    if device_col:
        dev_counts = df[device_col].value_counts().to_dict()
        df["shared_device_frequency"] = df[device_col].map(dev_counts).fillna(1)
    else:
        df["shared_device_frequency"] = 1
        
    if "card1" in df.columns:
        card_counts = df["card1"].value_counts().to_dict()
        df["card_clustering_degree"] = df["card1"].map(card_counts).fillna(1)
    else:
        df["card_clustering_degree"] = 1
        
    if "addr1" in df.columns:
        addr_counts = df["addr1"].value_counts().to_dict()
        df["address_velocity"] = df["addr1"].map(addr_counts).fillna(1)
    else:
        df["address_velocity"] = 1
        
    target_col = "isFraud"
    feature_cols = ["TransactionAmt", "shared_device_frequency", "card_clustering_degree", "address_velocity"]
    
    for c in ["card2", "card3", "card5", "addr2", "dist1"]:
        if c in df.columns:
            feature_cols.append(c)
            df[c] = df[c].fillna(df[c].median())
            
    df = df.dropna(subset=[target_col])
    X = df[feature_cols].fillna(0)
    y = df[target_col]
    
    print(f"Dataset shape: {X.shape}, Fraud rate: {y.mean():.4%}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print("Training XGBoost model...")
    scale_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=scale_weight,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n=== IEEE-CIS Evaluation Metrics ===")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"PR-AUC:    {average_precision_score(y_test, y_prob):.4f}")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"Confusion Matrix -> TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
    
    joblib.dump(model, "ieee_sentinel_model.pkl")
    print("Saved model to ieee_sentinel_model.pkl")

if __name__ == "__main__":
    run_pipeline()