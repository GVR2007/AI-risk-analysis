#!/usr/bin/env python3
"""
SENTINELGRAPH - AGENTIC EXPLAINER MODULE
========================================
Translates raw model risk probabilities and graph topologies 
into investigator-facing forensic decision cards for BFSI ops.
"""

import pandas as pd
import numpy as np


class SentinelGraphExplainer:
    def __init__(self, df, predictions, threshold=0.1052):
        self.df = df.copy()
        self.df["predicted_prob"] = predictions
        self.threshold = threshold

    def extract_abuse_ring_subgraph(self, transaction_id):
        """Extracts the relational network footprint around a flagged transaction."""
        match_rows = self.df[self.df["TransactionID"] == transaction_id]
        if match_rows.empty:
            return {"error": "Transaction ID not found in dataset."}

        tx = match_rows.iloc[0]
        device = tx.get("_device", "__MISSING__")
        card = tx.get("_card", "__MISSING__")
        address = tx.get("_address", "__MISSING__")

        # Isolate connected nodes in the transaction cluster
        cluster_df = self.df[
            (self.df["_device"] == device) | 
            (self.df["_card"] == card) | 
            (self.df["_address"] == address)
        ]

        prob = float(tx["predicted_prob"])
        risk_score = round(prob * 100, 1)
        total_accounts = int(cluster_df["_card"].nunique()) if "_card" in cluster_df.columns else 1
        total_txs = len(cluster_df)
        
        amt_col = "TransactionAmt" if "TransactionAmt" in cluster_df.columns else 0
        total_exposure_inr = float((cluster_df[amt_col] * 83.00).sum()) if isinstance(amt_col, str) else 0.0

        shared_devices = int(cluster_df["_device"].nunique()) if "_device" in cluster_df.columns else 1
        shared_ips = int(cluster_df.get("addr2", pd.Series([1])).nunique())
        shared_addresses = int(cluster_df["_address"].nunique()) if "_address" in cluster_df.columns else 1

        return {
            "transaction_id": int(transaction_id),
            "risk_score": risk_score,
            "status": "ABUSE RING DETECTED" if prob >= self.threshold else "MONITORED NORMAL",
            "total_accounts": total_accounts,
            "total_transactions": total_txs,
            "exposure_inr": f"₹{total_exposure_inr:,.2f}",
            "shared_devices": shared_devices,
            "shared_ips": shared_ips,
            "shared_addresses": shared_addresses,
            "cluster_velocity_flag": "High Velocity Syndicate" if total_txs > 3 else "Isolated Anomaly"
        }

    def render_investigator_card(self, transaction_id):
        """Renders the terminal-based investigator UI card for hackathon presentation."""
        data = self.extract_abuse_ring_subgraph(transaction_id)
        if "error" in data:
            return data["error"]

        card = f"""
        ┌────────────────────────────────────────────────────────┐
        │   🚨 {data['status']}                              │
        ├────────────────────────────────────────────────────────┤
        │ Transaction ID: {data['transaction_id']}                               │
        │ Risk Score:     {data['risk_score']}/100                               │
        │ Accounts Bound: {data['total_accounts']}                               │
        │ Transactions:   {data['total_transactions']}                               │
        │ Financial Risk: {data['exposure_inr']}                         │
        ├────────────────────────────────────────────────────────┤
        │ 🔗 Graph Telemetry Footprint:                          │
        │    • Shared Devices:   {data['shared_devices']}                              │
        │    • Shared Subnets:   {data['shared_ips']}                              │
        │    • Shared Addresses: {data['shared_addresses']}                              │
        │    • Syndicate Type:   {data['cluster_velocity_flag']}      │
        └────────────────────────────────────────────────────────┘
        """
        return card


# Example usage test for verification
if __name__ == "__main__":
    # Test stub to demonstrate the UI card output format
    print("SentinelGraph Explainer Module Initialized Successfully.")