#!/usr/bin/env python3
"""
SENTINELGRAPH - FORENSIC DECISION EXPLAINER MODULE
==================================================
Translates raw model risk probabilities and graph topologies
into investigator-facing forensic decision cards for BFSI ops.
(Template-driven explanation layer over existing model outputs - takes no
autonomous action).

This is the SINGLE CANONICAL definition of SentinelGraphExplainer. Other
scripts (e.g. run_sentinel_pipeline.py) should import this class rather than
redefining it, to avoid two copies drifting out of sync.

CHANGELOG:
    - render_investigator_card() now accepts a demo_mode flag. When the data
      fed to the explainer is fabricated/synthetic (e.g. a fallback stub used
      because real test data wasn't available), demo_mode=True renders a
      clearly labeled banner and disclaimer so a fabricated illustration is
      never mistaken for a real extraction.
"""

import pandas as pd
import numpy as np

USD_TO_INR = 83.00


class SentinelGraphExplainer:
    def __init__(self, df, predictions, threshold=0.105172):
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

        amt_col = "TransactionAmt" if "TransactionAmt" in cluster_df.columns else None
        total_exposure_inr = float((cluster_df[amt_col] * USD_TO_INR).sum()) if amt_col else 0.0

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
            "cluster_velocity_flag": "High Velocity Syndicate" if total_txs > 3 else "Isolated Anomaly",
        }

    def render_investigator_card(self, transaction_id, demo_mode=False):
        """
        Renders the terminal-based investigator UI card.

        demo_mode: set True when the underlying df/predictions passed to this
        explainer are fabricated/synthetic (e.g. a fallback stub used because
        real test data was unavailable). This adds a visible banner and
        disclaimer so the card can never be mistaken for a real extraction.
        """
        data = self.extract_abuse_ring_subgraph(transaction_id)
        if "error" in data:
            return data["error"]

        banner = "🚨 [SYNTHETIC DEMO DATA] " if demo_mode else "🚨 "

        card = f"""
        ┌────────────────────────────────────────────────────────┐
        │   {banner}{data['status']}
        ├────────────────────────────────────────────────────────┤
        │ Transaction ID: {data['transaction_id']}
        │ Risk Score:     {data['risk_score']}/100
        │ Accounts Bound: {data['total_accounts']}
        │ Transactions:   {data['total_transactions']}
        │ Financial Risk: {data['exposure_inr']}
        ├────────────────────────────────────────────────────────┤
        │ 🔗 Graph Telemetry Footprint:
        │    • Shared Devices:   {data['shared_devices']}
        │    • Shared Subnets:   {data['shared_ips']}
        │    • Shared Addresses: {data['shared_addresses']}
        │    • Syndicate Type:   {data['cluster_velocity_flag']}
        └────────────────────────────────────────────────────────┘
        """
        if demo_mode:
            card += (
                "\n        ⚠️  NOTE: This card uses illustrative synthetic ring data\n"
                "            to demonstrate the card FORMAT only. Only the flagged\n"
                "            Transaction ID and its real model score (if provided)\n"
                "            are genuine — surrounding entities were fabricated.\n"
            )
        return card


# Example usage test for verification
if __name__ == "__main__":
    # Minimal smoke test to confirm the module loads and renders correctly —
    # this stub is intentionally labeled with demo_mode=True since it is
    # obviously synthetic data, not a real extraction.
    print("SentinelGraph Explainer Module Initialized Successfully.")

    sample_df = pd.DataFrame({
        "TransactionID": [1000001, 1000002],
        "TransactionAmt": [250.0, 90.0],
        "_device": ["Test_Device_1", "Test_Device_1"],
        "_card": ["1111", "1111"],
        "_address": ["500", "500"],
    })
    sample_preds = [0.87, 0.34]

    explainer = SentinelGraphExplainer(sample_df, sample_preds, threshold=0.105172)
    print(explainer.render_investigator_card(1000001, demo_mode=True))