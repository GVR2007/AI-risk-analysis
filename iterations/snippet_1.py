import pandas as pd
import joblib
from sentinel_explainer import SentinelGraphExplainer

# 1. Load submission scores & test dataframe metadata
sub = pd.read_csv("submission.csv")
# Get the highest risk transaction ID flagged by your model
top_tx_id = int(sub.sort_values("isFraud", ascending=False).iloc[0]["TransactionID"])
top_prob = float(sub.sort_values("isFraud", ascending=False).iloc[0]["isFraud"])

print(f"Top Flagged Transaction ID: {top_tx_id} with Risk Probability: {top_prob:.4f}")

# 2. Re-instantiate explainer with test features or data context
# (Passing your processed test frame or dataframe slice)