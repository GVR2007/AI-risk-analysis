"""Package the notebook outputs the web app needs into app/model_store/ and app/static/figures/.

Run this once after re-running the notebooks:

    python app/export_artifacts.py

It copies the three deployed models, collects every results table into one results.json for the
"Model results" page, and copies the key figures. The app itself never retrains anything.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pandas as pd

APP = Path(__file__).resolve().parent
ROOT = APP.parent
MODELS = ROOT / "models"
STORE = APP / "model_store"
FIGS = APP / "static" / "figures"

DEPLOYED = {
    "reg_gradient_boosting_regressor.pkl": "loan_risk_gbr.pkl",      # tuned GBR, regression.ipynb
    "clf_final_model.pkl": "claim_fraud_bagging.pkl",               # tuned Bagging, classification.ipynb
    "clu_kmeans_bundle.pkl": "card_segments_kmeans.pkl",            # K-Means bundle, clustering.ipynb
    "clu_pca_sample.json": "card_segments_points.json",
}

FIGURES = [
    "reg_residual_pred_vs_actual.png", "reg_feature_importance_rf.png", "reg_correlation_heatmap.png",
    "clf_roc_curves_all10.png", "clf_pca_vs_encoded.png", "clf_final_model.png", "clf_feature_importance_rf.png",
    "clf_confusion_matrices_part_b.png",
    "clu_elbow_silhouette.png", "clu_dendrogram_ward.png", "clu_pca_2d_clusters.png", "clu_tsne_clusters.png",
    "clu_linkage_comparison.png", "clu_segment_profile_heatmap.png",
]


def regression_results() -> dict:
    """Read the 10-model table and the tuning results from the executed regression notebook."""
    nb = json.loads((ROOT / "notebooks" / "regression.ipynb").read_text())
    text = ""
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            for out in cell.get("outputs", []):
                text += "".join(out.get("text", []))
    rows = re.findall(r"^(.+?)\s+R2=([\d.]+)\s+RMSE=([\d.]+)\s+MAE=([\d.]+)", text, flags=re.M)
    table = [{"Model": m.strip(), "R2": float(r2), "RMSE": float(rmse), "MAE": float(mae)} for m, r2, rmse, mae in rows]
    table.sort(key=lambda r: r["R2"], reverse=True)
    tuned = re.findall(r"^(.+?)\n\s+Best params: (\{.*?\})\n\s+Baseline R2=([\d.]+)\s+->\s+Tuned R2=([\d.]+)", text, flags=re.M)
    tuning = [{"Model": m.strip(), "Best params": p, "Baseline R2": float(b), "Tuned R2": float(t)} for m, p, b, t in tuned]
    cv = re.findall(r"^(.+?): 5-fold CV R2 = ([\d.]+) \(\+/- ([\d.]+)\)", text, flags=re.M)
    cv_rows = [{"Model": m.strip(), "CV R2": float(s), "Std": float(sd)} for m, s, sd in cv]
    return {"table": table, "tuning": tuning, "cv": cv_rows}


def main():
    STORE.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    for src, dst in DEPLOYED.items():
        shutil.copy2(MODELS / src, STORE / dst)
        print(f"model_store/{dst}  <-  models/{src}")

    clf_final = pd.read_csv(MODELS / "clf_final_comparison.csv")
    clf_pca = pd.read_csv(MODELS / "clf_consolidated_pca_baseline.csv")
    results = {
        "regression": regression_results(),
        "classification": {
            "final": clf_final.drop(columns=["Rank"]).round(4).to_dict(orient="records"),
            "baseline_pca": clf_pca.drop(columns=["Rank"]).round(4).to_dict(orient="records"),
        },
        "clustering": {
            "metrics": pd.read_csv(MODELS / "clu_metrics.csv").round(4).to_dict(orient="records"),
            "k_selection": pd.read_csv(MODELS / "clu_k_selection.csv").round(4).to_dict(orient="records"),
            "segments": pd.read_csv(MODELS / "clu_segments.csv").round(4).to_dict(orient="records"),
        },
    }
    (STORE / "results.json").write_text(json.dumps(results, indent=1))
    print("model_store/results.json written")

    for name in FIGURES:
        if (MODELS / name).exists():
            shutil.copy2(MODELS / name, FIGS / name)
    print(f"{len(FIGURES)} figures copied to static/figures/")


if __name__ == "__main__":
    main()
