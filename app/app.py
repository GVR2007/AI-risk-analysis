"""AI Risk Analysis — web interface for the 23CSE301 ML capstone.

Run locally (from the repository root):

    pip install -r app/requirements.txt
    python app/app.py                 # http://127.0.0.1:5000

Production (Render / any host): gunicorn --chdir app app:app
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from risk_models import STORE, CardSegmentModel, ClaimFraudModel, LoanRiskModel  # noqa: E402

app = Flask(__name__)

loan = LoanRiskModel()
claims = ClaimFraudModel()
cards = CardSegmentModel()
RESULTS = json.loads((STORE / "results.json").read_text())


ALGORITHMS = {
    "loan": {"options": loan.algorithms, "default": loan.default_algorithm},
    "claim": {"options": claims.algorithms, "default": claims.default_algorithm},
    "card": {"options": cards.algorithms, "default": cards.default_algorithm},
}


@app.get("/")
def index():
    return render_template(
        "index.html",
        loan_options=loan.options,
        claim_options=claims.options,
        models_by_make=claims.models_by_make,
        results=RESULTS,
        algorithms=ALGORITHMS,
    )


@app.get("/api/meta")
def meta():
    return jsonify({"clustering": cards.meta(), "results": RESULTS, "algorithms": ALGORITHMS})


@app.post("/api/predict/<track>")
def predict(track):
    payload = request.get_json(silent=True) or {}
    model = {"loan": loan, "claim": claims, "card": cards}.get(track)
    if model is None:
        return jsonify({"error": f"Unknown model '{track}'. Use loan, claim or card."}), 404
    try:
        return jsonify(model.predict(payload))
    except (ValueError, KeyError, TypeError, ZeroDivisionError, FileNotFoundError) as exc:
        return jsonify({"error": f"Check the form values: {exc}"}), 400


@app.get("/api/sample/<track>")
def sample(track):
    model = {"loan": loan, "claim": claims, "card": cards}.get(track)
    if model is None:
        return jsonify({"error": f"Unknown model '{track}'."}), 404
    return jsonify(model.sample(request.args.get("kind", "random")))


@app.get("/healthz")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
