# AI Risk Analysis — ML Capstone (23CSE301)

End-to-end Machine Learning capstone covering **Regression**, **Classification**, and **Clustering** tracks, per the course guidelines (`23CSE301_ML_26_27_Capstone_Guidelines`).

## Datasets

| Track | Dataset | File | Target |
|---|---|---|---|
| Regression | Financial Risk for Loan Approval (synthetic loan-applicant data) | `data/financial_risk_loan_approval/Loan.csv` (20,000 rows x 36 cols) | `RiskScore` (continuous) |
| Classification | Insurance Claims Fraud | `data/insurance_claims/insurance_claims.csv` (1,000 rows x 40 cols) | `fraud_reported` (Y/N) |
| Clustering | *(added in Review 2)* | — | — |

## Repository structure

```
/
├── README.md                  # this file
├── requirements.txt           # pinned Python dependencies
├── data/                      # raw dataset files
├── notebooks/
│   ├── regression.ipynb       # Review 1 — full Regression track (10 algorithms)
│   ├── classification.ipynb   # Review 1 — Classification Part A (5 algorithms)
│   └── clustering.ipynb       # Review 2 (to be added)
├── models/                    # saved model artefacts / exported figures
└── app/                       # GUI / deployment code (bonus, optional)
```

## Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate      # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

To run the notebooks with Jupyter:

```bash
python -m ipykernel install --user --name=airisk --display-name="AI Risk (venv)"
jupyter notebook notebooks/
```

Select the **"AI Risk (venv)"** kernel when opening a notebook.

## Review 1 — status

- [x] Regression track: all 10 required algorithms trained & compared on `RiskScore` (R², RMSE, MAE), GridSearchCV tuning on the two best models, 5-fold CV, residual/predicted-vs-actual/feature-importance plots.
- [x] Classification Part A: Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM trained & compared on `fraud_reported` (Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices).
- [ ] Classification Part B + consolidated 10-algorithm comparison (Review 2)
- [ ] Clustering track (Review 2)
- [ ] Bonus GUI / deployment (optional)

Random seed `42` is used throughout for reproducibility. All preprocessing (encoding/scaling/imputation) is fit on the training split only, inside scikit-learn `Pipeline`/`ColumnTransformer` objects, to avoid data leakage.

## Generative AI assistance

Code scaffolding (pipeline structure, boilerplate plotting code) for this project was assisted by Claude (Anthropic), per the academic-integrity guidance in the course handout. All data analysis, interpretation, and feature-engineering decisions in the notebooks are the team's own.
