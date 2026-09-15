# AI Risk Analysis — ML Capstone (23CSE301)

End-to-end Machine Learning capstone covering **Regression**, **Classification**, and **Clustering** tracks, per the course guidelines (`23CSE301_ML_26_27_Capstone_Guidelines`).

## Datasets

| Track | Dataset | File | Target |
|---|---|---|---|
| Regression (official) | Financial Risk for Loan Approval (synthetic loan-applicant data) | `data/financial_risk_loan_approval/Loan.csv` (20,000 rows x 36 cols) | `RiskScore` (continuous) |
| Classification (official) | Insurance Claims Fraud | `data/insurance_claims/insurance_claims.csv` (1,000 rows x 40 cols) | `fraud_reported` (Y/N) |
| Regression (bonus/practice) | Kaggle Playground Series S4E12 — Insurance Premium Prediction | `data/playground-series-s4e12/train.csv` (1,200,000 rows x 21 cols) | `Premium Amount` (continuous) |
| Clustering | *(added in Review 2, dataset not yet assigned)* | — | — |

## Repository structure

```
/
├── README.md                          # this file
├── requirements.txt                   # pinned Python dependencies
├── data/
│   ├── financial_risk_loan_approval/  # official Regression dataset
│   ├── insurance_claims/              # official Classification dataset
│   └── playground-series-s4e12/       # bonus/practice Regression dataset (Kaggle)
├── notebooks/
│   ├── regression.ipynb                    # Review 1 — official Regression track (10 algorithms)
│   ├── classification.ipynb                # Review 1 — official Classification Part A (5 algorithms) + PCA
│   ├── regression_insurance_premium_1.ipynb  # bonus — Regression pipeline on the Kaggle dataset
│   └── clustering.ipynb                    # Review 2 (to be added)
├── models/                            # exported EDA/result figures referenced by the notebooks
├── contri/                            # per-member contribution folders (team submission scaffolding)
└── app/                               # GUI / deployment code (bonus, optional)
```

**Note:** `notebooks/dash1223.ipynb` is a scratch/working copy created during exploration and is **not** a graded deliverable — the maintained, up-to-date bonus notebook is `regression_insurance_premium_1.ipynb`.

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
- [x] Classification Part A: Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM trained & compared on `fraud_reported` (Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices), with a PCA dimensionality-reduction step as the shared preprocessing pipeline.
- [x] Bonus: Regression pipeline replicated end-to-end on a large (1.2M-row) external Kaggle dataset (`regression_insurance_premium_1.ipynb`) — see note below on its split strategy.
- [ ] Classification Part B + consolidated 10-algorithm comparison (Review 2)
- [ ] Clustering track (Review 2)
- [ ] Bonus GUI / deployment (optional)

Random seed `42` is used throughout for reproducibility. All preprocessing (encoding/scaling/imputation/PCA) is fit on the training split only, inside scikit-learn `Pipeline`/`ColumnTransformer` objects, to avoid data leakage.

### Bonus notebook: mixed full-data / subsample split

`regression_insurance_premium_1.ipynb` trains all 10 regression algorithms on `Premium Amount`, but at 1.2M rows not every algorithm is equally tractable:

- **Linear, Ridge, Lasso, ElasticNet, Polynomial Regression, KNN Regressor** — fast/scalable enough to train and evaluate on the **full 1.2M-row split** (960k train / 240k test).
- **Decision Tree, Random Forest, Gradient Boosting, SVR** — trained and evaluated on a fixed-seed **100,000-row subsample** (80k train / 20k test) instead, since repeated tree-building during `GridSearchCV` and SVR's O(n²)-O(n³) fit time are not tractable at full scale in a notebook.

This is a deliberate trade-off against using one identical test set for every algorithm, documented explicitly in the notebook (Section B3) — both splits are fixed-seed random samples of the same population, so R²/RMSE/MAE stay valid and comparable, just with different precision.

## Generative AI assistance

Code scaffolding (pipeline structure, boilerplate plotting code) for this project was assisted by Claude (Anthropic), per the academic-integrity guidance in the course handout. All data analysis, interpretation, and feature-engineering decisions in the notebooks are the team's own.

