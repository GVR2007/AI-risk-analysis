# AI Risk Analysis — ML Capstone (23CSE301)

End-to-end Machine Learning capstone covering **Regression**, **Classification**, and **Clustering** tracks, per the course guidelines (`23CSE301_ML_26_27_Capstone_Guidelines`), plus a web application (**Risk Desk**) that serves one trained model from each track.

## Problem statement

Risk assessment is one of the most widely deployed uses of machine learning in finance and insurance. The project frames all three tracks around **risk**:

| Track | Question | Target |
|---|---|---|
| Regression | How risky is this loan applicant? | continuous `RiskScore` |
| Classification | Is this insurance claim fraudulent? | `fraud_reported` (Y/N) |
| Clustering | Which credit-risk segment does this card holder belong to? | no labels — discover segments |

See [`PROBLEM_STATEMENT.md`](PROBLEM_STATEMENT.md) for the full problem statement and the reasoning behind each dataset choice.

## Datasets

| Track | Dataset | File | Size | Missing values |
|---|---|---|---|---|
| Regression | Financial Risk for Loan Approval (synthetic loan-applicant data) | `data/financial_risk_loan_approval/Loan.csv` | 20,000 x 36 | none |
| Classification | Insurance Claims Fraud (auto-insurance claims) | `data/insurance_claims/insurance_claims.csv` | 1,000 x 40 | 4 columns, encoded as `'?'` |
| **Clustering** | **Credit Card Dataset for Clustering** ("CC GENERAL", Kaggle, Arjun Bhasin) — real, anonymised 6-month usage of active credit-card holders | `data/credit_card_clustering/CC GENERAL.csv` | 8,950 x 18 | `MINIMUM_PAYMENTS` 313, `CREDIT_LIMIT` 1 |
| Regression (bonus/practice) | Kaggle Playground Series S4E12 — Insurance Premium Prediction | `data/playground-series-s4e12/train.csv` | 1,200,000 x 21 | yes |

The clustering file is the original Kaggle CSV (source: <https://www.kaggle.com/datasets/arjunbhasin2013/ccdata>; MD5 `8dc44d7b63b3492f3bbd874d2d3cc8ef`).

## Results summary

### Regression — `RiskScore` (Review 1)

| Rank | Model | R² | RMSE | MAE |
|---|---|---|---|---|
| 1 | **Gradient Boosting Regressor (tuned)** | **0.9072** | **2.4005** | **1.5945** |
| 2 | Gradient Boosting Regressor (baseline) | 0.8851 | 2.6707 | 1.8477 |
| 3 | Random Forest Regressor | 0.8758 | 2.7771 | 1.7478 |
| 4 | Polynomial Regression (deg=2) | 0.8660 | 2.8839 | 2.2261 |
| 5 | Support Vector Regressor (SVR) | 0.8454 | 3.0986 | 2.2616 |
| 6 | Linear Regression | 0.8199 | 3.3441 | 2.6192 |
| 7 | Ridge Regression | 0.8199 | 3.3442 | 2.6191 |
| 8 | Lasso Regression | 0.8187 | 3.3548 | 2.6205 |
| 9 | ElasticNet Regression | 0.8183 | 3.3584 | 2.6259 |
| 10 | K-Nearest Neighbors Regressor | 0.7469 | 3.9640 | 3.0709 |
| 11 | Decision Tree Regressor | 0.7160 | 4.1995 | 2.9874 |

5-fold CV R² of the tuned Gradient Boosting model: **0.9087 ± 0.0034**.

### Classification — `fraud_reported` (Review 1 Part A + Review 2 Part B)

Final consolidated table: every algorithm tuned with `GridSearchCV` (5-fold stratified CV, `f1_weighted`), with the feature representation (PCA-55 vs. one-hot encoded) treated as a hyperparameter. Ranked by cross-validated F1; test metrics are weighted averages on the same 200-claim test split.

| Rank | Model | Part | CV F1 | Accuracy | Precision | Recall | F1 | ROC-AUC | Fraud recall |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Bagging Classifier** | B | **0.8535** | 0.8150 | 0.8284 | 0.8150 | 0.8199 | 0.8166 | **0.7143** |
| 2 | Gradient Boosting Classifier | B | 0.8435 | 0.7850 | 0.7710 | 0.7850 | 0.7752 | 0.8204 | 0.4490 |
| 3 | Decision Tree Classifier | A | 0.8415 | 0.8300 | 0.8444 | 0.8300 | 0.8350 | 0.8471 | 0.7551 |
| 4 | MLP Classifier (Neural Network) | B | 0.8329 | 0.7600 | 0.7505 | 0.7600 | 0.7545 | 0.7761 | 0.4490 |
| 5 | Logistic Regression | A | 0.8267 | 0.7900 | 0.7776 | 0.7900 | 0.7814 | 0.8075 | 0.4694 |
| 6 | Support Vector Machine (SVC) | A | 0.8192 | 0.8300 | 0.8226 | 0.8300 | 0.8246 | 0.8138 | 0.5714 |
| 7 | Random Forest Classifier | B | 0.8051 | 0.7950 | 0.7936 | 0.7950 | 0.7943 | 0.8061 | 0.5714 |
| 8 | AdaBoost Classifier | B | 0.8032 | 0.7350 | 0.7013 | 0.7350 | 0.7110 | 0.8292 | 0.2653 |
| 9 | Naive Bayes (Gaussian) | A | 0.7190 | 0.6350 | 0.7241 | 0.6350 | 0.6595 | 0.6882 | 0.6327 |
| 10 | K-Nearest Neighbors | A | 0.6743 | 0.7600 | 0.7174 | 0.7600 | 0.7031 | 0.6010 | 0.1429 |

**Final model: tuned Bagging Classifier** (50 trees, `max_depth=5`, `max_samples=0.6`, encoded features). 5-fold CV on the full data: F1 **0.854 ± 0.013**, ROC-AUC **0.860 ± 0.014**. Against Review 1's best model (Logistic Regression on PCA), weighted F1 improves 0.742 → 0.820 and fraud recall 0.27 → 0.71.

### Clustering — card-holder risk segments (Review 2)

Same input for both algorithms (cleaned, log-transformed, scaled, PCA to 8 components = 91.3% variance), k = 4.

| Algorithm | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | Cluster sizes |
|---|---|---|---|---|
| **K-Means** | **0.2418** | **1.5148** | **2,803.0** | 3,204 / 2,374 / 1,773 / 1,598 |
| Agglomerative (Ward) | 0.1999 | 1.7153 | 2,370.8 | 2,520 / 2,359 / 2,309 / 1,761 |
| Agglomerative (complete) | 0.1380 | 1.5324 | 702.2 | 5,829 / 2,858 / 261 / 1 |
| Agglomerative (average) | 0.1919 | 1.0970 | 188.3 | 8,685 / 250 / 13 / 1 |
| Agglomerative (single) | 0.2834 | 0.6143 | 4.8 | 8,945 / 2 / 1 / 1 |

The four K-Means segments: **Active Transactors** (35.8%, low risk), **Dormant / Low-Activity** (19.8%, low risk), **Cash-Advance Revolvers** (26.5%, high risk), **Heavy Revolving Spenders** (17.9%, medium-high risk). K-Means is stable across 10 random seeds (mean ARI 0.987).

## Repository structure

```
/
├── README.md                          # this file
├── PROBLEM_STATEMENT.md               # problem statement & dataset-choice justification
├── requirements.txt                   # pinned Python dependencies (notebooks + app)
├── render.yaml                        # one-click deployment of the web app on Render
├── data/
│   ├── financial_risk_loan_approval/  # Regression dataset
│   ├── insurance_claims/              # Classification dataset
│   ├── credit_card_clustering/        # Clustering dataset (CC GENERAL)
│   └── playground-series-s4e12/       # bonus/practice Regression dataset (Kaggle)
├── notebooks/
│   ├── regression.ipynb                          # Review 1 — Regression track (10 algorithms)
│   ├── classification.ipynb                      # Review 1 Part A + Review 2 Part B (all 10 algorithms)
│   ├── clustering.ipynb                          # Review 2 — Clustering track (K-Means, Agglomerative)
│   ├── regression_insurance_premium_1.ipynb      # bonus — Regression pipeline on the Kaggle dataset
│   ├── regression_insurance_premium_no_pipeline.ipynb
│   └── classification_no_pipeline.ipynb          # Part A without sklearn Pipeline (manual preprocessing)
├── models/                            # exported figures, result tables (.csv) and trained models (.pkl, git-ignored)
├── app/                               # Risk Desk web app (Flask) — bonus GUI + deployment
│   ├── app.py                         # Flask routes / JSON API
│   ├── risk_models.py                 # loads the 3 models, mirrors notebook preprocessing
│   ├── export_artifacts.py            # copies the deployed models, tables and figures from models/
│   ├── model_store/                   # the 3 deployed models + results.json (committed)
│   ├── templates/, static/            # HTML, CSS, JS, bundled font and Chart.js
│   └── requirements.txt               # deploy-only dependencies
└── contri/                            # per-member contribution folders
```

**Note:** `notebooks/dash1223.ipynb` is a scratch/working copy and is **not** a graded deliverable.

## Environment setup

Python 3.11 or 3.12 is recommended (the pinned `numpy==2.0.2` / `scipy==1.13.1` have no wheels for 3.13).

```bash
python3 -m venv .venv
source .venv/bin/activate      # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## How to run

### Notebooks

```bash
python -m ipykernel install --user --name=airisk --display-name="AI Risk (venv)"
jupyter notebook notebooks/
```

Select the **"AI Risk (venv)"** kernel and run each notebook top to bottom (Kernel → Restart & Run All). On 2 CPU cores, `classification.ipynb` takes about 9 minutes (the 10-model grid search) and `clustering.ipynb` about 6 minutes (t-SNE and the dendrogram).

### Web app (Risk Desk)

```bash
python app/export_artifacts.py   # only needed after re-running the notebooks
python app/app.py                # open http://127.0.0.1:5000
```

The app has five pages: an overview, **Loan risk** (regression), **Claim fraud** (classification), **Card segments** (clustering) and **Model results** (all comparison tables and figures). Each tool page can load a real record from the dataset, shows the prediction on a risk scale, and lists the inputs that moved the result most (each input swapped for the typical value, one at a time). The app reproduces the notebook predictions: identical fraud probabilities and cluster labels, and R² 0.906 for the loan model on the same 4,000-row test split (0.907 in the notebook; the small gap is because the form recomputes monthly income and net worth with the dataset generator's formulas).

### Public deployment (Render, free tier)

1. Push the repository to GitHub (the three models in `app/model_store/` are committed; `.gitignore` excludes every other `.pkl`).
2. On <https://render.com>: **New → Blueprint**, choose the repository. `render.yaml` sets the build command (`pip install -r app/requirements.txt`) and the start command (`gunicorn --chdir app app:app`).
3. Render prints the public URL when the build finishes.

## Review status

- [x] Regression track: all 10 algorithms, comparison table, GridSearchCV on the two best models, 5-fold CV, residual / predicted-vs-actual / feature-importance plots.
- [x] Classification Part A (Review 1): Logistic Regression, KNN, Naive Bayes, Decision Tree, SVC.
- [x] Classification Part B (Review 2): Random Forest, AdaBoost, Gradient Boosting, Bagging (Decision Tree base), MLP, on the same split and pipeline as Part A.
- [x] Consolidated 10-algorithm table (Accuracy, Precision, Recall, F1, ROC-AUC), PCA-vs-encoded diagnostic, GridSearchCV for all 10, final model selection with documented improvement, 5-fold CV of the two best models.
- [x] Clustering track: K-Means (elbow curve + silhouette analysis) and Agglomerative (dendrogram + 4 linkage strategies); Silhouette, Davies-Bouldin and Calinski-Harabasz for all; PCA 2-D plot for every algorithm; t-SNE for both; segment profiling and stability check.
- [x] Bonus: web GUI (Flask) serving all three tracks, with deployment configuration for Render.

Random seed `42` is used throughout. For regression and classification, all preprocessing (imputation, encoding, scaling, PCA) is fitted on the training split only, inside scikit-learn `Pipeline` / `ColumnTransformer` objects. Clustering has no target, so its scaler and PCA are fitted on the full cleaned dataset (there is nothing to leak).

### Bonus notebook: mixed full-data / subsample split

`regression_insurance_premium_1.ipynb` trains all 10 regression algorithms on `Premium Amount`. Linear, Ridge, Lasso, ElasticNet, Polynomial Regression and KNN use the **full 1.2M-row split**. Decision Tree, Random Forest, Gradient Boosting and SVR use a fixed-seed **100,000-row subsample**, because repeated tree-building in `GridSearchCV` and SVR's O(n²)–O(n³) fit time are not tractable at full scale. Both are fixed-seed samples of the same population, so the metrics remain comparable (documented in the notebook, Section B3).

## Third-party assets

- Chart.js 4.4.1 (MIT) and the Archivo variable font (SIL Open Font License) are bundled in `app/static/` so the app works offline.

## Generative AI assistance

Claude (Anthropic) was used for code scaffolding (pipeline structure, plotting boilerplate) in Review 1. In Review 2 it was used for the Part B / clustering notebook code, for drafting the notebook commentary, and for building the web app. The team reviewed and verified every result. *(Team: adjust this paragraph so it describes exactly how you used AI, as the handout requires.)*
