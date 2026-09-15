# AI Risk Analysis — ML Capstone (23CSE301)

End-to-end Machine Learning capstone covering **Regression**, **Classification**, and **Clustering** tracks, per the course guidelines (`23CSE301_ML_26_27_Capstone_Guidelines`).

# Problem Statement — AI Risk Analysis Capstone (23CSE301)

## Overview

Risk assessment is one of the highest-value, most widely deployed applications of machine learning in the financial and insurance industries: lenders must price the risk of default before disbursing a loan, and insurers must separate genuine claims from fraudulent ones before paying out. Both problems share a common shape — a mix of numeric and categorical applicant/incident data, a target that is expensive to get wrong in either direction (approve a bad loan / pay a fraudulent claim vs. reject a good customer / delay a genuine claim), and messy, real-world data quality issues (missing fields, imbalanced outcomes, correlated features, potential leakage). That combination is exactly what the course guidelines ask the Regression and Classification tracks to demonstrate mastery of, which is why this project frames both tracks around **risk scoring**: predicting a continuous **loan risk score** (Regression) and detecting **fraudulent insurance claims** (Classification), with a third, self-selected dataset used to stress-test the same pipeline at a scale and difficulty the assigned datasets don't reach.

## Track 1 — Regression: Financial Risk for Loan Approval

**Dataset:** `data/financial_risk_loan_approval/Loan.csv` (20,000 rows, 36 columns) — instructor-assigned.
**Target:** `RiskScore` (continuous, ~28-84).

**Why this dataset is a strong fit for the Regression track, not just an assigned formality:**

- **A genuinely continuous, non-trivial target.** `RiskScore` is a composite score built from multiple weighted sub-factors (credit score bucket, DTI bucket, payment history, bankruptcy/default history, net worth, employment status — see `CSV Generation.py`), so it behaves like a real underwriting score rather than a linear function of one or two inputs. That non-linearity is exactly what motivates comparing 10 different regression algorithms rather than settling on Linear Regression alone (R²≈0.82 for plain linear models vs. R²≈0.91 for tuned Gradient Boosting in our results).
- **A built-in, teachable data-leakage trap.** `LoanApproved` is computed from a formula that also feeds into `RiskScore`, and the score's formula explicitly discounts approved loans by 20%. Using it as a naive predictor would silently leak the target and inflate every model's score — a realistic mistake to have to catch and document, not a contrived textbook example.
- **Real preprocessing decisions to justify.** Multicollinearity between `TotalAssets`/`NetWorth`/`SavingsAccountBalance`, monetary right-skew requiring outlier treatment, and five categorical fields needing encoding all force genuine Section B decisions (see `regression.ipynb`, Sections B1-B3) rather than a one-line `.fit()`.
- **Clean enough to focus on modelling, not just cleaning.** Zero missing values and zero duplicates at 20,000 rows means the track's 9 marks (guideline Section C) go toward algorithm comparison, tuning, and visualisation — not entirely consumed by data-wrangling as a 1M-row messy dataset would demand.

## Track 2 — Classification: Insurance Claims Fraud

**Dataset:** `data/insurance_claims/insurance_claims.csv` (1,000 rows, 40 columns) — instructor-assigned.
**Target:** `fraud_reported` (Y/N, ~25% fraud).

**Why this dataset is a strong fit for the Classification track:**

- **A real, costly, asymmetric business problem.** Unlike a synthetic toggle, insurance fraud detection has genuine, well-documented stakes: false negatives cost insurers real payouts, false positives cost genuine customers trust and delay. That's precisely why the guidelines mandate Accuracy *and* Precision/Recall/F1 *and* ROC-AUC *and* a confusion matrix (Section 3.2) instead of accuracy alone — this dataset is imbalanced (~75:25) exactly enough to make that requirement matter rather than be pro-forma.
- **Realistic missingness.** Four columns encode missing values as the literal string `'?'`, which must be detected and converted before any imputation — a small but genuine data-cleaning trap, not a pre-cleaned CSV.
- **High-cardinality categoricals that make dimensionality reduction meaningful.** `insured_hobbies`, `insured_occupation`, `auto_model` and others one-hot-expand the 35 raw columns to 142 encoded features — enough real dimensionality for the PCA analysis (Section C) to show a genuine, non-trivial trade-off (helps Naive Bayes and KNN, hurts Decision Tree) rather than a token 5-feature toy example.
- **A believable, checkable feature-engineering story.** `claim_to_premium_ratio` and `days_bind_to_incident` (claims filed suspiciously soon after a policy starts) are both real red flags insurers use in practice, giving the required "written justification of why it may improve model performance" (guideline B3) actual domain grounding instead of an arbitrary transformation.

## Track 3 (bonus) — Regression at Scale: Kaggle Playground Series S4E12

**Dataset:** `data/playground-series-s4e12/train.csv` (1,200,000 rows, 21 columns) — self-selected.
**Target:** `Premium Amount` (continuous).

This dataset was deliberately chosen — not assigned — to stress-test the same 10-algorithm pipeline built for Track 1 against two things the 20,000-row Loan dataset cannot: **genuine scale** (1.2M rows breaks the "just call `.fit()`" approach for SVR and GridSearchCV, forcing an explicit, documented full-data-vs-subsample strategy) and **a near-zero-signal target** (every numeric feature has |r| < 0.05 with `Premium Amount`). That second property is the real pedagogical point: it is easy to build a pipeline that looks correct on a dataset engineered to be learnable; it is more instructive to run the identical, honest pipeline on a dataset where the correct, defensible outcome is R² ≈ 0.02-0.05 and to document *why* that's the right result rather than a bug. Two independent implementations of this dataset's pipeline exist in this repo (`regression_insurance_premium_1.ipynb` using `Pipeline`/`ColumnTransformer`, and `regression_insurance_premium_no_pipeline.ipynb` doing every preprocessing step manually) and reach R² values agreeing to 3-4 decimal places — cross-validating that the weak result is a property of the data, not an implementation bug.

## Why not other commonly-used datasets

| Candidate | Why it was not used |
|---|---|
| **Titanic** (survival classification) | Only 891 rows; among the most solved/tutorialised datasets on the internet — the guidelines explicitly flag identical/near-identical notebooks across teams as an academic-integrity risk, and a dataset with thousands of public reference solutions raises exactly that risk. Binary target with no continuous-target option, so it can't serve a Regression track at all. |
| **Iris** (species classification) | 150 rows, 3 perfectly balanced classes, no missing values, no outliers, no categorical encoding needed — too clean to exercise Section B (cleaning/feature engineering) or justify comparing 10 algorithms; differences between models are noise-level. |
| **Boston Housing** (regression) | Removed from scikit-learn (≥1.2) and flagged as ethically problematic (an engineered feature assumes a racially discriminatory relationship); actively discouraged for new coursework. Also only ~506 rows — too small to support a genuine train/test split at the scale this project uses. |
| **Wine Quality** (regression/classification) | Reasonable dataset, but ordinal target (integer quality scores) makes "regression vs. classification" ambiguous in a way that would complicate satisfying two distinct track rubrics cleanly; also heavily tutorialised. |
| **MNIST / image datasets** | Wrong data modality — the guidelines' entire rubric (EDA on tabular distributions, correlation heatmaps, `ColumnTransformer`-style encoding/scaling, feature importance) assumes structured/tabular data, not images requiring CNNs. |

The two officially assigned datasets (Loan Approval, Insurance Fraud) already satisfy the rubric well on their own merits, as argued above; the bonus Kaggle dataset was added specifically because it offers something neither assigned dataset does — real scale and a legitimately hard, low-signal target — which is why it was worth the extra, ungraded effort rather than being redundant with Track 1.


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

