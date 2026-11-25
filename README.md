# Financial Fraud Detection System

An end-to-end academic portfolio prototype for scoring credit card transactions as likely legitimate or potentially fraudulent. It demonstrates data validation, exploratory analysis, leakage-aware preprocessing, imbalanced classification, threshold tuning, evaluation, explainability artifacts, a FastAPI API, and a React dashboard.

This is not a banking deployment or a definitive fraud decision system. The ULB dataset contains anonymised PCA components and only a limited two-day period.

## Dataset

Obtain `creditcard.csv` from the [ULB Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), accept Kaggle's terms, and place the file at `data/creditcard.csv`. The raw file is excluded by `.gitignore`; it is not bundled here. The expected dataset has 284,807 rows, 30 inputs (`Time`, `V1`–`V28`, `Amount`) and a `Class` target. The pipeline validates its contents instead of assuming these figures. No analysis or metric is included until it is generated from the real data.

## Stack and structure

Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib/seaborn, joblib, FastAPI, Pydantic, React, Vite, Recharts, Pytest.

```text
src/                 validation, preprocessing, EDA, training, evaluation, inference
backend/             FastAPI app and request schemas
frontend/            React/Vite dashboard
data/                place creditcard.csv here (ignored by Git)
models/              generated fitted inference pipeline (ignored)
reports/             generated JSON/CSV reports (ignored)
visualizations/      generated plots (ignored)
tests/               synthetic-fixture unit and API tests
```

## Install and run (Windows PowerShell)

Run commands from the project root. Python 3.10+ and Node.js 18+ are recommended. These instructions install Python packages into the active Python installation; a virtual environment is optional.

```powershell
cd F:\Financial-Fraud-Detection-Project
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

Validate the dataset and write `reports/data_quality.json`:

```powershell
py run_project.py validate
```

Run exploratory plots and observations:

```powershell
py run_project.py eda
```

Train and evaluate:

```powershell
py -m src.train
```

The 60/20/20 stratified random split reserves validation for candidate selection and threshold tuning, and holds test data aside for final evaluation. This split keeps the rare class in each partition, but random sampling across the short recorded period is not evidence of future-period performance. Amount is `log1p` transformed and scaled; the other features are scaled. Imbalance approaches include a dummy baseline, balanced logistic regression, random undersampling for a second logistic regression, balanced random forest and class-weighted XGBoost. Undersampling is applied to training rows only and trades information for balanced classes. SMOTE is not included: the compact baseline favors weighting and direct undersampling over synthetic points in anonymised PCA feature space. Logistic regularization is tuned with a small two-fold randomized search on a bounded training-only sample. Candidate ranking uses validation PR-AUC. The decision threshold maximizes precision among validation thresholds with recall of at least 0.80 (or best F1 if none meet that target). This makes the recall/false-positive tradeoff explicit; operational costs can justify a different objective.

Generated files include `reports/data_quality.json`, `eda_observations.json`, `model_comparison.csv`, `threshold_sweep.csv`, `permutation_importance.csv`, `test_metrics.json`, `training_summary.json`, `models/fraud_pipeline.joblib`, and PNG plots. PR-AUC is reported because it focuses on positive-class retrieval and precision under severe imbalance; accuracy can appear high while fraud is missed. Permutation importance is computed on a bounded validation sample and measures predictive reliance, not causation; PCA feature names remain anonymised. Model persistence contains all preprocessing. Evaluation artifacts and fitted models are generated locally and ignored by Git, avoiding claims that do not travel with the source code.

## Start the API and web app

After training, use two VS Code terminals. In the first terminal, start the backend from the project root:

```powershell
cd F:\Financial-Fraud-Detection-Project
py -m uvicorn backend.main:app --reload
```

Keep that terminal running. In a second terminal, start the frontend:

```powershell
cd F:\Financial-Fraud-Detection-Project\frontend
npm install
npm run dev
```

Open the local address Vite prints, usually `http://localhost:5173`; API docs are at `http://localhost:8000/docs`. The overview shows separate dataset, model, and API readiness checks. “Try a transaction” offers real labeled rows stored from the training dataset and an advanced editor for all 30 features. “Batch file” offers a blank CSV template and row-level validation. “Model results” explains held-out metrics and generated plots. “How it works” describes the training and inference flow in plain language. The app is not connected to a bank or live transaction stream.

Set `VITE_API_URL` if the API runs at another origin. Before training, prediction endpoints return HTTP 503 and the interface explains the missing setup step. The API offers `GET /`, `/health`, `/setup-status`, `/model-info`, `/demo-samples`, `/metrics`, `/visualizations`, `POST /predict`, and `POST /predict/batch` (up to 500 rows). Batch CSV headers must contain exactly the 30 feature names. Predictions can be downloaded as CSV.

## Testing

```powershell
py -m pytest -q
```

Tests use tiny synthetic fixtures only for unit and API behavior. Those fixtures are not training data and no model results are generated from them.

## Results and explainability

The checked-in local reports were generated from the Kaggle dataset at `data/creditcard.csv`. On the held-out test split, the selected Random Forest (balanced) model at threshold 0.08 achieved precision 0.6880, recall 0.8776, F1 0.7713, ROC-AUC 0.9669, and PR-AUC 0.8568. Its test confusion matrix was TN 56,825; FP 39; FN 12; TP 86. These are results for the fixed random split and should not be presented as proof of future-period or production performance. PR-AUC is useful because fraud is rare; accuracy alone would hide missed fraud. The dashboard labels source dataset counts separately from validation comparisons and final held-out test metrics.

The training code also generates threshold tables, test confusion/precision-recall/ROC plots, permutation importance, and EDA plots. Permutation importance describes predictive reliance, not causation. V1–V28 remain anonymized PCA components with no assigned business meaning.

## Limitations and future improvements

- The dataset covers approximately two days; temporal drift and future-period performance are unknown.
- PCA components are anonymised; users cannot enter standard card details.
- A stratified random split may leak neighboring time patterns across partitions.
- Threshold costs are illustrative and should be calibrated with a domain owner.
- Batch CSV parsing in the demo UI targets ordinary comma-separated numeric files; very large files and quoted multi-line CSV fields are not supported.
- This prototype is for learning and portfolio demonstration, not live transaction authorization.

Possible next steps include a forward-time evaluation on suitable data, calibrated probabilities, cost-based threshold selection, and local/global permutation importance with careful caveats.

## Acknowledgements

Dataset: ULB Machine Learning Group / Worldline, distributed via Kaggle. Author details are intentionally omitted; add your own contribution and acknowledgements before publishing.
