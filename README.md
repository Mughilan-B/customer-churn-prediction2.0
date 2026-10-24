# ✦ Churn AI — Customer Intelligence Platform

A Streamlit application for exploring telecom customer churn, scoring individual customers, reviewing retention opportunities, and scanning batches of customer records.

## Features

- **Executive dashboard:** churn rate, customer counts, monthly charges, tenure patterns, contract comparisons and risk overview.
- **Individual prediction:** score one hypothetical customer with Logistic Regression, Random Forest or XGBoost.
- **Customer 360 Explorer:** search customer IDs, filter by risk and contract, inspect a profile and export filtered records.
- **Bulk Risk Scanner:** upload a CSV, add churn probability/risk/action columns and download scored results.
- **What-if Simulator:** compare estimated scores across contract options for a hypothetical customer.
- **Model Insights:** inspect feature importance and churn patterns across payment method and tenure.
- **Retention playbook:** see transparent service-review prompts and suggested next steps.
- **CSV exports:** download prediction summaries, filtered customers and batch scoring results.

## Run locally

Use Python 3.10–3.12 where possible, then run:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

## Deploy on Streamlit Community Cloud

1. Open the Streamlit Community Cloud dashboard.
2. Choose **New app** and select this GitHub repository and the `main` branch.
3. Set the main file path to `app/app.py`.
4. Deploy. The app needs access to `data/Telco_Customer_Churn.csv`, the three files in `models/`, and `requirements.txt`.

## Bulk CSV format

For predictions, provide the feature columns used during model training, with the same column names and compatible category values as the supplied Telco dataset. The target `Churn` and identifier `customerID` are not prediction features. The app can offer a template from the Bulk Risk Scanner page.

## Important interpretation notes

- A churn probability is a **model estimate**, not certainty that a customer will leave.
- Retention prompts are practical review suggestions. They are not proof that an attribute caused churn.
- The What-if Simulator compares model scores for hypothetical inputs; it does not estimate the causal effect of changing a contract.
- Model performance figures should only be presented as final validation results after checking them against a held-out test set.
- Customer records can contain personal or commercially sensitive information. Only upload data you are authorized to process, and avoid sharing exported files publicly.

## Tech stack

Python · Streamlit · pandas · NumPy · scikit-learn · XGBoost · Plotly · SHAP/joblib

## Project layout

```text
app/app.py
data/Telco_Customer_Churn.csv
models/logistic_model.pkl
models/rf_model.pkl
models/xgb_model.pkl
requirements.txt
```
