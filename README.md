# ◈ LoanIQ — Loan Approval Engine

A two-stage Machine Learning pipeline wrapped in a polished **Streamlit** web UI that predicts loan approval and estimates the sanctioned amount for approved applications.

---

## How It Works

The prediction runs in two sequential stages:

| Stage | Model | Task | Output |
|-------|-------|------|--------|
| 1 | Random Forest Classifier | Should the loan be approved? | Approved / Rejected + confidence % |
| 2 | Random Forest Regressor | How much will be sanctioned? | Predicted loan amount (₹) |

Stage 2 only runs when Stage 1 predicts **Approved**. Both models are pre-trained `scikit-learn` pipelines loaded from `.pkl` files.

---

## Project Structure

```
loan-approval-app/
├── app.py                          ← Streamlit UI + prediction logic
├── main.py                         ← CLI entry point (for quick testing)
├── config.yaml                     ← Model paths & default input values
├── models/
│   ├── rf_classifier_pipeline.pkl  ← Stage 1: approval classifier
│   └── rf_regressor_pipeline.pkl   ← Stage 2: amount regressor
└── requirements.txt
```

---

## Input Features

| Feature | Type | Description |
|---------|------|-------------|
| `no_of_dependents` | Integer | Number of financial dependents |
| `education` | Categorical | `Graduate` / `Not Graduate` |
| `self_employed` | Categorical | `Yes` / `No` |
| `income_annum` | Float | Annual income (₹) |
| `loan_amount` | Float | Requested loan amount (₹) |
| `loan_term` | Integer | Repayment period (years) |
| `cibil_score` | Integer | Credit score (300–950) |
| `residential_assets_value` | Float | Value of residential property (₹) |
| `commercial_assets_value` | Float | Value of commercial property (₹) |
| `luxury_assets_value` | Float | Value of luxury assets (₹) |
| `bank_asset_value` | Float | Bank / liquid assets (₹) |

The UI also computes and displays three derived indicators in real-time:
- **Total Assets** — sum of all four asset fields
- **Loan-to-Asset (LTV) Ratio** — `loan_amount / total_assets`
- **Income Multiple** — `loan_amount / income_annum`

---

## Local Setup

### 1. Clone / unzip the project

```bash
cd loan-approval-app
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place model files

Copy your trained pipeline files into the `models/` directory:

```bash
cp /path/to/rf_classifier_pipeline.pkl models/
cp /path/to/rf_regressor_pipeline.pkl  models/
```

> **Note:** The paths in `config.yaml` are currently set to absolute Windows paths. Update them to point to your local `models/` directory, or simply ensure the files are at `models/rf_classifier_pipeline.pkl` and `models/rf_regressor_pipeline.pkl` relative to the project root (the `app.py` uses `Path(__file__).parent / "models"` which handles this automatically).

### 5. Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## CLI Usage

A minimal command-line interface is available in `main.py` for quick testing without the UI:

```bash
python main.py
```

It prompts for the number of dependents and runs a prediction with the remaining fields set to defaults from `config.yaml`.

---

## Configuration (`config.yaml`)

```yaml
models:
  classifier: models/rf_classifier_pipeline.pkl
  regressor:  models/rf_regressor_pipeline.pkl

ui:
  default_inputs:
    no_of_dependents: 1
    education: Graduate
    self_employed: No
    income_annum: 1200000
    loan_amount: 30000
    loan_term: 12
    cibil_score: 800
    residential_assets_value: 2000000
    commercial_assets_value: 2000000
    luxury_assets_value: 0
    bank_asset_value: 55000
```

---

## Dependencies

| Package | Version |
|---------|---------|
| `streamlit` | 1.55.0 |
| `pandas` | 2.2.2 |
| `numpy` | 1.26.4 |
| `scikit-learn` | 1.4.2 |
| `joblib` | 1.4.2 |

Install all at once:

```bash
pip install -r requirements.txt
```

---

## UI Features

- **Live credit score gauge** — colour-coded bar (Poor / Fair / Excellent) that updates as you move the CIBIL slider
- **Real-time financial summary** — total assets, LTV ratio, and income multiple update instantly as you enter values
- **Rejection tips** — if the application is rejected, the panel surfaces specific improvement areas (e.g. low CIBIL, high LTV, high income multiple)
- **Confidence bar** — visual display of the classifier's approval probability

---

## Deployment

The app can be deployed to any platform that supports Python and Streamlit.

### Streamlit Community Cloud (recommended for quick sharing)

1. Push the project to a public GitHub repository (exclude `.pkl` files if they are large — use Git LFS or host them externally)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set the main file path to `app.py`

### Railway / Render

```bash
# Start command
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## Model Notes

- Both pipelines are `scikit-learn` `Pipeline` objects that include their own preprocessing steps, so raw input DataFrames can be passed directly.
- The classifier uses `predict_proba` and applies a 0.5 threshold for the approval decision.
- The regressor receives the same feature set as the classifier, with `loan_amount` replaced by `loan_status = "approved"` as an additional column.
- Models are loaded once at startup via `@st.cache_resource` to avoid reloading on every interaction.
