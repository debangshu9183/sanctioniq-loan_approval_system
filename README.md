# Loan Approval Predictor

A two-stage ML pipeline deployed as a FastAPI web application.

| Stage | Model | Input | Output |
|-------|-------|-------|--------|
| 1 | Random Forest Classifier | Applicant features | Approved / Rejected + probability |
| 2 | Random Forest Regressor | Approved applicant features | Predicted sanctioned loan amount |

---

## Project Structure

```
loan-approval-app/
├── api/
│   └── index.py            ← FastAPI app (Vercel entry point)
├── src/
│   ├── config.py           ← All constants & column names
│   ├── schemas.py          ← Pydantic request/response models
│   ├── model_loader.py     ← Singleton pkl loader with caching
│   └── predictor.py        ← Two-stage prediction logic
├── static/
│   └── index.html          ← Single-page frontend UI
├── models/                 ← ⚠️  Place your .pkl files here (see below)
│   ├── rf_classifier_pipeline.pkl
│   └── rf_regressor_pipeline.pkl
├── requirements.txt
├── vercel.json
└── README.md
```

---

## Local Development

```bash
# 1. Clone / unzip the project
cd loan-approval-app

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy your model files into models/
cp /path/to/rf_classifier_pipeline.pkl models/
cp /path/to/rf_regressor_pipeline.pkl  models/

# 5. Run locally
uvicorn api.index:app --reload --port 8000
```

Open http://localhost:8000 in your browser.

---

## Deploy to Vercel

### Prerequisites
- [Vercel CLI](https://vercel.com/docs/cli): `npm i -g vercel`
- Vercel account (free tier works)

### Steps

```bash
# 1. Login
vercel login

# 2. Deploy (from project root)
vercel --prod
```

### ⚠️  Model File Size Warning

Vercel's Python serverless functions have a **50 MB compressed** limit.
The included models are:
- `rf_classifier_pipeline.pkl` ≈ 4.3 MB
- `rf_regressor_pipeline.pkl`  ≈ 11.6 MB

With `scikit-learn + numpy + pandas`, the total bundle approaches the limit.

**If deployment fails due to size:**
Option A — Upload models to **AWS S3 / Google Cloud Storage** and fetch them
at startup via `boto3` / `google-cloud-storage`.

Option B — Deploy to **Railway or Render** (no bundle size limit):
```bash
# Railway
railway init && railway up

# Render — connect your GitHub repo and set:
# Build Command:  pip install -r requirements.txt
# Start Command:  uvicorn api.index:app --host 0.0.0.0 --port $PORT
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/`         | Frontend UI |
| `GET`  | `/health`   | Service health check |
| `POST` | `/api/predict` | Run two-stage prediction |
| `GET`  | `/docs`     | Interactive Swagger UI |

### POST `/api/predict` — Request Body

```json
{
  "no_of_dependents": 2,
  "education": "Graduate",
  "self_employed": "No",
  "income_annum": 9600000,
  "loan_amount": 29900000,
  "loan_term": 12,
  "cibil_score": 778,
  "residential_assets_value": 2400000,
  "commercial_assets_value": 17600000,
  "luxury_assets_value": 22700000,
  "bank_asset_value": 8000000
}
```

### Response

```json
{
  "approved": true,
  "approval_probability": 0.97,
  "predicted_loan_amount": 28541234.0,
  "message": "Congratulations! Your loan application is likely to be approved. Estimated sanctioned amount: ₹28,541,234."
}
```
