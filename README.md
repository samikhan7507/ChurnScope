# ChurnScope — Production Customer Churn Prediction, SHAP Explainability & PWA

[![CI / CD Pipeline](https://github.com/samikhan7507/ChurnScope/actions/workflows/ci.yml/badge.svg)](https://github.com/samikhan7507/ChurnScope/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20%2B%20SHAP-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PWA Ready](https://img.shields.io/badge/PWA-Phone%20%2B%20Laptop%20Installable-F59E0B)](static/manifest.json)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **ChurnScope** is an end-to-end MLOps platform for telecom customer churn prediction and retention targeting. It combines a leak-free 41-feature scikit-learn + LightGBM/XGBoost training pipeline, probability calibration, champion–challenger registry gating, real-time `shap.TreeExplainer` explanations via FastAPI, Prometheus/Grafana observability, one-click Render deployment (`render.yaml` + `.env.example`), and an installable **Progressive Web App (PWA)** dashboard for desktop and mobile.

---

## Verified Model Performance (IBM Telco Customer Churn — 7,043 Customers)

Evaluated on a stratified **1,409-customer held-out test set** (70% train = 4,930 / 10% validation = 704 / 20% test = 1,409; churn prevalence = **26.54%**). Feature statistics, probability calibration (`sigmoid`), and the decision threshold (`0.2757`) are fit strictly on training/validation splits with zero test leakage.

| Metric | Production LightGBM (`v1`) | Source / Methodology |
|---|---|---|
| **5-Fold CV ROC-AUC** | **0.8472 ± 0.0113** | Stratified 5-fold CV on training split |
| **Test ROC-AUC** | **0.8462** | Held-out test set (`n = 1,409`) |
| **Test PR-AUC** | **0.6858** | 2.58× over 0.2654 random baseline |
| **F1 Score** | **0.6345** | Threshold `0.2757` selected on validation split |
| **Recall / Precision** | **79.14% / 52.95%** | Captures 296 of 374 actual churners (`FP=263, FN=78, TN=772`) |
| **Brier Score** | **0.1333** | Platt-calibrated probability accuracy |
| **Top-20% Decile Lift** | **2.61×** | Top 20% highest-risk cohort captures **52.1%** of all churners |
| **Estimated Net LTV Benefit** | **+$32,386 (231.7% ROI)** | Per-customer `MonthlyCharges × 6 mo × 35%` save rate − `$25` offer cost |
| **Test Coverage** | **90%** | 28 unit + API + PWA tests (`pytest --cov=src --cov=api`) |

### Multi-Model Benchmark (Same 41-Feature Pipeline & Split)

| Model | Test ROC-AUC | Test PR-AUC | Brier Score | Role |
|---|---|---|---|---|
| Dummy Classifier (prior) | 0.5000 | 0.2654 | 0.1950 | Baseline floor |
| Logistic Regression (scaled + 22 domain features) | 0.8545 | 0.6978 | 0.1303 | Linear benchmark |
| XGBoost (calibrated) | 0.8457 | 0.6872 | 0.1343 | Supported tree engine (`--algorithm xgboost`) |
| **LightGBM (calibrated + TreeExplainer SHAP)** | **0.8462** | **0.6858** | **0.1333** | Default production model (non-linear interactions + fast exact SHAP) |

**Top 5 Global SHAP Drivers:** `is_month_to_month`, `monthly_charge_per_tenure`, `charge_per_service`, `is_long_term_contract`, `OnlineSecurity`.

---

## Architecture

```text
data/raw/telco_churn.csv (7,043 rows)
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Leak-Free Training Pipeline (train_pipeline.py)                      │
│  1. DataLoader         : schema/value checks + Telco blank coercion  │
│  2. Stratified Split   : 70% train / 10% val / 20% test              │
│  3. FeatureEngineer    : 22 domain features (fit on X_train only)    │
│  4. ColumnTransformer  : median impute + scale + unknown-safe encode │
│  5. ChurnTrainer       : LightGBM/XGBoost + early stop + calibration │
│  6. ChurnEvaluator     : val-set threshold + LTV ROI + Lift + SHAP   │
│  7. Model Registry     : champion-challenger gate + versioned files  │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌──────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│ FastAPI Service  │   │ Installable PWA UI │   │ CLI Batch Scorer   │
│ (api/main.py)    │◄──┤ (run_dashboard.py  │   │ (predict.py)       │
│ /predict         │   │  streamlit_app.py) │   │ Chunked CSV/JSON/  │
│ /predict/explain │   │ /manifest.json     │   │ Parquet inference  │
│ /batch-predict   │   │ /sw.js (scope /)   │   │ (~16k rows/sec)    │
│ /model/metrics   │   │ Phone + Laptop app │   └────────────────────┘
│ /health/{live,   │   └────────────────────┘
│          ready}  │
│ /metrics (Prom)  │──────────► Prometheus + Grafana + Alert Rules
└──────────────────┘
```

---

## Quick Start (Local)

```bash
# 1. Clone and enter repo
git clone https://github.com/samikhan7507/ChurnScope.git
cd ChurnScope

# 2. One-command setup (creates .venv, installs deps, copies .env.example, downloads data & trains)
bash scripts/setup.sh --yes

# 3. Terminal 1 — Start FastAPI backend (http://localhost:8000/docs)
source .venv/bin/activate
make api

# 4. Terminal 2 — Start Installable PWA Dashboard (http://localhost:8501)
source .venv/bin/activate
make dashboard
```

---

## Install as an App (PWA — Laptop & Phone)

ChurnScope ships a full Web App Manifest (`static/manifest.json`), a root-scoped service worker (`/sw.js` with `Service-Worker-Allowed: /`), standard + maskable 192×192 and 512×512 icons, Apple touch icon, desktop (`wide`) + mobile (`narrow`) install screenshots, offline fallback (`static/offline.html`), and `<head>` injection via `run_dashboard.py`.

- **Laptop / Desktop (Chrome, Edge, Brave):** Open the dashboard over `https://` (e.g., your Render URL) or `http://localhost:8501`. Click the **install app icon (`⊕` / monitor-down arrow) at the right end of the browser address/search bar**, or click the amber **`INSTALL APP`** button in the dashboard top bar. Launches in a standalone desktop window with app shortcuts.
- **Android Phone (Chrome / Edge / Samsung Internet):** Open the dashboard URL and tap the **`INSTALL APP`** button, the bottom **"Install ChurnScope"** prompt, or `⋮ → Install app / Add to Home screen`. Uses adaptive maskable icons.
- **iPhone / iPad (Safari):** Open the dashboard URL, tap **Share (`↑`) → Add to Home Screen → Add**. Launches full-screen with the ChurnScope Apple touch icon and dark status bar.

---

## Deploy to Render (`render.yaml` + `.env.example`)

This repository includes both a **Render Blueprint** ([`render.yaml`](render.yaml)) and a complete environment variable template ([`.env.example`](.env.example)):

1. **Blueprint Deploy (Recommended):** In the Render Dashboard, click **New + → Blueprint**, connect `samikhan7507/ChurnScope`, and apply `render.yaml`. Render automatically provisions:
   - `churnscope-api` — downloads the IBM Telco dataset and trains the production model during build, exposes `/health/ready`, `/predict/explain`, and `/metrics`, and generates a shared `API_KEY`.
   - `churnscope-dashboard` — runs `python run_dashboard.py` on `$PORT`, automatically connects to `churnscope-api`, and serves the installable PWA over HTTPS.
2. **Environment Configuration:** Copy `.env.example` to `.env` for local overrides, or paste key/value pairs into **Render → Service → Environment**:
   - **API service keys:** `PYTHON_VERSION`, `APP_ENV`, `LOG_LEVEL`, `CONFIG_PATH`, `LOGGING_CONFIG_PATH`, `API_HOST`, `API_PORT`, `API_WORKERS`, `MODEL_ALGORITHM`, `MODEL_VERSION`, `DEFAULT_THRESHOLD`, `THRESHOLD_STRATEGY`, `MAX_BATCH_SIZE`, `AUTO_TRAIN_ON_STARTUP`, `AUTO_TRAIN_SAMPLES`, `API_KEY`, `DASHBOARD_URL`, `ALLOWED_ORIGINS`, `RATE_LIMIT_PER_MINUTE`, `ENABLE_METRICS`, `ENABLE_DOCS`.
   - **Dashboard PWA keys:** `DASHBOARD_HOST`, `DASHBOARD_PORT`, `API_BASE_URL`, `PUBLIC_API_URL`, `API_KEY`, `API_TIMEOUT_SECONDS`, `STREAMLIT_SERVER_HEADLESS`, `STREAMLIT_SERVER_ENABLE_STATIC_SERVING`, `STREAMLIT_SERVER_ENABLE_CORS`, `STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION`, `STREAMLIT_BROWSER_GATHER_USAGE_STATS`, `PWA_APP_NAME`, `PWA_SHORT_NAME`, `PWA_THEME_COLOR`, `PWA_BACKGROUND_COLOR`.

---

## Docker & Monitoring Stack

```bash
# Start FastAPI (port 8000) + PWA Dashboard (port 8501)
make docker-up

# Start full stack including Prometheus (9090) + provisioned Grafana (3000)
make docker-up-monitoring

# Tear down
make docker-down
```

---

## API Reference

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/health/live` | Public | Liveness probe (always `200` while process runs) |
| `GET` | `/health`, `/health/ready` | Public | Readiness probe (`200` when model loaded, `503` when degraded) |
| `POST` | `/predict` | `X-API-Key`* | Calibrated single-customer churn probability & risk tier |
| `POST` | `/predict/explain?top_k=12` | `X-API-Key`* | Prediction + real `TreeExplainer` SHAP values + retention actions |
| `POST` | `/batch-predict` | `X-API-Key`* | Vectorized batch scoring (up to `MAX_BATCH_SIZE`, default `1000`) |
| `GET` | `/model/info` | `X-API-Key`* | Loaded model version, threshold, feature count, and calibration |
| `GET` | `/model/metrics` | `X-API-Key`* | Full evaluation report, confusion matrix, ROI, global SHAP & registry |
| `GET` | `/metrics` | Public | Prometheus OpenMetrics scrape endpoint |
| `GET` | `/manifest.json`, `/sw.js` | Public | PWA Web App Manifest and root-scoped Service Worker |

*\*Authentication is optional when `API_KEY` is unset (local dev) and enforced automatically with constant-time comparison when `API_KEY` is configured.*

### Sample Request

```bash
curl -X POST http://localhost:8000/predict/explain \
  -H "Content-Type: application/json" \
  -d '{
    "customerID": "CUST-0042",
    "gender": "Female", "SeniorCitizen": "0", "Partner": "No", "Dependents": "No",
    "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.50, "TotalCharges": 286.50
  }'
```

---

## Project Structure

```text
ChurnScope/
├── api/                        # FastAPI service, strict Pydantic v2 schemas, Prometheus & auth middleware
├── config/                     # Central YAML configuration (with env overrides) & rotating logging config
├── monitoring/                 # Prometheus scrape/alert rules & provisioned Grafana dashboard
├── notebooks/                  # Reproducible EDA, feature engineering & multi-model benchmark scripts
├── scripts/                    # setup.sh, download_data.sh, retrain.sh & PWA asset generator
├── src/                        # DataLoader, FeatureEngineer (22), Preprocessor (41), Trainer, Evaluator, Predictor
├── static/                     # PWA manifest.json, sw.js, offline.html, icons (192/512/maskable/apple) & screenshots
├── tests/                      # 28 unit, integration, security & PWA tests (90% coverage)
├── .env.example                # Complete Render (API + Dashboard PWA) & local environment variables
├── render.yaml                 # Render Blueprint for two-service production deployment
├── Dockerfile                  # Multi-target non-root image (`production` API + `dashboard` PWA)
├── docker-compose.yml          # API + PWA Dashboard + optional Prometheus/Grafana profile
├── run_dashboard.py            # ASGI PWA server wrapping Streamlit with root /manifest.json & /sw.js
├── streamlit_app.py            # 4-tab responsive ChurnScope HUD (Single, Batch CSV, Live SHAP, Real Metrics)
├── train_pipeline.py           # End-to-end training, calibration, evaluation & registry CLI
└── predict.py                  # Unlimited chunked batch scoring CLI (CSV / Parquet / JSON)
```

---

## Testing & Code Quality

```bash
make lint        # ruff check + ruff format --check
make test        # pytest with >=80% coverage gate (currently 90%)
make predict     # score full 7,043-row dataset via CLI
```

---

## License

Released under the [MIT License](LICENSE).
