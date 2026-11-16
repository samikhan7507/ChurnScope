# ============================================================
# Makefile — ChurnScope Developer & Operations Commands
# ============================================================

.PHONY: help setup download-data train train-synthetic train-hpo train-xgboost \
        api api-prod dashboard pwa-assets predict predict-file \
        test test-unit test-api test-fast lint format lint-fix \
        eda features modeling docker-build docker-run docker-up \
        docker-up-monitoring docker-down docker-logs health sample-predict clean clean-all

PYTHON := python3
PORT := 8000
DASHBOARD_PORT := 8501
CODE_PATHS := src/ api/ tests/ train_pipeline.py predict.py streamlit_app.py run_dashboard.py scripts/generate_pwa_assets.py

help: ## Show all available commands
	@echo ""
	@echo "ChurnScope — Available Commands"
	@echo "================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ── Setup & Data ──────────────────────────────────────────────────────────
setup: ## Install dependencies, create directories, and copy .env.example
	pip install --upgrade pip
	pip install -r requirements.txt
	mkdir -p data/raw data/processed data/features
	mkdir -p models/trained models/artifacts models/registry
	mkdir -p logs results notebooks/plots
	@[ -f .env ] || cp .env.example .env
	@echo "Setup complete. Run 'make download-data && make train' or 'make train-synthetic'."

download-data: ## Download the real IBM Telco Customer Churn dataset (7,043 rows)
	bash scripts/download_data.sh

pwa-assets: ## Regenerate PWA icons and install-dialog screenshots in static/
	$(PYTHON) scripts/generate_pwa_assets.py

# ── Training ──────────────────────────────────────────────────────────────
train: ## Train LightGBM model on data/raw/telco_churn.csv
	$(PYTHON) train_pipeline.py

train-synthetic: ## Train LightGBM model on generated synthetic data
	$(PYTHON) train_pipeline.py --synthetic --n-samples 7043

train-hpo: ## Train with Optuna hyperparameter search
	$(PYTHON) train_pipeline.py --optimize-hpo

train-xgboost: ## Train using XGBoost instead of LightGBM
	$(PYTHON) train_pipeline.py --algorithm xgboost

# ── API & Installable PWA Dashboard ───────────────────────────────────────
api: ## Start FastAPI server with hot reload (port 8000)
	uvicorn api.main:app --host 0.0.0.0 --port $(PORT) --reload

api-prod: ## Start FastAPI server in production mode (2 workers)
	uvicorn api.main:app --host 0.0.0.0 --port $(PORT) --workers 2

dashboard: ## Start Streamlit HUD + root-scoped installable PWA (port 8501)
	DASHBOARD_PORT=$(DASHBOARD_PORT) $(PYTHON) run_dashboard.py

# ── Batch Inference ───────────────────────────────────────────────────────
predict: ## Score full data/raw/telco_churn.csv cohort via CLI
	$(PYTHON) predict.py --input data/raw/telco_churn.csv

predict-file: ## Score custom file: make predict-file INPUT=path/to/customers.csv
	$(PYTHON) predict.py --input $(INPUT)

# ── Testing & Quality ─────────────────────────────────────────────────────
test: ## Run full test suite with coverage gate (>=80%)
	pytest tests/ -v --cov=src --cov=api --cov-report=term-missing

test-unit: ## Run unit tests only
	pytest tests/test_data_loader.py tests/test_feature_engineering.py tests/test_preprocessor.py tests/test_trainer_evaluator.py tests/test_predictor.py -v

test-api: ## Run FastAPI & PWA integration tests
	pytest tests/test_api.py -v

test-fast: ## Fail-fast test run without coverage
	pytest tests/ -v -x

lint: ## Check lint rules and formatting with Ruff
	ruff check $(CODE_PATHS)
	ruff format --check $(CODE_PATHS)

format: ## Format code with Ruff
	ruff format $(CODE_PATHS)

lint-fix: ## Auto-fix lint findings and format code
	ruff check --fix $(CODE_PATHS)
	ruff format $(CODE_PATHS)

# ── Analysis Scripts ──────────────────────────────────────────────────────
eda: ## Generate EDA report and plots in notebooks/plots/
	$(PYTHON) notebooks/01_eda.py

features: ## Generate feature engineering analysis and plots
	$(PYTHON) notebooks/02_feature_engineering.py

modeling: ## Generate model comparison, calibration, and ROI plots
	$(PYTHON) notebooks/03_modeling.py

# ── Docker & Monitoring ───────────────────────────────────────────────────
docker-build: ## Build production API and PWA dashboard Docker images
	docker build --target production -t churnscope-api:latest .
	docker build --target dashboard -t churnscope-dashboard:latest .

docker-run: ## Run standalone API container (auto-trains if no model mounted)
	docker run --rm -p $(PORT):8000 -e AUTO_TRAIN_ON_STARTUP=true churnscope-api:latest

docker-up: ## Start API + PWA Dashboard with Docker Compose
	docker compose up --build -d
	@echo "API       : http://localhost:$(PORT) (docs: /docs)"
	@echo "Dashboard : http://localhost:$(DASHBOARD_PORT) (PWA installable)"

docker-up-monitoring: ## Start API + Dashboard + Prometheus + Grafana
	docker compose --profile monitoring up --build -d
	@echo "API        : http://localhost:$(PORT)"
	@echo "Dashboard  : http://localhost:$(DASHBOARD_PORT)"
	@echo "Prometheus : http://localhost:9090"
	@echo "Grafana    : http://localhost:3000"

docker-down: ## Stop Docker Compose stack
	docker compose --profile monitoring down

docker-logs: ## Tail FastAPI container logs
	docker compose logs -f churnscope-api

# ── Utilities ─────────────────────────────────────────────────────────────
health: ## Check running API readiness status
	curl -s http://localhost:$(PORT)/health/ready | $(PYTHON) -m json.tool

sample-predict: ## Send sample prediction + SHAP request to running API
	curl -s -X POST http://localhost:$(PORT)/predict/explain \
		-H "Content-Type: application/json" \
		-d '{"customerID":"DEMO-001","gender":"Female","SeniorCitizen":"0","Partner":"No","Dependents":"No","tenure":3,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":95.5,"TotalCharges":286.5}' \
		| $(PYTHON) -m json.tool

clean: ## Remove generated model binaries, caches, logs, and scratch results
	rm -rf models/trained/*.joblib models/artifacts/*.joblib
	rm -rf data/processed/* data/features/*
	rm -rf logs/*.log results/ notebooks/plots/*.png
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	@echo "Cleaned generated binaries and caches (kept tracked registry & eval JSON)."

clean-all: clean ## Remove all generated artifacts including downloaded raw data
	rm -rf data/raw/*.csv
	@echo "Cleaned all generated files and raw datasets."
