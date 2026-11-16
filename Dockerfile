# ============================================================
# ChurnScope — Multi-Stage Production Dockerfile
# Targets:
#   - dashboard  : Streamlit + PWA dashboard (port 8501 / $PORT)
#   - production : FastAPI + LightGBM/SHAP API (port 8000 / $PORT) [default]
# ============================================================

# ---- Stage 1: Dashboard Builder ----
FROM python:3.11-slim AS dashboard-builder
WORKDIR /build
COPY requirements-dashboard.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements-dashboard.txt

# ---- Stage 2: Dashboard & PWA Runtime ----
FROM python:3.11-slim AS dashboard
WORKDIR /app
RUN groupadd -r appuser && useradd -r -g appuser -m appuser
COPY --from=dashboard-builder /install /usr/local
COPY .streamlit/ ./.streamlit/
COPY static/ ./static/
COPY models/artifacts/evaluation_results.json ./models/artifacts/evaluation_results.json
COPY models/registry/model_registry.json ./models/registry/model_registry.json
COPY streamlit_app.py run_dashboard.py ./
RUN chown -R appuser:appuser /app
USER appuser
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_STATIC_SERVING=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; p = os.getenv('PORT', '8501'); urllib.request.urlopen(f'http://127.0.0.1:{p}/_stcore/health')"
CMD ["python", "run_dashboard.py"]

# ---- Stage 3: API Builder ----
FROM python:3.11-slim AS api-builder
WORKDIR /build
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    rm -rf /var/lib/apt/lists/*
COPY requirements-api.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements-api.txt

# ---- Stage 4: Production API Runtime (Default) ----
FROM python:3.11-slim AS production
WORKDIR /app
# libgomp1 is required by LightGBM and XGBoost OpenMP runtime on Debian slim.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -r appuser && useradd -r -g appuser -m appuser
COPY --from=api-builder /install /usr/local
COPY src/ ./src/
COPY api/ ./api/
COPY config/ ./config/
COPY static/ ./static/
COPY models/artifacts/evaluation_results.json ./models/artifacts/evaluation_results.json
COPY models/registry/model_registry.json ./models/registry/model_registry.json
COPY train_pipeline.py predict.py ./
RUN mkdir -p data/raw data/processed data/features \
             models/trained models/artifacts models/registry \
             logs results && \
    chown -R appuser:appuser /app
USER appuser
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CONFIG_PATH=/app/config/config.yaml \
    LOGGING_CONFIG_PATH=/app/config/logging.yaml \
    PORT=8000 \
    API_WORKERS=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; p = os.getenv('PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{p}/health/ready')"
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${API_WORKERS:-1} --log-level info"]
