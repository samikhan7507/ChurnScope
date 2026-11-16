"""
main.py — Production FastAPI service for ChurnScope.

Endpoints:
  GET  /                 — Service info and links
  GET  /health           — Combined service & model health (200 healthy, 503 degraded)
  GET  /health/live      — Liveness probe (always 200)
  GET  /health/ready     — Readiness probe (200 when model loaded, 503 otherwise)
  POST /predict          — Single-customer churn prediction
  POST /predict/explain  — Single-customer prediction + TreeExplainer SHAP + retention actions
  POST /batch-predict    — Batch customer scoring (up to max_batch_size)
  GET  /model/metrics    — Real held-out test metrics, curves, SHAP, and registry info
  GET  /metrics          — Prometheus OpenMetrics scrape endpoint
  GET  /manifest.json    — Progressive Web App (PWA) manifest
  GET  /sw.js            — Progressive Web App Service Worker (root scope)
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.middleware import (
    CHURN_MODEL_LOADED,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    record_prediction_metrics,
    verify_api_key,
)
from api.schemas import (
    BatchPredictionResponse,
    BatchPredictRequest,
    CustomerFeatures,
    ErrorResponse,
    ExplainPredictionResponse,
    HealthResponse,
    PredictionResponse,
)
from src import get_logger, load_config
from src.predictor import ChurnPredictor

logger = get_logger(__name__)
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _resolve_cors_origins(cfg: dict[str, Any]) -> tuple[list[str], bool]:
    """Return safe CORS origins; never combine wildcard '*' with allow_credentials=True."""
    origins = list(cfg.get("api", {}).get("cors_origins", []))
    if dashboard_url := os.getenv("DASHBOARD_URL", "").strip():
        origins.append(dashboard_url.rstrip("/"))
    if render_url := os.getenv("RENDER_EXTERNAL_URL", "").strip():
        origins.append(render_url.rstrip("/"))
    deduped = list(dict.fromkeys(origins)) or ["*"]
    if "*" in deduped:
        return ["*"], False
    return deduped, True


def _bootstrap_model_if_enabled(config: dict[str, Any]) -> bool:
    """Optionally train a baseline model on first boot (ideal for Render/Docker cold starts)."""
    api_cfg = config.get("api", {})
    if not api_cfg.get("auto_train_on_startup", False):
        return False

    raw_path = config["paths"]["raw_data"]
    use_synthetic = not os.path.exists(raw_path)
    n_samples = int(api_cfg.get("auto_train_samples", 5000))
    logger.warning(
        f"No trained model found and AUTO_TRAIN_ON_STARTUP=true. "
        f"Bootstrapping model (synthetic={use_synthetic}, n_samples={n_samples})..."
    )
    from train_pipeline import run_pipeline

    args = argparse.Namespace(
        synthetic=use_synthetic,
        save_synthetic=False,
        algorithm=config["model"].get("algorithm", "lightgbm"),
        optimize_hpo=False,
        threshold_strategy=config.get("threshold", {}).get("strategy", "f1"),
        force_promote=True,
        config=os.getenv("CONFIG_PATH", "config/config.yaml"),
        n_samples=n_samples,
    )
    run_pipeline(args)
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ChurnScope API...")
    config = load_config(os.getenv("CONFIG_PATH"))
    app.state.config = config
    app.state.predictor = None
    CHURN_MODEL_LOADED.set(0)

    try:
        predictor = ChurnPredictor(config)
        predictor.load()
        app.state.predictor = predictor
        CHURN_MODEL_LOADED.set(1)
        logger.info("ChurnPredictor loaded successfully. API ready.")
    except FileNotFoundError as exc:
        logger.warning(f"Initial model load skipped: {exc}")
        try:
            if _bootstrap_model_if_enabled(config):
                predictor = ChurnPredictor(config)
                predictor.load()
                app.state.predictor = predictor
                CHURN_MODEL_LOADED.set(1)
                logger.info("Auto-trained model loaded successfully. API ready.")
            else:
                logger.error(
                    "Running in degraded mode. Train a model with `python train_pipeline.py` "
                    "or set AUTO_TRAIN_ON_STARTUP=true."
                )
        except Exception:
            logger.exception("AUTO_TRAIN_ON_STARTUP failed; starting in degraded mode.")

    yield
    CHURN_MODEL_LOADED.set(0)
    logger.info("Shutting down ChurnScope API.")


def _initial_config() -> dict[str, Any]:
    try:
        return load_config()
    except FileNotFoundError:
        return {"api": {}}


_boot_cfg = _initial_config()
_api_cfg = _boot_cfg.get("api", {})
_cors_origins, _cors_credentials = _resolve_cors_origins(_boot_cfg)

app = FastAPI(
    title=str(_api_cfg.get("title", "ChurnScope Prediction API")),
    description=str(
        _api_cfg.get(
            "description",
            "Production ML API for customer churn prediction and real-time SHAP explainability.",
        )
    ),
    version=str(_api_cfg.get("version", "1.1.0")),
    docs_url=_api_cfg.get("docs_url", "/docs"),
    redoc_url=_api_cfg.get("redoc_url", "/redoc"),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Request-ID"],
    expose_headers=["X-Request-ID", "X-Response-Time-Ms"],
)
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=int(_api_cfg.get("rate_limit_per_minute", 120)),
)
app.add_middleware(RequestLoggingMiddleware)

if STATIC_DIR.exists():
    app.mount("/app/static", StaticFiles(directory=str(STATIC_DIR)), name="pwa-static")


def get_predictor(request: Request) -> ChurnPredictor:
    predictor: ChurnPredictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None or not predictor._is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run `python train_pipeline.py` first.",
        )
    return predictor


PredictorDep = Annotated[ChurnPredictor, Depends(get_predictor)]
AuthDep = Annotated[None, Depends(verify_api_key)]


def _build_health_payload(predictor: ChurnPredictor | None) -> tuple[HealthResponse, int]:
    if predictor is None or not predictor._is_loaded:
        return (
            HealthResponse(
                status="degraded",
                model_version="none",
                algorithm=None,
                is_calibrated=False,
                cv_roc_auc=None,
                val_roc_auc=None,
                n_features=None,
                trained_at=None,
                threshold=0.5,
                is_model_loaded=False,
            ),
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    info = predictor.get_model_info()
    return (
        HealthResponse(
            status="healthy",
            model_version=str(info.get("model_version", "unknown")),
            algorithm=info.get("algorithm"),
            is_calibrated=bool(info.get("is_calibrated", False)),
            cv_roc_auc=info.get("cv_roc_auc"),
            val_roc_auc=info.get("val_roc_auc"),
            n_features=info.get("n_features"),
            trained_at=info.get("trained_at"),
            threshold=float(info.get("threshold", 0.5)),
            is_model_loaded=True,
        ),
        status.HTTP_200_OK,
    )


@app.get("/", tags=["Root"])
def root() -> dict[str, Any]:
    return {
        "service": "ChurnScope Prediction API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "readiness": "/health/ready",
        "model_metrics": "/model/metrics",
        "prometheus_metrics": "/metrics",
        "pwa_manifest": "/manifest.json",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse}},
    tags=["Health"],
    summary="Combined API and model health check",
)
def health_check(request: Request) -> JSONResponse:
    predictor: ChurnPredictor | None = getattr(request.app.state, "predictor", None)
    payload, code = _build_health_payload(predictor)
    return JSONResponse(status_code=code, content=payload.model_dump())


@app.get("/health/live", tags=["Health"], summary="Container liveness probe")
def liveness_check() -> dict[str, str]:
    return {"status": "alive"}


@app.get(
    "/health/ready",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse}},
    tags=["Health"],
    summary="Readiness probe (200 only when trained model is loaded)",
)
def readiness_check(request: Request) -> JSONResponse:
    predictor: ChurnPredictor | None = getattr(request.app.state, "predictor", None)
    payload, code = _build_health_payload(predictor)
    return JSONResponse(status_code=code, content=payload.model_dump())


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={401: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["Prediction"],
    summary="Predict churn probability and risk tier for a single customer",
)
def predict_single(
    customer: CustomerFeatures,
    predictor: PredictorDep,
    _auth: AuthDep,
) -> PredictionResponse:
    try:
        result = predictor.predict(customer.model_dump())
        result["customer_id"] = customer.customerID
        record_prediction_metrics([result])
        return PredictionResponse(**result)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Input validation error: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Single prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed.",
        ) from exc


@app.post(
    "/predict/explain",
    response_model=ExplainPredictionResponse,
    responses={401: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["Prediction"],
    summary="Predict churn with real-time TreeExplainer SHAP attributions and retention actions",
)
def predict_and_explain(
    customer: CustomerFeatures,
    predictor: PredictorDep,
    _auth: AuthDep,
    top_k: int = 12,
) -> ExplainPredictionResponse:
    try:
        clamped_k = max(1, min(int(top_k), 41))
        result = predictor.explain(customer.model_dump(), top_k=clamped_k)
        result["customer_id"] = customer.customerID
        record_prediction_metrics([result])
        return ExplainPredictionResponse(**result)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Input validation error: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("SHAP explanation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction explanation failed.",
        ) from exc


@app.post(
    "/batch-predict",
    response_model=BatchPredictionResponse,
    responses={401: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["Prediction"],
    summary="Score up to 1,000 customers in a single vectorized batch",
)
def predict_batch(
    payload: BatchPredictRequest,
    predictor: PredictorDep,
    _auth: AuthDep,
) -> BatchPredictionResponse:
    try:
        customers_data = [c.model_dump() for c in payload.customers]
        results = predictor.predict_batch(pd.DataFrame(customers_data), enforce_limit=True)
        record_prediction_metrics(results)

        total = len(results)
        churn_count = sum(r["churn_prediction"] for r in results)
        tier_counts = Counter(r["risk_tier"] for r in results)
        avg_prob = sum(r["churn_probability"] for r in results) / max(total, 1)

        predictions = [
            PredictionResponse(**res, customer_id=customers_data[idx].get("customerID"))
            for idx, res in enumerate(results)
        ]
        return BatchPredictionResponse(
            total=total,
            churn_count=churn_count,
            churn_rate=round(churn_count / max(total, 1), 4),
            predictions=predictions,
            summary={
                "risk_tiers": dict(tier_counts),
                "avg_churn_probability": round(avg_prob, 4),
                "threshold_used": predictor.optimal_threshold,
            },
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed.",
        ) from exc


@app.get(
    "/model/metrics",
    tags=["Observability"],
    summary="Return real held-out test metrics, curves, SHAP importance, and registry metadata",
)
def model_metrics(request: Request, _auth: AuthDep) -> dict[str, Any]:
    config: dict[str, Any] = getattr(request.app.state, "config", None) or load_config()
    eval_path = os.path.join(config["paths"]["artifact_dir"], "evaluation_results.json")
    registry_path = config["paths"]["registry"]

    if not os.path.exists(eval_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation metrics not found. Run `python train_pipeline.py` first.",
        )

    with open(eval_path, encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)

    if os.path.exists(registry_path):
        with open(registry_path, encoding="utf-8") as fh:
            reg = json.load(fh)
        models = reg.get("models", [])
        active = next((m for m in reversed(models) if m.get("status") == "active"), None)
        data["registry"] = {
            "active_model": active or (models[-1] if models else None),
            "total_versions": len(models),
        }

    return data


@app.get("/metrics", include_in_schema=False, tags=["Observability"])
def prometheus_metrics(request: Request) -> Response:
    config: dict[str, Any] = getattr(request.app.state, "config", None) or _boot_cfg
    if not config.get("api", {}).get("enable_metrics", True):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metrics disabled.")
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── Progressive Web App (PWA) Endpoints ───────────────────────────────────────
@app.api_route("/manifest.json", methods=["GET", "HEAD"], include_in_schema=False)
@app.api_route("/manifest.webmanifest", methods=["GET", "HEAD"], include_in_schema=False)
def pwa_manifest() -> FileResponse:
    manifest_file = STATIC_DIR / "manifest.json"
    if not manifest_file.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Manifest not found.")
    return FileResponse(
        path=manifest_file,
        media_type="application/manifest+json",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.api_route("/sw.js", methods=["GET", "HEAD"], include_in_schema=False)
def pwa_service_worker() -> FileResponse:
    sw_file = STATIC_DIR / "sw.js"
    if not sw_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Service worker not found."
        )
    return FileResponse(
        path=sw_file,
        media_type="application/javascript",
        headers={
            "Service-Worker-Allowed": "/",
            "Cache-Control": "no-cache, no-store, must-revalidate",
        },
    )


@app.get("/apple-touch-icon.png", include_in_schema=False)
def apple_touch_icon() -> FileResponse:
    icon_file = STATIC_DIR / "apple-touch-icon.png"
    if not icon_file.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Icon not found.")
    return FileResponse(path=icon_file, media_type="image/png")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    icon_file = STATIC_DIR / "favicon-32.png"
    if not icon_file.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Favicon not found.")
    return FileResponse(path=icon_file, media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    api_section = cfg.get("api", {})
    uvicorn.run(
        "api.main:app",
        host=str(api_section.get("host", "0.0.0.0")),
        port=int(api_section.get("port", 8000)),
        reload=bool(api_section.get("reload", False)),
        log_level=str(cfg.get("logging", {}).get("level", "info")).lower(),
    )
