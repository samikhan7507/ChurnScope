"""
middleware.py — Observability, Prometheus instrumentation, rate limiting, and API-key security.
"""

from __future__ import annotations

import os
import re
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable

from fastapi import Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware

from src import get_logger

logger = get_logger(__name__)

_SAFE_REQ_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# ── Prometheus Metrics (matches monitoring/alert_rules.yml & grafana_dashboard.json) ──
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests handled by ChurnScope API",
    ["method", "handler", "status"],
)
HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "handler"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
CHURN_MODEL_LOADED = Gauge(
    "churn_model_loaded",
    "1 if a trained churn model is loaded and ready, 0 otherwise",
)
CHURN_PREDICTIONS_TOTAL = Counter(
    "churn_predictions_total",
    "Total customer churn predictions served",
    ["prediction", "risk_tier"],
)
CHURN_SCORE_HISTOGRAM = Histogram(
    "churn_prediction_probability",
    "Distribution of predicted customer churn probabilities",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)


def record_prediction_metrics(results: list[dict]) -> None:
    """Update Prometheus prediction counters and probability distribution."""
    for item in results:
        CHURN_PREDICTIONS_TOTAL.labels(
            prediction=str(item["churn_prediction"]),
            risk_tier=str(item["risk_tier"]),
        ).inc()
        CHURN_SCORE_HISTOGRAM.observe(float(item["churn_probability"]))


def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Enforce optional API-key authentication when the API_KEY environment variable is set."""
    expected = os.getenv("API_KEY", "").strip()
    if not expected:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key.strip(), expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Inject correlation IDs, record Prometheus telemetry, and sanitize unhandled 500s."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        incoming_id = request.headers.get("X-Request-ID", "")
        request_id = incoming_id if _SAFE_REQ_ID.match(incoming_id) else uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        path = request.url.path
        method = request.method
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            latency_s = time.perf_counter() - start
            logger.exception(f"[{request_id}] Unhandled error on {method} {path}")
            HTTP_REQUESTS_TOTAL.labels(method=method, handler=path, status="500").inc()
            HTTP_REQUEST_DURATION.labels(method=method, handler=path).observe(latency_s)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Internal server error", "request_id": request_id},
                headers={
                    "X-Request-ID": request_id,
                    "X-Response-Time-Ms": f"{latency_s * 1000:.1f}",
                },
            )

        latency_s = time.perf_counter() - start
        latency_ms = latency_s * 1000.0
        status_code = response.status_code

        HTTP_REQUESTS_TOTAL.labels(method=method, handler=path, status=str(status_code)).inc()
        HTTP_REQUEST_DURATION.labels(method=method, handler=path).observe(latency_s)

        if path not in {"/metrics", "/health/live"}:
            client_host = request.client.host if request.client else "unknown"
            log_fn = logger.info if status_code < 400 else logger.warning
            log_fn(
                f"[{request_id}] {method} {path} -> {status_code} "
                f"({latency_ms:.1f}ms) client={client_host}"
            )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{latency_ms:.1f}"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Lightweight sliding-window per-IP rate limiter for prediction endpoints."""

    _EXEMPT_PREFIXES = ("/health", "/metrics", "/docs", "/redoc", "/openapi", "/app/static")
    _EXEMPT_EXACT = {"/", "/manifest.json", "/manifest.webmanifest", "/sw.js", "/favicon.ico"}

    def __init__(self, app, requests_per_minute: int = 120) -> None:
        super().__init__(app)
        self.requests_per_minute = int(requests_per_minute)
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        limit = self.requests_per_minute
        if raw_env := os.getenv("RATE_LIMIT_PER_MINUTE"):
            try:
                limit = int(raw_env)
            except ValueError:
                pass

        path = request.url.path
        if limit <= 0 or path in self._EXEMPT_EXACT or path.startswith(self._EXEMPT_PREFIXES):
            return await call_next(request)

        client_ip = request.client.host if request.client else "anonymous"
        now = time.monotonic()
        cutoff = now - 60.0

        with self._lock:
            bucket = self._hits[client_ip]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= limit:
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"error": "Rate limit exceeded. Please retry shortly."},
                    headers={"Retry-After": "60"},
                )
            bucket.append(now)

        return await call_next(request)
