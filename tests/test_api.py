"""End-to-end integration tests for FastAPI endpoints, auth, metrics, and PWA manifest/SW."""

from __future__ import annotations

import copy

import pytest
import yaml
from fastapi.testclient import TestClient

from api.main import app
from tests.conftest import LOYAL_CUSTOMER, SAMPLE_CUSTOMER


@pytest.fixture
def api_client(trained_workspace, monkeypatch):
    monkeypatch.setenv("CONFIG_PATH", trained_workspace["config_path"])
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "500")
    with TestClient(app) as client:
        yield client


def test_root_and_health_probes(api_client):
    root = api_client.get("/")
    assert root.status_code == 200
    assert root.json()["pwa_manifest"] == "/manifest.json"
    assert "X-Request-ID" in root.headers
    assert "X-Response-Time-Ms" in root.headers

    # Upstream correlation ID is preserved when safe.
    echoed = api_client.get("/health/live", headers={"X-Request-ID": "req-abc-123"})
    assert echoed.status_code == 200
    assert echoed.headers["X-Request-ID"] == "req-abc-123"

    for endpoint in ("/health", "/health/ready"):
        resp = api_client.get(endpoint)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["is_model_loaded"] is True
        assert body["n_features"] == 41


def test_degraded_mode_returns_503(base_config, tmp_path, monkeypatch):
    empty_cfg = copy.deepcopy(base_config)
    empty_cfg["paths"]["model_dir"] = str(tmp_path / "no_models")
    empty_cfg["paths"]["artifact_dir"] = str(tmp_path / "no_artifacts")
    empty_cfg["paths"]["registry"] = str(tmp_path / "no_registry.json")
    empty_cfg["api"]["auto_train_on_startup"] = False
    cfg_file = tmp_path / "empty.yaml"
    cfg_file.write_text(yaml.safe_dump(empty_cfg), encoding="utf-8")

    monkeypatch.setenv("CONFIG_PATH", str(cfg_file))
    monkeypatch.setenv("AUTO_TRAIN_ON_STARTUP", "false")
    monkeypatch.delenv("API_KEY", raising=False)

    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        ready = client.get("/health/ready")
        assert ready.status_code == 503
        assert ready.json()["status"] == "degraded"
        assert client.post("/predict", json=SAMPLE_CUSTOMER).status_code == 503
        assert client.get("/model/metrics").status_code == 404


def test_predict_explain_and_batch_endpoints(api_client):
    pred = api_client.post("/predict", json=SAMPLE_CUSTOMER)
    assert pred.status_code == 200
    pdata = pred.json()
    assert 0.0 <= pdata["churn_probability"] <= 1.0
    assert pdata["customer_id"] == "TEST-001"
    assert pdata["churn_label"] in {"Churn", "No Churn"}

    expl = api_client.post("/predict/explain?top_k=8", json=SAMPLE_CUSTOMER)
    assert expl.status_code == 200
    edata = expl.json()
    assert len(edata["shap_values"]) == 8
    assert len(edata["recommendations"]) >= 1
    assert edata["churn_probability"] == pdata["churn_probability"]

    batch = api_client.post("/batch-predict", json={"customers": [SAMPLE_CUSTOMER, LOYAL_CUSTOMER]})
    assert batch.status_code == 200
    bdata = batch.json()
    assert bdata["total"] == 2
    assert len(bdata["predictions"]) == 2
    assert "risk_tiers" in bdata["summary"]


def test_strict_schema_validation(api_client):
    assert (
        api_client.post("/predict", json={**SAMPLE_CUSTOMER, "gender": "Unknown"}).status_code
        == 422
    )
    assert (
        api_client.post("/predict", json={**SAMPLE_CUSTOMER, "PaymentMethod": "Cash"}).status_code
        == 422
    )
    # Cross-field domain consistency checks
    inconsistent_phone = {**SAMPLE_CUSTOMER, "PhoneService": "No", "MultipleLines": "Yes"}
    assert api_client.post("/predict", json=inconsistent_phone).status_code == 422

    inconsistent_net = {**SAMPLE_CUSTOMER, "InternetService": "No", "StreamingTV": "Yes"}
    assert api_client.post("/predict", json=inconsistent_net).status_code == 422

    assert api_client.post("/batch-predict", json={"customers": []}).status_code == 422


def test_model_metrics_and_prometheus_endpoint(api_client):
    m_resp = api_client.get("/model/metrics")
    assert m_resp.status_code == 200
    metrics = m_resp.json()
    assert metrics["roc_auc"] > 0.70
    assert "confusion_matrix" in metrics
    assert "feature_importance" in metrics
    assert metrics["registry"]["total_versions"] >= 1

    prom = api_client.get("/metrics")
    assert prom.status_code == 200
    assert "churn_model_loaded 1.0" in prom.text
    assert "http_requests_total" in prom.text
    assert "churn_predictions_total" in prom.text


def test_api_key_authentication_and_rate_limit(api_client, monkeypatch):
    monkeypatch.setenv("API_KEY", "super-secret-render-key")
    assert api_client.post("/predict", json=SAMPLE_CUSTOMER).status_code == 401
    assert (
        api_client.post(
            "/predict", json=SAMPLE_CUSTOMER, headers={"X-API-Key": "wrong-key"}
        ).status_code
        == 401
    )
    ok = api_client.post(
        "/predict", json=SAMPLE_CUSTOMER, headers={"X-API-Key": "super-secret-render-key"}
    )
    assert ok.status_code == 200

    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
    # Already consumed requests from same client IP -> triggers 429 cleanly.
    limited = api_client.post("/predict", json=SAMPLE_CUSTOMER)
    assert limited.status_code == 429


def test_pwa_manifest_service_worker_and_icons(api_client):
    for m_path in ("/manifest.json", "/manifest.webmanifest"):
        resp = api_client.get(m_path)
        assert resp.status_code == 200
        assert "application/manifest+json" in resp.headers["content-type"]
        manifest = resp.json()
        assert manifest["short_name"] == "ChurnScope"
        assert manifest["display"] == "standalone"
        assert manifest["start_url"] == "/"
        assert manifest["scope"] == "/"
        sizes = {icon["sizes"] for icon in manifest["icons"]}
        assert {"192x192", "512x512"}.issubset(sizes)
        purposes = {icon["purpose"] for icon in manifest["icons"]}
        assert {"any", "maskable"}.issubset(purposes)
        form_factors = {shot["form_factor"] for shot in manifest["screenshots"]}
        assert form_factors == {"wide", "narrow"}

    sw = api_client.get("/sw.js")
    assert sw.status_code == 200
    assert sw.headers.get("service-worker-allowed") == "/"
    assert "javascript" in sw.headers["content-type"]

    for asset in (
        "/apple-touch-icon.png",
        "/favicon.ico",
        "/app/static/icon-192.png",
        "/app/static/icon-512.png",
        "/app/static/icon-maskable-512.png",
        "/app/static/offline.html",
    ):
        assert api_client.get(asset).status_code == 200


def test_dashboard_pwa_asgi_manifest_sw_and_head_injection(monkeypatch):
    """Verify the Streamlit PWA launcher enables phone + laptop address-bar install."""
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse
    from starlette.routing import Route

    import run_dashboard

    async def fake_index(_request):
        return HTMLResponse(
            "<!DOCTYPE html><html><head><title>Streamlit</title></head><body>app</body></html>"
        )

    stub_st = Starlette(routes=[Route("/", fake_index, methods=["GET"])])
    monkeypatch.setenv("PWA_APP_NAME", "ChurnScope Retention HUD")
    monkeypatch.setenv("PWA_SHORT_NAME", "ChurnScope")
    monkeypatch.setenv("PWA_THEME_COLOR", "#0b0e14")

    wrapped = run_dashboard.create_asgi_app(inner_app=stub_st)

    with TestClient(wrapped) as client:
        html_resp = client.get("/")
        assert html_resp.status_code == 200
        assert '<link rel="manifest" href="/manifest.json"' in html_resp.text
        assert 'name="apple-mobile-web-app-capable" content="yes"' in html_resp.text
        assert "beforeinstallprompt" in html_resp.text
        assert 'navigator.serviceWorker.register("/sw.js",{scope:"/"})' in html_resp.text

        m = client.get("/manifest.json")
        assert m.status_code == 200
        assert m.json()["name"] == "ChurnScope Retention HUD"
        assert m.json()["display"] == "standalone"
        assert client.head("/manifest.json").status_code == 200

        sw = client.get("/sw.js")
        assert sw.status_code == 200
        assert sw.headers["service-worker-allowed"] == "/"
        assert "no-cache" in sw.headers["cache-control"]
        assert client.head("/sw.js").status_code == 200

        assert client.get("/offline.html").status_code == 200
        assert client.get("/apple-touch-icon.png").status_code == 200
