"""Unit tests for the ChurnPredictor inference and real-time SHAP explainability engine."""

from __future__ import annotations

import copy

import pytest

from src.predictor import ChurnPredictor
from tests.conftest import LOYAL_CUSTOMER, SAMPLE_CUSTOMER


def test_singleton_and_single_prediction(trained_workspace):
    cfg = trained_workspace["config"]
    p1 = ChurnPredictor.get_instance(cfg)
    p2 = ChurnPredictor.get_instance(cfg)
    assert p1 is p2

    high_risk = p1.predict(SAMPLE_CUSTOMER)
    low_risk = p1.predict(LOYAL_CUSTOMER)

    assert 0.0 <= high_risk["churn_probability"] <= 1.0
    assert 0.0 <= high_risk["confidence"] <= 1.0
    assert high_risk["churn_probability"] > low_risk["churn_probability"]
    assert high_risk["churn_label"] in {"Churn", "No Churn"}
    assert high_risk["risk_tier"] in {"Low", "Medium", "High", "Critical"}

    # Risk tier must always align with binary churn prediction (no "Churn" + "Low").
    if high_risk["churn_prediction"] == 1:
        assert high_risk["risk_tier"] in {"Medium", "High", "Critical"}
    if low_risk["churn_prediction"] == 0:
        assert low_risk["risk_tier"] == "Low"


def test_real_shap_explainability(trained_workspace):
    predictor = ChurnPredictor(trained_workspace["config"]).load()
    explained = predictor.explain(SAMPLE_CUSTOMER, top_k=10)

    assert "base_value" in explained
    assert len(explained["shap_values"]) == 10
    assert len(explained["recommendations"]) >= 1

    # Confirm SHAP values match direction label and contain genuine non-zero contributions.
    non_zero = [c for c in explained["shap_values"] if abs(c["shap_value"]) > 0]
    assert len(non_zero) >= 5
    for contrib in explained["shap_values"]:
        expected_dir = "increases_risk" if contrib["shap_value"] >= 0 else "decreases_risk"
        assert contrib["direction"] == expected_dir


def test_batch_prediction_limits_and_metadata(trained_workspace):
    cfg = copy.deepcopy(trained_workspace["config"])
    cfg["api"]["max_batch_size"] = 2
    predictor = ChurnPredictor(cfg).load()

    res = predictor.predict_batch([SAMPLE_CUSTOMER, LOYAL_CUSTOMER], enforce_limit=True)
    assert len(res) == 2

    with pytest.raises(ValueError, match="exceeds maximum"):
        predictor.predict_batch([SAMPLE_CUSTOMER] * 3, enforce_limit=True)

    # CLI mode (enforce_limit=False) scores arbitrary batch sizes cleanly.
    unlimited = predictor.predict_batch([SAMPLE_CUSTOMER] * 5, enforce_limit=False, chunk_size=2)
    assert len(unlimited) == 5

    with pytest.raises(ValueError, match="empty"):
        predictor.predict_batch([])

    info = predictor.get_model_info()
    assert info["is_loaded"] is True
    assert info["n_features"] == 41


def test_unloaded_and_missing_model_errors(base_config, tmp_path):
    cfg = copy.deepcopy(base_config)
    cfg["paths"]["registry"] = str(tmp_path / "none.json")
    cfg["paths"]["model_dir"] = str(tmp_path / "empty_models")

    predictor = ChurnPredictor(cfg)
    with pytest.raises(RuntimeError, match="not loaded"):
        predictor.predict(SAMPLE_CUSTOMER)
    with pytest.raises(FileNotFoundError):
        predictor.load()
