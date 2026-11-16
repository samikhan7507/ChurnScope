"""Shared pytest fixtures for isolated, reproducible ChurnScope unit and integration tests."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from src import load_config
from src.predictor import ChurnPredictor
from train_pipeline import run_pipeline

SAMPLE_CUSTOMER: dict[str, Any] = {
    "customerID": "TEST-001",
    "gender": "Female",
    "SeniorCitizen": "0",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.5,
    "TotalCharges": 286.5,
}

LOYAL_CUSTOMER: dict[str, Any] = {
    "customerID": "TEST-LOYAL",
    "gender": "Male",
    "SeniorCitizen": "0",
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 68,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 69.5,
    "TotalCharges": 4726.0,
}


@pytest.fixture
def base_config() -> dict[str, Any]:
    return load_config("config/config.yaml", apply_env=False)


@pytest.fixture(scope="session")
def trained_workspace(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Train a fast, self-contained model + pipeline in an isolated temporary directory."""
    root: Path = tmp_path_factory.mktemp("churnscope_trained")
    cfg = copy.deepcopy(load_config("config/config.yaml", apply_env=False))
    cfg["paths"] = {
        "raw_data": str(root / "data" / "raw" / "telco_churn.csv"),
        "processed_data": str(root / "data" / "processed"),
        "features_data": str(root / "data" / "features"),
        "model_dir": str(root / "models" / "trained"),
        "artifact_dir": str(root / "models" / "artifacts"),
        "registry": str(root / "models" / "registry" / "model_registry.json"),
        "logging_config": "config/logging.yaml",
        "logs": str(root / "logs"),
    }
    cfg["model"]["lightgbm_params"]["n_estimators"] = 80
    cfg["model"]["cross_validation"]["n_splits"] = 3
    cfg["promotion_gate"]["min_roc_auc"] = 0.70

    cfg_path = root / "config.yaml"
    with cfg_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)

    args = argparse.Namespace(
        synthetic=True,
        save_synthetic=True,
        algorithm="lightgbm",
        optimize_hpo=False,
        threshold_strategy="f1",
        force_promote=True,
        config=str(cfg_path),
        n_samples=1200,
    )
    output = run_pipeline(args)
    return {
        "root": root,
        "config_path": str(cfg_path),
        "config": cfg,
        "pipeline_output": output,
    }


@pytest.fixture(autouse=True)
def _reset_singleton():
    ChurnPredictor.reset_instance()
    yield
    ChurnPredictor.reset_instance()
