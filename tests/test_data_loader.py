"""Unit tests for DataLoader ingestion, coercion, schema validation, and synthetic data."""

from __future__ import annotations

import pandas as pd
import pytest

from src import load_config
from src.data_loader import DataLoader


def test_load_csv_parquet_and_json(base_config, tmp_path):
    loader = DataLoader(base_config)
    df = DataLoader.generate_synthetic_data(n_samples=120, random_state=7)
    df["Churn"] = df["Churn"].map({1: "Yes", 0: "No"})

    csv_path = tmp_path / "sample.csv"
    pq_path = tmp_path / "sample.parquet"
    json_path = tmp_path / "sample.json"

    df.to_csv(csv_path, index=False)
    df.to_parquet(pq_path, index=False)
    df.to_json(json_path, orient="records")

    for src in (csv_path, pq_path, json_path):
        loaded = loader.load(str(src), require_target=True)
        assert len(loaded) == 120
        assert set(loaded["Churn"].unique()).issubset({0, 1})
        assert set(loaded["SeniorCitizen"].unique()).issubset({"0", "1"})


def test_telco_blank_total_charges_and_int_senior_coerced(base_config):
    loader = DataLoader(base_config)
    df = DataLoader.generate_synthetic_data(n_samples=20, random_state=3)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    df["TotalCharges"] = df["TotalCharges"].astype(str)
    df.loc[0, "tenure"] = 0
    df.loc[0, "TotalCharges"] = " "
    df.loc[1, "tenure"] = 10
    df.loc[1, "TotalCharges"] = "  "

    cleaned = loader.prepare_dataframe(df, require_target=True)
    assert cleaned.loc[0, "TotalCharges"] == 0.0
    assert pd.isna(cleaned.loc[1, "TotalCharges"])
    assert set(cleaned["SeniorCitizen"].unique()).issubset({"0", "1"})


def test_validation_rejects_bad_inputs(base_config, tmp_path):
    loader = DataLoader(base_config)

    with pytest.raises(FileNotFoundError):
        loader.load(str(tmp_path / "missing.csv"))

    bad_ext = tmp_path / "data.txt"
    bad_ext.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported data format"):
        loader.load(str(bad_ext))

    with pytest.raises(ValueError, match="empty"):
        loader.prepare_dataframe(pd.DataFrame())

    df = DataLoader.generate_synthetic_data(n_samples=15, random_state=1)
    with pytest.raises(ValueError, match="Missing required columns"):
        loader.prepare_dataframe(df.drop(columns=["Contract"]))

    bad_target = df.copy()
    bad_target["Churn"] = "Maybe"
    with pytest.raises(ValueError, match="Unexpected values in target"):
        loader.prepare_dataframe(bad_target, require_target=True)

    neg_df = df.copy()
    neg_df.loc[0, "MonthlyCharges"] = -5.0
    with pytest.raises(ValueError, match="negative"):
        loader.prepare_dataframe(neg_df)


def test_synthetic_data_is_domain_consistent():
    df = DataLoader.generate_synthetic_data(n_samples=500, random_state=42)
    no_phone = df[df["PhoneService"] == "No"]
    assert (no_phone["MultipleLines"] == "No phone service").all()

    no_net = df[df["InternetService"] == "No"]
    for col in (
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ):
        assert (no_net[col] == "No internet service").all()


def test_env_overrides_applied(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("PORT", "10000")
    monkeypatch.setenv("API_WORKERS", "3")
    monkeypatch.setenv("MAX_BATCH_SIZE", "250")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://a.onrender.com, https://b.onrender.com")
    monkeypatch.setenv("AUTO_TRAIN_ON_STARTUP", "true")
    monkeypatch.setenv("ENABLE_DOCS", "false")

    cfg = load_config("config/config.yaml", apply_env=True)
    assert cfg["logging"]["level"] == "DEBUG"
    assert cfg["api"]["port"] == 10000
    assert cfg["api"]["workers"] == 3
    assert cfg["api"]["max_batch_size"] == 250
    assert cfg["api"]["cors_origins"] == ["https://a.onrender.com", "https://b.onrender.com"]
    assert cfg["api"]["auto_train_on_startup"] is True
    assert cfg["api"]["docs_url"] is None
