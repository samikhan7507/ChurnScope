"""Unit tests for ChurnTrainer, ChurnEvaluator, champion-challenger gate, and batch CLI."""

from __future__ import annotations

import argparse
import copy
import json

import pytest

from predict import run_batch_inference
from src.data_loader import DataLoader
from src.evaluator import ChurnEvaluator
from src.preprocessor import ChurnPreprocessor
from src.trainer import ChurnTrainer


@pytest.fixture
def fast_splits(base_config, tmp_path):
    cfg = copy.deepcopy(base_config)
    cfg["paths"]["model_dir"] = str(tmp_path / "models")
    cfg["paths"]["artifact_dir"] = str(tmp_path / "artifacts")
    cfg["paths"]["registry"] = str(tmp_path / "registry" / "model_registry.json")
    cfg["model"]["lightgbm_params"]["n_estimators"] = 50
    cfg["model"]["xgboost_params"]["n_estimators"] = 50
    cfg["model"]["cross_validation"]["n_splits"] = 3
    df = DataLoader.generate_synthetic_data(n_samples=600, random_state=42)
    splits = ChurnPreprocessor(cfg).fit_transform(df)
    return cfg, splits


def test_xgboost_training_calibration_and_threshold_strategies(fast_splits, tmp_path):
    cfg, splits = fast_splits
    cfg["model"]["algorithm"] = "xgboost"

    trainer = ChurnTrainer(cfg)
    meta = trainer.train(splits)
    assert meta["algorithm"] == "xgboost"
    assert meta["is_calibrated"] is True
    assert 0.65 <= meta["cv_roc_auc_mean"] <= 1.0

    val_scores = trainer.predict_proba(splits["X_val"])
    test_scores = trainer.predict_proba(splits["X_test"])

    for strategy in ("f1", "roi", "custom_cost", "recall"):
        scfg = copy.deepcopy(cfg)
        scfg["threshold"]["strategy"] = strategy
        ev = ChurnEvaluator(scfg)
        res = ev.evaluate(
            y_true=splits["y_test"],
            y_scores=test_scores,
            y_val=splits["y_val"],
            y_val_scores=val_scores,
            val_monthly_charges=splits["val_monthly_charges"],
            test_monthly_charges=splits["test_monthly_charges"],
            feature_names=splits["feature_names"],
            model=trainer.base_model,
            X_test=splits["X_test"],
            training_metadata=meta,
        )
        assert 0.0 < res["threshold"] < 1.0
        assert res["threshold_source"] == "validation_set"
        assert "lift_at_20pct" in res["ranking_metrics"]
        assert "feature_importance" in res

    out_json = ev.save_results(str(tmp_path / "eval.json"))
    assert json.loads(open(out_json, encoding="utf-8").read())["roc_auc"] > 0.65


def test_optuna_hpo_and_champion_challenger_gate(fast_splits):
    cfg, splits = fast_splits
    cfg["optuna"]["enabled"] = True
    cfg["optuna"]["n_trials"] = 2

    champion = ChurnTrainer(cfg)
    champion.train(splits)
    champ_path, promoted = champion.save(version="v1", threshold=0.32)
    assert promoted is True

    # Simulate a severely degraded challenger that should be rejected by promotion gate.
    challenger = ChurnTrainer(cfg)
    challenger.model = champion.model
    challenger.base_model = champion.base_model
    challenger.training_metadata = {
        **champion.training_metadata,
        "val_roc_auc": 0.52,
        "cv_roc_auc_mean": 0.52,
    }
    _, promoted_bad = challenger.save(version="v2", threshold=0.50)
    assert promoted_bad is False

    with open(cfg["paths"]["registry"], encoding="utf-8") as fh:
        reg = json.load(fh)
    assert reg["latest"] == champ_path
    assert reg["models"][-1]["status"] == "rejected"


def test_cli_batch_inference_exceeding_1000_rows(trained_workspace, tmp_path):
    """predict.py must cleanly score >1,000 rows without hitting the API batch cap."""
    out_csv = tmp_path / "scored_1200.csv"
    args = argparse.Namespace(
        input=trained_workspace["config"]["paths"]["raw_data"],
        output=str(out_csv),
        threshold=None,
        model_path=None,
        chunk_size=500,
        config=trained_workspace["config_path"],
    )
    scored = run_batch_inference(args)
    assert len(scored) == 1200
    assert out_csv.exists()
    assert scored["churn_probability"].is_monotonic_decreasing
