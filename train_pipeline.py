"""
train_pipeline.py — End-to-end leak-free training orchestrator for ChurnScope.

Stages:
  1. Ingest and validate raw data (or generate domain-consistent synthetic data)
  2. Split train/val/test and fit unified feature-engineering + preprocessing pipeline on train
  3. Train LightGBM/XGBoost with stratified K-fold CV, early stopping, and calibration
  4. Tune decision threshold strictly on validation split and evaluate once on held-out test set
  5. Save versioned artifacts and apply champion-challenger promotion gate

Usage:
  python train_pipeline.py                        # train on config paths.raw_data
  python train_pipeline.py --synthetic            # train on synthetic dataset
  python train_pipeline.py --algorithm xgboost    # train XGBoost instead of LightGBM
  python train_pipeline.py --optimize-hpo         # run Optuna hyperparameter search
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

from src import get_logger, load_config
from src.data_loader import DataLoader
from src.evaluator import ChurnEvaluator
from src.feature_engineering import FeatureEngineer
from src.preprocessor import ChurnPreprocessor
from src.trainer import ChurnTrainer

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ChurnScope — End-to-End Training Pipeline")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic telecom customers instead of reading paths.raw_data.",
    )
    parser.add_argument(
        "--save-synthetic",
        action="store_true",
        help="Persist generated synthetic dataset to paths.raw_data (only if missing or explicit).",
    )
    parser.add_argument("--algorithm", choices=["lightgbm", "xgboost"], default=None)
    parser.add_argument("--optimize-hpo", action="store_true")
    parser.add_argument(
        "--threshold-strategy",
        choices=["f1", "roi", "custom_cost", "recall"],
        default=None,
    )
    parser.add_argument(
        "--force-promote",
        action="store_true",
        help="Promote model to active even if champion-challenger gate declines it.",
    )
    parser.add_argument(
        "--config", default=None, help="Path to YAML config (default: CONFIG_PATH)."
    )
    parser.add_argument("--n-samples", type=int, default=7043)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run_pipeline(args: argparse.Namespace | None = None) -> dict[str, Any]:
    """Execute the full training, validation, test evaluation, and artifact registration workflow."""
    start = time.perf_counter()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if args is None:
        args = parse_args()

    config = load_config(getattr(args, "config", None))
    if getattr(args, "algorithm", None):
        config["model"]["algorithm"] = args.algorithm
    if getattr(args, "optimize_hpo", False):
        config["optuna"]["enabled"] = True
    if getattr(args, "threshold_strategy", None):
        config["threshold"]["strategy"] = args.threshold_strategy

    logger.info("=" * 60)
    logger.info("CHURNSCOPE — TRAINING PIPELINE")
    logger.info(f"Started (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 1. Data Loading (never silently overwrite real raw data unless missing or requested).
    logger.info("[1/5] DATA LOADING & VALIDATION")
    loader = DataLoader(config)
    if getattr(args, "synthetic", False):
        n_samples = int(getattr(args, "n_samples", 7043))
        random_state = int(config["data"].get("random_state", 42))
        df_raw = DataLoader.generate_synthetic_data(n_samples=n_samples, random_state=random_state)
        raw_path = config["paths"]["raw_data"]
        if getattr(args, "save_synthetic", False) or not os.path.exists(raw_path):
            os.makedirs(os.path.dirname(raw_path) or ".", exist_ok=True)
            df_raw.to_csv(raw_path, index=False)
            logger.info(f"Saved synthetic dataset to: {raw_path}")
    else:
        df_raw = loader.load(require_target=True)

    # 2. Leak-free Split + Unified Feature Engineering & Preprocessing.
    logger.info("[2/5] LEAK-FREE SPLIT, FEATURE ENGINEERING & PREPROCESSING")
    preprocessor = ChurnPreprocessor(config)
    splits = preprocessor.fit_transform(df_raw)
    preprocessor.save_processed_splits(splits)

    # Save enriched training features snapshot for inspection/EDA.
    features_dir = config["paths"].get("features_data", "data/features/")
    os.makedirs(features_dir, exist_ok=True)
    fitted_fe: FeatureEngineer = preprocessor.pipeline.named_steps["feature_engineer"]
    enriched_train = fitted_fe.transform(splits["X_train_raw"])
    enriched_train.to_parquet(os.path.join(features_dir, "train_engineered.parquet"), index=False)

    version = str(config["model"].get("version", "v1"))
    version_tag = f"{config['model']['algorithm']}_{version}_{timestamp}"
    versioned_pipeline_path = preprocessor.save(versioned_tag=version_tag, update_canonical=False)

    # 3. Model Training + Cross-Validation + Calibration.
    logger.info("[3/5] MODEL TRAINING & CROSS-VALIDATION")
    trainer = ChurnTrainer(config)
    training_meta = trainer.train(splits)

    # 4. Validation Threshold Optimization + Held-Out Test Evaluation + Global SHAP.
    logger.info("[4/5] VALIDATION THRESHOLD TUNING & HELD-OUT TEST EVALUATION")
    evaluator = ChurnEvaluator(config)
    y_val_scores = trainer.predict_proba(splits["X_val"])
    y_test_scores = trainer.predict_proba(splits["X_test"])
    eval_results = evaluator.evaluate(
        y_true=splits["y_test"],
        y_scores=y_test_scores,
        y_val=splits["y_val"],
        y_val_scores=y_val_scores,
        val_monthly_charges=splits.get("val_monthly_charges"),
        test_monthly_charges=splits.get("test_monthly_charges"),
        feature_names=splits["feature_names"],
        model=trainer.base_model,
        X_test=splits["X_test"],
        training_metadata=training_meta,
    )

    # 5. Versioned Save + Champion-Challenger Promotion Gate.
    logger.info("[5/5] ARTIFACT REGISTRATION & PROMOTION GATE")
    model_path, promoted = trainer.save(
        version=version,
        timestamp=timestamp,
        pipeline_path=versioned_pipeline_path,
        threshold=eval_results["threshold"],
        eval_summary=eval_results,
        force_promote=bool(getattr(args, "force_promote", False)),
    )
    if promoted:
        preprocessor.save(update_canonical=True)
        evaluator.save_results()
    else:
        rejected_eval_path = os.path.join(
            config["paths"]["artifact_dir"], f"evaluation_rejected_{version_tag}.json"
        )
        evaluator.save_results(path=rejected_eval_path)
        logger.warning(f"Model not promoted; kept champion artifacts. Saved: {rejected_eval_path}")

    elapsed = time.perf_counter() - start
    biz = eval_results["business_metrics"]
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Total time        : {elapsed:.2f}s")
    logger.info(f"  Algorithm         : {config['model']['algorithm'].upper()}")
    logger.info(f"  Features used     : {training_meta['n_features']} (19 raw + 22 engineered)")
    logger.info(f"  CV ROC-AUC        : {training_meta['cv_roc_auc_mean']:.4f}")
    logger.info(f"  Val ROC-AUC       : {training_meta['val_roc_auc']:.4f}")
    logger.info(f"  Test ROC-AUC      : {eval_results['roc_auc']:.4f}")
    logger.info(f"  Test PR-AUC       : {eval_results['pr_auc']:.4f}")
    logger.info(f"  Test F1           : {eval_results['f1']:.4f}")
    logger.info(f"  Optimal threshold : {eval_results['threshold']:.4f}")
    logger.info(f"  Net LTV benefit   : ${biz['net_business_benefit_usd']:,.2f}")
    logger.info(f"  Promoted active   : {promoted}")
    logger.info(f"  Model path        : {model_path}")
    logger.info("=" * 60)

    return {
        "model_path": model_path,
        "pipeline_path": versioned_pipeline_path,
        "promoted": promoted,
        "eval_results": eval_results,
        "training_metadata": training_meta,
    }


if __name__ == "__main__":
    try:
        run_pipeline()
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("Training pipeline interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Training pipeline failed: {exc}", exc_info=True)
        sys.exit(1)
