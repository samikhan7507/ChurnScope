"""
trainer.py — Model training, cross-validation, Optuna HPO, calibration, and registry management.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src import get_logger

logger = get_logger(__name__)


def _calibrate_prefit(estimator: Any, X_val: np.ndarray, y_val: np.ndarray, method: str) -> Any:
    """Calibrate a pre-fitted classifier across scikit-learn 1.4–1.9+."""
    try:
        from sklearn.frozen import FrozenEstimator

        calibrator = CalibratedClassifierCV(FrozenEstimator(estimator), method=method)
    except ImportError:
        calibrator = CalibratedClassifierCV(estimator, cv="prefit", method=method)
    calibrator.fit(X_val, y_val)
    return calibrator


class ChurnTrainer:
    """Trains LightGBM or XGBoost churn models with CV, HPO, calibration, and versioning."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model_cfg = config["model"]
        self.paths = config["paths"]
        self.random_state: int = int(config.get("data", {}).get("random_state", 42))
        self.algorithm: str = str(self.model_cfg.get("algorithm", "lightgbm")).lower()
        if self.algorithm not in {"lightgbm", "xgboost"}:
            raise ValueError(
                f"Unsupported algorithm: {self.algorithm!r}. Use 'lightgbm' or 'xgboost'."
            )
        self.base_model: Any = None
        self.model: Any = None
        self.is_calibrated: bool = False
        self.training_metadata: dict[str, Any] = {}

    def train(self, splits: dict[str, Any]) -> dict[str, Any]:
        """Run optional HPO, stratified K-fold CV, early-stopped fit, and optional calibration."""
        X_train, y_train = splits["X_train"], splits["y_train"]
        X_val, y_val = splits["X_val"], splits["y_val"]
        feature_names = list(splits.get("feature_names", []))

        logger.info(
            f"Training {self.algorithm.upper()} | Train: {len(X_train):,} | "
            f"Val: {len(X_val):,} | Features: {X_train.shape[1]}"
        )
        start = time.perf_counter()
        run_id = uuid.uuid4().hex[:8]

        optuna_enabled = bool(self.config.get("optuna", {}).get("enabled", False))
        params = (
            self._run_optuna(X_train, y_train)
            if optuna_enabled
            else self._get_default_params(splits.get("class_weights"))
        )

        cv_scores = self._cross_validate(X_train, y_train, params)
        self.base_model = self._build_model(params, for_early_stopping=True)
        self._fit_model(self.base_model, X_train, y_train, X_val, y_val)

        cal_cfg = self.model_cfg.get("calibration", {})
        if cal_cfg.get("enabled", False) and len(np.unique(y_val)) == 2 and len(y_val) >= 30:
            method = str(cal_cfg.get("method", "sigmoid"))
            self.model = _calibrate_prefit(self.base_model, X_val, y_val, method=method)
            self.is_calibrated = True
            logger.info(f"Probability calibration applied on validation set (method={method!r})")
        else:
            self.model = self.base_model
            self.is_calibrated = False

        val_scores = self.predict_proba(X_val)
        val_roc_auc = float(roc_auc_score(y_val, val_scores))
        elapsed = time.perf_counter() - start

        self.training_metadata = {
            "run_id": run_id,
            "algorithm": self.algorithm,
            "params": params,
            "is_calibrated": self.is_calibrated,
            "cv_roc_auc_mean": float(cv_scores.mean()),
            "cv_roc_auc_std": float(cv_scores.std()),
            "cv_roc_auc_scores": [float(s) for s in cv_scores],
            "val_roc_auc": val_roc_auc,
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
            "test_samples": int(len(splits.get("y_test", []))),
            "n_features": int(X_train.shape[1]),
            "feature_names": feature_names,
            "training_time_seconds": round(elapsed, 2),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(
            f"Training finished in {elapsed:.2f}s | "
            f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} | "
            f"Val ROC-AUC: {val_roc_auc:.4f}"
        )
        return self.training_metadata

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict positive-class (churn) probabilities."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded. Call train() first.")
        return np.asarray(self.model.predict_proba(X)[:, 1], dtype=float)

    def save(
        self,
        version: str | None = None,
        timestamp: str | None = None,
        pipeline_path: str | None = None,
        threshold: float | None = None,
        eval_summary: dict[str, Any] | None = None,
        force_promote: bool = False,
    ) -> tuple[str, bool]:
        """Save versioned model bundle and promote to active if it passes the champion gate."""
        if self.model is None:
            raise RuntimeError("No model to save. Call train() first.")

        version = version or str(self.model_cfg.get("version", "v1"))
        ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"churn_model_{self.algorithm}_{version}_{ts}.joblib"

        model_dir = self.paths["model_dir"]
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, filename)

        artifact = {
            "model": self.model,
            "base_model": self.base_model,
            "is_calibrated": self.is_calibrated,
            "algorithm": self.algorithm,
            "version": version,
            "threshold": threshold,
            "pipeline_path": pipeline_path,
            "metadata": self.training_metadata,
            "eval_summary": eval_summary or {},
        }
        joblib.dump(artifact, save_path)
        logger.info(f"Model artifact saved: {save_path}")

        promoted = self._update_registry(
            model_path=save_path,
            version=version,
            pipeline_path=pipeline_path,
            threshold=threshold,
            eval_summary=eval_summary,
            force_promote=force_promote,
        )
        return save_path, promoted

    def _get_default_params(self, class_weights: dict[int, float] | None = None) -> dict[str, Any]:
        key = f"{self.algorithm}_params"
        params = dict(self.model_cfg.get(key, {}))
        params.pop("use_label_encoder", None)
        params.setdefault("random_state", self.random_state)
        params.setdefault("n_jobs", -1)
        if (
            self.algorithm == "xgboost"
            and "scale_pos_weight" not in params
            and class_weights
            and 1 in class_weights
        ):
            params["scale_pos_weight"] = round(float(class_weights[1]), 4)
        return params

    def _build_model(self, params: dict[str, Any], for_early_stopping: bool = False) -> Any:
        es_cfg = self.model_cfg.get("early_stopping", {})
        use_es = for_early_stopping and bool(es_cfg.get("enabled", True))
        es_rounds = int(es_cfg.get("rounds", 50))

        if self.algorithm == "lightgbm":
            import lightgbm as lgb

            clean = {k: v for k, v in params.items() if k != "verbose"}
            return lgb.LGBMClassifier(**clean, verbose=-1)

        if self.algorithm == "xgboost":
            from xgboost import XGBClassifier

            clean = {
                k: v
                for k, v in params.items()
                if k
                not in {"class_weight", "use_label_encoder", "verbosity", "early_stopping_rounds"}
            }
            clean.setdefault("eval_metric", "logloss")
            if use_es:
                clean["early_stopping_rounds"] = es_rounds
            return XGBClassifier(**clean, verbosity=0)

        raise ValueError(f"Unsupported algorithm: {self.algorithm!r}")

    def _fit_model(
        self,
        estimator: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        es_cfg = self.model_cfg.get("early_stopping", {})
        use_es = bool(es_cfg.get("enabled", True)) and len(X_val) > 0
        rounds = int(es_cfg.get("rounds", 50))

        if self.algorithm == "lightgbm" and use_es:
            import lightgbm as lgb

            estimator.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(rounds, verbose=False), lgb.log_evaluation(-1)],
            )
        elif self.algorithm == "xgboost" and use_es:
            estimator.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            estimator.fit(X_train, y_train)

    def _cross_validate(self, X: np.ndarray, y: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        cv_cfg = self.model_cfg.get("cross_validation", {})
        n_splits = int(cv_cfg.get("n_splits", 5))
        scoring = str(cv_cfg.get("scoring", "roc_auc"))
        logger.info(f"Running {n_splits}-fold stratified CV (scoring={scoring!r})...")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        estimator = self._build_model(params, for_early_stopping=False)
        # Keep outer loop sequential so LightGBM/XGBoost OpenMP threads never oversubscribe CPUs.
        scores = cross_val_score(estimator, X, y, cv=skf, scoring=scoring, n_jobs=1)
        for idx, score in enumerate(scores, start=1):
            logger.info(f"  Fold {idx}: {score:.4f}")
        return scores

    def _run_optuna(self, X_train: np.ndarray, y_train: np.ndarray) -> dict[str, Any]:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        cfg = self.config.get("optuna", {})
        metric = str(cfg.get("metric", "roc_auc"))
        direction = str(cfg.get("direction", "maximize"))
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)

        def objective(trial: optuna.Trial) -> float:
            if self.algorithm == "lightgbm":
                trial_params: dict[str, Any] = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                    "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "subsample_freq": 1,
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
                    "class_weight": "balanced",
                    "random_state": self.random_state,
                    "n_jobs": 1,
                }
            else:
                n_neg = int((y_train == 0).sum())
                n_pos = max(int((y_train == 1).sum()), 1)
                trial_params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
                    "scale_pos_weight": round(n_neg / n_pos, 4),
                    "eval_metric": "logloss",
                    "random_state": self.random_state,
                    "n_jobs": 1,
                }
            est = self._build_model(trial_params, for_early_stopping=False)
            return float(
                cross_val_score(est, X_train, y_train, cv=skf, scoring=metric, n_jobs=1).mean()
            )

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(
            objective,
            n_trials=int(cfg.get("n_trials", 40)),
            timeout=int(cfg.get("timeout", 600)),
        )
        best = dict(study.best_params)
        best["random_state"] = self.random_state
        best["n_jobs"] = -1
        if self.algorithm == "lightgbm":
            best["class_weight"] = "balanced"
            best["subsample_freq"] = 1
        else:
            best["eval_metric"] = "logloss"
            best["scale_pos_weight"] = round(
                int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1), 4
            )
        logger.info(f"Optuna best {metric}: {study.best_value:.4f}")
        return best

    def _should_promote(
        self, registry: dict[str, Any], candidate_auc: float | None, force_promote: bool
    ) -> bool:
        """Champion-challenger promotion gate to block degraded models from becoming active."""
        if force_promote:
            return True
        gate_cfg = self.config.get("promotion_gate", {})
        if not gate_cfg.get("enabled", True) or candidate_auc is None:
            return True

        min_auc = float(gate_cfg.get("min_roc_auc", 0.70))
        max_drop = float(gate_cfg.get("max_roc_auc_drop", 0.015))

        if candidate_auc < min_auc:
            logger.warning(
                f"Promotion blocked: candidate ROC-AUC ({candidate_auc:.4f}) < minimum ({min_auc:.4f})"
            )
            return False

        # Only compare against an active champion whose binary actually exists on disk.
        active_models = [
            m
            for m in registry.get("models", [])
            if m.get("status") == "active" and os.path.exists(str(m.get("model_path", "")))
        ]
        if active_models:
            champion = active_models[-1]
            champ_auc = champion.get("val_roc_auc") or champion.get("cv_roc_auc")
            if champ_auc is not None and candidate_auc < (float(champ_auc) - max_drop):
                logger.warning(
                    f"Promotion blocked: candidate ROC-AUC ({candidate_auc:.4f}) dropped more than "
                    f"{max_drop:.4f} below active champion ({float(champ_auc):.4f})"
                )
                return False

        return True

    def _update_registry(
        self,
        model_path: str,
        version: str,
        pipeline_path: str | None = None,
        threshold: float | None = None,
        eval_summary: dict[str, Any] | None = None,
        force_promote: bool = False,
    ) -> bool:
        registry_path = self.paths["registry"]
        os.makedirs(os.path.dirname(registry_path) or ".", exist_ok=True)
        registry: dict[str, Any] = {}
        if os.path.exists(registry_path):
            with open(registry_path, encoding="utf-8") as fh:
                registry = json.load(fh)

        candidate_auc = self.training_metadata.get("val_roc_auc") or self.training_metadata.get(
            "cv_roc_auc_mean"
        )
        promoted = self._should_promote(registry, candidate_auc, force_promote=force_promote)

        models_list: list[dict[str, Any]] = registry.setdefault("models", [])
        if promoted:
            for existing in models_list:
                if existing.get("status") == "active":
                    existing["status"] = "archived"

        entry: dict[str, Any] = {
            "run_id": self.training_metadata.get("run_id"),
            "model_path": model_path,
            "pipeline_path": pipeline_path,
            "version": version,
            "algorithm": self.algorithm,
            "is_calibrated": self.is_calibrated,
            "cv_roc_auc": self.training_metadata.get("cv_roc_auc_mean"),
            "cv_roc_auc_std": self.training_metadata.get("cv_roc_auc_std"),
            "val_roc_auc": self.training_metadata.get("val_roc_auc"),
            "test_roc_auc": (eval_summary or {}).get("roc_auc"),
            "test_pr_auc": (eval_summary or {}).get("pr_auc"),
            "test_f1": (eval_summary or {}).get("f1"),
            "threshold": threshold,
            "n_features": self.training_metadata.get("n_features"),
            "train_samples": self.training_metadata.get("train_samples"),
            "val_samples": self.training_metadata.get("val_samples"),
            "test_samples": self.training_metadata.get("test_samples"),
            "trained_at": self.training_metadata.get("trained_at"),
            "status": "active" if promoted else "rejected",
        }
        models_list.append(entry)

        if promoted:
            registry["latest"] = model_path
            registry["latest_pipeline"] = pipeline_path
            registry["latest_threshold"] = threshold

        with open(registry_path, "w", encoding="utf-8") as fh:
            json.dump(registry, fh, indent=2)
        logger.info(f"Model registry updated: {registry_path} (promoted={promoted})")
        return promoted
