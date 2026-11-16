"""
preprocessor.py — Leak-free split + feature engineering + preprocessing pipeline.

Architecture:
  1. Split raw dataset into train / val / test FIRST (stratified).
  2. Fit a unified sklearn Pipeline([feature_engineer, preprocessor]) on X_train ONLY.
  3. Transform X_train, X_val, and X_test using all 19 raw + 22 engineered features (= 41).
  4. Persist the fitted pipeline + feature names + training reference distributions via joblib.
"""

from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src import get_logger
from src.feature_engineering import FeatureEngineer

logger = get_logger(__name__)


class ChurnPreprocessor:
    """Manages leak-free train/val/test splitting and end-to-end feature preprocessing."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.data_cfg = config["data"]
        self.paths = config["paths"]
        self.target_col: str = self.data_cfg["target_column"]
        self.id_col: str = self.data_cfg["customer_id_column"]
        self.numeric_feats: list[str] = list(self.data_cfg["numeric_features"])
        self.categorical_feats: list[str] = list(self.data_cfg["categorical_features"])
        self.test_size: float = float(self.data_cfg.get("test_size", 0.2))
        self.val_size: float = float(self.data_cfg.get("val_size", 0.1))
        self.random_state: int = int(self.data_cfg.get("random_state", 42))
        self.stratify: bool = bool(self.data_cfg.get("stratify", True))
        self.pipeline: Pipeline | None = None
        self.feature_names: list[str] = []
        self.reference_stats: dict[str, Any] = {}

    def fit_transform(self, df: pd.DataFrame) -> dict[str, Any]:
        """Split data without leakage, fit unified pipeline on train, and return splits."""
        logger.info("Starting leak-free preprocessing pipeline...")

        # Keep only raw columns before splitting so pre-computed features never leak stats.
        fe_probe = FeatureEngineer(self.config)
        preexisting_eng = [c for c in fe_probe.get_new_feature_names() if c in df.columns]
        clean_df = df.drop(columns=[self.id_col, *preexisting_eng], errors="ignore")

        if self.target_col not in clean_df.columns:
            raise ValueError(f"Target column {self.target_col!r} not found in DataFrame.")

        X = clean_df.drop(columns=[self.target_col])
        y = clean_df[self.target_col].astype(int)

        holdout_size = self.test_size + self.val_size
        strat_full = y if self.stratify else None
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=holdout_size,
            random_state=self.random_state,
            stratify=strat_full,
        )

        val_frac = self.val_size / holdout_size
        strat_temp = y_temp if self.stratify else None
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=1.0 - val_frac,
            random_state=self.random_state,
            stratify=strat_temp,
        )

        logger.info(
            f"Data split complete | Train: {len(X_train):,} | "
            f"Val: {len(X_val):,} | Test: {len(X_test):,}"
        )

        # Inspect columns created on X_train to register all raw + engineered features.
        sample_enriched = FeatureEngineer(self.config).fit_transform(X_train)
        self.pipeline = self._build_pipeline(available_columns=list(sample_enriched.columns))

        X_train_p = self.pipeline.fit_transform(X_train)
        X_val_p = self.pipeline.transform(X_val)
        X_test_p = self.pipeline.transform(X_test)
        self.feature_names = self._get_feature_names()
        self.reference_stats = self._compute_reference_stats(X_train)

        logger.info(
            f"Transformed feature matrix: {len(self.feature_names)} features "
            f"| Train churn rate: {y_train.mean():.2%}"
        )

        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        pos_weight = float(n_neg / max(n_pos, 1))
        class_weights = {0: 1.0, 1: pos_weight}

        val_monthly = (
            pd.to_numeric(X_val["MonthlyCharges"], errors="coerce").fillna(65.0).to_numpy()
            if "MonthlyCharges" in X_val.columns
            else None
        )
        test_monthly = (
            pd.to_numeric(X_test["MonthlyCharges"], errors="coerce").fillna(65.0).to_numpy()
            if "MonthlyCharges" in X_test.columns
            else None
        )

        return {
            "X_train": X_train_p,
            "X_val": X_val_p,
            "X_test": X_test_p,
            "y_train": y_train.to_numpy(),
            "y_val": y_val.to_numpy(),
            "y_test": y_test.to_numpy(),
            "val_monthly_charges": val_monthly,
            "test_monthly_charges": test_monthly,
            "X_train_raw": X_train,
            "X_val_raw": X_val,
            "X_test_raw": X_test,
            "feature_names": self.feature_names,
            "class_weights": class_weights,
        }

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new customer records using the already-fitted unified pipeline."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not fitted. Call fit_transform() or load() first.")
        fe_step = self.pipeline.named_steps.get("feature_engineer")
        drop_cols = [self.id_col, self.target_col]
        if fe_step is not None:
            drop_cols.extend(c for c in fe_step.get_new_feature_names() if c in df.columns)
        clean = df.drop(columns=drop_cols, errors="ignore")
        return self.pipeline.transform(clean)

    def save(
        self,
        path: str | None = None,
        versioned_tag: str | None = None,
        update_canonical: bool = True,
    ) -> str:
        """Save the fitted pipeline artifact (plus optional versioned copy for rollback)."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not fitted. Call fit_transform() before save().")
        artifact_dir = self.paths["artifact_dir"]
        os.makedirs(artifact_dir, exist_ok=True)
        payload = {
            "pipeline": self.pipeline,
            "feature_names": self.feature_names,
            "reference_stats": self.reference_stats,
        }

        saved_path: str | None = None
        if versioned_tag:
            versioned_path = os.path.join(
                artifact_dir, f"preprocessing_pipeline_{versioned_tag}.joblib"
            )
            joblib.dump(payload, versioned_path)
            logger.info(f"Versioned pipeline saved: {versioned_path}")
            saved_path = versioned_path

        if update_canonical or not versioned_tag:
            canonical_path = path or os.path.join(artifact_dir, "preprocessing_pipeline.joblib")
            os.makedirs(os.path.dirname(canonical_path) or ".", exist_ok=True)
            joblib.dump(payload, canonical_path)
            logger.info(f"Preprocessing pipeline saved: {canonical_path}")
            if saved_path is None:
                saved_path = canonical_path

        return str(saved_path)

    def load(self, path: str | None = None) -> ChurnPreprocessor:
        """Load a previously saved preprocessing pipeline."""
        load_path = path or os.path.join(
            self.paths["artifact_dir"], "preprocessing_pipeline.joblib"
        )
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Preprocessing pipeline not found: {load_path}")
        artifact = joblib.load(load_path)
        self.pipeline = artifact["pipeline"]
        self.feature_names = list(artifact.get("feature_names", []))
        self.reference_stats = dict(artifact.get("reference_stats", {}))
        logger.info(
            f"Preprocessing pipeline loaded: {load_path} ({len(self.feature_names)} features)"
        )
        return self

    def save_processed_splits(self, splits: dict[str, Any]) -> None:
        """Persist processed feature matrices and labels to config paths.processed_data."""
        proc_dir = self.paths.get("processed_data", "data/processed/")
        os.makedirs(proc_dir, exist_ok=True)
        feat_names = splits["feature_names"]
        for split_name in ("train", "val", "test"):
            split_df = pd.DataFrame(splits[f"X_{split_name}"], columns=feat_names)
            split_df[self.target_col] = splits[f"y_{split_name}"]
            split_df.to_parquet(os.path.join(proc_dir, f"{split_name}.parquet"), index=False)
        logger.info(f"Processed train/val/test splits saved to: {proc_dir}")

    def _build_pipeline(self, available_columns: list[str]) -> Pipeline:
        fe = FeatureEngineer(self.config)
        eng_numeric = [c for c in fe.get_new_feature_names() if c in available_columns]
        raw_numeric = [c for c in self.numeric_feats if c in available_columns]
        all_numeric = raw_numeric + [c for c in eng_numeric if c not in raw_numeric]
        all_categorical = [c for c in self.categorical_feats if c in available_columns]

        num_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        cat_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
            ]
        )
        col_transformer = ColumnTransformer(
            transformers=[
                ("num", num_transformer, all_numeric),
                ("cat", cat_transformer, all_categorical),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        logger.info(
            f"Pipeline configured: {len(raw_numeric)} raw numeric + "
            f"{len(eng_numeric)} engineered + {len(all_categorical)} categorical "
            f"= {len(all_numeric) + len(all_categorical)} total features"
        )
        return Pipeline(
            [
                ("feature_engineer", fe),
                ("preprocessor", col_transformer),
            ]
        )

    def _get_feature_names(self) -> list[str]:
        if self.pipeline is None:
            return []
        preprocessor = self.pipeline.named_steps["preprocessor"]
        return [str(name) for name in preprocessor.get_feature_names_out()]

    def _compute_reference_stats(self, X_train: pd.DataFrame) -> dict[str, Any]:
        """Record training-split summary distributions for baseline monitoring."""
        stats: dict[str, Any] = {"n_train": int(len(X_train)), "numeric": {}, "categorical": {}}
        for col in self.numeric_feats:
            if col in X_train.columns:
                series = pd.to_numeric(X_train[col], errors="coerce").dropna()
                if len(series) > 0:
                    stats["numeric"][col] = {
                        "mean": round(float(series.mean()), 4),
                        "std": round(float(series.std()), 4),
                        "median": round(float(series.median()), 4),
                        "p10": round(float(series.quantile(0.10)), 4),
                        "p90": round(float(series.quantile(0.90)), 4),
                    }
        for col in self.categorical_feats:
            if col in X_train.columns:
                dist = X_train[col].value_counts(normalize=True).round(4).to_dict()
                stats["categorical"][col] = {str(k): float(v) for k, v in dist.items()}
        return stats
