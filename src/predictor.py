"""
predictor.py — Thread-safe inference and real-time SHAP explainability engine for ChurnScope.

Loads the model artifact, paired leak-free preprocessing pipeline, and validation-tuned threshold
once per worker process, delivering ~2–5ms predictions and ~5–15ms TreeExplainer SHAP attributions.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src import get_logger
from src.data_loader import DataLoader
from src.evaluator import extract_positive_shap

logger = get_logger(__name__)

FEATURE_LABELS: dict[str, str] = {
    "Contract": "Contract Type",
    "tenure": "Tenure (Months)",
    "tenure_band": "Tenure Lifecycle Band",
    "log_tenure": "Log Tenure",
    "is_new_customer": "New Customer (≤6m)",
    "is_loyal_customer": "Loyal Customer (≥24m)",
    "MonthlyCharges": "Monthly Charges",
    "TotalCharges": "Total Charges",
    "log_monthly_charges": "Log Monthly Charges",
    "monthly_charge_per_tenure": "Charge / Tenure Ratio",
    "charge_discrepancy": "Billing Discrepancy",
    "charge_per_service": "Charge per Service",
    "is_high_value": "High Monthly Spend",
    "InternetService": "Internet Service",
    "has_internet": "Has Internet",
    "has_fiber": "Fiber Optic Internet",
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support",
    "protection_services_count": "Protection Add-Ons Count",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
    "PhoneService": "Phone Service",
    "MultipleLines": "Multiple Lines",
    "total_services": "Total Subscribed Services",
    "service_penetration": "Service Penetration",
    "stickiness_score": "Service Stickiness Score",
    "high_risk_flag": "New + High-Spend Flag",
    "is_month_to_month": "Month-to-Month Contract",
    "is_long_term_contract": "Two-Year Contract",
    "PaymentMethod": "Payment Method",
    "is_electronic_check": "Electronic Check Payment",
    "is_auto_payment": "Automatic Payment",
    "PaperlessBilling": "Paperless Billing",
    "is_paperless": "Paperless Billing Flag",
    "highest_churn_risk_profile": "M2M + E-Check Risk Combo",
    "SeniorCitizen": "Senior Citizen",
    "Partner": "Has Partner",
    "Dependents": "Has Dependents",
    "gender": "Gender",
}


class ChurnPredictor:
    """In-process singleton inference + SHAP attribution engine."""

    _instance: ChurnPredictor | None = None
    _lock = threading.Lock()

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.paths = config["paths"]
        self.threshold_cfg = config.get("threshold", {})
        self.data_loader = DataLoader(config)
        self.model: Any = None
        self.base_model: Any = None
        self.preprocessing_pipeline: Any = None
        self._shap_explainer: Any = None
        self.feature_names: list[str] = []
        self.optimal_threshold: float = float(self.threshold_cfg.get("default", 0.5))
        self.model_version: str = "unknown"
        self.model_path: str | None = None
        self.model_metadata: dict[str, Any] = {}
        self._is_loaded: bool = False

    @classmethod
    def get_instance(cls, config: dict[str, Any]) -> ChurnPredictor:
        """Return a thread-safe per-worker singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = cls(config)
                    instance.load()
                    cls._instance = instance
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Clear cached singleton (used by tests and hot reloads)."""
        with cls._lock:
            cls._instance = None

    def load(
        self, model_path: str | None = None, pipeline_path: str | None = None
    ) -> ChurnPredictor:
        """Load versioned model artifact, paired preprocessing pipeline, and optimal threshold."""
        resolved_model = model_path or self._get_latest_model_path()
        if not os.path.exists(resolved_model):
            raise FileNotFoundError(f"Model not found: {resolved_model}")

        artifact = joblib.load(resolved_model)
        self.model = artifact["model"]
        self.base_model = artifact.get("base_model") or self.model
        self.model_version = str(artifact.get("version", "v1"))
        self.model_metadata = dict(artifact.get("metadata", {}))
        self.model_path = resolved_model
        self._shap_explainer = None
        logger.info(f"Model loaded: {resolved_model} (version={self.model_version})")

        bundled_pipeline = artifact.get("pipeline_path")
        fallback_pipeline = os.path.join(
            self.paths["artifact_dir"], "preprocessing_pipeline.joblib"
        )
        if pipeline_path:
            resolved_pipeline = pipeline_path
        elif bundled_pipeline and os.path.exists(str(bundled_pipeline)):
            resolved_pipeline = str(bundled_pipeline)
        else:
            resolved_pipeline = fallback_pipeline

        if not os.path.exists(resolved_pipeline):
            raise FileNotFoundError(f"Preprocessing pipeline not found: {resolved_pipeline}")

        pipe_artifact = joblib.load(resolved_pipeline)
        self.preprocessing_pipeline = pipe_artifact["pipeline"]
        self.feature_names = list(pipe_artifact.get("feature_names", []))
        logger.info(
            f"Preprocessing pipeline loaded: {resolved_pipeline} "
            f"({len(self.feature_names)} features)"
        )

        # Prefer threshold bundled inside the model artifact for atomic rollback safety.
        bundled_threshold = artifact.get("threshold")
        if bundled_threshold is not None:
            self.optimal_threshold = float(bundled_threshold)
        else:
            eval_path = os.path.join(self.paths["artifact_dir"], "evaluation_results.json")
            if os.path.exists(eval_path):
                with open(eval_path, encoding="utf-8") as fh:
                    ev = json.load(fh)
                if ev.get("threshold") is not None:
                    self.optimal_threshold = float(ev["threshold"])

        logger.info(f"ChurnPredictor ready | Active threshold: {self.optimal_threshold:.4f}")
        self._is_loaded = True
        return self

    def predict(self, customer_data: dict[str, Any] | pd.DataFrame) -> dict[str, Any]:
        """Score a single customer record."""
        self._check_loaded()
        df = (
            pd.DataFrame([customer_data])
            if isinstance(customer_data, dict)
            else customer_data.iloc[:1].copy()
        )
        X = self._prepare_features(df)
        proba = float(self.model.predict_proba(X)[0, 1])
        pred = int(proba >= self.optimal_threshold)
        return self._format_output(proba, pred)

    def predict_batch(
        self,
        customers: list[dict[str, Any]] | pd.DataFrame,
        enforce_limit: bool = False,
        chunk_size: int = 5000,
    ) -> list[dict[str, Any]]:
        """Score multiple customers; enforces max_batch_size for API calls, chunks large CLI jobs."""
        self._check_loaded()
        df = pd.DataFrame(customers) if isinstance(customers, list) else customers.copy()
        if df.empty:
            raise ValueError("Customer batch cannot be empty.")

        max_batch = int(self.config.get("api", {}).get("max_batch_size", 1000))
        if enforce_limit and len(df) > max_batch:
            raise ValueError(f"Batch size {len(df)} exceeds maximum allowed ({max_batch}).")

        all_probas: list[np.ndarray] = []
        step = max(1, int(chunk_size))
        for start in range(0, len(df), step):
            chunk = df.iloc[start : start + step]
            X_chunk = self._prepare_features(chunk)
            all_probas.append(np.asarray(self.model.predict_proba(X_chunk)[:, 1], dtype=float))

        probas = np.concatenate(all_probas)
        preds = (probas >= self.optimal_threshold).astype(int)
        results = [
            self._format_output(float(p), int(pr)) for p, pr in zip(probas, preds, strict=True)
        ]

        churn_n = sum(r["churn_prediction"] for r in results)
        logger.info(
            f"Batch scored: {churn_n}/{len(results)} predicted churners "
            f"({churn_n / len(results):.1%})"
        )
        return results

    def explain(
        self, customer_data: dict[str, Any] | pd.DataFrame, top_k: int = 12
    ) -> dict[str, Any]:
        """Compute prediction plus genuine TreeExplainer per-customer SHAP attributions."""
        self._check_loaded()
        raw_dict = (
            dict(customer_data)
            if isinstance(customer_data, dict)
            else customer_data.iloc[0].to_dict()
        )
        df = pd.DataFrame([raw_dict])
        X = self._prepare_features(df)

        proba = float(self.model.predict_proba(X)[0, 1])
        pred = int(proba >= self.optimal_threshold)
        output = self._format_output(proba, pred)

        explainer = self._get_shap_explainer()
        sv_matrix = extract_positive_shap(explainer.shap_values(X))
        sv_row = sv_matrix[0]

        expected = explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            exp_arr = np.asarray(expected, dtype=float).ravel()
            base_value = float(exp_arr[1] if len(exp_arr) > 1 else exp_arr[0])
        else:
            base_value = float(expected)

        names = self.feature_names or [f"feature_{i}" for i in range(len(sv_row))]
        order = np.argsort(np.abs(sv_row))[::-1][: max(1, min(top_k, len(sv_row)))]

        contributions = [
            {
                "feature": names[idx],
                "display_name": FEATURE_LABELS.get(names[idx], names[idx]),
                "shap_value": round(float(sv_row[idx]), 4),
                "direction": "increases_risk" if sv_row[idx] >= 0 else "decreases_risk",
            }
            for idx in order
        ]

        output["base_value"] = round(base_value, 4)
        output["shap_values"] = contributions
        output["recommendations"] = self._generate_recommendations(raw_dict, contributions)
        return output

    def get_model_info(self) -> dict[str, Any]:
        """Return active model version, performance metadata, and operational threshold."""
        return {
            "model_version": self.model_version,
            "algorithm": self.model_metadata.get("algorithm", "unknown"),
            "is_calibrated": bool(self.model_metadata.get("is_calibrated", False)),
            "cv_roc_auc": self.model_metadata.get("cv_roc_auc_mean"),
            "val_roc_auc": self.model_metadata.get("val_roc_auc"),
            "n_features": self.model_metadata.get("n_features") or len(self.feature_names),
            "trained_at": self.model_metadata.get("trained_at"),
            "threshold": self.optimal_threshold,
            "is_loaded": self._is_loaded,
        }

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        cleaned = self.data_loader.prepare_dataframe(df, require_target=False)
        drop_cols = [self.data_loader.id_col, self.data_loader.target_col]
        fe_step = self.preprocessing_pipeline.named_steps.get("feature_engineer")
        if fe_step is not None:
            drop_cols.extend(c for c in fe_step.get_new_feature_names() if c in cleaned.columns)
        features_df = cleaned.drop(columns=drop_cols, errors="ignore")
        return self.preprocessing_pipeline.transform(features_df)

    def _get_shap_explainer(self) -> Any:
        if self._shap_explainer is None:
            import shap

            self._shap_explainer = shap.TreeExplainer(self.base_model)
        return self._shap_explainer

    def _get_latest_model_path(self) -> str:
        registry_path = self.paths["registry"]
        if os.path.exists(registry_path):
            with open(registry_path, encoding="utf-8") as fh:
                registry = json.load(fh)
            latest = registry.get("latest")
            if latest and os.path.exists(latest):
                return str(latest)

        model_dir = self.paths["model_dir"]
        if os.path.exists(model_dir):
            files = sorted(Path(model_dir).glob("*.joblib"), reverse=True)
            if files:
                return str(files[0])

        raise FileNotFoundError(
            f"No trained model found in registry ({registry_path}) or {model_dir}. "
            "Run `python train_pipeline.py` first."
        )

    def _format_output(self, proba: float, prediction: int) -> dict[str, Any]:
        thr = float(self.optimal_threshold)
        if proba >= thr:
            confidence = (proba - thr) / max(1.0 - thr, 1e-6)
        else:
            confidence = (thr - proba) / max(thr, 1e-6)

        return {
            "churn_probability": round(proba, 4),
            "churn_prediction": prediction,
            "churn_label": "Churn" if prediction == 1 else "No Churn",
            "risk_tier": self._get_risk_tier(proba, thr),
            "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 4),
            "threshold_used": round(thr, 4),
        }

    @staticmethod
    def _get_risk_tier(proba: float, threshold: float) -> str:
        """Assign a risk tier aligned with the active classification threshold."""
        if proba < threshold:
            return "Low"
        span = max(1.0 - threshold, 0.15)
        medium_cap = threshold + 0.30 * span
        high_cap = threshold + 0.65 * span
        if proba >= high_cap:
            return "Critical"
        if proba >= medium_cap:
            return "High"
        return "Medium"

    @staticmethod
    def _generate_recommendations(
        customer: dict[str, Any], shap_contributions: list[dict[str, Any]]
    ) -> list[str]:
        """Translate positive SHAP risk drivers and account profile into retention actions."""
        positive_features = {
            c["feature"] for c in shap_contributions if float(c["shap_value"]) > 0.0
        }
        recs: list[str] = []

        if customer.get("Contract") == "Month-to-month" and (
            {"Contract", "is_month_to_month", "highest_churn_risk_profile"} & positive_features
        ):
            recs.append(
                "Offer a 1-year contract upgrade with a loyalty discount — removes the top M2M churn driver."
            )
        if (
            customer.get("InternetService") == "Fiber optic"
            and customer.get("OnlineSecurity") == "No"
        ):
            recs.append(
                "Bundle complimentary Online Security for 3 months on Fiber optic to boost service stickiness."
            )
        if customer.get("TechSupport") == "No" and customer.get("InternetService") != "No":
            recs.append(
                "Provide priority Tech Support onboarding — reduces technical dissatisfaction churn."
            )
        if customer.get("PaymentMethod") == "Electronic check":
            recs.append(
                "Incentivize switch from Electronic check to Auto-Pay ($5/mo credit) to cut payment friction."
            )
        tenure_val = customer.get("tenure")
        if isinstance(tenure_val, (int, float)) and tenure_val <= 12:
            recs.append(
                "Enroll in first-year customer success check-in — early-tenure accounts face elevated risk."
            )
        monthly_val = customer.get("MonthlyCharges")
        if isinstance(monthly_val, (int, float)) and monthly_val >= 80:
            recs.append(
                "Run a proactive plan right-sizing review to improve perceived price-to-value ratio."
            )

        if not recs:
            recs.append("Maintain standard engagement — account displays strong retention signals.")

        return recs[:4]

    def _check_loaded(self) -> None:
        if not self._is_loaded or self.model is None or self.preprocessing_pipeline is None:
            raise RuntimeError("Predictor not loaded. Call load() first.")
