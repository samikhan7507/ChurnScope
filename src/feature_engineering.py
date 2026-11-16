"""
feature_engineering.py — Stateful, leak-free domain feature transformer for ChurnScope.

Implements scikit-learn's BaseEstimator + TransformerMixin contract so aggregate statistics
(such as the MonthlyCharges median used for `is_high_value`) are learned ONLY on the training
split during `fit()` and applied identically to validation, test, batch, and single-customer
API requests during `transform()`.

Creates 22 distinct engineered features across 5 domain groups:
  1. Tenure lifecycle (4)    : tenure_band, is_new_customer, is_loyal_customer, log_tenure
  2. Charge ratios (4)       : monthly_charge_per_tenure, charge_discrepancy,
                               is_high_value, log_monthly_charges
  3. Service adoption (5)    : total_services, service_penetration, has_internet,
                               has_fiber, protection_services_count
  4. Interactions (3)        : charge_per_service, stickiness_score, high_risk_flag
  5. Contract & billing (6)  : is_month_to_month, is_long_term_contract, is_electronic_check,
                               is_auto_payment, is_paperless, highest_churn_risk_profile
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src import get_logger

logger = get_logger(__name__)

DEFAULT_MONTHLY_CHARGES_MEDIAN = 70.35

YES_NO_SERVICE_COLS = (
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)

PROTECTION_COLS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain-specific telecom feature generator compatible with sklearn Pipelines."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config
        self.monthly_charges_median_: float = DEFAULT_MONTHLY_CHARGES_MEDIAN
        self.is_fitted_: bool = False

    @property
    def _fe_cfg(self) -> dict[str, Any]:
        if isinstance(self.config, dict):
            return self.config.get("feature_engineering", {})
        return {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> FeatureEngineer:
        """Fit training-split reference statistics (e.g., MonthlyCharges median)."""
        if "MonthlyCharges" in X.columns:
            mc = pd.to_numeric(X["MonthlyCharges"], errors="coerce").dropna()
            if len(mc) > 0:
                self.monthly_charges_median_ = float(mc.median())
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate engineered features using stored training statistics (no data leakage)."""
        df = X.copy()
        fe_cfg = self._fe_cfg

        if fe_cfg.get("create_tenure_bands", True):
            df = self._tenure_features(df)
        if fe_cfg.get("create_charge_ratios", True):
            df = self._charge_ratio_features(df)
        if fe_cfg.get("create_service_count", True):
            df = self._service_count_features(df)
        if fe_cfg.get("create_interaction_features", True):
            df = self._interaction_features(df)
        if fe_cfg.get("create_contract_flags", True):
            df = self._contract_risk_features(df)

        return df

    def _tenure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "tenure" not in df.columns:
            return df
        tenure = pd.to_numeric(df["tenure"], errors="coerce")
        valid = tenure.fillna(0).clip(lower=0)
        df["tenure_band"] = np.select(
            [valid <= 6, valid <= 12, valid <= 24, valid <= 48],
            [0, 1, 2, 3],
            default=4,
        ).astype(int)
        df["is_new_customer"] = (valid <= 6).astype(int)
        df["is_loyal_customer"] = (valid >= 24).astype(int)
        df["log_tenure"] = np.log1p(valid)
        return df

    def _charge_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "MonthlyCharges" not in df.columns:
            return df
        monthly = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        median_ref = float(getattr(self, "monthly_charges_median_", DEFAULT_MONTHLY_CHARGES_MEDIAN))

        if "tenure" in df.columns:
            tenure = pd.to_numeric(df["tenure"], errors="coerce").clip(lower=0)
            df["monthly_charge_per_tenure"] = monthly / (tenure + 1.0)
            if "TotalCharges" in df.columns:
                total = pd.to_numeric(df["TotalCharges"], errors="coerce")
                expected = monthly * tenure
                df["charge_discrepancy"] = (total - expected) / (expected + 1.0)

        df["is_high_value"] = (monthly > median_ref).fillna(False).astype(int)
        df["log_monthly_charges"] = np.log1p(monthly.clip(lower=0))
        return df

    def _service_count_features(self, df: pd.DataFrame) -> pd.DataFrame:
        service_flags: list[pd.Series] = []

        for col in YES_NO_SERVICE_COLS:
            if col in df.columns:
                service_flags.append((df[col] == "Yes").astype(int))

        if "InternetService" in df.columns:
            has_internet = (df["InternetService"].notna() & (df["InternetService"] != "No")).astype(
                int
            )
            has_fiber = (df["InternetService"] == "Fiber optic").astype(int)
            service_flags.append(has_internet)
            df["has_internet"] = has_internet
            df["has_fiber"] = has_fiber

        if not service_flags:
            return df

        total_services = sum(service_flags)
        df["total_services"] = total_services.astype(int)
        df["service_penetration"] = df["total_services"] / float(len(service_flags))

        avail_prot = [col for col in PROTECTION_COLS if col in df.columns]
        if avail_prot:
            df["protection_services_count"] = sum(
                (df[col] == "Yes").astype(int) for col in avail_prot
            ).astype(int)

        return df

    def _interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "MonthlyCharges" in df.columns and "total_services" in df.columns:
            monthly = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
            df["charge_per_service"] = monthly / (df["total_services"] + 1.0)
        if "total_services" in df.columns and "tenure" in df.columns:
            tenure = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).clip(lower=0)
            df["stickiness_score"] = df["total_services"] * np.log1p(tenure)
        if "is_new_customer" in df.columns and "is_high_value" in df.columns:
            df["high_risk_flag"] = (
                (df["is_new_customer"] == 1) & (df["is_high_value"] == 1)
            ).astype(int)
        return df

    def _contract_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Contract" in df.columns:
            df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
            df["is_long_term_contract"] = (df["Contract"] == "Two year").astype(int)
        if "PaymentMethod" in df.columns:
            df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
            df["is_auto_payment"] = (
                df["PaymentMethod"]
                .astype(str)
                .str.contains("automatic", case=False, na=False)
                .astype(int)
            )
        if "PaperlessBilling" in df.columns:
            df["is_paperless"] = (df["PaperlessBilling"] == "Yes").astype(int)
        if "is_month_to_month" in df.columns and "is_electronic_check" in df.columns:
            df["highest_churn_risk_profile"] = (
                (df["is_month_to_month"] == 1) & (df["is_electronic_check"] == 1)
            ).astype(int)
        return df

    def get_new_feature_names(self) -> list[str]:
        """Return the ordered list of engineered features enabled by the active config."""
        fe_cfg = self._fe_cfg
        names: list[str] = []
        if fe_cfg.get("create_tenure_bands", True):
            names.extend(["tenure_band", "is_new_customer", "is_loyal_customer", "log_tenure"])
        if fe_cfg.get("create_charge_ratios", True):
            names.extend(
                [
                    "monthly_charge_per_tenure",
                    "charge_discrepancy",
                    "is_high_value",
                    "log_monthly_charges",
                ]
            )
        if fe_cfg.get("create_service_count", True):
            names.extend(
                [
                    "total_services",
                    "service_penetration",
                    "has_internet",
                    "has_fiber",
                    "protection_services_count",
                ]
            )
        if fe_cfg.get("create_interaction_features", True):
            names.extend(["charge_per_service", "stickiness_score", "high_risk_flag"])
        if fe_cfg.get("create_contract_flags", True):
            names.extend(
                [
                    "is_month_to_month",
                    "is_long_term_contract",
                    "is_electronic_check",
                    "is_auto_payment",
                    "is_paperless",
                    "highest_churn_risk_profile",
                ]
            )
        return names
