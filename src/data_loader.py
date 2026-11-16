"""
data_loader.py — Multi-source data ingestion, type coercion, and schema validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import get_logger, load_config

logger = get_logger(__name__)

TARGET_MAP: dict[Any, int] = {
    "Yes": 1,
    "No": 0,
    "yes": 1,
    "no": 0,
    "1": 1,
    "0": 0,
    1: 1,
    0: 0,
}


class DataLoader:
    """Ingests customer churn tables from disk and enforces data contracts."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.data_cfg = config["data"]
        self.paths = config["paths"]
        self.target_col: str = self.data_cfg["target_column"]
        self.id_col: str = self.data_cfg["customer_id_column"]

    def load(self, source: str | None = None, require_target: bool = True) -> pd.DataFrame:
        """Load dataset from CSV/Parquet/JSON/Excel, coerce dtypes, and validate schema."""
        resolved = source or self.paths["raw_data"]
        logger.info(f"Loading data from: {resolved}")
        df = self._read_source(resolved)
        df = self.prepare_dataframe(df, require_target=require_target)
        self._log_data_quality(df)
        logger.info(f"Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]} cols")
        return df

    def prepare_dataframe(self, df: pd.DataFrame, require_target: bool = False) -> pd.DataFrame:
        """Apply consistent type coercion and schema checks for both training and inference."""
        if df.empty:
            raise ValueError("Input dataset is empty (0 rows).")
        cleaned = self._coerce_types(df, require_target=require_target)
        self._validate_schema(cleaned, require_target=require_target)
        return cleaned

    def _read_source(self, source: str) -> pd.DataFrame:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Data source not found: {source}")
        ext = path.suffix.lower()
        if ext == ".csv":
            return pd.read_csv(path)
        if ext in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if ext == ".json":
            return pd.read_json(path)
        if ext in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        raise ValueError(f"Unsupported data format: {ext!r}")

    def _coerce_types(self, df: pd.DataFrame, require_target: bool = False) -> pd.DataFrame:
        out = df.copy()

        # Strip surrounding whitespace on categorical columns (works on pandas 2.x and 3.x).
        for col in self.data_cfg.get("categorical_features", []):
            if col in out.columns and col != "SeniorCitizen":
                mask = out[col].notna()
                out.loc[mask, col] = out.loc[mask, col].astype(str).str.strip()

        if "tenure" in out.columns:
            out["tenure"] = pd.to_numeric(out["tenure"], errors="coerce")

        if "MonthlyCharges" in out.columns:
            out["MonthlyCharges"] = pd.to_numeric(out["MonthlyCharges"], errors="coerce")

        if "TotalCharges" in out.columns:
            out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
            # In the IBM Telco dataset, brand-new customers (tenure == 0) have blank TotalCharges.
            if "tenure" in out.columns:
                zero_tenure_blank = (out["tenure"] == 0) & out["TotalCharges"].isna()
                if zero_tenure_blank.any():
                    out.loc[zero_tenure_blank, "TotalCharges"] = 0.0

        if "SeniorCitizen" in out.columns:
            numeric_sc = pd.to_numeric(out["SeniorCitizen"], errors="coerce")
            normalized = pd.Series(index=out.index, dtype="object")
            valid_sc = numeric_sc.notna()
            normalized.loc[valid_sc] = numeric_sc.loc[valid_sc].astype(int).astype(str)
            normalized.loc[~valid_sc] = np.nan
            out["SeniorCitizen"] = normalized

        if self.target_col in out.columns:
            raw_target = out[self.target_col]
            if pd.api.types.is_string_dtype(raw_target) or raw_target.dtype == object:
                stripped = raw_target.astype(str).str.strip()
                mapped = stripped.map(TARGET_MAP)
            else:
                mapped = raw_target.map(TARGET_MAP)

            if mapped.isna().any():
                bad_vals = raw_target[mapped.isna()].unique().tolist()[:5]
                raise ValueError(
                    f"Unexpected values in target column {self.target_col!r}: {bad_vals}"
                )
            out[self.target_col] = mapped.astype(int)
        elif require_target:
            raise ValueError(f"Missing required target column: {self.target_col!r}")

        return out

    def _validate_schema(self, df: pd.DataFrame, require_target: bool = False) -> None:
        required = list(self.data_cfg.get("numeric_features", [])) + list(
            self.data_cfg.get("categorical_features", [])
        )
        if require_target:
            required.append(self.target_col)
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        for num_col in self.data_cfg.get("numeric_features", []):
            non_null = df[num_col].dropna()
            if (non_null < 0).any():
                raise ValueError(f"Numeric column {num_col!r} contains negative values.")

    def _log_data_quality(self, df: pd.DataFrame) -> None:
        if self.id_col in df.columns:
            dup_count = int(df[self.id_col].duplicated().sum())
            if dup_count > 0:
                logger.warning(f"Duplicate {self.id_col!r} values detected: {dup_count}")

        nulls = df.isnull().sum()
        for col, n_null in nulls[nulls > 0].items():
            logger.warning(f"Nulls in {col!r}: {n_null} ({n_null / len(df) * 100:.2f}%)")

        if self.target_col in df.columns:
            logger.info(f"Target churn rate: {df[self.target_col].mean():.2%}")

    @staticmethod
    def generate_synthetic_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
        """Generate domain-consistent telecom customer records matching the IBM Telco schema."""
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")

        rng = np.random.default_rng(random_state)

        def yn(prob_yes: float) -> np.ndarray:
            return rng.choice(["Yes", "No"], size=n_samples, p=[prob_yes, 1.0 - prob_yes])

        gender = rng.choice(["Male", "Female"], size=n_samples)
        senior = rng.choice(["0", "1"], size=n_samples, p=[0.84, 0.16])
        partner = yn(0.48)
        dependents = np.where(partner == "Yes", yn(0.46), yn(0.15))
        contract = rng.choice(
            ["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.21, 0.24]
        )

        # Tenure correlated with contract length.
        tenure = np.where(
            contract == "Month-to-month",
            rng.integers(0, 48, size=n_samples),
            np.where(
                contract == "One year",
                rng.integers(6, 65, size=n_samples),
                rng.integers(18, 73, size=n_samples),
            ),
        )

        phone_service = yn(0.90)
        multiple_lines = np.where(phone_service == "No", "No phone service", yn(0.47))

        internet_svc = rng.choice(
            ["DSL", "Fiber optic", "No"], size=n_samples, p=[0.34, 0.44, 0.22]
        )
        has_net = internet_svc != "No"

        def net_addon(prob_yes: float) -> np.ndarray:
            return np.where(has_net, yn(prob_yes), "No internet service")

        online_security = net_addon(0.37)
        online_backup = net_addon(0.44)
        device_protection = net_addon(0.44)
        tech_support = net_addon(0.37)
        streaming_tv = net_addon(0.49)
        streaming_movies = net_addon(0.49)

        paperless = yn(0.59)
        payment_method = rng.choice(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            size=n_samples,
            p=[0.34, 0.23, 0.22, 0.21],
        )

        # Realistic monthly pricing built from subscribed services.
        base_charge = (
            np.where(phone_service == "Yes", 20.0, 0.0)
            + np.where(multiple_lines == "Yes", 5.0, 0.0)
            + np.where(
                internet_svc == "DSL", 25.0, np.where(internet_svc == "Fiber optic", 50.0, 0.0)
            )
            + 5.0 * (online_security == "Yes")
            + 5.0 * (online_backup == "Yes")
            + 5.0 * (device_protection == "Yes")
            + 5.0 * (tech_support == "Yes")
            + 10.0 * (streaming_tv == "Yes")
            + 10.0 * (streaming_movies == "Yes")
            + rng.normal(0.0, 2.5, size=n_samples)
        )
        monthly_charges = base_charge.clip(18.25, 118.75).round(2)
        total_charges = np.where(
            tenure == 0,
            0.0,
            (tenure * monthly_charges + rng.normal(0.0, 35.0, size=n_samples)).clip(18.0),
        ).round(2)

        # Realistic logistic churn data-generating process (~26.5% churn rate).
        logit = (
            -2.15
            + 1.35 * (contract == "Month-to-month")
            - 0.95 * (contract == "Two year")
            + 0.85 * (internet_svc == "Fiber optic")
            - 0.70 * (internet_svc == "No")
            + 0.45 * (payment_method == "Electronic check")
            - 0.45 * (online_security == "Yes")
            - 0.45 * (tech_support == "Yes")
            + 0.30 * (paperless == "Yes")
            + 0.25 * (senior == "1")
            - 0.20 * (dependents == "Yes")
            - 0.032 * tenure
            + 0.008 * (monthly_charges - 65.0)
        )
        churn_prob = 1.0 / (1.0 + np.exp(-logit))
        churn = (rng.uniform(0.0, 1.0, size=n_samples) < churn_prob).astype(int)

        df = pd.DataFrame(
            {
                "customerID": [f"CUST-{i:05d}" for i in range(n_samples)],
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure.astype(int),
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_svc,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
                "Churn": churn,
            }
        )
        logger.info(
            f"Generated synthetic data: {n_samples:,} rows | Churn rate: {df['Churn'].mean():.2%}"
        )
        return df


if __name__ == "__main__":
    cfg = load_config()
    out_path = Path(cfg["paths"]["raw_data"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_df = DataLoader.generate_synthetic_data(n_samples=7043)
    synthetic_df.to_csv(out_path, index=False)
    print(
        f"Saved synthetic dataset to {out_path}: {len(synthetic_df):,} rows "
        f"| Churn rate: {synthetic_df['Churn'].mean():.1%}"
    )
