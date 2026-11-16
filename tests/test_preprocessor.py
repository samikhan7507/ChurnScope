"""Unit tests for the leak-free ChurnPreprocessor pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from src.data_loader import DataLoader
from src.preprocessor import ChurnPreprocessor


@pytest.fixture
def raw_df():
    return DataLoader.generate_synthetic_data(n_samples=500, random_state=99)


class TestChurnPreprocessor:
    def test_split_sizes_and_all_41_features_retained(self, base_config, raw_df):
        preprocessor = ChurnPreprocessor(base_config)
        result = preprocessor.fit_transform(raw_df)

        total = len(raw_df)
        train_n = result["X_train"].shape[0]
        val_n = result["X_val"].shape[0]
        test_n = result["X_test"].shape[0]

        assert train_n + val_n + test_n == total
        assert train_n > test_n > val_n

        # 3 raw numeric + 22 engineered + 16 categorical = 41 total features (none dropped!)
        assert result["X_train"].shape[1] == 41
        assert result["X_val"].shape[1] == 41
        assert result["X_test"].shape[1] == 41
        assert len(result["feature_names"]) == 41
        assert "Churn" not in result["feature_names"]
        assert "customerID" not in result["feature_names"]
        assert "monthly_charge_per_tenure" in result["feature_names"]
        assert "is_month_to_month" in result["feature_names"]

    def test_no_leakage_and_handles_missing_and_unseen_values(self, base_config, raw_df):
        dirty = raw_df.copy()
        dirty.loc[0:5, "TotalCharges"] = np.nan
        dirty.loc[6:8, "PaymentMethod"] = np.nan

        preprocessor = ChurnPreprocessor(base_config)
        result = preprocessor.fit_transform(dirty)

        for split_key in ("X_train", "X_val", "X_test"):
            assert not np.isnan(result[split_key]).any()

        for y_key in ("y_train", "y_val", "y_test"):
            assert set(np.unique(result[y_key])).issubset({0, 1})

        # Unseen category at inference time maps cleanly to unknown_value (-1) without failing.
        unseen = raw_df.head(5).copy()
        unseen.loc[unseen.index[0], "PaymentMethod"] = "Crypto wallet"
        transformed = preprocessor.transform(unseen)
        assert transformed.shape == (5, 41)
        assert not np.isnan(transformed).any()

        # Verify FeatureEngineer median was fit on X_train only, not the entire DataFrame.
        fe_step = preprocessor.pipeline.named_steps["feature_engineer"]
        expected_train_median = float(result["X_train_raw"]["MonthlyCharges"].median())
        assert fe_step.monthly_charges_median_ == pytest.approx(expected_train_median)

    def test_save_load_roundtrip_and_processed_splits(self, base_config, raw_df, tmp_path):
        base_config["paths"]["processed_data"] = str(tmp_path / "processed")
        base_config["paths"]["artifact_dir"] = str(tmp_path / "artifacts")

        preprocessor = ChurnPreprocessor(base_config)
        splits = preprocessor.fit_transform(raw_df)
        preprocessor.save_processed_splits(splits)
        assert (tmp_path / "processed" / "train.parquet").exists()
        assert (tmp_path / "processed" / "val.parquet").exists()
        assert (tmp_path / "processed" / "test.parquet").exists()

        save_path = str(tmp_path / "pipeline.joblib")
        preprocessor.save(path=save_path)

        reloaded = ChurnPreprocessor(base_config).load(save_path)
        sample = raw_df.head(25)
        np.testing.assert_array_almost_equal(
            preprocessor.transform(sample), reloaded.transform(sample)
        )

    def test_unfitted_errors(self, base_config, raw_df):
        pp = ChurnPreprocessor(base_config)
        with pytest.raises(RuntimeError):
            pp.transform(raw_df.head(2))
        with pytest.raises(RuntimeError):
            pp.save()
