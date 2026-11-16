"""Unit tests for leak-free FeatureEngineer domain features and single-vs-batch parity."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer


def test_exact_22_distinct_engineered_features(base_config):
    df = DataLoader.generate_synthetic_data(n_samples=300, random_state=11)
    fe = FeatureEngineer(base_config).fit(df)
    enriched = fe.transform(df)

    expected = fe.get_new_feature_names()
    assert len(expected) == 22
    assert len(set(expected)) == 22
    assert enriched.shape[1] == df.shape[1] + 22
    for col in expected:
        assert col in enriched.columns

    # Verify charge ratios are genuinely distinct (no identical duplicate columns).
    assert not np.allclose(
        enriched["monthly_charge_per_tenure"].to_numpy(),
        enriched["charge_per_service"].to_numpy(),
    )


def test_single_customer_vs_batch_parity_no_inference_skew(base_config):
    """is_high_value and high_risk_flag must match exactly whether scored 1-by-1 or in a batch."""
    train_df = DataLoader.generate_synthetic_data(n_samples=400, random_state=21)
    test_df = DataLoader.generate_synthetic_data(n_samples=25, random_state=22)

    # Ensure at least one new high-spend customer in test_df.
    test_df.loc[0, "tenure"] = 2
    test_df.loc[0, "MonthlyCharges"] = 115.0

    fe = FeatureEngineer(base_config).fit(train_df)
    batch_out = fe.transform(test_df)

    single_row_out = fe.transform(test_df.iloc[[0]])
    assert int(single_row_out.iloc[0]["is_high_value"]) == 1
    assert int(single_row_out.iloc[0]["high_risk_flag"]) == 1

    for idx in range(5):
        one = fe.transform(test_df.iloc[[idx]])
        pd.testing.assert_series_equal(
            one.iloc[0][fe.get_new_feature_names()],
            batch_out.iloc[idx][fe.get_new_feature_names()],
            check_names=False,
        )


def test_internet_service_counted_in_total_services(base_config):
    row = pd.DataFrame(
        [
            {
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "tenure": 12,
                "MonthlyCharges": 85.0,
                "TotalCharges": 1020.0,
                "Contract": "Month-to-month",
                "PaymentMethod": "Electronic check",
                "PaperlessBilling": "Yes",
            }
        ]
    )
    out = FeatureEngineer(base_config).fit_transform(row)
    # PhoneService + Fiber optic + OnlineSecurity + TechSupport = 4 subscribed services
    assert int(out.loc[0, "total_services"]) == 4
    assert int(out.loc[0, "protection_services_count"]) == 2
    assert int(out.loc[0, "has_internet"]) == 1
    assert int(out.loc[0, "has_fiber"]) == 1
    assert int(out.loc[0, "highest_churn_risk_profile"]) == 1


def test_feature_group_toggles_respected(base_config):
    cfg = copy.deepcopy(base_config)
    cfg["feature_engineering"]["create_contract_flags"] = False
    cfg["feature_engineering"]["create_interaction_features"] = False

    df = DataLoader.generate_synthetic_data(n_samples=50, random_state=5)
    fe = FeatureEngineer(cfg).fit(df)
    out = fe.transform(df)
    assert len(fe.get_new_feature_names()) == 13
    assert "is_month_to_month" not in out.columns
    assert "stickiness_score" not in out.columns
