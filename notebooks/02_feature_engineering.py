"""
02_feature_engineering.py — Analysis of ChurnScope's 22 leak-free domain features.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib

if not os.getenv("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

from src import load_config
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer


def main() -> None:
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(ROOT_DIR / "config" / "config.yaml"))
    raw_path = ROOT_DIR / cfg["paths"]["raw_data"]
    df_raw = (
        DataLoader(cfg).load(str(raw_path), require_target=True)
        if raw_path.exists()
        else DataLoader.generate_synthetic_data(n_samples=7043, random_state=42)
    )

    fe = FeatureEngineer(cfg)
    df_eng = fe.fit_transform(df_raw)
    new_feats = fe.get_new_feature_names()

    print("=" * 60)
    print("CHURNSCOPE — FEATURE ENGINEERING DEEP DIVE")
    print("=" * 60)
    print(f"Raw columns        : {df_raw.shape[1]}")
    print(f"Enriched columns   : {df_eng.shape[1]}")
    print(f"Engineered features: {len(new_feats)}")
    print(f"Train median ($)   : ${fe.monthly_charges_median_:.2f}")

    # 1. Contract & Payment Risk Flags
    risk_flags = [
        ("is_month_to_month", "Month-to-Month"),
        ("is_electronic_check", "Electronic Check"),
        ("is_auto_payment", "Auto Payment"),
        ("highest_churn_risk_profile", "M2M + E-Check Combo"),
    ]
    overall = float(df_eng["Churn"].mean())
    risk_rates = {
        lbl: float(df_eng.loc[df_eng[col] == 1, "Churn"].mean())
        for col, lbl in risk_flags
        if col in df_eng.columns
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(
        list(risk_rates.keys()),
        list(risk_rates.values()),
        color=["#E85D5D" if v > overall else "#4C9BE8" for v in risk_rates.values()],
    )
    axes[0].axvline(overall, color="#555", linestyle="--", label=f"Baseline ({overall:.1%})")
    axes[0].set_xlabel("Churn Rate When Flag = 1")
    axes[0].set_title("Churn Rate Across Engineered Risk Flags")
    axes[0].legend()

    pivot = df_eng.groupby(["is_month_to_month", "is_electronic_check"])["Churn"].mean().unstack()
    pivot.index = ["1yr / 2yr Contract", "Month-to-Month"]
    pivot.columns = ["Other Payment", "Electronic Check"]
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="Reds", ax=axes[1], linewidths=0.5)
    axes[1].set_title("Contract × Payment Method Interaction")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "fe_01_risk_flags.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Target Correlation Ranking (Engineered vs Raw)
    numeric_cols = df_eng.select_dtypes(include=[np.number]).columns.tolist()
    corr = df_eng[numeric_cols].corr()["Churn"].drop("Churn").abs().sort_values(ascending=False)
    top = corr.head(20)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    bar_colors = ["#E85D5D" if f in new_feats else "#4C9BE8" for f in top.index]
    ax.barh(range(len(top)), top.values[::-1], color=bar_colors[::-1])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index[::-1])
    ax.set_xlabel("|Pearson Correlation with Churn|")
    ax.set_title("Top 20 Numeric Features Correlated with Churn")
    ax.legend(
        handles=[
            Patch(facecolor="#E85D5D", label="Engineered feature"),
            Patch(facecolor="#4C9BE8", label="Raw feature"),
        ],
        loc="lower right",
    )
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "fe_02_target_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nTop 12 Features by |Correlation with Churn|:")
    print(corr.head(12).round(4).to_string())
    print(f"\nSaved plots to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
