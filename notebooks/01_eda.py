"""
01_eda.py — Exploratory Data Analysis for ChurnScope.

Loads the real IBM Telco dataset (`data/raw/telco_churn.csv`) when present or falls back to
domain-consistent synthetic data, saves 6 publication-ready plots into `notebooks/plots/`,
and prints a verified data-quality & business-insights summary.
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
import pandas as pd
import seaborn as sns

from src import load_config
from src.data_loader import DataLoader

PALETTE = {"No Churn": "#4C9BE8", "Churn": "#E85D5D"}


def main() -> None:
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(ROOT_DIR / "config" / "config.yaml"))
    raw_path = ROOT_DIR / cfg["paths"]["raw_data"]
    if raw_path.exists():
        df = DataLoader(cfg).load(str(raw_path), require_target=True)
        source_label = f"IBM Telco Dataset ({raw_path.name})"
    else:
        df = DataLoader.generate_synthetic_data(n_samples=7043, random_state=42)
        source_label = "Synthetic Telecom Cohort"

    df["Churn_Label"] = df["Churn"].map({1: "Churn", 0: "No Churn"})

    print("=" * 60)
    print(f"CHURNSCOPE — EXPLORATORY DATA ANALYSIS ({source_label})")
    print("=" * 60)
    print(f"Shape      : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Churn rate : {df['Churn'].mean():.2%} ({int(df['Churn'].sum()):,} churners)")

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    print("\nNumeric Summary:")
    print(df[num_cols].describe().round(2).to_string())

    # 1. Target Distribution (colors aligned dynamically with label order)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Target Distribution — Customer Churn", fontsize=14, fontweight="bold")
    order = ["No Churn", "Churn"]
    counts = df["Churn_Label"].value_counts().reindex(order)
    axes[0].pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=[PALETTE[lbl] for lbl in counts.index],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[0].set_title("Overall Churn Split")

    sns.countplot(
        data=df,
        x="Churn_Label",
        hue="Churn_Label",
        palette=PALETTE,
        order=order,
        legend=False,
        ax=axes[1],
    )
    for patch in axes[1].patches:
        h = patch.get_height()
        axes[1].annotate(
            f"{int(h):,}\n({h / len(df):.1%})",
            (patch.get_x() + patch.get_width() / 2.0, h),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axes[1].set_title("Customer Count by Churn Status")
    axes[1].set_xlabel("")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "01_target_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Numeric Distributions
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Numeric Feature Distributions by Churn Status", fontsize=14, fontweight="bold")
    for idx, col in enumerate(num_cols):
        for lbl in order:
            subset = df.loc[df["Churn_Label"] == lbl, col].dropna()
            subset.plot.kde(ax=axes[0, idx], label=lbl, color=PALETTE[lbl], linewidth=2)
        axes[0, idx].set_title(f"{col} — Density")
        axes[0, idx].legend()
        sns.boxplot(
            data=df,
            x="Churn_Label",
            y=col,
            hue="Churn_Label",
            palette=PALETTE,
            order=order,
            legend=False,
            ax=axes[1, idx],
        )
        axes[1, idx].set_xlabel("")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "02_numeric_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Categorical Churn Rates
    cat_cols = [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "TechSupport",
        "OnlineSecurity",
        "PaperlessBilling",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    base_rate = float(df["Churn"].mean())
    for ax, col in zip(axes.ravel(), cat_cols, strict=True):
        rates = df.groupby(col)["Churn"].mean().sort_values(ascending=False)
        colors = ["#E85D5D" if v > base_rate else "#4C9BE8" for v in rates.values]
        bars = ax.bar(range(len(rates)), rates.values, color=colors)
        ax.axhline(base_rate, color="#666", linestyle="--", linewidth=1.1)
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels(rates.index, rotation=22, ha="right", fontsize=9)
        ax.set_ylim(0, min(1.0, float(rates.max()) * 1.28))
        ax.set_title(col)
        for bar, val in zip(bars, rates.values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.1%}",
                ha="center",
                fontsize=8,
            )
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "03_categorical_churn_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Tenure Bands (include_lowest=True retains 0-month signups)
    fig, ax = plt.subplots(figsize=(11, 5))
    df["tenure_bin"] = pd.cut(
        df["tenure"],
        bins=[0, 6, 12, 24, 36, 48, 75],
        labels=["0-6m", "7-12m", "13-24m", "25-36m", "37-48m", "49m+"],
        include_lowest=True,
    )
    t_stats = df.groupby("tenure_bin", observed=True)["Churn"].agg(["mean", "count"])
    colors = ["#E85D5D" if v > base_rate else "#4C9BE8" for v in t_stats["mean"]]
    bars = ax.bar(range(len(t_stats)), t_stats["mean"], color=colors)
    ax.axhline(base_rate, color="#555", linestyle="--", label=f"Overall ({base_rate:.1%})")
    ax.set_xticks(range(len(t_stats)))
    ax.set_xticklabels(t_stats.index)
    ax.set_ylabel("Churn Rate")
    ax.set_title("Churn Rate Across Customer Tenure Lifecycle")
    ax.legend()
    for bar, (_, row) in zip(bars, t_stats.iterrows(), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{row['mean']:.1%}\n(n={int(row['count']):,})",
            ha="center",
            fontsize=9,
        )
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "04_tenure_churn_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(7, 5.5))
    corr = df[["Churn", *num_cols]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax
    )
    ax.set_title("Numeric Feature Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "05_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    mtm = df.loc[df["Contract"] == "Month-to-month", "Churn"].mean()
    two_yr = max(df.loc[df["Contract"] == "Two year", "Churn"].mean(), 1e-4)
    fiber = df.loc[df["InternetService"] == "Fiber optic", "Churn"].mean()
    echeck = df.loc[df["PaymentMethod"] == "Electronic check", "Churn"].mean()
    new_c = df.loc[df["tenure"] <= 6, "Churn"].mean()
    loyal_c = df.loc[df["tenure"] >= 24, "Churn"].mean()

    print("\nVerified Business Insights:")
    print(f"  1. Month-to-month ({mtm:.1%}) vs Two-year ({two_yr:.1%}) = {mtm / two_yr:.1f}x risk")
    print(f"  2. Early tenure <=6m ({new_c:.1%}) vs loyal >=24m ({loyal_c:.1%})")
    print(f"  3. Fiber optic churn rate     : {fiber:.1%}")
    print(f"  4. Electronic check churn rate: {echeck:.1%}")
    print(f"\nSaved all EDA plots to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
