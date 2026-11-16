"""
03_modeling.py — Multi-model benchmark, leak-free threshold tuning, calibration, SHAP, and ROI.
"""

from __future__ import annotations

import copy
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
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src import load_config
from src.data_loader import DataLoader
from src.evaluator import ChurnEvaluator
from src.preprocessor import ChurnPreprocessor
from src.trainer import ChurnTrainer


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(ROOT_DIR / "config" / "config.yaml"))
    raw_path = ROOT_DIR / cfg["paths"]["raw_data"]
    df = (
        DataLoader(cfg).load(str(raw_path), require_target=True)
        if raw_path.exists()
        else DataLoader.generate_synthetic_data(n_samples=7043, random_state=42)
    )

    splits = ChurnPreprocessor(cfg).fit_transform(df)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    # 1. Multi-model benchmark (Dummy -> LogisticRegression -> XGBoost -> LightGBM)
    rows: list[dict[str, object]] = []

    dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
    d_scores = dummy.predict_proba(X_test)[:, 1]
    rows.append(
        {
            "Model": "Dummy (Baseline)",
            "ROC-AUC": round(float(roc_auc_score(y_test, d_scores)), 4),
            "PR-AUC": round(float(average_precision_score(y_test, d_scores)), 4),
            "Brier": round(float(brier_score_loss(y_test, d_scores)), 4),
        }
    )

    lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
    lr_scores = lr.predict_proba(X_test)[:, 1]
    rows.append(
        {
            "Model": "Logistic Regression",
            "ROC-AUC": round(float(roc_auc_score(y_test, lr_scores)), 4),
            "PR-AUC": round(float(average_precision_score(y_test, lr_scores)), 4),
            "Brier": round(float(brier_score_loss(y_test, lr_scores)), 4),
        }
    )

    xgb_cfg = copy.deepcopy(cfg)
    xgb_cfg["model"]["algorithm"] = "xgboost"
    xgb_trainer = ChurnTrainer(xgb_cfg)
    xgb_trainer.train(splits)
    xgb_scores = xgb_trainer.predict_proba(X_test)
    rows.append(
        {
            "Model": "XGBoost (Calibrated)",
            "ROC-AUC": round(float(roc_auc_score(y_test, xgb_scores)), 4),
            "PR-AUC": round(float(average_precision_score(y_test, xgb_scores)), 4),
            "Brier": round(float(brier_score_loss(y_test, xgb_scores)), 4),
        }
    )

    lgb_trainer = ChurnTrainer(cfg)
    meta = lgb_trainer.train(splits)
    val_scores = lgb_trainer.predict_proba(splits["X_val"])
    test_scores = lgb_trainer.predict_proba(X_test)
    rows.append(
        {
            "Model": "LightGBM (Calibrated)",
            "ROC-AUC": round(float(roc_auc_score(y_test, test_scores)), 4),
            "PR-AUC": round(float(average_precision_score(y_test, test_scores)), 4),
            "Brier": round(float(brier_score_loss(y_test, test_scores)), 4),
        }
    )

    print("\n" + "=" * 60)
    print("MULTI-MODEL HELD-OUT TEST BENCHMARK (41 FEATURES)")
    print("=" * 60)
    print(pd.DataFrame(rows).to_string(index=False))

    # 2. Leak-free validation threshold tuning + full evaluation
    evaluator = ChurnEvaluator(cfg)
    results = evaluator.evaluate(
        y_true=y_test,
        y_scores=test_scores,
        y_val=splits["y_val"],
        y_val_scores=val_scores,
        val_monthly_charges=splits["val_monthly_charges"],
        test_monthly_charges=splits["test_monthly_charges"],
        feature_names=splits["feature_names"],
        model=lgb_trainer.base_model,
        X_test=X_test,
        training_metadata=meta,
    )

    # 3. ROC, PR, and Calibration Curves
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fpr, tpr, _ = roc_curve(y_test, test_scores)
    axes[0].plot(fpr, tpr, color="#4C9BE8", lw=2.2, label=f"LightGBM ({results['roc_auc']:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")

    prec, rec, _ = precision_recall_curve(y_test, test_scores)
    axes[1].plot(rec, prec, color="#E85D5D", lw=2.2, label=f"PR-AUC = {results['pr_auc']:.4f}")
    axes[1].axhline(y_test.mean(), color="#777", ls="--", label=f"Base = {y_test.mean():.3f}")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    c_true, c_pred = calibration_curve(y_test, test_scores, n_bins=10)
    axes[2].plot([0, 1], [0, 1], "k--", lw=1)
    axes[2].plot(
        c_pred, c_true, "o-", color="#27AE60", lw=2, label=f"Brier = {results['brier_score']:.4f}"
    )
    axes[2].set_title("Probability Calibration Curve")
    axes[2].legend()
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "model_01_performance_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Global SHAP Top 15 Plot
    fi = results.get("feature_importance", [])[:15]
    if fi:
        fig, ax = plt.subplots(figsize=(9, 6))
        names = [x["feature"] for x in fi][::-1]
        vals = [x["mean_abs_shap"] for x in fi][::-1]
        ax.barh(names, vals, color="#4C9BE8")
        ax.set_xlabel("Mean |SHAP Value| (log-odds)")
        ax.set_title("Top 15 Global SHAP Features — LightGBM")
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "model_02_shap_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 5. Customer-level LTV Net Benefit Curve
    grid = np.linspace(0.05, 0.90, 150)
    mc = splits["test_monthly_charges"]
    months = float(cfg["business"]["expected_retained_months"])
    cost = float(cfg["business"]["retention_offer_cost"])
    rate = float(cfg["business"]["retention_success_rate"])
    net_curve = [
        float(
            (mc[(test_scores >= t) & (y_test == 1)] * months * rate).sum()
            - (test_scores >= t).sum() * cost
        )
        for t in grid
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grid, net_curve, color="#27AE60", lw=2.3, label="Net LTV Benefit ($)")
    ax.axvline(
        results["threshold"],
        color="#E67E22",
        ls="--",
        label=f"Val Threshold ({results['threshold']:.3f})",
    )
    ax.axhline(0, color="#777", ls=":")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Estimated Net Benefit (USD)")
    ax.set_title("Retention Campaign Net LTV Benefit vs Decision Threshold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "model_03_business_roi.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved modeling & SHAP plots to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
