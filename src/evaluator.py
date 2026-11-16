"""
evaluator.py — Leak-free threshold optimization, classification/ranking metrics, SHAP, and LTV ROI.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import get_logger

logger = get_logger(__name__)


def extract_positive_shap(raw_shap: Any) -> np.ndarray:
    """Normalize SHAP output across 2D, list-of-2D, and 3D `(n, features, 2)` formats."""
    if isinstance(raw_shap, list):
        return np.asarray(raw_shap[1], dtype=float)
    arr = np.asarray(raw_shap, dtype=float)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        return arr[:, :, 1]
    return arr


class ChurnEvaluator:
    """Evaluates churn classifiers on held-out splits with validation-tuned decision thresholds."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.threshold_cfg = config.get("threshold", {})
        self.biz_cfg = config.get("business", {})
        self.random_state: int = int(config.get("data", {}).get("random_state", 42))
        self.optimal_threshold: float = float(self.threshold_cfg.get("default", 0.5))
        self.evaluation_results: dict[str, Any] = {}

    def optimize_threshold(
        self,
        y_val: np.ndarray,
        y_val_scores: np.ndarray,
        monthly_charges: np.ndarray | None = None,
    ) -> float:
        """Select optimal decision threshold strictly on validation predictions (no test leakage)."""
        strategy = str(self.threshold_cfg.get("strategy", "f1")).lower()
        y_val = np.asarray(y_val, dtype=int)
        y_val_scores = np.asarray(y_val_scores, dtype=float)

        prec_v, rec_v, thresholds = precision_recall_curve(y_val, y_val_scores)
        if len(thresholds) == 0:
            self.optimal_threshold = float(self.threshold_cfg.get("default", 0.5))
            return self.optimal_threshold

        logger.info(f"Optimizing threshold on validation set (strategy={strategy!r})...")

        if strategy == "f1":
            num = 2.0 * prec_v[:-1] * rec_v[:-1]
            denom = prec_v[:-1] + rec_v[:-1]
            f1s = np.divide(num, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
            best_idx = int(np.argmax(f1s))
            opt = float(thresholds[best_idx])
            logger.info(f"Validation F1-optimal threshold: {opt:.4f} (Val F1={f1s[best_idx]:.4f})")

        elif strategy == "roi":
            months = float(self.biz_cfg.get("expected_retained_months", 6))
            offer_cost = float(self.biz_cfg.get("retention_offer_cost", 25.0))
            ret_rate = float(self.biz_cfg.get("retention_success_rate", 0.35))
            avg_rev = float(self.biz_cfg.get("avg_monthly_revenue", 65.0))
            rev_vec = (
                np.asarray(monthly_charges, dtype=float)
                if monthly_charges is not None and len(monthly_charges) == len(y_val)
                else np.full(len(y_val), avg_rev, dtype=float)
            )
            grid = np.linspace(0.05, 0.95, 181)
            benefits = []
            for t in grid:
                targeted = y_val_scores >= t
                saved = float((rev_vec[targeted & (y_val == 1)] * months * ret_rate).sum())
                cost = float(targeted.sum() * offer_cost)
                benefits.append(saved - cost)
            best_idx = int(np.argmax(benefits))
            opt = float(grid[best_idx])
            logger.info(
                f"Validation ROI-optimal threshold: {opt:.4f} (Val Net=${benefits[best_idx]:,.2f})"
            )

        elif strategy == "custom_cost":
            cc = self.threshold_cfg.get("cost_matrix", {})
            fn_cost = float(cc.get("fn_cost", 5.0))
            fp_cost = float(cc.get("fp_cost", 1.0))
            grid = np.linspace(0.05, 0.95, 181)
            costs = []
            for t in grid:
                y_pred = (y_val_scores >= t).astype(int)
                _, fp_n, fn_n, _ = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
                costs.append(fn_cost * fn_n + fp_cost * fp_n)
            best_idx = int(np.argmin(costs))
            opt = float(grid[best_idx])
            logger.info(f"Validation cost-optimal threshold: {opt:.4f}")

        elif strategy == "recall":
            min_recall = float(self.threshold_cfg.get("min_recall", 0.80))
            eligible = rec_v[:-1] >= min_recall
            if eligible.any():
                best_idx = int(np.argmax(prec_v[:-1][eligible]))
                opt = float(thresholds[eligible][best_idx])
            else:
                opt = float(thresholds[int(np.argmax(rec_v[:-1]))])
            logger.info(f"Validation recall-optimal threshold: {opt:.4f}")

        else:
            opt = float(self.threshold_cfg.get("default", 0.5))

        self.optimal_threshold = round(opt, 4)
        return self.optimal_threshold

    def evaluate(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        threshold: float | None = None,
        y_val: np.ndarray | None = None,
        y_val_scores: np.ndarray | None = None,
        val_monthly_charges: np.ndarray | None = None,
        test_monthly_charges: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        model: Any = None,
        X_test: np.ndarray | None = None,
        training_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute full evaluation suite on the held-out test set."""
        logger.info("Running held-out test evaluation...")
        y_true = np.asarray(y_true, dtype=int)
        y_scores = np.clip(np.asarray(y_scores, dtype=float), 1e-7, 1.0 - 1e-7)

        threshold_source = "explicit"
        if threshold is None:
            if (
                self.threshold_cfg.get("optimize", True)
                and y_val is not None
                and y_val_scores is not None
            ):
                threshold = self.optimize_threshold(
                    y_val, y_val_scores, monthly_charges=val_monthly_charges
                )
                threshold_source = "validation_set"
            else:
                threshold = float(self.threshold_cfg.get("default", 0.5))
                threshold_source = "config_default"

        self.optimal_threshold = float(threshold)
        y_pred = (y_scores >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_idx = np.linspace(0, len(fpr) - 1, min(100, len(fpr))).astype(int)

        prec_v, rec_v, _ = precision_recall_curve(y_true, y_scores)
        pr_idx = np.linspace(0, len(prec_v) - 1, min(100, len(prec_v))).astype(int)

        cal_true, cal_pred = calibration_curve(y_true, y_scores, n_bins=10, strategy="uniform")

        results: dict[str, Any] = {
            "threshold": round(float(threshold), 4),
            "threshold_strategy": str(self.threshold_cfg.get("strategy", "f1")),
            "threshold_source": threshold_source,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "roc_auc": float(roc_auc_score(y_true, y_scores)),
            "pr_auc": float(average_precision_score(y_true, y_scores)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "log_loss": float(log_loss(y_true, y_scores)),
            "brier_score": float(brier_score_loss(y_true, y_scores)),
            "n_test": int(len(y_true)),
            "positive_rate": float(y_true.mean()),
            "predicted_positive_rate": float(y_pred.mean()),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
            "ranking_metrics": self._compute_ranking_metrics(y_true, y_scores),
            "business_metrics": self._compute_business_metrics(
                tn=int(tn),
                fp=int(fp),
                fn=int(fn),
                tp=int(tp),
                y_true=y_true,
                y_pred=y_pred,
                monthly_charges=test_monthly_charges,
            ),
            "classification_report": classification_report(
                y_true,
                y_pred,
                labels=[0, 1],
                target_names=["Non-churner", "Churner"],
                output_dict=True,
                zero_division=0,
            ),
            "roc_curve": {
                "fpr": [round(float(v), 5) for v in fpr[roc_idx]],
                "tpr": [round(float(v), 5) for v in tpr[roc_idx]],
            },
            "pr_curve": {
                "precision": [round(float(v), 5) for v in prec_v[pr_idx]],
                "recall": [round(float(v), 5) for v in rec_v[pr_idx]],
            },
            "calibration_curve": {
                "mean_predicted": [round(float(v), 5) for v in cal_pred],
                "fraction_positives": [round(float(v), 5) for v in cal_true],
            },
        }

        if training_metadata:
            results["model_summary"] = {
                "algorithm": training_metadata.get("algorithm"),
                "is_calibrated": training_metadata.get("is_calibrated", False),
                "cv_roc_auc_mean": training_metadata.get("cv_roc_auc_mean"),
                "cv_roc_auc_std": training_metadata.get("cv_roc_auc_std"),
                "val_roc_auc": training_metadata.get("val_roc_auc"),
                "n_features": training_metadata.get("n_features"),
                "n_train": training_metadata.get("train_samples"),
                "n_val": training_metadata.get("val_samples"),
                "n_test": int(len(y_true)),
                "trained_at": training_metadata.get("trained_at"),
            }

        if model is not None and X_test is not None and feature_names is not None:
            shap_importance = self._compute_shap(model, X_test, feature_names)
            if shap_importance:
                results["feature_importance"] = shap_importance

        self.evaluation_results = results
        self._log_summary(results)
        return results

    def save_results(self, path: str | None = None) -> str:
        """Write evaluation results JSON to models/artifacts/evaluation_results.json."""
        default_dir = self.config.get("paths", {}).get("artifact_dir", "models/artifacts/")
        out_path = path or os.path.join(default_dir, "evaluation_results.json")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(self.evaluation_results, fh, indent=2, default=str)
        logger.info(f"Evaluation results saved: {out_path}")
        return out_path

    def _compute_ranking_metrics(
        self, y_true: np.ndarray, y_scores: np.ndarray
    ) -> dict[str, float]:
        """Compute Precision@K, Recall@K, and Lift@K for capacity-constrained retention teams."""
        fractions = self.biz_cfg.get("top_k_fractions", [0.10, 0.20])
        order = np.argsort(y_scores)[::-1]
        y_sorted = y_true[order]
        base_rate = float(y_true.mean()) if len(y_true) > 0 else 0.0
        total_pos = int(y_true.sum())
        metrics: dict[str, float] = {}

        for frac in fractions:
            pct = int(round(float(frac) * 100))
            k = max(1, int(round(len(y_true) * float(frac))))
            top_k = y_sorted[:k]
            p_at_k = float(top_k.mean())
            r_at_k = float(top_k.sum() / total_pos) if total_pos > 0 else 0.0
            lift = float(p_at_k / base_rate) if base_rate > 0 else 0.0
            metrics[f"precision_at_{pct}pct"] = round(p_at_k, 4)
            metrics[f"recall_at_{pct}pct"] = round(r_at_k, 4)
            metrics[f"lift_at_{pct}pct"] = round(lift, 2)

        return metrics

    def _compute_business_metrics(
        self,
        tn: int,
        fp: int,
        fn: int,
        tp: int,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        monthly_charges: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Estimate retention campaign economics using multi-month customer LTV."""
        avg_rev = float(self.biz_cfg.get("avg_monthly_revenue", 65.0))
        months = float(self.biz_cfg.get("expected_retained_months", 6))
        offer_cost = float(self.biz_cfg.get("retention_offer_cost", 25.0))
        ret_rate = float(self.biz_cfg.get("retention_success_rate", 0.35))

        total_churners = int(y_true.sum())
        if monthly_charges is not None and len(monthly_charges) == len(y_true):
            mc = np.asarray(monthly_charges, dtype=float)
            tp_mask = (y_true == 1) & (y_pred == 1)
            revenue_saved = float((mc[tp_mask] * months * ret_rate).sum())
            revenue_at_risk = float((mc[y_true == 1] * months).sum())
        else:
            revenue_saved = tp * avg_rev * months * ret_rate
            revenue_at_risk = total_churners * avg_rev * months

        offer_cost_total = (tp + fp) * offer_cost
        net_benefit = revenue_saved - offer_cost_total
        roi_pct = (net_benefit / offer_cost_total * 100.0) if offer_cost_total > 0 else 0.0

        return {
            "total_churners_in_test": total_churners,
            "churners_identified": tp,
            "churners_missed": fn,
            "non_churners_targeted": fp,
            "true_negatives": tn,
            "churn_coverage_pct": round(tp / total_churners * 100.0, 1) if total_churners else 0.0,
            "targeting_efficiency_pct": round(tp / (tp + fp) * 100.0, 1) if (tp + fp) else 0.0,
            "expected_retained_months": int(months),
            "retention_offer_cost_per_customer_usd": round(offer_cost, 2),
            "assumed_retention_success_rate": round(ret_rate, 2),
            "estimated_revenue_saved_usd": round(revenue_saved, 2),
            "retention_offer_cost_usd": round(offer_cost_total, 2),
            "net_business_benefit_usd": round(net_benefit, 2),
            "campaign_roi_pct": round(roi_pct, 1),
            "revenue_at_risk_without_model_usd": round(revenue_at_risk, 2),
        }

    def _compute_shap(
        self,
        model: Any,
        X_test: np.ndarray,
        feature_names: list[str],
        max_samples: int = 500,
    ) -> list[dict[str, Any]] | None:
        """Compute reproducible global mean(|SHAP|) feature importance via TreeExplainer."""
        try:
            import shap

            rng = np.random.default_rng(self.random_state)
            n = min(max_samples, X_test.shape[0])
            idx = rng.choice(X_test.shape[0], size=n, replace=False)
            explainer = shap.TreeExplainer(model)
            sv = extract_positive_shap(explainer.shap_values(X_test[idx]))
            mean_abs = np.abs(sv).mean(axis=0)
            imp_df = pd.DataFrame(
                {
                    "feature": feature_names[: len(mean_abs)],
                    "mean_abs_shap": [round(float(v), 6) for v in mean_abs],
                }
            ).sort_values("mean_abs_shap", ascending=False)
            top5 = ", ".join(imp_df["feature"].head(5).tolist())
            logger.info(f"Top 5 global SHAP features: {top5}")
            return imp_df.to_dict(orient="records")
        except ImportError:
            logger.warning("SHAP package not installed — skipping feature importance.")
            return None
        except Exception as exc:
            logger.warning(f"Global SHAP calculation failed: {exc}")
            return None

    def _log_summary(self, results: dict[str, Any]) -> None:
        cm = results["confusion_matrix"]
        biz = results["business_metrics"]
        rank = results.get("ranking_metrics", {})
        logger.info("=" * 58)
        logger.info("HELD-OUT TEST EVALUATION SUMMARY")
        logger.info(f"  Threshold ({results['threshold_source']}) : {results['threshold']:.4f}")
        logger.info(f"  Accuracy    : {results['accuracy']:.4f}")
        logger.info(f"  ROC-AUC     : {results['roc_auc']:.4f}")
        logger.info(f"  PR-AUC      : {results['pr_auc']:.4f}")
        logger.info(f"  F1 Score    : {results['f1']:.4f}")
        logger.info(f"  Precision   : {results['precision']:.4f}")
        logger.info(f"  Recall      : {results['recall']:.4f}")
        logger.info(f"  Brier Score : {results['brier_score']:.4f}")
        logger.info(
            f"  TP/FP/FN/TN : {cm['true_positive']}/{cm['false_positive']}/"
            f"{cm['false_negative']}/{cm['true_negative']}"
        )
        if "lift_at_20pct" in rank:
            logger.info(
                f"  Top-20% Lift: {rank['lift_at_20pct']:.2f}x "
                f"(Recall@20%={rank['recall_at_20pct']:.1%})"
            )
        logger.info(
            f"  Net Benefit : ${biz['net_business_benefit_usd']:,.2f} "
            f"(ROI: {biz['campaign_roi_pct']:.1f}%)"
        )
        logger.info("=" * 58)
