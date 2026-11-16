"""
predict.py — Production CLI batch inference for ChurnScope.

Loads any-sized customer CSV/Parquet/JSON/Excel file, coerces and validates columns,
scores customers in memory-safe chunks, and outputs risk-ranked predictions.

Usage:
  python predict.py --input data/raw/telco_churn.csv
  python predict.py --input data/raw/telco_churn.csv --output results/scores.csv
  python predict.py --input data/raw/telco_churn.csv --threshold 0.45
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import get_logger, load_config
from src.data_loader import DataLoader
from src.predictor import ChurnPredictor

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ChurnScope — Batch Inference CLI")
    parser.add_argument("--input", required=True, help="Path to customer CSV/Parquet/JSON/Excel.")
    parser.add_argument("--output", default=None, help="Output CSV path (default: results/...).")
    parser.add_argument(
        "--threshold", type=float, default=None, help="Override decision threshold."
    )
    parser.add_argument(
        "--model-path", default=None, help="Optional path to specific .joblib model."
    )
    parser.add_argument("--chunk-size", type=int, default=5000, help="Rows scored per batch chunk.")
    parser.add_argument("--config", default=None, help="Path to config YAML.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run_batch_inference(args: argparse.Namespace | None = None) -> pd.DataFrame:
    """Execute batch churn inference and save sorted results to CSV."""
    if args is None:
        args = parse_args()

    config = load_config(args.config)
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    if args.threshold is not None and not (0.0 < args.threshold < 1.0):
        raise ValueError(f"Threshold must be strictly between 0 and 1, got {args.threshold}")

    loader = DataLoader(config)
    df = loader.load(source=args.input, require_target=False)

    predictor = ChurnPredictor(config)
    predictor.load(model_path=args.model_path)
    if args.threshold is not None:
        predictor.optimal_threshold = float(args.threshold)
        logger.info(f"Using custom threshold override: {predictor.optimal_threshold:.4f}")

    start = time.perf_counter()
    results = predictor.predict_batch(
        df,
        enforce_limit=False,
        chunk_size=int(getattr(args, "chunk_size", 5000)),
    )
    elapsed = max(time.perf_counter() - start, 1e-6)
    throughput = len(results) / elapsed
    logger.info(f"Scored {len(results):,} customers in {elapsed:.2f}s ({throughput:,.0f} rows/sec)")

    out_df = df.copy()
    out_df["churn_probability"] = [r["churn_probability"] for r in results]
    out_df["churn_prediction"] = [r["churn_prediction"] for r in results]
    out_df["churn_label"] = [r["churn_label"] for r in results]
    out_df["risk_tier"] = [r["risk_tier"] for r in results]
    out_df["confidence"] = [r["confidence"] for r in results]
    out_df["threshold_used"] = [r["threshold_used"] for r in results]
    out_df = out_df.sort_values("churn_probability", ascending=False).reset_index(drop=True)

    output_path = args.output
    if not output_path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = f"results/churn_predictions_{ts}.csv"

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_file, index=False)

    churn_n = int(out_df["churn_prediction"].sum())
    logger.info(f"Predictions written to: {out_file}")
    logger.info(f"Predicted churners: {churn_n:,}/{len(out_df):,} ({churn_n / len(out_df):.1%})")
    logger.info("Risk tier breakdown:")
    for tier, count in out_df["risk_tier"].value_counts().items():
        logger.info(f"  {tier:<10}: {count:>6,} ({count / len(out_df) * 100:.1f}%)")

    return out_df


if __name__ == "__main__":
    run_batch_inference()
