import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.target_rows import (
    build_test_target_rows,
    build_validation_target_rows,
    order_predictions_by_target_rows,
)


DEFAULT_PRED_FILE = "multi_target_ranker_predictions.csv"
DEFAULT_TARGET_FILE = "multi_target_valid_labels.parquet"
DEFAULT_OUTPUT_FILE = "multi_target_ranker_predictions_ordered.csv"
DEFAULT_K = 20


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Order multi-target predictions by target rows.")
    parser.add_argument("--pred-file", default=DEFAULT_PRED_FILE)
    parser.add_argument("--target-file", default=DEFAULT_TARGET_FILE)
    parser.add_argument("--events-file", help="Build target rows from events by expanding every session to all types.")
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    return parser.parse_args(argv)


def load_target_rows(output_dir, target_file, events_file):
    if events_file:
        events_path = output_dir / events_file
        if not events_path.exists():
            raise FileNotFoundError(f"Events file not found: {events_path}")
        return build_test_target_rows(pd.read_parquet(events_path))

    target_path = output_dir / target_file
    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")
    return build_validation_target_rows(pd.read_parquet(target_path))


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(__file__).resolve().parents[2] / "outputs"
    pred_path = output_dir / args.pred_file
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    predictions = pd.read_csv(pred_path)
    target_rows = load_target_rows(output_dir, args.target_file, args.events_file)
    ordered = order_predictions_by_target_rows(predictions, target_rows, args.k)
    ordered.to_csv(output_dir / args.output_file, index=False)

    empty_rows = int((ordered["predictions"] == "").sum())
    print(f"Ordered predictions saved to {args.output_file}")
    print(f"Output rows: {len(ordered):,}")
    print(f"Empty prediction rows: {empty_rows:,}")
