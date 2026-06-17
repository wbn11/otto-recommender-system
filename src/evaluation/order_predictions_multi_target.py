import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.target_rows import (
    load_target_rows_from_file,
    order_predictions_by_target_rows,
)


DEFAULT_PRED_FILE = "multi_target_ranker_predictions.csv"
DEFAULT_LABELS_FILE = "multi_target_valid_labels.parquet"
DEFAULT_OUTPUT_FILE = "multi_target_ranker_predictions_ordered.csv"
DEFAULT_K = 20


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Order multi-target predictions by target rows.")
    parser.add_argument("--pred-file", default=DEFAULT_PRED_FILE)
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--labels-file", help=f"Validation labels file under outputs/. Default: {DEFAULT_LABELS_FILE}")
    target_group.add_argument("--test-events-file", help="Test events file under outputs/. Target rows are expanded to all types.")
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(__file__).resolve().parents[2] / "outputs"
    pred_path = output_dir / args.pred_file
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    predictions = pd.read_csv(pred_path)
    labels_file = args.labels_file if args.test_events_file else (args.labels_file or DEFAULT_LABELS_FILE)
    target_rows = load_target_rows_from_file(output_dir, labels_file, args.test_events_file)
    ordered = order_predictions_by_target_rows(predictions, target_rows, args.k)
    ordered.to_csv(output_dir / args.output_file, index=False)

    empty_rows = int((ordered["predictions"] == "").sum())
    print(f"Ordered predictions saved to {args.output_file}")
    print(f"Output rows: {len(ordered):,}")
    print(f"Empty prediction rows: {empty_rows:,}")
