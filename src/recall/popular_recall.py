import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.target_rows import load_target_rows_from_file


EVENT_TYPES = ("clicks", "carts", "orders")
DEFAULT_TRAIN_FILE = "train_events.parquet"
DEFAULT_LABELS_FILE = "valid_labels.parquet"
DEFAULT_OUTPUT_FILE = "popular_predictions.csv"
DEFAULT_K = 20


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate multi-target popular item recall.")
    parser.add_argument(
        "--train-file",
        default=DEFAULT_TRAIN_FILE,
        help=f"Train events file under outputs/. Default: {DEFAULT_TRAIN_FILE}",
    )
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--labels-file",
        help=f"Validation labels file under outputs/. Default: {DEFAULT_LABELS_FILE}",
    )
    target_group.add_argument(
        "--test-events-file",
        help="Test events file under outputs/. Target rows are expanded to all types.",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Prediction output file under outputs/. Default: {DEFAULT_OUTPUT_FILE}",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Number of items per prediction row. Default: {DEFAULT_K}",
    )
    return parser.parse_args(argv)


def load_inputs(output_dir, train_file, labels_file, test_events_file):
    train_path = output_dir / train_file

    if not train_path.exists():
        raise FileNotFoundError(f"Train events file not found: {train_path}")

    train_events = pd.read_parquet(train_path)
    target_rows = load_target_rows_from_file(output_dir, labels_file, test_events_file)

    required_train_columns = {"aid", "type"}

    missing_train_columns = required_train_columns - set(train_events.columns)

    if missing_train_columns:
        raise ValueError(f"{train_path} missing columns: {sorted(missing_train_columns)}")

    return train_events, target_rows


def build_type_popular_items(train_events, k):
    fallback_items = train_events["aid"].value_counts().head(k).index.tolist()
    popular_by_type = {}

    for event_type in EVENT_TYPES:
        type_items = (
            train_events.loc[train_events["type"] == event_type, "aid"]
            .value_counts()
            .head(k)
            .index
            .tolist()
        )
        popular_by_type[event_type] = fill_to_k(type_items, fallback_items, k)

    return popular_by_type


def fill_to_k(items, fallback_items, k):
    filled = []
    seen = set()

    for item in items + fallback_items:
        if len(filled) >= k:
            break
        if item in seen:
            continue

        filled.append(item)
        seen.add(item)

    return filled


def build_predictions(target_rows, popular_by_type):
    rows = []

    for session, event_type in zip(target_rows["session"], target_rows["type"]):
        items = popular_by_type.get(event_type, [])
        rows.append({
            "session": session,
            "type": event_type,
            "predictions": " ".join(map(str, items)),
        })

    return pd.DataFrame(rows, columns=["session", "type", "predictions"])


def main(argv=None):
    args = parse_args(argv)
    root = Path(__file__).resolve().parent.parent.parent
    output_dir = root / "outputs"

    labels_file = args.labels_file if args.test_events_file else (args.labels_file or DEFAULT_LABELS_FILE)
    train_events, target_rows = load_inputs(output_dir, args.train_file, labels_file, args.test_events_file)
    popular_by_type = build_type_popular_items(train_events, args.k)
    predictions = build_predictions(target_rows, popular_by_type)
    predictions.to_csv(output_dir / args.output_file, index=False)

    print(f"Multi-target popular predictions saved to {args.output_file}")
    print(f"Output rows: {len(predictions):,}")
    for event_type in EVENT_TYPES:
        print(f"{event_type}: {len(popular_by_type[event_type])} items")
