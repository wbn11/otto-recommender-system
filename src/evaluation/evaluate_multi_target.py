import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import load_config


_CFG = load_config()
DEFAULT_PRED_FILE = "multi_target_popular_predictions.csv"
DEFAULT_LABELS_FILE = "multi_target_valid_labels.parquet"
DEFAULT_K = _CFG.get("eval", {}).get("k", 20)

TYPE_WEIGHTS = _CFG.get("eval", {}).get("type_weights", {
    "clicks": 0.10,
    "carts": 0.30,
    "orders": 0.60,
})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate multi-target recall predictions.")
    parser.add_argument(
        "--pred-file",
        default=DEFAULT_PRED_FILE,
        help=f"Prediction CSV file under outputs/. Default: {DEFAULT_PRED_FILE}",
    )
    parser.add_argument(
        "--labels-file",
        default=DEFAULT_LABELS_FILE,
        help=f"Label CSV file under outputs/. Default: {DEFAULT_LABELS_FILE}",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Evaluate Recall@K. Default: {DEFAULT_K}",
    )
    return parser.parse_args(argv)


def parse_items(value):
    if pd.isna(value):
        return []

    items = []
    for token in str(value).split():
        if token.isdigit():
            items.append(int(token))

    return items


def load_inputs(output_dir, labels_file, pred_file):
    labels_path = output_dir / labels_file
    predictions_path = output_dir / pred_file

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {predictions_path}")

    labels = pd.read_parquet(labels_path)
    predictions = pd.read_csv(predictions_path)

    required_label_columns = {"session", "type", "labels"}
    required_prediction_columns = {"session", "type", "predictions"}

    missing_label_columns = required_label_columns - set(labels.columns)
    missing_prediction_columns = required_prediction_columns - set(predictions.columns)

    if missing_label_columns:
        raise ValueError(f"{labels_path} missing columns: {sorted(missing_label_columns)}")
    if missing_prediction_columns:
        raise ValueError(f"{predictions_path} missing columns: {sorted(missing_prediction_columns)}")

    return labels, predictions


def add_recall_column(labels_df, predictions_df, k):
    merged = labels_df.merge(predictions_df, on=["session", "type"], how="left")
    hits = []
    denominators = []
    recalls = []

    for _, row in merged.iterrows():
        true_items = set(parse_items(row["labels"]))
        pred_items = parse_items(row["predictions"])[:k]
        denominator = min(len(true_items), k)
        hit_count = len(true_items.intersection(pred_items))
        recall = hit_count / denominator if denominator else 0.0

        hits.append(hit_count)
        denominators.append(denominator)
        recalls.append(recall)

    merged["hits"] = hits
    merged["denominator"] = denominators
    merged["recall"] = recalls
    return merged


def summarize_by_type(merged_df, type_weights):
    rows = []

    for event_type, weight in type_weights.items():
        type_df = merged_df[merged_df["type"] == event_type]
        hits = int(type_df["hits"].sum()) if not type_df.empty else 0
        denominator = int(type_df["denominator"].sum()) if not type_df.empty else 0
        rows.append({
            "type": event_type,
            "rows": len(type_df),
            "hits": hits,
            "denominator": denominator,
            "recall": hits / denominator if denominator else 0.0,
            "weight": weight,
        })

    for row in rows:
        row["contribution"] = row["recall"] * row["weight"]

    return rows


def summarize_overall(merged_df):
    rows = len(merged_df)
    hits = int(merged_df["hits"].sum()) if rows else 0
    denominator = int(merged_df["denominator"].sum()) if rows else 0
    recall = hits / denominator if denominator else 0.0
    return {
        "rows": rows,
        "hits": hits,
        "denominator": denominator,
        "recall": recall,
    }


def print_summary(pred_file, k, overall, by_type_rows):
    print(f"Evaluating {pred_file}...")
    print(f"Overall Recall@{k}: {overall['recall']:.4f}")
    print()
    print("By Type:")
    print(f"{'type':<8} {'rows':>8} {'hits':>8} {'total':>8} {f'recall@{k}':>10} {'weight':>8} {'contribution':>13}")

    for row in by_type_rows:
        print(
            f"{row['type']:<8} "
            f"{row['rows']:>8,} "
            f"{row['hits']:>8,} "
            f"{row['denominator']:>8,} "
            f"{row['recall']:>10.4f} "
            f"{row['weight']:>8.2f} "
            f"{row['contribution']:>13.4f}"
        )

    weighted_score = sum(row["contribution"] for row in by_type_rows)
    print()
    print(f"Weighted Score: {weighted_score:.4f}")


def main(argv=None):
    args = parse_args(argv)
    root = Path(__file__).resolve().parent.parent.parent
    output_dir = root / "outputs"

    labels, predictions = load_inputs(output_dir, args.labels_file, args.pred_file)
    merged = add_recall_column(labels, predictions, args.k)
    overall = summarize_overall(merged)
    by_type_rows = summarize_by_type(merged, TYPE_WEIGHTS)
    print_summary(args.pred_file, args.k, overall, by_type_rows)
