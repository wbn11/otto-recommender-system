import argparse
from pathlib import Path

import pandas as pd


DEFAULT_PRED_FILE = "fusion_predictions.csv"
DEFAULT_LABELS_FILE = "valid_labels.csv"
DEFAULT_K = 20

TYPE_WEIGHTS = {
    "clicks": 0.10,
    "carts": 0.30,
    "orders": 0.60,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate recall predictions.")
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


def parse_predictions(pred_str):
    if pd.isna(pred_str):
        return []

    predictions = []
    for token in str(pred_str).split():
        if token.isdigit():
            predictions.append(int(token))

    return predictions


def load_inputs(output_dir, labels_file, pred_file):
    labels_path = output_dir / labels_file
    predictions_path = output_dir / pred_file

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {predictions_path}")

    labels = pd.read_csv(labels_path)
    predictions = pd.read_csv(predictions_path)
    return labels, predictions


def add_hit_column(labels_df, predictions_df, k):
    merged = labels_df.merge(predictions_df, on="session", how="left")
    hits = []

    for _, row in merged.iterrows():
        label_aid = row["label_aid"]
        pred_list = parse_predictions(row["predictions"])
        hits.append(label_aid in pred_list[:k])

    merged["hit"] = hits
    return merged


def summarize_overall(merged_df):
    labels = len(merged_df)
    hits = int(merged_df["hit"].sum())
    recall = hits / labels if labels else 0.0
    return {
        "labels": labels,
        "hits": hits,
        "recall": recall,
    }


def summarize_by_type(merged_df, type_weights):
    rows = []

    for label_type, weight in type_weights.items():
        type_df = merged_df[merged_df["label_type"] == label_type]
        labels = len(type_df)
        hits = int(type_df["hit"].sum())
        recall = hits / labels if labels else 0.0
        contribution = recall * weight

        rows.append({
            "type": label_type,
            "labels": labels,
            "hits": hits,
            "recall": recall,
            "weight": weight,
            "contribution": contribution,
        })

    return rows


def print_summary(pred_file, k, overall, by_type_rows):
    print(f"Evaluating {pred_file}...")
    print(f"Overall Recall@{k}: {overall['recall']:.4f}")
    print()
    print("By Type:")
    print(f"{'type':<8} {'labels':>8} {'hits':>8} {f'recall@{k}':>10} {'weight':>8} {'contribution':>13}")

    for row in by_type_rows:
        print(
            f"{row['type']:<8} "
            f"{row['labels']:>8,} "
            f"{row['hits']:>8,} "
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
    merged = add_hit_column(labels, predictions, args.k)
    overall = summarize_overall(merged)
    by_type_rows = summarize_by_type(merged, TYPE_WEIGHTS)
    print_summary(args.pred_file, args.k, overall, by_type_rows)
