import argparse
from pathlib import Path

import pandas as pd

DEFAULT_PRED_FILE = "fusion_predictions.csv"
DEFAULT_LABELS_FILE = "valid_labels.csv"
DEFAULT_K = 20


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
    """字符串转商品列表"""
    if pd.isna(pred_str):
        return []
    return list(map(int, pred_str.split()))

def recall_at_k(labels_df,predictions_df,k):
    """计算 Recall@K"""
    merged = labels_df.merge(predictions_df,on="session",how="left")

    hits = 0
    for _, row in merged.iterrows():
        label_aid = row["label_aid"]
        pred_list = parse_predictions(row["predictions"])
        if label_aid in pred_list[:k]:
            hits += 1
    recall = hits / len(merged)
    return recall


def main(argv=None):
    args = parse_args(argv)
    ROOT = Path(__file__).resolve().parent.parent.parent
    output_dir = ROOT / "outputs"
    labels_path = output_dir / args.labels_file
    predictions_path = output_dir / args.pred_file

    labels = pd.read_csv(labels_path)
    predictions = pd.read_csv(predictions_path)

    recall = recall_at_k(labels,predictions,k=args.k)
    print(f"Evaluating {args.pred_file}...")
    print(f"Recall@{args.k}: {recall:.4f}")

