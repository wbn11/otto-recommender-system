"""生成 Kaggle submission 文件。

读取 session/type/predictions 格式的测试预测，
转换为 Kaggle 要求的 session_type/labels 格式。
"""

import argparse
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.target_rows import normalize_prediction_items


DEFAULT_PRED_FILE = "test_ranker_predictions.csv"
DEFAULT_OUTPUT_FILE = "submission.csv"
DEFAULT_K = 20


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build Kaggle submission from multi-target predictions.")
    parser.add_argument("--pred-file", default=DEFAULT_PRED_FILE)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    return parser.parse_args(argv)


def build_submission(predictions, k):
    required_columns = {"session", "type", "predictions"}
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise ValueError(f"predictions missing columns: {sorted(missing_columns)}")

    # Kaggle expects the target type to be encoded in one session_type column.
    submission = pd.DataFrame({
        "session_type": predictions["session"].astype(str) + "_" + predictions["type"].astype(str),
        "labels": predictions["predictions"].apply(lambda value: normalize_prediction_items(value, k)),
    })
    return submission[["session_type", "labels"]]


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(__file__).resolve().parents[2] / "outputs"
    pred_path = output_dir / args.pred_file
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    predictions = pd.read_csv(pred_path)
    submission = build_submission(predictions, args.k)
    submission.to_csv(output_dir / args.output_file, index=False)

    empty_rows = int((submission["labels"] == "").sum())
    print(f"Submission saved to {args.output_file}")
    print(f"Rows: {len(submission):,}")
    print(f"Empty label rows: {empty_rows:,}")
