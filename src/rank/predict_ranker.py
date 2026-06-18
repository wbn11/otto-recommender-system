import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rank.common import build_predictions_from_scores, get_output_dir
from utils.target_rows import load_target_rows_from_file


DEFAULT_CANDIDATES_FILE = "ranker_train_data.parquet"
DEFAULT_LABELS_FILE = "valid_labels.parquet"
DEFAULT_MODEL_FILE = "lgbm_ranker.txt"
DEFAULT_OUTPUT_FILE = "ranker_predictions.csv"
DEFAULT_K = 20


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Predict multi-target LightGBM ranker results.")
    parser.add_argument("--candidates-file", default=DEFAULT_CANDIDATES_FILE)
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--labels-file", help=f"Validation labels file under outputs/. Default: {DEFAULT_LABELS_FILE}")
    target_group.add_argument("--test-events-file", help="Test events file under outputs/. Target rows are expanded to all types.")
    parser.add_argument("--model-file", default=DEFAULT_MODEL_FILE)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    return parser.parse_args(argv)


def load_lightgbm():
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            "lightgbm is not installed. Activate the project environment and run "
            "`python -m pip install lightgbm` or `python -m pip install -r requirements.txt`."
        ) from exc
    return lgb


def main(argv=None):
    args = parse_args(argv)
    lgb = load_lightgbm()
    output_dir = get_output_dir()

    candidates_path = output_dir / args.candidates_file
    labels_file = args.labels_file if args.test_events_file else (args.labels_file or DEFAULT_LABELS_FILE)
    model_path = output_dir / args.model_file
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    candidates = pd.read_parquet(candidates_path)
    target_rows = load_target_rows_from_file(output_dir, labels_file, args.test_events_file)
    model = lgb.Booster(model_file=str(model_path))
    feature_columns = model.feature_name()

    missing_features = set(feature_columns) - set(candidates.columns)
    if missing_features:
        raise ValueError(f"Candidates missing model features: {sorted(missing_features)}")

    candidates["ranker_score"] = model.predict(
        candidates[feature_columns].fillna(0),
        num_iteration=model.best_iteration,
    )
    predictions = build_predictions_from_scores(candidates, target_rows, "ranker_score", args.k)
    predictions.to_csv(output_dir / args.output_file, index=False)

    print(f"Ranker predictions saved to {args.output_file}")
    print(f"Rows: {len(predictions):,}")
