import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rank.build_ranker_train_data import (
    add_history_features,
    add_stat_features,
    cast_existing_feature_dtypes,
    require_columns,
)
from rank.common import get_output_dir


DEFAULT_CANDIDATES_FILE = "test_recall_candidates.parquet"
DEFAULT_TEST_EVENTS_FILE = "test_events.parquet"
DEFAULT_OUTPUT_FILE = "test_ranker_data.parquet"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build multi-target ranker inference data from recall candidates.")
    parser.add_argument("--candidates-file", default=DEFAULT_CANDIDATES_FILE)
    parser.add_argument("--test-events-file", default=DEFAULT_TEST_EVENTS_FILE)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args(argv)


def load_inputs(output_dir, args):
    candidates_path = output_dir / args.candidates_file
    events_path = output_dir / args.test_events_file
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    candidates = pd.read_parquet(candidates_path)
    events = pd.read_parquet(events_path)
    require_columns(candidates, ["session", "type", "aid", "target_type_id"], args.candidates_file)
    require_columns(events, ["session", "aid", "type", "ts"], args.test_events_file)
    return candidates, events


def build_ranker_inference_data(candidates, events):
    duplicate_count = int(candidates.duplicated(["session", "type", "aid"]).sum())
    if duplicate_count:
        raise ValueError(f"Recall candidates contain duplicate (session,type,aid) rows: {duplicate_count:,}")
    if "label" in candidates.columns:
        candidates = candidates.drop(columns=["label"])

    candidates = add_stat_features(candidates, events)
    candidates = add_history_features(candidates, events)
    candidates = cast_existing_feature_dtypes(candidates)
    return candidates


def main(argv=None):
    args = parse_args(argv)
    output_dir = get_output_dir()
    candidates, events = load_inputs(output_dir, args)

    input_rows = len(candidates)
    inference_data = build_ranker_inference_data(candidates, events)
    inference_data.to_parquet(output_dir / args.output_file, index=False)

    duplicate_count = int(inference_data.duplicated(["session", "type", "aid"]).sum())
    group_count = inference_data[["session", "type"]].drop_duplicates().shape[0]
    print(f"Ranker inference data saved to {args.output_file}")
    print(f"Rows: {len(inference_data):,}")
    print(f"Input candidate rows: {input_rows:,}")
    print(f"Groups: {group_count:,}")
    print(f"Contains label column: {'label' in inference_data.columns}")
    print(f"Duplicate (session,type,aid) rows: {duplicate_count:,}")
