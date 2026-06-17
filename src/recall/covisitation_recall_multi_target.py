import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.target_rows import load_target_rows_from_file


DEFAULT_TRAIN_FILE = "multi_target_train_events.parquet"
DEFAULT_LABELS_FILE = "multi_target_valid_labels.parquet"
DEFAULT_COVIS_FILE = "multi_target_covis_topk.pkl"
DEFAULT_OUTPUT_FILE = "multi_target_covisitation_predictions.csv"
DEFAULT_K = 20


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate multi-target co-visitation recall.")
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
        "--covis-file",
        default=DEFAULT_COVIS_FILE,
        help=f"Co-visitation pickle file under outputs/. Default: {DEFAULT_COVIS_FILE}",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Prediction output file under outputs/. Default: {DEFAULT_OUTPUT_FILE}",
    )
    parser.add_argument(
        "--detail-file",
        help="Optional parquet output with columns session,type,aid,rank,covis_score.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Number of items per prediction row. Default: {DEFAULT_K}",
    )
    return parser.parse_args(argv)


def recommend(session_items, covis_topk, k):
    scores = defaultdict(float)

    for idx, aid in enumerate(reversed(session_items)):
        weight = 1.0 / (idx + 1)
        for neighbor, count in covis_topk.get(aid, []):
            scores[neighbor] += weight * count

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]


def load_inputs(output_dir, train_file, labels_file, test_events_file, covis_file):
    train_path = output_dir / train_file
    covis_path = output_dir / covis_file

    if not train_path.exists():
        raise FileNotFoundError(f"Train events file not found: {train_path}")
    if not covis_path.exists():
        raise FileNotFoundError(f"Co-visitation file not found: {covis_path}")

    train_events = pd.read_parquet(train_path)
    target_rows = load_target_rows_from_file(output_dir, labels_file, test_events_file)

    with open(covis_path, "rb") as f:
        covis_topk = pickle.load(f)

    return train_events, target_rows, covis_topk


def build_session_recommendations(valid_sessions, session_items, covis_topk, k):
    recs_by_session = {}

    for session in tqdm(valid_sessions, desc="Generating multi-target covis recall"):
        recs = recommend(session_items.get(session, []), covis_topk, k)
        recs_by_session[session] = recs

    return recs_by_session


def build_predictions(target_rows, recs_by_session):
    rows = []

    for session, event_type in zip(target_rows["session"], target_rows["type"]):
        recs = recs_by_session.get(session, [])
        rows.append({
            "session": session,
            "type": event_type,
            "predictions": " ".join(str(aid) for aid, _ in recs),
        })

    return pd.DataFrame(rows, columns=["session", "type", "predictions"])


def build_details(target_rows, recs_by_session):
    rows = defaultdict(list)

    for session, event_type in zip(target_rows["session"], target_rows["type"]):
        for rank, (aid, score) in enumerate(recs_by_session.get(session, []), start=1):
            rows["session"].append(session)
            rows["type"].append(event_type)
            rows["aid"].append(aid)
            rows["rank"].append(rank)
            rows["covis_score"].append(score)

    details = pd.DataFrame(rows, columns=["session", "type", "aid", "rank", "covis_score"])
    if not details.empty:
        details["rank"] = details["rank"].astype("int16")
        details["covis_score"] = details["covis_score"].astype("float32")
    return details


def main(argv=None):
    args = parse_args(argv)
    root = Path(__file__).resolve().parent.parent.parent
    output_dir = root / "outputs"

    labels_file = args.labels_file if args.test_events_file else (args.labels_file or DEFAULT_LABELS_FILE)
    train_events, target_rows, covis_topk = load_inputs(
        output_dir,
        args.train_file,
        labels_file,
        args.test_events_file,
        args.covis_file,
    )

    session_items = (
        train_events.sort_values("ts")
        .groupby("session")["aid"]
        .apply(list)
        .to_dict()
    )
    target_sessions = target_rows["session"].drop_duplicates().tolist()
    recs_by_session = build_session_recommendations(target_sessions, session_items, covis_topk, args.k)
    predictions = build_predictions(target_rows, recs_by_session)
    predictions.to_csv(output_dir / args.output_file, index=False)

    if args.detail_file:
        details = build_details(target_rows, recs_by_session)
        details.to_parquet(output_dir / args.detail_file, index=False)
        print(f"Multi-target covisitation details saved to {args.detail_file}")
        print(f"Detail rows: {len(details):,}")

    empty_rows = int((predictions["predictions"] == "").sum())
    print(f"Multi-target covisitation predictions saved to {args.output_file}")
    print(f"Output rows: {len(predictions):,}")
    print(f"Empty prediction rows: {empty_rows:,}")
