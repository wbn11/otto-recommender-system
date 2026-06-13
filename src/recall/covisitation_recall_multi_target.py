import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm


DEFAULT_TRAIN_FILE = "multi_target_train_events.csv"
DEFAULT_LABELS_FILE = "multi_target_valid_labels.csv"
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
    parser.add_argument(
        "--labels-file",
        default=DEFAULT_LABELS_FILE,
        help=f"Validation labels file under outputs/. Default: {DEFAULT_LABELS_FILE}",
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
    return [aid for aid, _ in ranked[:k]]


def load_inputs(output_dir, train_file, labels_file, covis_file):
    train_path = output_dir / train_file
    labels_path = output_dir / labels_file
    covis_path = output_dir / covis_file

    if not train_path.exists():
        raise FileNotFoundError(f"Train events file not found: {train_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Validation labels file not found: {labels_path}")
    if not covis_path.exists():
        raise FileNotFoundError(f"Co-visitation file not found: {covis_path}")

    train_events = pd.read_csv(train_path)
    valid_labels = pd.read_csv(labels_path)

    with open(covis_path, "rb") as f:
        covis_topk = pickle.load(f)

    return train_events, valid_labels, covis_topk


def build_session_recommendations(valid_sessions, session_items, covis_topk, k):
    recs_by_session = {}

    for session in tqdm(valid_sessions, desc="Generating multi-target covis recall"):
        recs = recommend(session_items.get(session, []), covis_topk, k)
        recs_by_session[session] = " ".join(map(str, recs))

    return recs_by_session


def build_predictions(valid_labels, recs_by_session):
    rows = []

    for session, event_type in zip(valid_labels["session"], valid_labels["type"]):
        rows.append({
            "session": session,
            "type": event_type,
            "predictions": recs_by_session.get(session, ""),
        })

    return pd.DataFrame(rows, columns=["session", "type", "predictions"])


def main(argv=None):
    args = parse_args(argv)
    root = Path(__file__).resolve().parent.parent.parent
    output_dir = root / "outputs"

    train_events, valid_labels, covis_topk = load_inputs(
        output_dir,
        args.train_file,
        args.labels_file,
        args.covis_file,
    )

    session_items = (
        train_events.sort_values("ts")
        .groupby("session")["aid"]
        .apply(list)
        .to_dict()
    )
    valid_sessions = valid_labels["session"].drop_duplicates().tolist()
    recs_by_session = build_session_recommendations(valid_sessions, session_items, covis_topk, args.k)
    predictions = build_predictions(valid_labels, recs_by_session)
    predictions.to_csv(output_dir / args.output_file, index=False)

    empty_rows = int((predictions["predictions"] == "").sum())
    print(f"Multi-target covisitation predictions saved to {args.output_file}")
    print(f"Output rows: {len(predictions):,}")
    print(f"Empty prediction rows: {empty_rows:,}")
