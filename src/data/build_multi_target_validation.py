import argparse
from pathlib import Path

import pandas as pd


EVENT_TYPES = ("clicks", "carts", "orders")
DEFAULT_NROWS = 100000
DEFAULT_HISTORY_RATIO = 0.8
DEFAULT_TRAIN_FILE = "multi_target_train_events.csv"
DEFAULT_LABELS_FILE = "multi_target_valid_labels.csv"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build multi-target validation data.")
    parser.add_argument(
        "--nrows",
        type=int,
        default=DEFAULT_NROWS,
        help=f"Number of sessions to read from raw jsonl. Default: {DEFAULT_NROWS}",
    )
    parser.add_argument(
        "--history-ratio",
        type=float,
        default=DEFAULT_HISTORY_RATIO,
        help=f"Session prefix ratio used as history. Default: {DEFAULT_HISTORY_RATIO}",
    )
    parser.add_argument(
        "--train-file",
        default=DEFAULT_TRAIN_FILE,
        help=f"Output train events file under outputs/. Default: {DEFAULT_TRAIN_FILE}",
    )
    parser.add_argument(
        "--labels-file",
        default=DEFAULT_LABELS_FILE,
        help=f"Output validation labels file under outputs/. Default: {DEFAULT_LABELS_FILE}",
    )
    return parser.parse_args(argv)


def load_data(data_file, nrows=None):
    return pd.read_json(data_file, lines=True, nrows=nrows)


def expand_events(df):
    rows = []
    for _, row in df.iterrows():
        session_id = row["session"]

        for event in row["events"]:
            rows.append({
                "session": session_id,
                "aid": event["aid"],
                "ts": event["ts"],
                "type": event["type"],
            })

    return pd.DataFrame(rows)


def split_history_future(group, history_ratio):
    group = group.sort_values("ts")
    split_idx = int(len(group) * history_ratio)
    split_idx = max(1, min(split_idx, len(group) - 1))
    return group.iloc[:split_idx], group.iloc[split_idx:]


def unique_in_order(values):
    seen = set()
    unique_values = []

    for value in values:
        if value in seen:
            continue

        unique_values.append(value)
        seen.add(value)

    return unique_values


def build_multi_target_validation(events_df, history_ratio):
    train_parts = []
    label_rows = []
    skipped_sessions = 0

    for session, group in events_df.groupby("session"):
        if len(group) < 2:
            skipped_sessions += 1
            continue

        history, future = split_history_future(group, history_ratio)
        train_parts.append(history)

        for event_type in EVENT_TYPES:
            labels = unique_in_order(future.loc[future["type"] == event_type, "aid"].tolist())
            if not labels:
                continue

            label_rows.append({
                "session": session,
                "type": event_type,
                "labels": " ".join(map(str, labels)),
            })

    train_events = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    valid_labels = pd.DataFrame(label_rows, columns=["session", "type", "labels"])
    return train_events, valid_labels, skipped_sessions


def print_summary(train_events, valid_labels, skipped_sessions):
    print(f"Train events: {len(train_events):,}")
    print(f"Label rows: {len(valid_labels):,}")
    print(f"Train sessions: {train_events['session'].nunique():,}" if not train_events.empty else "Train sessions: 0")
    print(f"Label sessions: {valid_labels['session'].nunique():,}" if not valid_labels.empty else "Label sessions: 0")
    print(f"Skipped sessions: {skipped_sessions:,}")

    if valid_labels.empty:
        return

    print("Labels by type:")
    for event_type in EVENT_TYPES:
        type_labels = valid_labels[valid_labels["type"] == event_type]
        label_count = type_labels["labels"].str.split().str.len().sum() if not type_labels.empty else 0
        print(f"  {event_type:<6} rows={len(type_labels):,} labels={int(label_count):,}")


def main(argv=None):
    args = parse_args(argv)

    if not 0 < args.history_ratio < 1:
        raise ValueError("--history-ratio must be between 0 and 1")

    root = Path(__file__).resolve().parent.parent.parent
    data_file = root / "data" / "otto-recsys-train.jsonl"
    output_dir = root / "outputs"
    output_dir.mkdir(exist_ok=True)

    df = load_data(data_file, nrows=args.nrows)
    events_df = expand_events(df)
    train_events, valid_labels, skipped_sessions = build_multi_target_validation(
        events_df,
        args.history_ratio,
    )

    train_events.to_csv(output_dir / args.train_file, index=False)
    valid_labels.to_csv(output_dir / args.labels_file, index=False)

    print_summary(train_events, valid_labels, skipped_sessions)
