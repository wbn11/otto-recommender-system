"""构建 LightGBM 排序训练数据。

读取召回候选池、训练历史和验证标签，为每个候选 item 打 label，
并补充统计特征和 session history 特征。
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rank.common import TYPE2ID, get_output_dir, parse_items


DEFAULT_CANDIDATES_FILE = "recall_candidates.parquet"
DEFAULT_TRAIN_FILE = "train_events.parquet"
DEFAULT_LABELS_FILE = "valid_labels.parquet"
DEFAULT_OUTPUT_FILE = "ranker_train_data.parquet"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build multi-target ranker training data from recall candidates.")
    parser.add_argument("--candidates-file", default=DEFAULT_CANDIDATES_FILE)
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--labels-file", default=DEFAULT_LABELS_FILE)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args(argv)


def require_columns(df, columns, file_name):
    missing_columns = set(columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"{file_name} missing columns: {sorted(missing_columns)}")


def load_inputs(output_dir, args):
    candidates_path = output_dir / args.candidates_file
    train_path = output_dir / args.train_file
    labels_path = output_dir / args.labels_file

    for path in [candidates_path, train_path, labels_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    candidates = pd.read_parquet(candidates_path)
    train_events = pd.read_parquet(train_path)
    labels = pd.read_parquet(labels_path)

    require_columns(candidates, ["session", "type", "aid", "target_type_id"], args.candidates_file)
    require_columns(train_events, ["session", "aid", "type", "ts"], args.train_file)
    require_columns(labels, ["session", "type", "labels"], args.labels_file)
    return candidates, train_events, labels


def build_positive_labels(labels):
    rows = defaultdict(list)

    # Expand each validation label into positive (session, type, aid) triples.
    for session, event_type, label_value in zip(labels["session"], labels["type"], labels["labels"]):
        target_type_id = TYPE2ID.get(event_type, 0)
        if not target_type_id:
            continue

        for aid in parse_items(label_value):
            rows["session"].append(int(session))
            rows["target_type_id"].append(target_type_id)
            rows["aid"].append(int(aid))

    positives = pd.DataFrame(rows)
    if positives.empty:
        return pd.DataFrame(columns=["session", "target_type_id", "aid", "label"])

    positives = positives.drop_duplicates(["session", "target_type_id", "aid"])
    positives["label"] = 1
    positives["target_type_id"] = positives["target_type_id"].astype("int8")
    positives["label"] = positives["label"].astype("int8")
    return positives


def add_labels(candidates, labels):
    positives = build_positive_labels(labels)
    # Keep all recalled candidates; candidates not in positives become negatives.
    candidates = candidates.merge(
        positives,
        on=["session", "target_type_id", "aid"],
        how="left",
        copy=False,
    )
    candidates["label"] = candidates["label"].fillna(0).astype("int8")
    return candidates


def build_event_stats(train_events):
    # Global item/session counts are simple but strong ranking priors.
    item_counts = train_events.groupby("aid", sort=False).size()
    item_type_counts = train_events.groupby(["aid", "type"], sort=False).size().unstack(fill_value=0)
    session_counts = train_events.groupby("session", sort=False).size()
    session_type_counts = train_events.groupby(["session", "type"], sort=False).size().unstack(fill_value=0)

    for event_type in TYPE2ID:
        if event_type not in item_type_counts.columns:
            item_type_counts[event_type] = 0
        if event_type not in session_type_counts.columns:
            session_type_counts[event_type] = 0

    return item_counts, item_type_counts, session_counts, session_type_counts


def add_stat_features(candidates, train_events):
    item_counts, item_type_counts, session_counts, session_type_counts = build_event_stats(train_events)

    candidates["item_popularity"] = candidates["aid"].map(item_counts).fillna(0).astype("int32")
    candidates["session_len"] = candidates["session"].map(session_counts).fillna(0).astype("int16")

    for event_type in TYPE2ID:
        candidates[f"item_{event_type}_count"] = (
            candidates["aid"].map(item_type_counts[event_type]).fillna(0).astype("int32")
        )
        candidates[f"session_{event_type}_count"] = (
            candidates["session"].map(session_type_counts[event_type]).fillna(0).astype("int16")
        )

    return candidates


def build_history_features(train_events):
    events = train_events.sort_values(["session", "ts"], kind="mergesort").copy()
    events["event_pos"] = events.groupby("session", sort=False).cumcount()
    events["session_len_for_pos"] = events.groupby("session", sort=False)["aid"].transform("size")

    # Count repeated interactions and keep the latest occurrence per session-item pair.
    counts = (
        events.groupby(["session", "aid"], sort=False)
        .size()
        .reset_index(name="session_aid_count")
    )
    last_events = events.groupby(["session", "aid"], sort=False).tail(1)
    last_events = last_events[["session", "aid", "type", "event_pos", "session_len_for_pos"]].copy()
    last_events["aid_last_pos_from_end"] = (
        last_events["session_len_for_pos"] - last_events["event_pos"] - 1
    )
    last_events["aid_last_type_id"] = last_events["type"].map(TYPE2ID).fillna(0)

    history = counts.merge(
        last_events[["session", "aid", "aid_last_pos_from_end", "aid_last_type_id"]],
        on=["session", "aid"],
        how="left",
        copy=False,
    )
    history["session_aid_count"] = history["session_aid_count"].astype("int16")
    history["aid_last_pos_from_end"] = history["aid_last_pos_from_end"].astype("int16")
    history["aid_last_type_id"] = history["aid_last_type_id"].astype("int8")
    return history


def add_history_features(candidates, train_events):
    history = build_history_features(train_events)
    candidates = candidates.merge(history, on=["session", "aid"], how="left", copy=False)
    # Missing history means the candidate was recalled from other users/items, not this session.
    candidates["in_session_history"] = candidates["session_aid_count"].notna().astype("int8")
    candidates["session_aid_count"] = candidates["session_aid_count"].fillna(0).astype("int16")
    candidates["aid_last_pos_from_end"] = candidates["aid_last_pos_from_end"].fillna(-1).astype("int16")
    candidates["aid_last_type_id"] = candidates["aid_last_type_id"].fillna(0).astype("int8")
    return candidates


def cast_existing_feature_dtypes(candidates):
    # Compact dtypes reduce memory pressure during LightGBM training.
    int8_columns = [column for column in candidates.columns if column.startswith("from_")]
    int8_columns.extend(["label", "source_count", "target_type_id"])
    for column in int8_columns:
        if column in candidates.columns:
            candidates[column] = candidates[column].fillna(0).astype("int8")

    for column in [column for column in candidates.columns if column.endswith("_rank") or column == "min_rank"]:
        candidates[column] = candidates[column].fillna(0).astype("int16")
    for column in [
        column
        for column in candidates.columns
        if column.endswith("_score") or column.endswith("_score_norm")
    ]:
        candidates[column] = candidates[column].fillna(0).astype("float32")

    return candidates


def build_ranker_train_data(candidates, train_events, labels):
    duplicate_count = int(candidates.duplicated(["session", "type", "aid"]).sum())
    if duplicate_count:
        raise ValueError(f"Recall candidates contain duplicate (session,type,aid) rows: {duplicate_count:,}")

    candidates = add_labels(candidates, labels)
    candidates = add_stat_features(candidates, train_events)
    candidates = add_history_features(candidates, train_events)
    candidates = cast_existing_feature_dtypes(candidates)
    return candidates


def main(argv=None):
    args = parse_args(argv)
    output_dir = get_output_dir()
    candidates, train_events, labels = load_inputs(output_dir, args)

    input_rows = len(candidates)
    train_data = build_ranker_train_data(candidates, train_events, labels)
    output_path = output_dir / args.output_file
    train_data.to_parquet(output_path, index=False)

    duplicate_count = int(train_data.duplicated(["session", "type", "aid"]).sum())
    group_count = train_data[["session", "type"]].drop_duplicates().shape[0]
    positive_count = int(train_data["label"].sum())
    print(f"Ranker train data saved to {args.output_file}")
    print(f"Rows: {len(train_data):,}")
    print(f"Input candidate rows: {input_rows:,}")
    print(f"Groups: {group_count:,}")
    print(f"Positive rows: {positive_count:,}")
    print(f"Duplicate (session,type,aid) rows: {duplicate_count:,}")
