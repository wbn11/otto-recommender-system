import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rank.common import (
    TYPE2ID,
    TYPE_WEIGHTS,
    get_output_dir,
    parse_items,
    parse_source_args,
    load_prediction_map,
)


DEFAULT_TRAIN_FILE = "multi_target_train_events.parquet"
DEFAULT_LABELS_FILE = "multi_target_valid_labels.parquet"
DEFAULT_OUTPUT_FILE = "multi_target_ranker_candidates.parquet"
DEFAULT_K = 20
DEFAULT_EVAL_K = 20


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build multi-target ranker candidates.")
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--labels-file", default=DEFAULT_LABELS_FILE)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Max candidates loaded from each source.")
    parser.add_argument("--eval-k", type=int, default=DEFAULT_EVAL_K, help="K used for candidate oracle recall.")
    parser.add_argument(
        "--source",
        action="append",
        help="Recall source in name=file format. Can be passed multiple times.",
    )
    parser.add_argument(
        "--source-detail",
        action="append",
        help="Optional raw-score detail source in name=file format. Used by --feature-group raw.",
    )
    parser.add_argument(
        "--feature-group",
        action="append",
        choices=["overlap", "raw", "history"],
        help="Additional feature group to enable. Can be passed multiple times.",
    )
    return parser.parse_args(argv)


def load_inputs(output_dir, args):
    train_path = output_dir / args.train_file
    labels_path = output_dir / args.labels_file
    if not train_path.exists():
        raise FileNotFoundError(f"Train events file not found: {train_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    train_events = pd.read_parquet(train_path)
    labels = pd.read_parquet(labels_path)
    sources = parse_source_args(args.source)
    source_details = parse_source_args(args.source_detail) if args.source_detail else {}
    prediction_maps = {
        source_name: load_prediction_map(output_dir, file_name, args.k)
        for source_name, file_name in sources.items()
    }
    return train_events, labels, sources, source_details, prediction_maps


def build_event_stats(train_events):
    item_counts = train_events.groupby("aid").size()
    item_type_counts = train_events.groupby(["aid", "type"]).size().unstack(fill_value=0)
    session_counts = train_events.groupby("session").size()
    session_type_counts = train_events.groupby(["session", "type"]).size().unstack(fill_value=0)

    for event_type in TYPE2ID:
        if event_type not in item_type_counts.columns:
            item_type_counts[event_type] = 0
        if event_type not in session_type_counts.columns:
            session_type_counts[event_type] = 0

    return item_counts, item_type_counts, session_counts, session_type_counts


def append_candidate_rows(labels, prediction_maps, source_names, source_k, eval_k):
    rows = defaultdict(list)
    oracle_hits = {event_type: 0 for event_type in TYPE_WEIGHTS}
    oracle_denominators = {event_type: 0 for event_type in TYPE_WEIGHTS}

    source_feature_names = []
    for source_name in source_names:
        source_feature_names.extend([
            f"from_{source_name}",
            f"{source_name}_rank",
            f"{source_name}_score",
        ])

    for session, event_type, label_value in tqdm(
        zip(labels["session"], labels["type"], labels["labels"]),
        total=len(labels),
        desc="Building ranker candidates",
    ):
        session = int(session)
        true_items = set(parse_items(label_value))
        candidates = {}

        for source_name in source_names:
            items = prediction_maps[source_name].get((session, event_type), [])
            for rank, aid in enumerate(items[:source_k], start=1):
                candidate = candidates.setdefault(
                    aid,
                    {feature_name: 0 for feature_name in source_feature_names},
                )
                candidate[f"from_{source_name}"] = 1
                candidate[f"{source_name}_rank"] = rank
                candidate[f"{source_name}_score"] = 1.0 / rank

        oracle_hits[event_type] += min(len(true_items.intersection(candidates)), eval_k)
        oracle_denominators[event_type] += min(len(true_items), eval_k)

        for aid, features in candidates.items():
            rows["session"].append(session)
            rows["type"].append(event_type)
            rows["aid"].append(int(aid))
            rows["label"].append(1 if aid in true_items else 0)
            rows["target_type_id"].append(TYPE2ID[event_type])
            for feature_name in source_feature_names:
                rows[feature_name].append(features[feature_name])

    return pd.DataFrame(rows), oracle_hits, oracle_denominators


def add_stat_features(candidates, item_counts, item_type_counts, session_counts, session_type_counts):
    candidates["item_popularity"] = candidates["aid"].map(item_counts).fillna(0).astype("int32")
    candidates["session_len"] = candidates["session"].map(session_counts).fillna(0).astype("int16")

    for event_type in TYPE2ID:
        candidates[f"item_{event_type}_count"] = (
            candidates["aid"].map(item_type_counts[event_type]).fillna(0).astype("int32")
        )
        candidates[f"session_{event_type}_count"] = (
            candidates["session"].map(session_type_counts[event_type]).fillna(0).astype("int16")
        )

    int8_columns = [col for col in candidates.columns if col.startswith("from_")]
    int8_columns.append("label")
    int8_columns.append("target_type_id")
    for column in int8_columns:
        candidates[column] = candidates[column].astype("int8")

    for column in [col for col in candidates.columns if col.endswith("_rank")]:
        candidates[column] = candidates[column].astype("int16")
    for column in [col for col in candidates.columns if col.endswith("_score")]:
        candidates[column] = candidates[column].astype("float32")

    return candidates


def add_overlap_features(candidates, source_names):
    from_columns = [f"from_{source_name}" for source_name in source_names]
    rank_columns = [f"{source_name}_rank" for source_name in source_names]
    score_columns = [f"{source_name}_score" for source_name in source_names]

    candidates["source_count"] = candidates[from_columns].sum(axis=1).astype("int8")
    rank_values = candidates[rank_columns].where(candidates[rank_columns] > 0)
    candidates["min_rank"] = rank_values.min(axis=1).fillna(0).astype("int16")
    candidates["rrf_score"] = candidates[score_columns].sum(axis=1).astype("float32")
    return candidates


def find_detail_score_column(details, source_name):
    candidates = [
        f"{source_name}_raw_score",
        f"{source_name}_score",
        "score",
    ]
    for column in candidates:
        if column in details.columns:
            return column
    raise ValueError(
        f"Detail file for {source_name} must include one of these score columns: {candidates}"
    )


def add_raw_score_features(candidates, output_dir, source_details):
    if not source_details:
        raise ValueError("--feature-group raw requires at least one --source-detail name=file argument.")

    for source_name, file_name in source_details.items():
        detail_path = output_dir / file_name
        if not detail_path.exists():
            raise FileNotFoundError(f"Source detail file not found: {detail_path}")

        details = pd.read_parquet(detail_path)
        required_columns = {"session", "type", "aid"}
        missing_columns = required_columns - set(details.columns)
        if missing_columns:
            raise ValueError(f"{detail_path} missing columns: {sorted(missing_columns)}")

        score_column = find_detail_score_column(details, source_name)
        raw_column = f"{source_name}_raw_score"
        details = details[["session", "type", "aid", score_column]].rename(
            columns={score_column: raw_column}
        )
        details = details.drop_duplicates(["session", "type", "aid"], keep="first")
        candidates = candidates.merge(details, on=["session", "type", "aid"], how="left")
        candidates[raw_column] = candidates[raw_column].fillna(0).astype("float32")

    return candidates


def build_history_features(train_events):
    events = train_events.sort_values(["session", "ts"], kind="mergesort").copy()
    events["event_pos"] = events.groupby("session", sort=False).cumcount()
    events["session_len_for_pos"] = events.groupby("session", sort=False)["aid"].transform("size")

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
    )
    history["session_aid_count"] = history["session_aid_count"].astype("int16")
    history["aid_last_pos_from_end"] = history["aid_last_pos_from_end"].astype("int16")
    history["aid_last_type_id"] = history["aid_last_type_id"].astype("int8")
    return history


def add_history_features(candidates, train_events):
    history = build_history_features(train_events)
    candidates = candidates.merge(history, on=["session", "aid"], how="left")
    candidates["in_session_history"] = candidates["session_aid_count"].notna().astype("int8")
    candidates["session_aid_count"] = candidates["session_aid_count"].fillna(0).astype("int16")
    candidates["aid_last_pos_from_end"] = candidates["aid_last_pos_from_end"].fillna(-1).astype("int16")
    candidates["aid_last_type_id"] = candidates["aid_last_type_id"].fillna(0).astype("int8")
    return candidates


def apply_feature_groups(candidates, train_events, output_dir, source_names, source_details, feature_groups):
    feature_groups = set(feature_groups or [])
    if "overlap" in feature_groups:
        candidates = add_overlap_features(candidates, source_names)
    if "raw" in feature_groups:
        candidates = add_raw_score_features(candidates, output_dir, source_details)
    if "history" in feature_groups:
        candidates = add_history_features(candidates, train_events)
    return candidates


def print_oracle_summary(oracle_hits, oracle_denominators, eval_k):
    print(f"Candidate oracle Recall@{eval_k}")
    weighted_score = 0.0
    for event_type, type_weight in TYPE_WEIGHTS.items():
        denominator = oracle_denominators[event_type]
        recall = oracle_hits[event_type] / denominator if denominator else 0.0
        weighted_score += recall * type_weight
        print(
            f"{event_type}: recall={recall:.4f}, "
            f"hits={oracle_hits[event_type]:,}, total={denominator:,}"
        )
    print(f"Weighted Score: {weighted_score:.4f}")


def main(argv=None):
    args = parse_args(argv)
    output_dir = get_output_dir()
    train_events, labels, sources, source_details, prediction_maps = load_inputs(output_dir, args)
    source_names = list(sources)
    feature_groups = args.feature_group or []

    item_counts, item_type_counts, session_counts, session_type_counts = build_event_stats(train_events)
    candidates, oracle_hits, oracle_denominators = append_candidate_rows(
        labels=labels,
        prediction_maps=prediction_maps,
        source_names=source_names,
        source_k=args.k,
        eval_k=args.eval_k,
    )
    candidates = add_stat_features(
        candidates,
        item_counts=item_counts,
        item_type_counts=item_type_counts,
        session_counts=session_counts,
        session_type_counts=session_type_counts,
    )
    candidates = apply_feature_groups(
        candidates=candidates,
        train_events=train_events,
        output_dir=output_dir,
        source_names=source_names,
        source_details=source_details,
        feature_groups=feature_groups,
    )

    output_path = output_dir / args.output_file
    candidates.to_parquet(output_path, index=False)

    print(f"Ranker candidates saved to {args.output_file}")
    print(f"Rows: {len(candidates):,}")
    print(f"Groups: {candidates[['session', 'type']].drop_duplicates().shape[0]:,}")
    print(f"Positive rows: {int(candidates['label'].sum()):,}")
    print(f"Sources: {', '.join(source_names)}")
    print(f"Feature groups: {', '.join(feature_groups) if feature_groups else 'base'}")
    print_oracle_summary(oracle_hits, oracle_denominators, args.eval_k)
