"""合并多路召回候选。

读取 popular、co-visitation、DSSM 预测结果，
按 session/type/aid 合并为排序候选池并生成召回源特征。
"""

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.target_rows import parse_unique_prediction_items


TYPE2ID = {
    "clicks": 1,
    "carts": 2,
    "orders": 3,
}
TYPE_ORDER = tuple(TYPE2ID)
SOURCE_FILES = {
    "popular": "popular_predictions.csv",
    "covis": "covisitation_predictions.csv",
    "dssm": "dssm_predictions.csv",
}
SOURCE_COLUMNS = [
    "from_popular", "popular_rank", "popular_score",
    "from_covis", "covis_rank", "covis_score",
    "from_dssm", "dssm_rank", "dssm_score",
]
RAW_SCORE_NORM_COLUMNS = ["covis_raw_score_norm", "dssm_raw_score_norm"]
BASE_OUTPUT_COLUMNS = [
    "session", "type", "aid",
    *SOURCE_COLUMNS,
    "source_count", "min_rank", "rrf_score", "target_type_id",
]
DEFAULT_OUTPUT_FILE = "recall_candidates.parquet"
DEFAULT_K = 50


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build merged multi-target recall candidates.")
    parser.add_argument("--popular-file", default=SOURCE_FILES["popular"])
    parser.add_argument("--covis-file", default=SOURCE_FILES["covis"])
    parser.add_argument("--dssm-file", default=SOURCE_FILES["dssm"])
    parser.add_argument("--covis-detail-file", help="Optional covis detail parquet with raw covis scores.")
    parser.add_argument("--dssm-detail-file", help="Optional DSSM detail parquet with raw cosine scores.")
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Max candidates loaded from each source.")
    return parser.parse_args(argv)


def load_predictions(output_dir, file_name, k):
    path = output_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_csv(path)
    required_columns = {"session", "type", "predictions"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{path} missing columns: {sorted(missing_columns)}")

    return df[["session", "type", "predictions"]].copy()


def empty_candidate_features():
    return {column: 0 for column in SOURCE_COLUMNS}


def add_source_candidates(candidates, predictions, source_name, k):
    from_col = f"from_{source_name}"
    rank_col = f"{source_name}_rank"
    score_col = f"{source_name}_score"

    for session, event_type, prediction in tqdm(
        zip(predictions["session"], predictions["type"], predictions["predictions"]),
        total=len(predictions),
        desc=f"Loading {source_name} candidates",
        leave=False,
    ):
        session = int(session)
        for rank, aid in enumerate(parse_unique_prediction_items(prediction)[:k], start=1):
            # One candidate row can accumulate features from multiple recall sources.
            key = (session, event_type, int(aid))
            candidate = candidates.setdefault(key, empty_candidate_features())
            candidate[from_col] = 1
            candidate[rank_col] = rank
            candidate[score_col] = 1.0 / rank


def find_detail_score_column(details, source_name):
    score_columns = [f"{source_name}_score", "score"]
    for column in score_columns:
        if column in details.columns:
            return column
    raise ValueError(f"{source_name} detail file must include one of: {score_columns}")


def add_normalized_raw_scores(candidates, details, source_name):
    required_columns = {"session", "type", "aid"}
    missing_columns = required_columns - set(details.columns)
    if missing_columns:
        raise ValueError(f"{source_name} detail file missing columns: {sorted(missing_columns)}")

    score_column = find_detail_score_column(details, source_name)
    raw_norm_column = f"{source_name}_raw_score_norm"
    details = details[["session", "type", "aid", score_column]].copy()

    grouped_scores = details.groupby(["session", "type"], sort=False)[score_column]
    min_scores = grouped_scores.transform("min")
    max_scores = grouped_scores.transform("max")
    denominator = max_scores - min_scores
    # Normalize raw source scores within each target row so LightGBM can compare them safely.
    details[raw_norm_column] = np.where(
        denominator > 0,
        (details[score_column] - min_scores) / denominator,
        1.0,
    ).astype("float32")

    matched_rows = 0
    for session, event_type, aid, score in tqdm(
        zip(details["session"], details["type"], details["aid"], details[raw_norm_column]),
        total=len(details),
        desc=f"Loading {source_name} normalized raw scores",
        leave=False,
    ):
        candidate = candidates.get((int(session), event_type, int(aid)))
        if candidate is None:
            continue
        candidate[raw_norm_column] = float(score)
        matched_rows += 1

    return matched_rows


def load_and_add_normalized_raw_scores(output_dir, candidates, source_detail_files):
    matched_counts = {}
    for source_name, file_name in source_detail_files.items():
        if not file_name:
            continue

        path = output_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"{source_name} detail file not found: {path}")

        details = pd.read_parquet(path)
        matched_counts[source_name] = add_normalized_raw_scores(candidates, details, source_name)

    return matched_counts


def get_output_columns(include_raw_score_norm):
    if not include_raw_score_norm:
        return BASE_OUTPUT_COLUMNS

    return [
        "session", "type", "aid",
        *SOURCE_COLUMNS,
        *RAW_SCORE_NORM_COLUMNS,
        "source_count", "min_rank", "rrf_score", "target_type_id",
    ]


def candidates_to_frame(candidates, include_raw_score_norm):
    output_columns = get_output_columns(include_raw_score_norm)
    row_count = len(candidates)
    if not row_count:
        return pd.DataFrame(columns=output_columns)

    columns = {
        "session": np.empty(row_count, dtype=np.int64),
        "aid": np.empty(row_count, dtype=np.int64),
        "from_popular": np.zeros(row_count, dtype=np.int8),
        "popular_rank": np.zeros(row_count, dtype=np.int16),
        "popular_score": np.zeros(row_count, dtype=np.float32),
        "from_covis": np.zeros(row_count, dtype=np.int8),
        "covis_rank": np.zeros(row_count, dtype=np.int16),
        "covis_score": np.zeros(row_count, dtype=np.float32),
        "from_dssm": np.zeros(row_count, dtype=np.int8),
        "dssm_rank": np.zeros(row_count, dtype=np.int16),
        "dssm_score": np.zeros(row_count, dtype=np.float32),
        "covis_raw_score_norm": np.zeros(row_count, dtype=np.float32),
        "dssm_raw_score_norm": np.zeros(row_count, dtype=np.float32),
        "source_count": np.zeros(row_count, dtype=np.int8),
        "min_rank": np.zeros(row_count, dtype=np.int16),
        "rrf_score": np.zeros(row_count, dtype=np.float32),
        "target_type_id": np.zeros(row_count, dtype=np.int8),
    }
    type_codes = np.zeros(row_count, dtype=np.int8)

    for idx, ((session, event_type, aid), features) in enumerate(candidates.items()):
        # RRF-style source features summarize how many channels recalled the same item.
        source_flags = [features[f"from_{source_name}"] for source_name in SOURCE_FILES]
        ranks = [
            features[f"{source_name}_rank"]
            for source_name in SOURCE_FILES
            if features[f"{source_name}_rank"] > 0
        ]
        scores = [features[f"{source_name}_score"] for source_name in SOURCE_FILES]
        target_type_id = TYPE2ID.get(event_type, 0)

        columns["session"][idx] = session
        type_codes[idx] = target_type_id
        columns["aid"][idx] = aid
        for column in SOURCE_COLUMNS:
            columns[column][idx] = features[column]
        if include_raw_score_norm:
            for column in RAW_SCORE_NORM_COLUMNS:
                columns[column][idx] = features.get(column, 0.0)
        columns["source_count"][idx] = sum(source_flags)
        columns["min_rank"][idx] = min(ranks) if ranks else 0
        columns["rrf_score"][idx] = sum(scores)
        columns["target_type_id"][idx] = target_type_id

    candidates.clear()
    gc.collect()

    columns["type"] = pd.Categorical.from_codes(type_codes - 1, categories=TYPE_ORDER)
    return pd.DataFrame(columns, columns=output_columns, copy=False)


def build_recall_candidates(output_dir, source_files, source_detail_files, k):
    candidates = {}
    for source_name, file_name in source_files.items():
        predictions = load_predictions(output_dir, file_name, k)
        add_source_candidates(candidates, predictions, source_name, k)

    matched_counts = load_and_add_normalized_raw_scores(output_dir, candidates, source_detail_files)
    include_raw_score_norm = any(source_detail_files.values())
    return candidates_to_frame(candidates, include_raw_score_norm), matched_counts


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(__file__).resolve().parents[2] / "outputs"
    source_files = {
        "popular": args.popular_file,
        "covis": args.covis_file,
        "dssm": args.dssm_file,
    }
    source_detail_files = {
        "covis": args.covis_detail_file,
        "dssm": args.dssm_detail_file,
    }

    candidates, matched_counts = build_recall_candidates(output_dir, source_files, source_detail_files, args.k)
    candidates.to_parquet(output_dir / args.output_file, index=False)

    duplicate_count = int(candidates.duplicated(["session", "type", "aid"]).sum()) if not candidates.empty else 0
    group_count = candidates[["session", "type"]].drop_duplicates().shape[0] if not candidates.empty else 0
    print(f"Recall candidates saved to {args.output_file}")
    print(f"Rows: {len(candidates):,}")
    print(f"Groups: {group_count:,}")
    print(f"Duplicate (session,type,aid) rows: {duplicate_count:,}")
    print(f"Sources: {', '.join(source_files)}")
    for source_name, matched_count in matched_counts.items():
        print(f"{source_name} normalized raw score rows matched: {matched_count:,}")
