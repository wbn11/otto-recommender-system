from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_SOURCE_FILES = {
    "popular": "multi_target_popular_predictions.csv",
    "covis": "multi_target_covisitation_predictions.csv",
    "dssm": "multi_target_dssm_predictions.csv",
}
TYPE2ID = {
    "clicks": 1,
    "carts": 2,
    "orders": 3,
}
TYPE_WEIGHTS = {
    "clicks": 0.10,
    "carts": 0.30,
    "orders": 0.60,
}
ID_COLUMNS = {"session", "type", "aid", "label"}


def parse_items(value):
    if pd.isna(value):
        return []

    items = []
    seen = set()
    for token in str(value).split():
        if not token.isdigit() or token in seen:
            continue
        item = int(token)
        items.append(item)
        seen.add(token)
    return items


def parse_source_args(source_args):
    if not source_args:
        return DEFAULT_SOURCE_FILES.copy()

    sources = {}
    for value in source_args:
        if "=" not in value:
            raise ValueError(f"Source must use name=file format: {value}")
        name, file_name = value.split("=", 1)
        name = name.strip()
        file_name = file_name.strip()
        if not name or not file_name:
            raise ValueError(f"Source must use name=file format: {value}")
        sources[name] = file_name

    return sources


def load_prediction_map(output_dir, file_name, k):
    path = output_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_csv(path)
    required_columns = {"session", "type", "predictions"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{path} missing columns: {sorted(missing_columns)}")

    predictions = {}
    for session, event_type, prediction in zip(df["session"], df["type"], df["predictions"]):
        predictions[(int(session), event_type)] = parse_items(prediction)[:k]

    return predictions


def get_feature_columns(df):
    return [col for col in df.columns if col not in ID_COLUMNS]


def build_predictions_from_scores(candidates, labels, score_col, k):
    ranked = candidates.sort_values(
        ["session", "type", score_col, "aid"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    top = ranked.groupby(["session", "type"], sort=False).head(k)
    predictions = (
        top.groupby(["session", "type"], sort=False)["aid"]
        .apply(lambda values: " ".join(map(str, values.tolist())))
        .reset_index(name="predictions")
    )

    rows = labels[["session", "type"]].merge(predictions, on=["session", "type"], how="left")
    rows["predictions"] = rows["predictions"].fillna("")
    return rows[["session", "type", "predictions"]]


def evaluate_predictions(labels, predictions, k):
    merged = labels.merge(predictions, on=["session", "type"], how="left")
    summary = {}

    for event_type, type_weight in TYPE_WEIGHTS.items():
        type_df = merged[merged["type"] == event_type]
        hits = 0
        denominator = 0

        for _, row in type_df.iterrows():
            true_items = set(parse_items(row["labels"]))
            pred_items = parse_items(row["predictions"])[:k]
            hits += len(true_items.intersection(pred_items))
            denominator += min(len(true_items), k)

        recall = hits / denominator if denominator else 0.0
        summary[event_type] = {
            "hits": hits,
            "denominator": denominator,
            "recall": recall,
            "contribution": recall * type_weight,
        }

    weighted_score = sum(row["contribution"] for row in summary.values())
    return summary, weighted_score


def print_eval_summary(title, summary, weighted_score):
    print(title)
    for event_type in ("clicks", "carts", "orders"):
        row = summary[event_type]
        print(f"{event_type}: recall={row['recall']:.4f}, hits={row['hits']:,}, total={row['denominator']:,}")
    print(f"Weighted Score: {weighted_score:.4f}")


def get_output_dir():
    return Path(__file__).resolve().parents[2] / "outputs"
