from __future__ import annotations

import pandas as pd


EVENT_TYPES = ("clicks", "carts", "orders")
TARGET_COLUMNS = ["session", "type"]
PREDICTION_COLUMNS = ["session", "type", "predictions"]


def select_target_columns(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    missing_columns = set(TARGET_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"{source_name} missing columns: {sorted(missing_columns)}")

    return df[TARGET_COLUMNS].copy()


def build_validation_target_rows(labels_df: pd.DataFrame) -> pd.DataFrame:
    return select_target_columns(labels_df, "validation labels")


def build_test_target_rows(events_df: pd.DataFrame) -> pd.DataFrame:
    if "session" not in events_df.columns:
        raise ValueError("events missing columns: ['session']")

    sessions = events_df["session"].drop_duplicates().tolist()
    rows = [
        {"session": session, "type": event_type}
        for session in sessions
        for event_type in EVENT_TYPES
    ]
    return pd.DataFrame(rows, columns=TARGET_COLUMNS)


def load_target_rows_from_file(
    output_dir,
    labels_file: str | None = None,
    test_events_file: str | None = None,
) -> pd.DataFrame:
    if labels_file and test_events_file:
        raise ValueError("--labels-file and --test-events-file are mutually exclusive.")
    if not labels_file and not test_events_file:
        raise ValueError("Either labels_file or test_events_file must be provided.")

    if test_events_file:
        events_path = output_dir / test_events_file
        if not events_path.exists():
            raise FileNotFoundError(f"Events file not found: {events_path}")
        return build_test_target_rows(pd.read_parquet(events_path))

    labels_path = output_dir / labels_file
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    return build_validation_target_rows(pd.read_parquet(labels_path))


def parse_unique_prediction_items(value) -> list[str]:
    if pd.isna(value):
        return []

    items = []
    seen = set()
    for token in str(value).split():
        if not token.isdigit() or token in seen:
            continue
        items.append(token)
        seen.add(token)
    return items


def normalize_prediction_items(value, k: int) -> str:
    return " ".join(parse_unique_prediction_items(value)[:k])


def order_predictions_by_target_rows(predictions: pd.DataFrame, target_rows: pd.DataFrame, k: int) -> pd.DataFrame:
    missing_columns = set(PREDICTION_COLUMNS) - set(predictions.columns)
    if missing_columns:
        raise ValueError(f"predictions missing columns: {sorted(missing_columns)}")

    ordered_targets = select_target_columns(target_rows, "target rows")
    normalized = predictions[PREDICTION_COLUMNS].copy()
    normalized["predictions"] = normalized["predictions"].apply(lambda value: normalize_prediction_items(value, k))
    normalized = normalized.drop_duplicates(TARGET_COLUMNS, keep="first")

    ordered = ordered_targets.merge(normalized, on=TARGET_COLUMNS, how="left")
    ordered["predictions"] = ordered["predictions"].fillna("")
    return ordered[PREDICTION_COLUMNS]
