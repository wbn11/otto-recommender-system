"""Canonical schemas and categorical ids used throughout the new pipeline."""

from __future__ import annotations

try:
    import pyarrow as pa
except ImportError as exc:  # pragma: no cover - reported by environment check
    raise ImportError("pyarrow is required for canonical OTTO datasets") from exc


PAD_ID = 0
UNK_ID = 1

EVENT_TYPE_TO_ID = {
    "clicks": 1,
    "carts": 2,
    "orders": 3,
}
EVENT_ID_TO_TYPE = {value: key for key, value in EVENT_TYPE_TO_ID.items()}

EVENT_SCHEMA = pa.schema([
    pa.field("session", pa.int64(), nullable=False),
    pa.field("aid", pa.int64(), nullable=False),
    pa.field("ts", pa.int64(), nullable=False),
    pa.field("event_type", pa.int8(), nullable=False),
])

SESSION_SCHEMA = pa.schema([
    pa.field("session", pa.int64(), nullable=False),
    pa.field("start_ts", pa.int64(), nullable=False),
    pa.field("end_ts", pa.int64(), nullable=False),
    pa.field("event_count", pa.int32(), nullable=False),
    pa.field("click_count", pa.int32(), nullable=False),
    pa.field("cart_count", pa.int32(), nullable=False),
    pa.field("order_count", pa.int32(), nullable=False),
])

QUERY_SCHEMA = pa.schema([
    pa.field("session", pa.int64(), nullable=False),
    pa.field("aids", pa.list_(pa.int64()), nullable=False),
    pa.field("timestamps", pa.list_(pa.int64()), nullable=False),
    pa.field("event_types", pa.list_(pa.int8()), nullable=False),
])

LABEL_SCHEMA = pa.schema([
    pa.field("session", pa.int64(), nullable=False),
    pa.field("target_type", pa.int8(), nullable=False),
    pa.field("aid", pa.int64(), nullable=False),
])

POPULAR_SCHEMA = pa.schema([
    pa.field("target_type", pa.int8(), nullable=False),
    pa.field("aid", pa.int64(), nullable=False),
    pa.field("source", pa.string(), nullable=False),
    pa.field("source_rank", pa.int32(), nullable=False),
    pa.field("source_score", pa.float32(), nullable=False),
])

RECALL_SCHEMA = pa.schema([
    pa.field("session", pa.int64(), nullable=False),
    pa.field("target_type", pa.int8(), nullable=False),
    pa.field("aid", pa.int64(), nullable=False),
    pa.field("source", pa.string(), nullable=False),
    pa.field("source_rank", pa.int32(), nullable=False),
    pa.field("source_score", pa.float32(), nullable=False),
])

COVIS_MATRIX_SCHEMA = pa.schema([
    pa.field("aid", pa.int64(), nullable=False),
    pa.field("neighbor_aid", pa.int64(), nullable=False),
    pa.field("source_rank", pa.int32(), nullable=False),
    pa.field("source_score", pa.float32(), nullable=False),
])

SESSION_RECALL_SCHEMA = pa.schema([
    pa.field("session", pa.int64(), nullable=False),
    pa.field("aid", pa.int64(), nullable=False),
    pa.field("source", pa.string(), nullable=False),
    pa.field("source_rank", pa.int32(), nullable=False),
    pa.field("source_score", pa.float32(), nullable=False),
])

ITEM_VOCAB_SCHEMA = pa.schema([
    pa.field("aid", pa.int64(), nullable=False),
    pa.field("item_id", pa.int32(), nullable=False),
])

DSSM_SEQUENCE_SCHEMA = pa.schema([
    pa.field("session", pa.int64(), nullable=False),
    pa.field("item_ids", pa.list_(pa.int32()), nullable=False),
    pa.field("event_types", pa.list_(pa.int8()), nullable=False),
])

FUSED_CANDIDATE_SCHEMA = pa.schema([
    pa.field("session", pa.int64(), nullable=False),
    pa.field("target_type", pa.int8(), nullable=False),
    pa.field("aid", pa.int64(), nullable=False),
    pa.field("candidate_rank", pa.int16(), nullable=False),
    pa.field("selected_source", pa.int8(), nullable=False),
    pa.field("from_popular", pa.int8(), nullable=False),
    pa.field("popular_rank", pa.int16(), nullable=False),
    pa.field("popular_score", pa.float32(), nullable=False),
    pa.field("from_revisit", pa.int8(), nullable=False),
    pa.field("revisit_rank", pa.int16(), nullable=False),
    pa.field("revisit_score", pa.float32(), nullable=False),
    pa.field("from_type_covis", pa.int8(), nullable=False),
    pa.field("type_covis_rank", pa.int16(), nullable=False),
    pa.field("type_covis_score", pa.float32(), nullable=False),
    pa.field("from_buy2buy", pa.int8(), nullable=False),
    pa.field("buy2buy_rank", pa.int16(), nullable=False),
    pa.field("buy2buy_score", pa.float32(), nullable=False),
    pa.field("from_time_covis", pa.int8(), nullable=False),
    pa.field("time_covis_rank", pa.int16(), nullable=False),
    pa.field("time_covis_score", pa.float32(), nullable=False),
    pa.field("from_dssm", pa.int8(), nullable=False),
    pa.field("dssm_rank", pa.int16(), nullable=False),
    pa.field("dssm_score", pa.float32(), nullable=False),
    pa.field("source_count", pa.int8(), nullable=False),
    pa.field("best_source_rank", pa.int16(), nullable=False),
])


# M8 uses one explicit schema for both ranker training and inference.  The
# first six columns identify a row; the remaining 49 columns are registered
# model features in ``features/registry.py``.
RANKER_FEATURE_SCHEMA_VERSION = "otto_ranker_features_v1"

RANKER_FEATURE_SCHEMA = pa.schema([
    pa.field("session", pa.int64(), nullable=False),
    pa.field("target_type", pa.int8(), nullable=False),
    pa.field("aid", pa.int64(), nullable=False),
    pa.field("candidate_rank", pa.int16(), nullable=False),
    pa.field("selected_source", pa.int8(), nullable=False),
    pa.field("label", pa.int8(), nullable=False),
    # Base (1)
    pa.field("target_type_id", pa.int8(), nullable=False),
    # Session (8)
    pa.field("session_length", pa.int16(), nullable=False),
    pa.field("session_click_count", pa.int16(), nullable=False),
    pa.field("session_cart_count", pa.int16(), nullable=False),
    pa.field("session_order_count", pa.int16(), nullable=False),
    pa.field("session_duration_ms", pa.int64(), nullable=False),
    pa.field("session_last_event_type", pa.int8(), nullable=False),
    pa.field("session_hour_utc", pa.int8(), nullable=False),
    pa.field("session_weekday_utc", pa.int8(), nullable=False),
    # Item (4)
    pa.field("item_total_count", pa.int32(), nullable=False),
    pa.field("item_click_count", pa.int32(), nullable=False),
    pa.field("item_cart_count", pa.int32(), nullable=False),
    pa.field("item_order_count", pa.int32(), nullable=False),
    # Recall source (20)
    pa.field("from_popular", pa.int8(), nullable=False),
    pa.field("popular_rank", pa.int16(), nullable=False),
    pa.field("popular_score", pa.float32(), nullable=False),
    pa.field("from_revisit", pa.int8(), nullable=False),
    pa.field("revisit_rank", pa.int16(), nullable=False),
    pa.field("revisit_score", pa.float32(), nullable=False),
    pa.field("from_type_covis", pa.int8(), nullable=False),
    pa.field("type_covis_rank", pa.int16(), nullable=False),
    pa.field("type_covis_score", pa.float32(), nullable=False),
    pa.field("from_buy2buy", pa.int8(), nullable=False),
    pa.field("buy2buy_rank", pa.int16(), nullable=False),
    pa.field("buy2buy_score", pa.float32(), nullable=False),
    pa.field("from_time_covis", pa.int8(), nullable=False),
    pa.field("time_covis_rank", pa.int16(), nullable=False),
    pa.field("time_covis_score", pa.float32(), nullable=False),
    pa.field("from_dssm", pa.int8(), nullable=False),
    pa.field("dssm_rank", pa.int16(), nullable=False),
    pa.field("dssm_score", pa.float32(), nullable=False),
    pa.field("source_count", pa.int8(), nullable=False),
    pa.field("best_source_rank", pa.int16(), nullable=False),
    # Session-candidate interaction (14)
    pa.field("candidate_seen", pa.int8(), nullable=False),
    pa.field("candidate_occurrence_count", pa.int16(), nullable=False),
    pa.field("candidate_last_position", pa.int16(), nullable=False),
    pa.field("candidate_distance_to_end", pa.int16(), nullable=False),
    pa.field("candidate_last_event_type", pa.int8(), nullable=False),
    pa.field("last_item_type_covis_score", pa.float32(), nullable=False),
    pa.field("last_item_buy2buy_score", pa.float32(), nullable=False),
    pa.field("last_item_time_covis_score", pa.float32(), nullable=False),
    pa.field("recent5_type_covis_max", pa.float32(), nullable=False),
    pa.field("recent5_type_covis_mean", pa.float32(), nullable=False),
    pa.field("recent5_buy2buy_max", pa.float32(), nullable=False),
    pa.field("recent5_buy2buy_mean", pa.float32(), nullable=False),
    pa.field("recent5_time_covis_max", pa.float32(), nullable=False),
    pa.field("recent5_time_covis_mean", pa.float32(), nullable=False),
    # Temporal (2)
    pa.field("item_recent_1d_count", pa.int32(), nullable=False),
    pa.field("item_recent_7d_count", pa.int32(), nullable=False),
])


TOP20_PREDICTION_SCHEMA = pa.schema([
    pa.field("session", pa.int64(), nullable=False),
    pa.field("target_type", pa.int8(), nullable=False),
    pa.field("aids", pa.list_(pa.int64(), 20), nullable=False),
    pa.field("scores", pa.list_(pa.float32(), 20), nullable=False),
])
