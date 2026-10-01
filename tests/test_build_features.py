import json

import pyarrow as pa
import pyarrow.parquet as pq

from data.schemas import (
    COVIS_MATRIX_SCHEMA,
    EVENT_SCHEMA,
    FUSED_CANDIDATE_SCHEMA,
    LABEL_SCHEMA,
    QUERY_SCHEMA,
    RANKER_FEATURE_SCHEMA,
)
from features.build_features import build_features
from features.registry import FEATURE_GROUPS, FEATURE_NAMES
from utils.config import resolve_project_config


def _candidate(session, target_type, aid, rank):
    present = int(aid == 30)
    return {
        "session": session,
        "target_type": target_type,
        "aid": aid,
        "candidate_rank": rank,
        "selected_source": 5 if aid == 30 else 1,
        "from_popular": 0,
        "popular_rank": 0,
        "popular_score": 0.0,
        "from_revisit": 1 - present,
        "revisit_rank": rank if not present else 0,
        "revisit_score": 1.0 if not present else 0.0,
        "from_type_covis": present,
        "type_covis_rank": rank if present else 0,
        "type_covis_score": 5.0 if present else 0.0,
        "from_buy2buy": 0,
        "buy2buy_rank": 0,
        "buy2buy_score": 0.0,
        "from_time_covis": 0,
        "time_covis_rank": 0,
        "time_covis_score": 0.0,
        "from_dssm": present,
        "dssm_rank": rank if present else 0,
        "dssm_score": 0.8 if present else 0.0,
        "source_count": 2 if present else 1,
        "best_source_rank": rank,
    }


def test_build_features_point_in_time_schema_and_values(tmp_path):
    queries = tmp_path / "queries.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "session": 1,
                    "aids": [10, 20],
                    "timestamps": [101, 102],
                    "event_types": [1, 2],
                },
                {
                    "session": 2,
                    # Regression case: filtering to recent5 must happen before
                    # casting the rank to int8.  A 130-event query previously
                    # failed when DuckDB tried to cast position 128.
                    "aids": [11] * 129 + [21],
                    "timestamps": list(range(103, 233)),
                    "event_types": [1] * 129 + [3],
                },
            ],
            schema=QUERY_SCHEMA,
        ),
        queries,
    )
    labels = tmp_path / "labels.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"session": 1, "target_type": 1, "aid": 30},
                {"session": 2, "target_type": 3, "aid": 30},
            ],
            schema=LABEL_SCHEMA,
        ),
        labels,
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"session": 100, "aid": 10, "ts": 90, "event_type": 1},
                {"session": 100, "aid": 20, "ts": 91, "event_type": 2},
                {"session": 100, "aid": 30, "ts": 92, "event_type": 3},
                {"session": 101, "aid": 30, "ts": 93, "event_type": 1},
            ],
            schema=EVENT_SCHEMA,
        ),
        snapshot / "part-00000.parquet",
    )

    candidates = tmp_path / "candidates"
    candidates.mkdir()
    for bucket, session in ((0, 2), (1, 1)):
        for target_type in (1, 2, 3):
            rows = [
                _candidate(session, target_type, 20 if session == 1 else 21, 1),
                _candidate(session, target_type, 30, 2),
            ]
            pq.write_table(
                pa.Table.from_pylist(rows, schema=FUSED_CANDIDATE_SCHEMA),
                candidates / f"session-bucket-{bucket:05d}-target-{target_type}.parquet",
            )

    matrix_rows = {
        "type": [
            {"aid": 10, "neighbor_aid": 30, "source_rank": 1, "source_score": 1.0},
            {"aid": 20, "neighbor_aid": 30, "source_rank": 1, "source_score": 5.0},
            {"aid": 21, "neighbor_aid": 30, "source_rank": 1, "source_score": 4.0},
        ],
        "buy": [
            {"aid": 20, "neighbor_aid": 30, "source_rank": 1, "source_score": 7.0},
        ],
        "time": [
            {"aid": 10, "neighbor_aid": 30, "source_rank": 1, "source_score": 2.0},
            {"aid": 20, "neighbor_aid": 30, "source_rank": 1, "source_score": 6.0},
        ],
    }
    matrix_dirs = {}
    for name, rows in matrix_rows.items():
        directory = tmp_path / name
        directory.mkdir()
        pq.write_table(
            pa.Table.from_pylist(rows, schema=COVIS_MATRIX_SCHEMA),
            directory / "bucket-00000.parquet",
        )
        matrix_dirs[name] = directory

    config = resolve_project_config("configs/experiments/pipeline_smoke.yaml")
    config["candidate_k"] = 2
    config["runtime"].update({"workers": 1, "duckdb_memory_limit_gb": 1})
    output = tmp_path / "features"
    report = build_features(
        candidates,
        queries,
        labels,
        snapshot,
        matrix_dirs["type"],
        matrix_dirs["buy"],
        matrix_dirs["time"],
        output,
        config,
        dataset_name="ranker",
    )

    table = pq.read_table(output / "parts")
    assert table.schema.names == RANKER_FEATURE_SCHEMA.names
    assert table.schema.types == RANKER_FEATURE_SCHEMA.types
    assert table.num_rows == 12
    assert len(FEATURE_NAMES) == 49
    assert {name: len(values) for name, values in FEATURE_GROUPS.items()} == {
        "base": 1,
        "session": 8,
        "item": 4,
        "recall": 20,
        "interaction": 14,
        "temporal": 2,
    }
    row = next(
        item
        for item in table.to_pylist()
        if item["session"] == 1 and item["target_type"] == 1 and item["aid"] == 30
    )
    assert row["label"] == 1
    assert row["session_length"] == 2
    assert row["session_click_count"] == 1
    assert row["session_cart_count"] == 1
    assert row["candidate_seen"] == 0
    assert row["candidate_distance_to_end"] == -1
    assert row["last_item_type_covis_score"] == 5.0
    assert row["recent5_type_covis_max"] == 5.0
    assert row["recent5_type_covis_mean"] == 3.0
    assert row["last_item_buy2buy_score"] == 7.0
    assert row["last_item_time_covis_score"] == 6.0
    assert row["item_total_count"] == 2
    assert row["item_recent_1d_count"] == 2
    registry = json.loads((output / "feature_registry.json").read_text(encoding="utf-8"))
    assert registry["feature_count"] == 49
    assert report["assertions"]["passed"] is True
    assert report["output_data"]["positive_rows"] == 2
    assert report["point_in_time"]["snapshot_max_ts"] < report["point_in_time"]["cutoff_ts"]
