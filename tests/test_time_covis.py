import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.schemas import EVENT_SCHEMA
from recall.build_type_covis import build_type_covis


HOUR_MS = 60 * 60 * 1000


def test_time_covis_uses_closest_pair_per_session_and_24_hour_limit(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    events = [
        {"session": 1, "aid": 10, "ts": 0, "event_type": 1},
        {"session": 1, "aid": 20, "ts": HOUR_MS, "event_type": 1},
        {"session": 1, "aid": 20, "ts": 2 * HOUR_MS, "event_type": 2},
        {"session": 1, "aid": 30, "ts": 25 * HOUR_MS, "event_type": 3},
        {"session": 2, "aid": 10, "ts": 30 * HOUR_MS, "event_type": 1},
        {"session": 2, "aid": 20, "ts": 30 * HOUR_MS, "event_type": 3},
    ]
    pq.write_table(
        pa.Table.from_pylist(events, schema=EVENT_SCHEMA),
        snapshot / "part-00000.parquet",
    )

    output = tmp_path / "time_covis"
    report = build_type_covis(
        snapshot,
        output,
        dataset_name="ranker",
        recent_events_per_session=30,
        topk_neighbors=10,
        pair_buckets=2,
        event_type_weights={1: 1.0, 2: 1.0, 3: 1.0},
        workers=1,
        memory_limit_gb=1,
        source_name="time_covis",
        pair_score_mode="time_decay",
        max_time_difference_ms=24 * HOUR_MS,
        time_decay_ms=HOUR_MS,
        pair_semantics="test",
    )

    files = sorted((output / "matrix").glob("bucket-*.parquet"))
    matrix = pa.concat_tables([pq.read_table(path) for path in files]).to_pylist()
    scores = {
        (row["aid"], row["neighbor_aid"]): row["source_score"] for row in matrix
    }
    expected_10_20 = 1.0 + math.exp(-1.0)
    assert scores[(10, 20)] == pytest.approx(expected_10_20, rel=1e-6)
    assert scores[(20, 10)] == pytest.approx(expected_10_20, rel=1e-6)
    assert (10, 30) not in scores
    assert (30, 10) not in scores
    assert scores[(20, 30)] == pytest.approx(math.exp(-23.0), rel=1e-6)
    assert scores[(30, 20)] == pytest.approx(math.exp(-23.0), rel=1e-6)
    assert report["source"] == "time_covis"
    assert report["configuration"]["pair_score_mode"] == "time_decay"
    assert report["assertions"]["passed"] is True
