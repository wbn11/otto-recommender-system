import pyarrow as pa
import pyarrow.parquet as pq

from data.schemas import COVIS_MATRIX_SCHEMA, EVENT_SCHEMA
from recall.build_type_covis import build_type_covis


def test_type_covis_uses_unique_directed_pairs_and_behavior_weights(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    events = [
        {"session": 1, "aid": 10, "ts": 1, "event_type": 1},
        {"session": 1, "aid": 20, "ts": 2, "event_type": 1},
        {"session": 1, "aid": 20, "ts": 3, "event_type": 2},
        {"session": 1, "aid": 30, "ts": 4, "event_type": 3},
        {"session": 2, "aid": 10, "ts": 5, "event_type": 1},
        {"session": 2, "aid": 20, "ts": 6, "event_type": 3},
        {"session": 3, "aid": 40, "ts": 7, "event_type": 1},
    ]
    pq.write_table(
        pa.Table.from_pylist(events, schema=EVENT_SCHEMA),
        snapshot / "part-00000.parquet",
    )

    output = tmp_path / "type_covis"
    report = build_type_covis(
        snapshot,
        output,
        dataset_name="ranker",
        recent_events_per_session=30,
        topk_neighbors=2,
        pair_buckets=2,
        workers=1,
        memory_limit_gb=1,
    )

    files = sorted((output / "matrix").glob("bucket-*.parquet"))
    matrix = pa.concat_tables([pq.read_table(path) for path in files])
    assert matrix.schema.names == COVIS_MATRIX_SCHEMA.names
    assert matrix.schema.types == COVIS_MATRIX_SCHEMA.types
    rows = matrix.to_pylist()
    item_10 = sorted(
        (row for row in rows if row["aid"] == 10),
        key=lambda row: row["source_rank"],
    )
    assert [(row["neighbor_aid"], row["source_score"]) for row in item_10] == [
        (20, 9.0),
        (30, 6.0),
    ]
    item_20 = sorted(
        (row for row in rows if row["aid"] == 20),
        key=lambda row: row["source_rank"],
    )
    assert [(row["neighbor_aid"], row["source_score"]) for row in item_20] == [
        (30, 6.0),
        (10, 2.0),
    ]
    assert all(row["aid"] != row["neighbor_aid"] for row in rows)
    assert report["assertions"]["passed"] is True
    assert report["snapshot_stats"]["sessions"] == 3
    assert report["partial_pairs"]["retained"] is False
    assert not (output / "partial_pairs").exists()
