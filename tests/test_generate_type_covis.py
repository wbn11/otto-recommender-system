import json

import pyarrow as pa
import pyarrow.parquet as pq

from data.schemas import (
    COVIS_MATRIX_SCHEMA,
    LABEL_SCHEMA,
    QUERY_SCHEMA,
    SESSION_RECALL_SCHEMA,
)
from recall.generate_type_covis import generate_type_covis


def test_generate_type_covis_ranks_neighbors_and_reports_zero_candidate_sessions(tmp_path):
    type_covis = tmp_path / "type_covis"
    matrix_dir = type_covis / "matrix"
    matrix_dir.mkdir(parents=True)
    matrix_rows = [
        {"aid": 10, "neighbor_aid": 20, "source_rank": 1, "source_score": 9.0},
        {"aid": 10, "neighbor_aid": 30, "source_rank": 2, "source_score": 6.0},
    ]
    pq.write_table(
        pa.Table.from_pylist(matrix_rows, schema=COVIS_MATRIX_SCHEMA),
        matrix_dir / "bucket-00000.parquet",
    )
    (type_covis / "build_report.json").write_text(
        json.dumps({"snapshot_stats": {"max_ts": 9}}), encoding="utf-8"
    )

    queries_path = tmp_path / "queries.parquet"
    labels_path = tmp_path / "labels.parquet"
    queries = [
        {"session": 100, "aids": [10], "timestamps": [10], "event_types": [1]},
        {"session": 200, "aids": [999], "timestamps": [11], "event_types": [1]},
    ]
    labels = [
        {"session": 100, "target_type": 1, "aid": 20},
        {"session": 100, "target_type": 2, "aid": 30},
        {"session": 100, "target_type": 3, "aid": 40},
        {"session": 200, "target_type": 1, "aid": 998},
    ]
    pq.write_table(pa.Table.from_pylist(queries, schema=QUERY_SCHEMA), queries_path)
    pq.write_table(pa.Table.from_pylist(labels, schema=LABEL_SCHEMA), labels_path)

    output = tmp_path / "recall"
    report = generate_type_covis(
        matrix_dir,
        queries_path,
        labels_path,
        output,
        dataset_name="ranker",
        recent_events=30,
        topk=2,
        eval_ks=(1, 2),
        aid_buckets=2,
        matrix_buckets_per_pass=1,
        workers=1,
        memory_limit_gb=1,
    )

    files = sorted((output / "candidates").glob("*.parquet"))
    candidates = pa.concat_tables([pq.read_table(path) for path in files])
    assert candidates.schema.names == SESSION_RECALL_SCHEMA.names
    assert candidates.schema.types == SESSION_RECALL_SCHEMA.types
    assert candidates.to_pylist() == [
        {
            "session": 100,
            "aid": 20,
            "source": "type_covis",
            "source_rank": 1,
            "source_score": 9.0,
        },
        {
            "session": 100,
            "aid": 30,
            "source": "type_covis",
            "source_rank": 2,
            "source_score": 6.0,
        },
    ]
    assert report["metrics"]["recall_at_2"]["weighted"] == 0.35
    assert report["candidate_statistics"]["sessions_without_candidates"] == 1
    assert report["candidate_statistics"]["session_coverage"] == 0.5
    assert report["leakage_assertions"]["passed"] is True
