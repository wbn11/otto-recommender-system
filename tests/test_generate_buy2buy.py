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


def test_generate_buy2buy_uses_all_query_behaviors_and_buy2buy_source(tmp_path):
    buy2buy = tmp_path / "buy2buy"
    matrix_dir = buy2buy / "matrix"
    matrix_dir.mkdir(parents=True)
    matrix_rows = [
        {"aid": 10, "neighbor_aid": 20, "source_rank": 1, "source_score": 2.0},
    ]
    pq.write_table(
        pa.Table.from_pylist(matrix_rows, schema=COVIS_MATRIX_SCHEMA),
        matrix_dir / "bucket-00000.parquet",
    )
    (buy2buy / "build_report.json").write_text(
        json.dumps({"source": "buy2buy", "snapshot_stats": {"max_ts": 9}}),
        encoding="utf-8",
    )

    queries_path = tmp_path / "queries.parquet"
    labels_path = tmp_path / "labels.parquet"
    queries = [
        {"session": 100, "aids": [10], "timestamps": [10], "event_types": [1]},
    ]
    labels = [
        {"session": 100, "target_type": 2, "aid": 20},
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
        topk=1,
        eval_ks=(1,),
        aid_buckets=2,
        matrix_buckets_per_pass=1,
        workers=1,
        memory_limit_gb=1,
        source_name="buy2buy",
    )

    files = sorted((output / "candidates").glob("*.parquet"))
    candidates = pa.concat_tables([pq.read_table(path) for path in files])
    assert candidates.schema.names == SESSION_RECALL_SCHEMA.names
    assert candidates.schema.types == SESSION_RECALL_SCHEMA.types
    assert candidates.to_pylist() == [
        {
            "session": 100,
            "aid": 20,
            "source": "buy2buy",
            "source_rank": 1,
            "source_score": 2.0,
        }
    ]
    assert report["source"] == "buy2buy"
    assert report["metrics"]["recall_at_1"]["by_target_type"]["2"] == 1.0
    assert report["leakage_assertions"]["passed"] is True
