import pyarrow as pa
import pyarrow.parquet as pq

from data.schemas import (
    FUSED_CANDIDATE_SCHEMA,
    LABEL_SCHEMA,
    POPULAR_SCHEMA,
    QUERY_SCHEMA,
    RECALL_SCHEMA,
    SESSION_RECALL_SCHEMA,
)
from recall.fuse_candidates import SOURCE_IDS, fuse_candidates
from utils.config import resolve_project_config


def _write_session_source(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SESSION_RECALL_SCHEMA), path)


def test_candidate_fusion_round_robin_backfill_and_features(tmp_path):
    queries = tmp_path / "queries.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"session": 1, "aids": [10], "timestamps": [101], "event_types": [1]},
                {"session": 2, "aids": [11], "timestamps": [102], "event_types": [1]},
            ],
            schema=QUERY_SCHEMA,
        ),
        queries,
    )
    labels = tmp_path / "labels.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"session": 1, "target_type": 1, "aid": 50},
                {"session": 1, "target_type": 2, "aid": 200},
                {"session": 2, "target_type": 3, "aid": 300},
            ],
            schema=LABEL_SCHEMA,
        ),
        labels,
    )
    popular = tmp_path / "popular.parquet"
    popular_rows = []
    for target_type, aids in {
        1: [10, 100, 101, 102, 103],
        2: [200, 201, 202, 203, 204],
        3: [300, 301, 302, 303, 304],
    }.items():
        for rank, aid in enumerate(aids, start=1):
            popular_rows.append(
                {
                    "target_type": target_type,
                    "aid": aid,
                    "source": "popular",
                    "source_rank": rank,
                    "source_score": float(10 - rank),
                }
            )
    pq.write_table(pa.Table.from_pylist(popular_rows, schema=POPULAR_SCHEMA), popular)

    revisit = tmp_path / "revisit.parquet"
    revisit_rows = []
    for session, aid in ((1, 10), (2, 11)):
        for target_type in (1, 2, 3):
            revisit_rows.append(
                {
                    "session": session,
                    "target_type": target_type,
                    "aid": aid,
                    "source": "revisit",
                    "source_rank": 1,
                    "source_score": 1.0,
                }
            )
    pq.write_table(pa.Table.from_pylist(revisit_rows, schema=RECALL_SCHEMA), revisit)

    type_dir = tmp_path / "type"
    buy_dir = tmp_path / "buy"
    time_dir = tmp_path / "time"
    _write_session_source(
        type_dir / "session-bucket-00001.parquet",
        [
            {"session": 1, "aid": 10, "source": "type_covis", "source_rank": 1,
             "source_score": 2.0},
            {"session": 1, "aid": 20, "source": "type_covis", "source_rank": 2,
             "source_score": 1.0},
        ],
    )
    _write_session_source(
        buy_dir / "session-bucket-00001.parquet",
        [{"session": 1, "aid": 30, "source": "buy2buy", "source_rank": 1,
          "source_score": 1.5}],
    )
    _write_session_source(
        time_dir / "session-bucket-00001.parquet",
        [{"session": 1, "aid": 40, "source": "time_covis", "source_rank": 1,
          "source_score": 1.2}],
    )

    dssm_dir = tmp_path / "dssm"
    dssm_dir.mkdir()
    dssm_schema = pa.schema(
        [
            pa.field("session", pa.int64(), nullable=False),
            pa.field("target_type", pa.int8(), nullable=False),
            pa.field("aids", pa.list_(pa.int64(), 2), nullable=False),
            pa.field("scores", pa.list_(pa.float32(), 2), nullable=False),
        ]
    )
    dssm_rows = []
    for session, aids in ((1, [50, 60]), (2, [51, 61])):
        for target_type in (1, 2, 3):
            dssm_rows.append(
                {
                    "session": session,
                    "target_type": target_type,
                    "aids": aids,
                    "scores": [0.9, 0.8],
                }
            )
    pq.write_table(pa.Table.from_pylist(dssm_rows, schema=dssm_schema), dssm_dir / "part-00000.parquet")

    config = resolve_project_config("configs/experiments/debug.yaml")
    config["candidate_k"] = 5
    config["recall"]["eval_ks"] = [2, 5]
    config["runtime"].update({"workers": 1, "duckdb_memory_limit_gb": 1})
    output = tmp_path / "fused"
    report = fuse_candidates(
        queries,
        labels,
        popular,
        revisit,
        type_dir,
        buy_dir,
        time_dir,
        dssm_dir,
        output,
        config,
        dataset_name="ranker",
        session_buckets=2,
    )

    table = pq.read_table(output / "parts")
    assert table.schema.names == FUSED_CANDIDATE_SCHEMA.names
    assert table.schema.types == FUSED_CANDIDATE_SCHEMA.types
    assert table.num_rows == 2 * 3 * 5
    rows = table.to_pylist()
    group = [row for row in rows if row["session"] == 1 and row["target_type"] == 1]
    assert [row["aid"] for row in sorted(group, key=lambda row: row["candidate_rank"])] == [
        10,
        30,
        40,
        50,
        20,
    ]
    repeated = next(row for row in group if row["aid"] == 10)
    assert repeated["selected_source"] == SOURCE_IDS["revisit"]
    assert repeated["from_popular"] == 1
    assert repeated["from_revisit"] == 1
    assert repeated["from_type_covis"] == 1
    assert repeated["source_count"] == 3
    sparse_group = [row for row in rows if row["session"] == 2 and row["target_type"] == 1]
    assert sum(row["selected_source"] == SOURCE_IDS["popular"] for row in sparse_group) == 2
    assert report["assertions"]["passed"] is True
    assert report["candidates"]["average_per_group"] == 5
    assert set(report["metrics"]) == {"candidate_recall_at_2", "candidate_recall_at_5"}
