import csv
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq

from data.schemas import ITEM_VOCAB_SCHEMA, LABEL_SCHEMA, QUERY_SCHEMA
from evaluation.analyze_dssm_history import analyze_dssm_history
from utils.config import resolve_project_config


def test_dssm_history_analysis_reproduces_overall_metrics(tmp_path):
    data_dir = tmp_path / "dssm-data"
    data_dir.mkdir()
    vocab_path = data_dir / "item_vocab.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"aid": 10, "item_id": 2},
                {"aid": 20, "item_id": 3},
                {"aid": 30, "item_id": 4},
            ],
            schema=ITEM_VOCAB_SCHEMA,
        ),
        vocab_path,
    )
    (data_dir / "build_report.json").write_text(
        json.dumps(
            {
                "dataset": "ranker",
                "vocabulary": {
                    "path": "item_vocab.parquet",
                    "sha256": hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )

    queries = tmp_path / "queries.parquet"
    query_rows = [
        {"session": 1, "aids": [999], "timestamps": [1], "event_types": [1]},
        {"session": 2, "aids": [10, 999], "timestamps": [1, 2], "event_types": [1, 2]},
        {
            "session": 3,
            "aids": [10, 20, 30],
            "timestamps": [1, 2, 3],
            "event_types": [1, 2, 3],
        },
        {
            "session": 4,
            # The two unknown events are outside the DSSM last-50 window.
            "aids": [999, 999] + [10, 20] * 25,
            "timestamps": list(range(1, 53)),
            "event_types": [1, 2] * 26,
        },
    ]
    pq.write_table(pa.Table.from_pylist(query_rows, schema=QUERY_SCHEMA), queries)

    labels = tmp_path / "labels.parquet"
    label_rows = [
        {
            "session": session,
            "target_type": target_type,
            # Empty-history labels are deliberately outside the FAISS vocabulary.
            "aid": 999 if session == 1 else 10,
        }
        for session in range(1, 5)
        for target_type in (1, 2, 3)
    ]
    pq.write_table(pa.Table.from_pylist(label_rows, schema=LABEL_SCHEMA), labels)

    candidate_root = tmp_path / "dssm-recall"
    candidate_dir = candidate_root / "candidates"
    candidate_dir.mkdir(parents=True)
    candidate_schema = pa.schema(
        [
            pa.field("session", pa.int64(), nullable=False),
            pa.field("target_type", pa.int8(), nullable=False),
            pa.field("aids", pa.list_(pa.int64(), 2), nullable=False),
            pa.field("scores", pa.list_(pa.float32(), 2), nullable=False),
        ]
    )
    candidate_rows = []
    for session in range(1, 5):
        for target_type in (1, 2, 3):
            aids = [20, 30] if session == 1 else [10, 20]
            candidate_rows.append(
                {
                    "session": session,
                    "target_type": target_type,
                    "aids": aids,
                    "scores": [0.9, 0.8],
                }
            )
    pq.write_table(
        pa.Table.from_pylist(candidate_rows, schema=candidate_schema),
        candidate_dir / "part-00000.parquet",
    )
    (candidate_root / "report.json").write_text(
        json.dumps(
            {
                "dataset": "ranker",
                "configuration": {"max_history": 50, "topk": 2},
                "index": {"items": 3},
            }
        ),
        encoding="utf-8",
    )

    config = resolve_project_config("configs/experiments/pipeline_smoke.yaml")
    config["dssm"]["max_sequence_length"] = 50
    config["recall"]["eval_ks"] = [1, 2]
    config["runtime"].update({"workers": 1, "duckdb_memory_limit_gb": 1})
    output = tmp_path / "analysis"
    report = analyze_dssm_history(
        data_dir,
        candidate_root,
        queries,
        labels,
        output,
        config,
        dataset_name="ranker",
    )

    assert report["assertions"]["passed"] is True
    assert report["evaluated_sessions"] == 4
    assert {
        bucket: report["history_distribution"][bucket]["sessions"]
        for bucket in ("empty", "short", "medium", "long")
    } == {"empty": 1, "short": 1, "medium": 1, "long": 1}
    assert report["history_distribution"]["empty"]["unknown_history_events"] == 1
    assert report["history_distribution"]["short"]["unknown_history_events"] == 1
    assert report["history_distribution"]["long"]["raw_history_events"] == 50
    assert report["history_distribution"]["long"]["unknown_history_events"] == 0
    assert report["metrics"]["empty"]["recall_at_1"]["weighted"] == 0.0
    assert (
        report["label_retrievability"]["empty"]["weighted_retrievable_ceiling"]
        == 0.0
    )
    assert report["label_retrievability"]["empty"]["by_target_type"]["3"] == {
        "labels": 1,
        "retrievable_labels": 0,
        "retrievable_rate": 0.0,
    }
    assert report["metrics"]["short"]["recall_at_1"]["weighted"] == 1.0
    assert report["metrics"]["overall"]["recall_at_1"]["weighted"] == 0.75
    assert (
        report["metrics"]["overall"]["recall_at_1"][
            "weighted_retrievable_ceiling"
        ]
        == 0.75
    )
    assert (
        report["metrics"]["overall"]["recall_at_1"][
            "weighted_recall_given_retrievable"
        ]
        == 1.0
    )
    assert report["metrics"]["overall"]["recall_at_1"]["denominators"] == {
        "1": 4,
        "2": 4,
        "3": 4,
    }

    history = pq.read_table(output / "session_history.parquet")
    assert history["effective_history_length"].to_pylist() == [0, 1, 3, 50]
    with (output / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5 * 2 * 4
    assert any(
        row["history_bucket"] == "long"
        and row["k"] == "2"
        and row["target_type"] == "weighted"
        for row in rows
    )
    empty_click = next(
        row
        for row in rows
        if row["history_bucket"] == "empty"
        and row["k"] == "1"
        and row["target_type"] == "clicks"
    )
    assert empty_click["retrievable_labels"] == "0"
    assert empty_click["retrievable_rate"] == "0.0"
