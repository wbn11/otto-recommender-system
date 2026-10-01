"""Stream feature shards through the selected LambdaRank model.

Each input shard contains one target type and is ordered by session and
candidate rank.  It is loaded independently, scored, reduced from K=100 to
Top20, and released before the next shard.  The full 540M-row validation table
therefore never resides in memory at once.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.schemas import (
    LABEL_SCHEMA,
    RANKER_FEATURE_SCHEMA,
    RANKER_FEATURE_SCHEMA_VERSION,
    TOP20_PREDICTION_SCHEMA,
)
from features.registry import FEATURE_NAMES
from utils.config import resolve_project_config


TARGET_WEIGHTS = {1: 0.10, 2: 0.30, 3: 0.60}
_FEATURE_RE = re.compile(r"session-bucket-(\d+)-target-([123])\.parquet$")


def _quote(value: str | Path) -> str:
    return str(value).replace("'", "''")


def _path(path: Path) -> str:
    return path.resolve().as_posix()


def _sql_files(paths: Sequence[Path]) -> str:
    return "[" + ",".join(f"'{_quote(_path(path))}'" for path in paths) + "]"


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _check_schema(path: Path, expected: pa.Schema, name: str) -> None:
    actual = pq.read_schema(path)
    if actual.names != expected.names or actual.types != expected.types:
        raise AssertionError(f"Unexpected {name} schema in {path}: {actual}")


def _feature_files(
    path: str | Path,
    *,
    max_buckets: int | None,
) -> tuple[Path, list[tuple[int, int, Path]], bool]:
    root = Path(path).resolve()
    files: list[tuple[int, int, Path]] = []
    for file in sorted((root / "parts").glob("session-bucket-*-target-*.parquet")):
        match = _FEATURE_RE.fullmatch(file.name)
        if match is None:
            continue
        bucket, target_type = int(match.group(1)), int(match.group(2))
        if max_buckets is not None and bucket >= max_buckets:
            continue
        _check_schema(file, RANKER_FEATURE_SCHEMA, "ranker feature")
        files.append((bucket, target_type, file))
    if not files:
        raise FileNotFoundError(f"No canonical ranker feature shards below {root}")
    buckets = sorted({bucket for bucket, _, _ in files})
    missing = [
        (bucket, target_type)
        for bucket in buckets
        for target_type in (1, 2, 3)
        if not any(item[:2] == (bucket, target_type) for item in files)
    ]
    if missing:
        raise ValueError(f"Missing feature shards: {missing}")
    limited = max_buckets is not None
    return root, files, limited


def _validate_model(model_dir: Path, model) -> tuple[str, ...]:
    schema_path = model_dir / "feature_schema.json"
    if not schema_path.is_file():
        raise FileNotFoundError(schema_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema.get("schema_version") != RANKER_FEATURE_SCHEMA_VERSION:
        raise AssertionError("Model and validation feature schema versions differ")
    feature_names = tuple(schema.get("model_feature_names", ()))
    if not feature_names:
        raise AssertionError("Saved model does not contain a feature list")
    if len(set(feature_names)) != len(feature_names):
        raise AssertionError("Saved model feature names are not unique")
    if set(feature_names) - set(FEATURE_NAMES):
        raise AssertionError("Saved model contains features outside the canonical registry")
    canonical_subset = tuple(name for name in FEATURE_NAMES if name in set(feature_names))
    if feature_names != canonical_subset:
        raise AssertionError("Saved model features do not follow canonical registry order")
    if tuple(model.feature_name()) != feature_names:
        raise AssertionError("LightGBM model feature order differs from its saved schema")
    return feature_names


def _topk_indices(score_matrix, aid_matrix, topk: int):
    import numpy as np

    unordered = np.argpartition(-score_matrix, topk - 1, axis=1)[:, :topk]
    selected_scores = np.take_along_axis(score_matrix, unordered, axis=1)
    selected_aids = np.take_along_axis(aid_matrix, unordered, axis=1)
    order = np.lexsort((selected_aids, -selected_scores), axis=1)
    indices = np.take_along_axis(unordered, order, axis=1)
    return indices


def _score_file(
    model,
    feature_file: Path,
    output_file: Path,
    *,
    candidate_k: int,
    topk: int,
    compression: str,
    row_group_size: int,
    max_groups_per_file: int | None,
    feature_names: Sequence[str],
) -> dict[str, int | float]:
    import numpy as np

    started = time.perf_counter()
    columns = ["session", "target_type", "aid", "candidate_rank", *feature_names]
    table = pq.read_table(feature_file, columns=columns)
    if max_groups_per_file is not None:
        table = table.slice(0, max_groups_per_file * candidate_k)
    rows = table.num_rows
    if rows == 0 or rows % candidate_k:
        raise AssertionError(f"{feature_file} has {rows} rows, not complete K={candidate_k} groups")
    groups = rows // candidate_k
    sessions = table.column("session").to_numpy(zero_copy_only=False).reshape(groups, candidate_k)
    targets = table.column("target_type").to_numpy(zero_copy_only=False).reshape(
        groups, candidate_k
    )
    aids = table.column("aid").to_numpy(zero_copy_only=False).reshape(groups, candidate_k)
    ranks = table.column("candidate_rank").to_numpy(zero_copy_only=False).reshape(
        groups, candidate_k
    )
    if not np.all(sessions == sessions[:, :1]):
        raise AssertionError(f"Session rows are not contiguous in {feature_file}")
    if not np.all(targets == targets[:, :1]):
        raise AssertionError(f"Target rows are not contiguous in {feature_file}")
    expected_ranks = np.arange(1, candidate_k + 1, dtype=ranks.dtype)
    if not np.all(ranks == expected_ranks):
        raise AssertionError(f"Candidate ranks are not 1..{candidate_k} in {feature_file}")

    feature_frame = table.select(list(feature_names)).to_pandas(
        split_blocks=True, self_destruct=True
    )
    scores = np.asarray(model.predict(feature_frame), dtype=np.float32).reshape(
        groups, candidate_k
    )
    indices = _topk_indices(scores, aids, topk)
    top_aids = np.take_along_axis(aids, indices, axis=1).astype(np.int64, copy=False)
    top_scores = np.take_along_axis(scores, indices, axis=1).astype(np.float32, copy=False)
    if np.any(np.diff(top_scores, axis=1) > 0):
        raise AssertionError("Top-K model scores are not monotonically descending")
    session_array = pa.array(sessions[:, 0], type=pa.int64())
    target_array = pa.array(targets[:, 0], type=pa.int8())
    aid_array = pa.FixedSizeListArray.from_arrays(
        pa.array(top_aids.reshape(-1), type=pa.int64()), topk
    )
    score_array = pa.FixedSizeListArray.from_arrays(
        pa.array(top_scores.reshape(-1), type=pa.float32()), topk
    )
    output_table = pa.Table.from_arrays(
        [session_array, target_array, aid_array, score_array],
        schema=TOP20_PREDICTION_SCHEMA,
    )
    pq.write_table(
        output_table,
        output_file,
        compression=compression,
        row_group_size=row_group_size,
    )
    _check_schema(output_file, TOP20_PREDICTION_SCHEMA, "Top20 prediction")
    del feature_frame, scores, indices, top_aids, top_scores, output_table, table
    gc.collect()
    return {
        "candidate_rows": rows,
        "prediction_groups": groups,
        "prediction_items": groups * topk,
        "runtime_seconds": round(time.perf_counter() - started, 6),
    }


def _evaluate(
    prediction_files: Sequence[Path],
    labels: Path,
    *,
    topk: int,
    workers: int,
    memory_limit_gb: int,
    temp_dir: Path,
) -> tuple[dict[str, Any], dict[str, int]]:
    connection = duckdb.connect()
    temp_dir.mkdir(parents=True, exist_ok=True)
    connection.execute(f"SET threads={int(workers)}")
    connection.execute(f"SET memory_limit='{int(memory_limit_gb)}GB'")
    connection.execute(f"SET temp_directory='{_quote(_path(temp_dir))}'")
    prediction_relation = f"read_parquet({_sql_files(prediction_files)}, union_by_name=true)"
    invalid_groups = int(
        connection.execute(
            f"""
            SELECT count(*) FROM {prediction_relation}
            WHERE list_unique(aids) != {int(topk)}
               OR array_length(aids) != {int(topk)}
               OR array_length(scores) != {int(topk)}
            """
        ).fetchone()[0]
    )
    rows = connection.execute(
        f"""
        WITH prediction_groups AS (
            SELECT session, target_type FROM {prediction_relation}
        ), label_counts AS (
            SELECT labels.session, labels.target_type, count(*)::BIGINT AS label_count
            FROM read_parquet('{_quote(_path(labels))}') labels
            INNER JOIN prediction_groups USING (session, target_type)
            GROUP BY labels.session, labels.target_type
        ), predicted_items AS (
            SELECT session, target_type, unnest(aids)::BIGINT AS aid
            FROM {prediction_relation}
        ), hit_counts AS (
            SELECT
                labels.session,
                labels.target_type,
                count(*)::BIGINT AS hits
            FROM read_parquet('{_quote(_path(labels))}') labels
            INNER JOIN predicted_items USING (session, target_type, aid)
            GROUP BY labels.session, labels.target_type
        )
        SELECT
            label_counts.target_type,
            sum(coalesce(hit_counts.hits, 0))::BIGINT AS hits,
            sum(least(label_counts.label_count, {int(topk)}))::BIGINT AS denominator
        FROM label_counts
        LEFT JOIN hit_counts USING (session, target_type)
        GROUP BY label_counts.target_type
        ORDER BY label_counts.target_type
        """
    ).fetchall()
    connection.close()
    by_target: dict[str, Any] = {}
    weighted = 0.0
    for target_type, hits, denominator in rows:
        recall = int(hits) / int(denominator) if denominator else 0.0
        by_target[str(int(target_type))] = {
            "hits": int(hits),
            "denominator": int(denominator),
            "recall": round(recall, 8),
        }
        weighted += TARGET_WEIGHTS[int(target_type)] * recall
    return {
        "recall_at_20": {
            "by_target_type": by_target,
            "weighted": round(weighted, 8),
        }
    }, {"invalid_prediction_groups": invalid_groups}


def predict_lambdarank(
    features_dir: str | Path,
    model_dir: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
    config: Mapping[str, Any],
    *,
    max_buckets: int | None = None,
    max_groups_per_file: int | None = None,
) -> dict[str, Any]:
    """Generate and evaluate Top20 predictions from one feature shard at a time."""

    import lightgbm as lgb

    feature_root, feature_files, limited = _feature_files(
        features_dir, max_buckets=max_buckets
    )
    model_root = Path(model_dir).resolve()
    labels = Path(labels_path).resolve()
    destination = Path(output_dir).resolve()
    model_path = model_root / "model.txt"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not labels.is_file():
        raise FileNotFoundError(labels)
    _check_schema(labels, LABEL_SCHEMA, "label")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite prediction output: {destination}")
    model = lgb.Booster(model_file=str(model_path))
    model_feature_names = _validate_model(model_root, model)

    candidate_k = int(config["candidate_k"])
    topk = int(config["ranker"]["eval_at"])
    if topk != 20:
        raise ValueError("The canonical prediction schema currently requires eval_at=20")
    runtime = config["runtime"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    staging = destination.parent / f".{destination.name}.building-{token}"
    parts_dir = staging / "parts"
    work_dir = staging / "work"
    parts_dir.mkdir(parents=True)
    started = time.perf_counter()
    output_files: list[Path] = []
    file_reports: list[dict[str, Any]] = []
    total_candidate_rows = 0
    total_groups = 0
    try:
        for position, (bucket, target_type, feature_file) in enumerate(feature_files, start=1):
            output_file = (
                parts_dir
                / f"session-bucket-{bucket:05d}-target-{target_type}.parquet"
            )
            stats = _score_file(
                model,
                feature_file,
                output_file,
                candidate_k=candidate_k,
                topk=topk,
                compression=str(runtime["parquet_compression"]),
                row_group_size=int(runtime["parquet_row_group_size"]),
                max_groups_per_file=max_groups_per_file,
                feature_names=model_feature_names,
            )
            output_files.append(output_file)
            total_candidate_rows += int(stats["candidate_rows"])
            total_groups += int(stats["prediction_groups"])
            file_reports.append({
                "bucket": bucket,
                "target_type": target_type,
                **stats,
            })
            print(
                f"[lambdarank-predict] file={position}/{len(feature_files)} "
                f"groups={total_groups:,} elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
        metrics, violations = _evaluate(
            output_files,
            labels,
            topk=topk,
            workers=int(runtime["workers"]),
            memory_limit_gb=int(runtime["duckdb_memory_limit_gb"]),
            temp_dir=work_dir / "duckdb_tmp",
        )
        if violations["invalid_prediction_groups"]:
            raise AssertionError(f"Invalid Top20 predictions: {violations}")
        report = {
            "dataset": "valid",
            "model": str(model_root),
            "model_iteration": int(model.current_iteration()),
            "features": str(feature_root),
            "labels": str(labels),
            "output": str(destination),
            "configuration": {
                "candidate_k": candidate_k,
                "topk": topk,
                "streaming_unit": "one session-bucket and target-type shard",
                "max_buckets": max_buckets,
                "max_groups_per_file": max_groups_per_file,
                "model_feature_count": len(model_feature_names),
                "model_feature_names": list(model_feature_names),
            },
            "predictions": {
                "files": len(output_files),
                "groups": total_groups,
                "candidate_rows_scored": total_candidate_rows,
                "items_retained": total_groups * topk,
                "disk_bytes": _directory_size(parts_dir),
                "limited": limited or max_groups_per_file is not None,
            },
            "metrics": metrics,
            "assertions": {
                "passed": not any(violations.values()),
                "feature_order_matches_model": (
                    tuple(model.feature_name()) == model_feature_names
                ),
                "one_top20_row_per_group": True,
                "unique_aids_per_group": violations["invalid_prediction_groups"] == 0,
                "final_valid_not_used_for_training": True,
                "violations": violations,
            },
            "file_reports": file_reports,
            "timing": {
                "total_seconds": round(time.perf_counter() - started, 6),
            },
        }
        shutil.rmtree(work_dir, ignore_errors=True)
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.replace(destination)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream LambdaRank Top20 predictions.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-buckets", type=int)
    parser.add_argument("--max-groups-per-file", type=int)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    config_path = args.config or os.environ.get("OTTO_CONFIG_PATH")
    if not config_path:
        raise ValueError("Provide --config or run through the experiment runner")
    experiment_dir = os.environ.get("OTTO_EXPERIMENT_DIR")
    if args.output_dir:
        output_dir = args.output_dir
    elif experiment_dir:
        output_dir = Path(experiment_dir) / "predictions" / "valid" / "lambdarank"
    else:
        raise ValueError("Provide --output-dir or run through the experiment runner")
    report = predict_lambdarank(
        args.features_dir,
        args.model_dir,
        args.labels,
        output_dir,
        resolve_project_config(config_path),
        max_buckets=args.max_buckets,
        max_groups_per_file=args.max_groups_per_file,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
