"""Train the unified M9 LightGBM LambdaRank model.

Only groups containing at least one recalled positive are useful to a pairwise
ranking objective.  This trainer filters all-zero groups, keeps every one of
the existing K candidates in an eligible group, and makes a deterministic
session-level train/early-stopping split inside the ranker window.  The final
validation window is never read here.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import duckdb
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.schemas import RANKER_FEATURE_SCHEMA, RANKER_FEATURE_SCHEMA_VERSION
from features.registry import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    feature_names_for_groups,
    registry_payload_for_features,
)
from utils.config import resolve_project_config


TARGET_WEIGHTS = {1: 0.10, 2: 0.30, 3: 0.60}


def _quote(value: str | Path) -> str:
    return str(value).replace("'", "''")


def _path(path: Path) -> str:
    return path.resolve().as_posix()


def _sql_files(paths: Sequence[Path]) -> str:
    return "[" + ",".join(f"'{_quote(_path(path))}'" for path in paths) + "]"


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _feature_files(path: str | Path) -> tuple[Path, list[Path]]:
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted((root / "parts").glob("session-bucket-*-target-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No M8 feature shards below {root}")
    for file in files:
        schema = pq.read_schema(file)
        if schema.names != RANKER_FEATURE_SCHEMA.names or schema.types != RANKER_FEATURE_SCHEMA.types:
            raise AssertionError(f"Unexpected ranker feature schema in {file}: {schema}")
    registry_path = root / "feature_registry.json"
    if not registry_path.is_file():
        raise FileNotFoundError(registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registered_names = tuple(feature["name"] for feature in registry["features"])
    if registry.get("schema_version") != RANKER_FEATURE_SCHEMA_VERSION:
        raise AssertionError("Feature schema version differs from the training code")
    if registered_names != FEATURE_NAMES:
        raise AssertionError("Feature registry differs from the canonical 49-feature order")
    return root, files


def _configure_duckdb(
    connection: duckdb.DuckDBPyConnection,
    *,
    workers: int,
    memory_limit_gb: int,
    temp_dir: Path,
) -> None:
    temp_dir.mkdir(parents=True, exist_ok=True)
    connection.execute(f"SET threads={int(workers)}")
    connection.execute(f"SET memory_limit='{int(memory_limit_gb)}GB'")
    connection.execute(f"SET temp_directory='{_quote(_path(temp_dir))}'")
    connection.execute("SET preserve_insertion_order=false")


def _prepare_eligible_groups(
    connection: duckdb.DuckDBPyConnection,
    files: Sequence[Path],
    *,
    seed: int,
    max_groups: int | None,
) -> dict[str, int]:
    feature_relation = f"read_parquet({_sql_files(files)}, union_by_name=true)"
    connection.execute(
        f"""
        CREATE TEMP TABLE all_group_stats AS
        SELECT
            session::BIGINT AS session,
            target_type::TINYINT AS target_type,
            count(*)::INTEGER AS candidate_count,
            sum(label)::INTEGER AS positive_count
        FROM {feature_relation}
        GROUP BY session, target_type
        """
    )
    totals = connection.execute(
        """
        SELECT
            count(*)::BIGINT,
            sum(candidate_count)::BIGINT,
            count(*) FILTER (WHERE positive_count > 0)::BIGINT,
            sum(candidate_count) FILTER (WHERE positive_count > 0)::BIGINT,
            sum(positive_count)::BIGINT
        FROM all_group_stats
        """
    ).fetchone()
    limit_sql = f"LIMIT {int(max_groups)}" if max_groups else ""
    connection.execute(
        f"""
        CREATE TEMP TABLE eligible_groups AS
        SELECT session, target_type, candidate_count, positive_count
        FROM all_group_stats
        WHERE positive_count > 0
        ORDER BY hash(session, target_type, {int(seed)}), session, target_type
        {limit_sql}
        """
    )
    selected = connection.execute(
        """
        SELECT count(*)::BIGINT, sum(candidate_count)::BIGINT, sum(positive_count)::BIGINT
        FROM eligible_groups
        """
    ).fetchone()
    return {
        "input_groups": int(totals[0]),
        "input_rows": int(totals[1]),
        "eligible_groups_before_limit": int(totals[2]),
        "eligible_rows_before_limit": int(totals[3]),
        "input_positive_rows": int(totals[4]),
        "selected_groups": int(selected[0]),
        "selected_rows": int(selected[1]),
        "selected_positive_rows": int(selected[2]),
        "all_zero_groups_dropped": int(totals[0] - totals[2]),
    }


def _load_split(
    connection: duckdb.DuckDBPyConnection,
    files: Sequence[Path],
    *,
    seed: int,
    validation: bool,
    feature_names: Sequence[str],
):
    import pandas as pd

    split_condition = "= 0" if validation else "!= 0"
    selected_columns = [
        "features.session",
        "features.target_type",
        "features.aid",
        "features.candidate_rank",
        "features.label",
        *(f"features.{name}" for name in feature_names),
    ]
    query = f"""
        SELECT {','.join(selected_columns)}
        FROM read_parquet({_sql_files(files)}, union_by_name=true) features
        INNER JOIN eligible_groups eligible
          ON eligible.session = features.session
         AND eligible.target_type = features.target_type
        WHERE hash(features.session, {int(seed)}) % 10 {split_condition}
        ORDER BY features.session, features.target_type, features.candidate_rank
    """
    arrow = connection.execute(query).fetch_arrow_table()
    frame = arrow.to_pandas(split_blocks=True, self_destruct=True)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        split_name = "validation" if validation else "training"
        raise ValueError(f"Internal {split_name} split is empty")
    return frame


def _group_sizes(frame) -> list[int]:
    return (
        frame.groupby(["session", "target_type"], sort=False, observed=True)
        .size()
        .astype("int32")
        .tolist()
    )


def _within_candidate_recall(frame, scores, k: int) -> dict[str, Any]:
    ranked = frame[["session", "target_type", "aid", "label"]].copy()
    ranked["score"] = scores
    ranked = ranked.sort_values(
        ["session", "target_type", "score", "aid"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    top = ranked.groupby(["session", "target_type"], sort=False, observed=True).head(k)
    result: dict[str, Any] = {"k": k, "by_target_type": {}}
    weighted = 0.0
    for target_type, weight in TARGET_WEIGHTS.items():
        denominator = int(ranked.loc[ranked["target_type"] == target_type, "label"].sum())
        hits = int(top.loc[top["target_type"] == target_type, "label"].sum())
        recall = hits / denominator if denominator else 0.0
        result["by_target_type"][str(target_type)] = {
            "hits": hits,
            "denominator": denominator,
            "recall": round(recall, 8),
        }
        weighted += weight * recall
    result["weighted"] = round(weighted, 8)
    result["scope"] = "recalled positives inside eligible groups; not final offline recall"
    return result


def train_lambdarank(
    features_dir: str | Path,
    output_dir: str | Path,
    config: Mapping[str, Any],
    *,
    max_groups: int | None = None,
    num_boost_round: int | None = None,
    feature_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Train a unified LambdaRank model from a canonical M8 feature subset."""

    import lightgbm as lgb
    import numpy as np

    feature_root, files = _feature_files(features_dir)
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite LambdaRank output: {destination}")
    ranker_config = config["ranker"]
    runtime_config = config["runtime"]
    seed = int(config["seed"])
    rounds = int(num_boost_round or ranker_config["num_boost_round"])
    early_stopping_rounds = int(ranker_config["early_stopping_rounds"])
    eval_at = int(ranker_config["eval_at"])
    selected_features = tuple(feature_names or FEATURE_NAMES)
    # This also validates uniqueness, membership and canonical order.
    selected_registry = registry_payload_for_features(selected_features)
    if max_groups is not None and max_groups < 20:
        raise ValueError("max_groups must be at least 20 to create both internal splits")

    destination.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    staging = destination.parent / f".{destination.name}.building-{token}"
    work_dir = staging / "work"
    staging.mkdir(parents=True)
    connection = duckdb.connect()
    _configure_duckdb(
        connection,
        workers=int(runtime_config["workers"]),
        memory_limit_gb=int(runtime_config["duckdb_memory_limit_gb"]),
        temp_dir=work_dir / "duckdb_tmp",
    )
    started = time.perf_counter()
    try:
        group_report = _prepare_eligible_groups(
            connection, files, seed=seed, max_groups=max_groups
        )
        load_started = time.perf_counter()
        train_frame = _load_split(
            connection,
            files,
            seed=seed,
            validation=False,
            feature_names=selected_features,
        )
        valid_frame = _load_split(
            connection,
            files,
            seed=seed,
            validation=True,
            feature_names=selected_features,
        )
        connection.close()
        shutil.rmtree(work_dir, ignore_errors=True)
        load_seconds = time.perf_counter() - load_started

        train_groups = _group_sizes(train_frame)
        valid_groups = _group_sizes(valid_frame)
        train_labels = train_frame["label"].to_numpy(dtype=np.float32, copy=True)
        valid_labels = valid_frame["label"].to_numpy(dtype=np.float32, copy=True)
        train_weights = train_frame["target_type"].map(TARGET_WEIGHTS).to_numpy(
            dtype=np.float32, copy=True
        )
        valid_weights = valid_frame["target_type"].map(TARGET_WEIGHTS).to_numpy(
            dtype=np.float32, copy=True
        )
        train_features = train_frame.loc[:, list(selected_features)]
        valid_features = valid_frame.loc[:, list(selected_features)]
        categorical_features = (
            ["target_type_id"] if "target_type_id" in selected_features else []
        )
        train_set = lgb.Dataset(
            train_features,
            label=train_labels,
            group=train_groups,
            weight=train_weights,
            params={"data_random_seed": seed},
            feature_name=list(selected_features),
            categorical_feature=categorical_features,
            free_raw_data=True,
        )
        train_set.construct()
        del train_features, train_labels, train_weights, train_frame
        gc.collect()
        valid_set = lgb.Dataset(
            valid_features,
            label=valid_labels,
            group=valid_groups,
            weight=valid_weights,
            reference=train_set,
            params={"data_random_seed": seed},
            feature_name=list(selected_features),
            categorical_feature=categorical_features,
            free_raw_data=False,
        )
        valid_set.construct()

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [eval_at],
            "learning_rate": float(ranker_config["learning_rate"]),
            "num_leaves": int(ranker_config["num_leaves"]),
            "min_data_in_leaf": int(ranker_config["min_data_in_leaf"]),
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": int(runtime_config["workers"]),
            "verbosity": -1,
        }
        evaluation_result: dict[str, Any] = {}
        train_started = time.perf_counter()
        model = lgb.train(
            params,
            train_set,
            num_boost_round=rounds,
            valid_sets=[valid_set],
            valid_names=["internal_valid"],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=True),
                lgb.log_evaluation(period=10),
                lgb.record_evaluation(evaluation_result),
            ],
        )
        training_seconds = time.perf_counter() - train_started
        model_path = staging / "model.txt"
        model.save_model(str(model_path), num_iteration=model.best_iteration)

        predictions = model.predict(valid_features, num_iteration=model.best_iteration)
        recall = _within_candidate_recall(valid_frame, predictions, eval_at)
        importance = sorted(
            zip(
                selected_features,
                model.feature_importance(importance_type="gain"),
                model.feature_importance(importance_type="split"),
            ),
            key=lambda row: (-float(row[1]), row[0]),
        )
        with (staging / "feature_importance.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["feature", "gain", "split"])
            writer.writerows(importance)
        schema_payload = selected_registry
        schema_payload["model_feature_names"] = list(model.feature_name())
        (staging / "feature_schema.json").write_text(
            json.dumps(schema_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report = {
            "model": "unified_lightgbm_lambdarank",
            "output": str(destination),
            "features": str(feature_root),
            "schema_version": RANKER_FEATURE_SCHEMA_VERSION,
            "feature_count": len(selected_features),
            "selected_feature_groups": selected_registry["enabled_groups"],
            "selected_feature_names": list(selected_features),
            "group_filter": group_report,
            "split": {
                "rule": "hash(session, seed) % 10; zero=internal_valid",
                "seed": seed,
                "train_rows": int(sum(train_groups)),
                "train_groups": len(train_groups),
                "internal_valid_rows": int(sum(valid_groups)),
                "internal_valid_groups": len(valid_groups),
                "session_level": True,
            },
            "parameters": params,
            "requested_num_boost_round": rounds,
            "early_stopping_rounds": early_stopping_rounds,
            "best_iteration": int(model.best_iteration),
            "best_score": model.best_score,
            "evaluation_history": evaluation_result,
            "internal_valid_within_candidate_recall": recall,
            "artifacts": {
                "model": "model.txt",
                "feature_schema": "feature_schema.json",
                "feature_importance": "feature_importance.csv",
                "model_bytes": model_path.stat().st_size,
            },
            "assertions": {
                "passed": (
                    len(model.feature_name()) == len(selected_features)
                    and tuple(model.feature_name()) == selected_features
                    and min(train_groups + valid_groups) > 0
                ),
                "all_zero_groups_removed": True,
                "feature_order_matches_registry": (
                    tuple(model.feature_name()) == selected_features
                ),
                "final_valid_not_used": True,
            },
            "timing": {
                "load_seconds": round(load_seconds, 6),
                "training_seconds": round(training_seconds, 6),
                "total_seconds": round(time.perf_counter() - started, 6),
            },
        }
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        del valid_features, valid_labels, valid_weights, valid_frame, valid_set, train_set
        gc.collect()
        staging.replace(destination)
        return report
    except Exception:
        try:
            connection.close()
        except Exception:
            pass
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train unified LightGBM LambdaRank.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--num-boost-round", type=int)
    parser.add_argument(
        "--feature-groups",
        nargs="+",
        choices=tuple(FEATURE_GROUPS),
        help="Optional whole feature groups to train; defaults to all 49 features.",
    )
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
        output_dir = Path(experiment_dir) / "models" / "lambdarank"
    else:
        raise ValueError("Provide --output-dir or run through the experiment runner")
    report = train_lambdarank(
        args.features_dir,
        output_dir,
        resolve_project_config(config_path),
        max_groups=args.max_groups,
        num_boost_round=args.num_boost_round,
        feature_names=(
            feature_names_for_groups(args.feature_groups)
            if args.feature_groups
            else None
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
