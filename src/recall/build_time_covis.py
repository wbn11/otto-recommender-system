"""Build a time-decayed co-visitation matrix."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recall.build_type_covis import build_type_covis
from utils.config import resolve_project_config


MILLISECONDS_PER_HOUR = 60 * 60 * 1000


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a time-decayed CoVis Top-K matrix.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", choices=("ranker", "valid"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--recent-events", type=int)
    parser.add_argument("--topk-neighbors", type=int)
    parser.add_argument("--pair-buckets", type=int)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    config_path = args.config or os.environ.get("OTTO_CONFIG_PATH")
    if not config_path:
        raise ValueError("Provide --config or run through the experiment runner")
    config = resolve_project_config(config_path)
    experiment_dir = os.environ.get("OTTO_EXPERIMENT_DIR")
    if args.output_dir:
        output_dir = args.output_dir
    elif experiment_dir:
        output_dir = Path(experiment_dir) / "recall" / args.dataset_name / "time_covis"
    else:
        raise ValueError("Provide --output-dir or run through the experiment runner")

    time_config = config["recall"]["time_covis"]
    max_time_difference_ms = round(
        float(time_config["max_time_difference_hours"]) * MILLISECONDS_PER_HOUR
    )
    time_decay_ms = round(
        float(time_config["time_decay_hours"]) * MILLISECONDS_PER_HOUR
    )
    report = build_type_covis(
        args.snapshot_dir,
        output_dir,
        dataset_name=args.dataset_name,
        recent_events_per_session=(
            args.recent_events or time_config["recent_events_per_session"]
        ),
        topk_neighbors=args.topk_neighbors or time_config["topk_neighbors"],
        pair_buckets=args.pair_buckets or time_config["pair_buckets"],
        event_type_weights={1: 1.0, 2: 1.0, 3: 1.0},
        compression=config["runtime"]["parquet_compression"],
        row_group_size=config["runtime"]["parquet_row_group_size"],
        workers=config["runtime"]["workers"],
        memory_limit_gb=config["runtime"]["duckdb_memory_limit_gb"],
        source_name="time_covis",
        allowed_event_types=(1, 2, 3),
        pair_score_mode="time_decay",
        max_time_difference_ms=max_time_difference_ms,
        time_decay_ms=time_decay_ms,
        pair_semantics=(
            "directed unique item pairs per session within the recent event window; "
            "each session contributes the maximum exp(-abs(delta_ts)/decay) for the "
            "pair when abs(delta_ts) is within the configured maximum"
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
