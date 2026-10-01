"""Generate and evaluate session candidates from a Buy2Buy matrix."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.schemas import EVENT_TYPE_TO_ID
from recall.generate_type_covis import generate_type_covis
from utils.config import resolve_project_config


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate candidates from Buy2Buy Top-K.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dataset-name", choices=("ranker", "valid"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--topk", type=int)
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
        output_dir = (
            Path(experiment_dir) / "recall" / args.dataset_name / "buy2buy" / "recall"
        )
    else:
        raise ValueError("Provide --output-dir or run through the experiment runner")

    recall_config = config["recall"]
    buy_config = recall_config["buy2buy"]
    behavior_weights = {
        EVENT_TYPE_TO_ID[name]: float(value)
        for name, value in buy_config["query_event_type_weights"].items()
    }
    target_weights = {
        EVENT_TYPE_TO_ID[name]: float(value)
        for name, value in recall_config["type_weights"].items()
    }
    report = generate_type_covis(
        args.matrix_dir,
        args.queries,
        args.labels,
        output_dir,
        dataset_name=args.dataset_name,
        recent_events=buy_config["recent_events_per_session"],
        recency_scale_events=buy_config["query_recency_scale_events"],
        event_type_weights=behavior_weights,
        topk=args.topk or recall_config["topk_per_source"],
        eval_ks=recall_config["eval_ks"],
        target_weights=target_weights,
        aid_buckets=buy_config["pair_buckets"],
        compression=config["runtime"]["parquet_compression"],
        row_group_size=config["runtime"]["parquet_row_group_size"],
        workers=config["runtime"]["workers"],
        memory_limit_gb=config["runtime"]["duckdb_memory_limit_gb"],
        source_name="buy2buy",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
