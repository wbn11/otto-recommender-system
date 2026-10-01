"""Unified command entrypoint for the maintained OTTO pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import cast


@dataclass(frozen=True)
class Task:
    path: str
    description: str
    example: str


SRC_DIR = Path(__file__).resolve().parents[1]

TASKS = {
    "check-environment": Task(
        "tools/check_environment.py",
        "Validate Python, CUDA, FAISS and pipeline dependencies.",
        "python src/pipeline/run.py check-environment --require-gpu",
    ),
    "ingest-events": Task(
        "data/ingest.py",
        "Stream raw OTTO JSONL into canonical Parquet shards.",
        "python src/pipeline/run.py ingest-events --dataset train",
    ),
    "build-time-splits": Task(
        "data/build_time_splits.py",
        "Build point-in-time snapshots, prefix queries and future labels.",
        "python src/pipeline/run.py build-time-splits "
        "--events-dir EVENTS --sessions-dir SESSIONS",
    ),
    "prepare-dssm-data": Task(
        "data/prepare_dssm_data.py",
        "Build the item vocabulary and streaming DSSM sequences.",
        "python src/pipeline/run.py prepare-dssm-data "
        "--snapshot-dir SNAPSHOT/events --dataset-name ranker",
    ),
    "train-dssm-baseline": Task(
        "models/train_dssm_baseline.py",
        "Train the fixed-position DSSM comparison model.",
        "python src/pipeline/run.py train-dssm-baseline --data-dir DSSM_DATA",
    ),
    "train-dssm-attention": Task(
        "models/train_dssm_attention.py",
        "Train the final target-aware attention DSSM.",
        "python src/pipeline/run.py train-dssm-attention --data-dir DSSM_DATA",
    ),
    "dssm-recall": Task(
        "recall/generate_dssm.py",
        "Generate target-conditioned DSSM Top-K candidates with FAISS FlatIP.",
        "python src/pipeline/run.py dssm-recall --data-dir DSSM_DATA "
        "--checkpoint MODEL.pt --queries QUERIES --labels LABELS "
        "--dataset-name ranker",
    ),
    "analyze-dssm-history": Task(
        "evaluation/analyze_dssm_history.py",
        "Evaluate DSSM recall by effective query-history length.",
        "python src/pipeline/run.py analyze-dssm-history --data-dir DSSM_DATA "
        "--candidates DSSM_RECALL --queries QUERIES --labels LABELS "
        "--dataset-name ranker",
    ),
    "build-popular-revisit": Task(
        "recall/build_popular_revisit.py",
        "Build and evaluate point-in-time Popular and Revisit recall.",
        "python src/pipeline/run.py build-popular-revisit "
        "--snapshot-dir SNAPSHOT/events --queries QUERIES --labels LABELS "
        "--dataset-name ranker",
    ),
    "build-type-covis": Task(
        "recall/build_type_covis.py",
        "Build the type-weighted co-visitation Top-K matrix.",
        "python src/pipeline/run.py build-type-covis "
        "--snapshot-dir SNAPSHOT/events --dataset-name ranker",
    ),
    "type-covis-recall": Task(
        "recall/generate_type_covis.py",
        "Generate candidates from a Type-CoVis matrix.",
        "python src/pipeline/run.py type-covis-recall --matrix-dir MATRIX "
        "--queries QUERIES --labels LABELS --dataset-name ranker",
    ),
    "build-buy2buy": Task(
        "recall/build_buy2buy.py",
        "Build the cart/order-only Buy2Buy Top-K matrix.",
        "python src/pipeline/run.py build-buy2buy "
        "--snapshot-dir SNAPSHOT/events --dataset-name ranker",
    ),
    "buy2buy-recall": Task(
        "recall/generate_buy2buy.py",
        "Generate candidates from a Buy2Buy matrix.",
        "python src/pipeline/run.py buy2buy-recall --matrix-dir MATRIX "
        "--queries QUERIES --labels LABELS --dataset-name ranker",
    ),
    "build-time-covis": Task(
        "recall/build_time_covis.py",
        "Build the time-decayed co-visitation Top-K matrix.",
        "python src/pipeline/run.py build-time-covis "
        "--snapshot-dir SNAPSHOT/events --dataset-name ranker",
    ),
    "time-covis-recall": Task(
        "recall/generate_time_covis.py",
        "Generate candidates from a Time-CoVis matrix.",
        "python src/pipeline/run.py time-covis-recall --matrix-dir MATRIX "
        "--queries QUERIES --labels LABELS --dataset-name ranker",
    ),
    "fuse-candidates": Task(
        "recall/fuse_candidates.py",
        "Fuse six recall sources into the bounded Top-K candidate table.",
        "python src/pipeline/run.py fuse-candidates --queries QUERIES "
        "--labels LABELS --popular POPULAR --revisit REVISIT "
        "--type-covis TYPE --buy2buy BUY --time-covis TIME "
        "--dssm DSSM --dataset-name ranker",
    ),
    "build-features": Task(
        "features/build_features.py",
        "Build the versioned point-in-time LambdaRank feature table.",
        "python src/pipeline/run.py build-features --candidates CANDIDATES "
        "--queries QUERIES --labels LABELS --snapshot-dir SNAPSHOT/events "
        "--type-covis-matrix TYPE/matrix --buy2buy-matrix BUY/matrix "
        "--time-covis-matrix TIME/matrix --dataset-name ranker",
    ),
    "train-lambdarank": Task(
        "rank/train_lambdarank.py",
        "Train the unified 49-feature LightGBM LambdaRank model.",
        "python src/pipeline/run.py train-lambdarank --features-dir FEATURES",
    ),
    "select-lambdarank-features": Task(
        "rank/select_lambdarank_features.py",
        "Select LambdaRank feature groups by backward elimination.",
        "python src/pipeline/run.py select-lambdarank-features "
        "--features-dir FEATURES --max-allowed-ndcg-drop 0.0002",
    ),
    "predict-lambdarank": Task(
        "rank/predict_lambdarank.py",
        "Stream feature shards through LambdaRank and evaluate Top20.",
        "python src/pipeline/run.py predict-lambdarank --features-dir FEATURES "
        "--model-dir MODEL --labels LABELS",
    ),
}


TASK_GROUPS = {
    "Environment": ("check-environment",),
    "Data": ("ingest-events", "build-time-splits", "prepare-dssm-data"),
    "Traditional recall": (
        "build-popular-revisit",
        "build-type-covis",
        "type-covis-recall",
        "build-buy2buy",
        "buy2buy-recall",
        "build-time-covis",
        "time-covis-recall",
    ),
    "DSSM": (
        "train-dssm-baseline",
        "train-dssm-attention",
        "dssm-recall",
        "analyze-dssm-history",
    ),
    "Ranking": (
        "fuse-candidates",
        "build-features",
        "train-lambdarank",
        "select-lambdarank-features",
        "predict-lambdarank",
    ),
}


def load_task_module(task_name: str, task: Task) -> ModuleType:
    module_path = SRC_DIR / task.path
    module_name = f"otto_task_{task_name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load task module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def run_task(task_name: str, task_args: list[str] | None = None) -> None:
    task = TASKS[task_name]
    module = load_task_module(task_name, task)
    task_main = getattr(module, "main", None)
    if not callable(task_main):
        raise AttributeError(f"{task.path} does not expose a callable main()")

    task_callable = cast(Callable[..., object], task_main)
    arguments = task_args or []
    if arguments and not inspect.signature(task_callable).parameters:
        raise ValueError(f"{task_name} does not accept task arguments: {arguments}")
    if inspect.signature(task_callable).parameters:
        task_callable(arguments)
    else:
        task_callable()


def print_tasks() -> None:
    print("Tasks:")
    task_name_width = max(len(task_name) for task_name in TASKS)
    for group_name, task_names in TASK_GROUPS.items():
        print(f"  {group_name}:")
        for task_name in task_names:
            task = TASKS[task_name]
            print(
                f"    {task_name:<{task_name_width}} "
                f"{task.description} Example: {task.example}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified entrypoint for the maintained OTTO pipeline."
    )
    parser.add_argument("task", nargs="?", choices=sorted(TASKS))
    parser.add_argument("task_args", nargs=argparse.REMAINDER)
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list or args.task is None:
        print_tasks()
        return
    run_task(args.task, args.task_args)


if __name__ == "__main__":
    main()
