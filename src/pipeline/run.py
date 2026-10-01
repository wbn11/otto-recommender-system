"""统一项目命令入口。

注册数据构建、召回、排序、评估和 test submission 任务，
并提供 validation/ranker/test/all 等流程级 workflow。
"""

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


@dataclass(frozen=True)
class WorkflowStep:
    task_name: str
    args: tuple[str, ...] = ()


@dataclass(frozen=True)
class Workflow:
    description: str
    steps: tuple[WorkflowStep, ...]
    example: str


SRC_DIR = Path(__file__).resolve().parents[1]

TASKS = {
    "check-environment": Task(
        "tools/check_environment.py",
        "Validate Python, CUDA, FAISS and pipeline dependencies.",
        "python src/pipeline/run.py check-environment --require-gpu",
    ),
    "build-validation": Task(
        "data/build_validation.py",
        "Build multi-target validation files.",
        "python src/pipeline/run.py build-validation",
    ),
    "ingest-events": Task(
        "data/ingest.py",
        "Stream raw OTTO JSONL into canonical Parquet shards.",
        "python src/pipeline/run.py ingest-events "
        "--config configs/experiments/debug.yaml --max-sessions 1000",
    ),
    "build-time-splits": Task(
        "data/build_time_splits.py",
        "Build point-in-time snapshots, prefix queries and future labels.",
        "python src/pipeline/run.py build-time-splits "
        "--events-dir EVENTS --sessions-dir SESSIONS",
    ),
    "prepare-dssm-data": Task(
        "data/prepare_dssm_data.py",
        "Build a deterministic item vocabulary and streaming DSSM session sequences.",
        "python src/pipeline/run.py prepare-dssm-data --snapshot-dir SNAPSHOT/events --dataset-name ranker",
    ),
    "inspect-dssm-loader": Task(
        "models/inspect_dssm_loader.py",
        "Check streaming next-item batches against the DSSM data report.",
        "python src/pipeline/run.py inspect-dssm-loader --data-dir DSSM_DATA",
    ),
    "train-dssm-baseline": Task(
        "models/train_dssm_baseline.py",
        "Train fixed-position DSSM from streaming Parquet with safe in-batch negatives.",
        "python src/pipeline/run.py train-dssm-baseline --data-dir DSSM_DATA --max-batches 10",
    ),
    "train-dssm-attention": Task(
        "models/train_dssm_attention.py",
        "Train target-aware attention DSSM with learned recency embeddings.",
        "python src/pipeline/run.py train-dssm-attention "
        "--data-dir DSSM_DATA --max-batches 10",
    ),
    "dssm-recall": Task(
        "recall/generate_dssm.py",
        "Generate target-conditioned DSSM Top-K lists with exact FAISS FlatIP.",
        "python src/pipeline/run.py dssm-recall --data-dir DSSM_DATA "
        "--checkpoint MODEL.pt --queries QUERIES.parquet "
        "--labels LABELS.parquet --dataset-name ranker",
    ),
    "analyze-dssm-history": Task(
        "evaluation/analyze_dssm_history.py",
        "Evaluate persisted DSSM candidates by effective query-history length.",
        "python src/pipeline/run.py analyze-dssm-history --data-dir DSSM_DATA "
        "--candidates DSSM_RECALL --queries QUERIES.parquet "
        "--labels LABELS.parquet --dataset-name ranker",
    ),
    "fuse-candidates": Task(
        "recall/fuse_candidates.py",
        "Fuse six recall sources into a bounded candidate table.",
        "python src/pipeline/run.py fuse-candidates --queries QUERIES "
        "--labels LABELS --popular POPULAR --revisit REVISIT "
        "--type-covis TYPE --buy2buy BUY --time-covis TIME "
        "--dssm DSSM --dataset-name ranker",
    ),
    "build-features": Task(
        "features/build_features.py",
        "Build the versioned point-in-time LightGBM feature table.",
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
        "Select LambdaRank feature groups with deterministic backward elimination.",
        "python src/pipeline/run.py select-lambdarank-features "
        "--features-dir FEATURES --max-allowed-ndcg-drop 0.0002",
    ),
    "predict-lambdarank": Task(
        "rank/predict_lambdarank.py",
        "Stream feature shards through LambdaRank and evaluate Top20.",
        "python src/pipeline/run.py predict-lambdarank --features-dir FEATURES "
        "--model-dir MODEL --labels LABELS",
    ),
    "build-popular-revisit": Task(
        "recall/build_popular_revisit.py",
        "Build and evaluate snapshot-safe Popular and Revisit recall.",
        "python src/pipeline/run.py build-popular-revisit --snapshot-dir SNAPSHOT/events "
        "--queries QUERIES.parquet --labels LABELS.parquet --dataset-name ranker",
    ),
    "build-type-covis": Task(
        "recall/build_type_covis.py",
        "Build a disk-backed type-weighted co-visitation Top-K matrix.",
        "python src/pipeline/run.py build-type-covis --snapshot-dir SNAPSHOT/events --dataset-name ranker",
    ),
    "type-covis-recall": Task(
        "recall/generate_type_covis.py",
        "Generate and evaluate session candidates from a type-CoVis matrix.",
        "python src/pipeline/run.py type-covis-recall --matrix-dir MATRIX "
        "--queries QUERIES --labels LABELS --dataset-name ranker",
    ),
    "build-buy2buy": Task(
        "recall/build_buy2buy.py",
        "Build a cart/order-only Buy2Buy Top-K matrix.",
        "python src/pipeline/run.py build-buy2buy --snapshot-dir SNAPSHOT/events --dataset-name ranker",
    ),
    "buy2buy-recall": Task(
        "recall/generate_buy2buy.py",
        "Generate and evaluate session candidates from a Buy2Buy matrix.",
        "python src/pipeline/run.py buy2buy-recall --matrix-dir MATRIX "
        "--queries QUERIES --labels LABELS --dataset-name ranker",
    ),
    "build-time-covis": Task(
        "recall/build_time_covis.py",
        "Build a time-decayed co-visitation Top-K matrix.",
        "python src/pipeline/run.py build-time-covis --snapshot-dir SNAPSHOT/events --dataset-name ranker",
    ),
    "time-covis-recall": Task(
        "recall/generate_time_covis.py",
        "Generate and evaluate session candidates from a time-CoVis matrix.",
        "python src/pipeline/run.py time-covis-recall --matrix-dir MATRIX "
        "--queries QUERIES --labels LABELS --dataset-name ranker",
    ),
    "build-test-events": Task(
        "data/build_test_events.py",
        "Build multi-target test events.",
        "python src/pipeline/run.py build-test-events --nrows 1000",
    ),
    "popular-recall": Task(
        "recall/popular_recall.py",
        "Generate multi-target popular item recall.",
        "python src/pipeline/run.py popular-recall",
    ),
    "build-covis-matrix": Task(
        "recall/build_covis_matrix.py",
        "Build multi-target co-visitation top-k matrix.",
        "python src/pipeline/run.py build-covis-matrix --top-k 50",
    ),
    "covisitation-recall": Task(
        "recall/covisitation_recall.py",
        "Generate multi-target co-visitation recall from saved matrix.",
        "python src/pipeline/run.py covisitation-recall --k 50",
    ),
    "fusion-recall": Task(
        "recall/fusion_recall.py",
        "Fuse multi-source multi-target recall predictions.",
        "python src/pipeline/run.py fusion-recall",
    ),
    "build-recall-candidates": Task(
        "recall/build_recall_candidates.py",
        "Merge multi-source recall predictions into a candidate pool.",
        "python src/pipeline/run.py build-recall-candidates",
    ),
    "build-ranker-train-data": Task(
        "rank/build_ranker_train_data.py",
        "Build multi-target ranker training data from recall candidates.",
        "python src/pipeline/run.py build-ranker-train-data",
    ),
    "build-ranker-inference-data": Task(
        "rank/build_ranker_inference_data.py",
        "Build multi-target ranker inference data from recall candidates.",
        "python src/pipeline/run.py build-ranker-inference-data",
    ),
    "train-ranker": Task(
        "rank/train_ranker.py",
        "Train the multi-target LightGBM ranker.",
        "python src/pipeline/run.py train-ranker",
    ),
    "ranker-predict": Task(
        "rank/predict_ranker.py",
        "Generate multi-target ranker predictions.",
        "python src/pipeline/run.py ranker-predict",
    ),
    "evaluate": Task(
        "evaluation/evaluate.py",
        "Evaluate multi-target recall predictions.",
        "python src/pipeline/run.py evaluate --pred-file ranker_predictions.csv",
    ),
    "build-submission": Task(
        "evaluation/build_submission.py",
        "Build Kaggle submission from predictions.",
        "python src/pipeline/run.py build-submission --pred-file test_ranker_predictions.csv",
    ),
    "analyze-recall-candidates": Task(
        "evaluation/analyze_recall_candidates.py",
        "Analyze multi-target recall candidate oracle recall.",
        "python src/pipeline/run.py analyze-recall-candidates",
    ),
}


# The original end-to-end workflows depended on the retired pandas DSSM and
# are intentionally removed. New milestone workflows are registered only once
# all of their strict point-in-time stages have been implemented.
WORKFLOWS: dict[str, Workflow] = {}
WORKFLOW_GROUPS: dict[str, tuple[str, ...]] = {}

TASK_GROUPS = {
    "Environment": (
        "check-environment",
    ),
    "Data": (
        "ingest-events",
        "build-time-splits",
        "prepare-dssm-data",
        "inspect-dssm-loader",
        "train-dssm-baseline",
        "train-dssm-attention",
        "build-validation",
        "build-test-events",
    ),
    "Recall": (
        "build-popular-revisit",
        "build-type-covis",
        "type-covis-recall",
        "build-buy2buy",
        "buy2buy-recall",
        "build-time-covis",
        "time-covis-recall",
        "dssm-recall",
        "fuse-candidates",
        "popular-recall",
        "build-covis-matrix",
        "covisitation-recall",
        "fusion-recall",
        "build-recall-candidates",
    ),
    "Ranker": (
        "build-features",
        "train-lambdarank",
        "select-lambdarank-features",
        "predict-lambdarank",
        "build-ranker-train-data",
        "build-ranker-inference-data",
        "train-ranker",
        "ranker-predict",
    ),
    "Evaluation": (
        "analyze-dssm-history",
        "analyze-recall-candidates",
        "evaluate",
        "build-submission",
    ),
}


def load_task_module(task_name: str, task: Task) -> ModuleType:
    module_path = SRC_DIR / task.path
    module_name = f"otto_task_{task_name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load task module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    # Some runtime features (notably dataclasses on Python 3.10) resolve type
    # information through sys.modules while the module is being executed.
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
    task_args = task_args or []
    main_params = inspect.signature(task_callable).parameters

    if task_args and not main_params:
        raise ValueError(f"{task_name} does not accept task arguments: {task_args}")

    if main_params:
        task_callable(task_args)
    else:
        task_callable()


def run_workflow(workflow_name: str) -> None:
    workflow = WORKFLOWS[workflow_name]
    print(f"Running workflow: {workflow_name}")
    print(f"{workflow.description}")

    for index, step in enumerate(workflow.steps, start=1):
        # Workflows call the same task runner as single-task execution.
        args_text = " ".join(step.args)
        command_text = f"{step.task_name} {args_text}".strip()
        print(f"\n[{index}/{len(workflow.steps)}] {command_text}")
        run_task(step.task_name, list(step.args))


def print_tasks() -> None:
    if WORKFLOWS:
        print("Workflows:")
        workflow_name_width = max(len(workflow_name) for workflow_name in WORKFLOWS)
        for group_name, workflow_names in WORKFLOW_GROUPS.items():
            print(f"  {group_name}:")
            for workflow_name in workflow_names:
                workflow = WORKFLOWS[workflow_name]
                print(
                    f"    {workflow_name:<{workflow_name_width}} "
                    f"{workflow.description} Example: {workflow.example}"
                )
        print()
    print("Tasks:")
    task_name_width = max(len(task_name) for task_name in TASKS)
    for group_name, task_names in TASK_GROUPS.items():
        print(f"  {group_name}:")
        for task_name in task_names:
            task = TASKS[task_name]
            print(f"    {task_name:<{task_name_width}} {task.description} Example: {task.example}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified entrypoint for OTTO recommendation experiments.")
    parser.add_argument("task", nargs="?", choices=sorted(TASKS), help="Task to run.")
    parser.add_argument("task_args", nargs=argparse.REMAINDER, help="Arguments passed to the selected task.")
    parser.add_argument("--workflow", choices=sorted(WORKFLOWS), help="Workflow preset to run.")
    parser.add_argument("--list", action="store_true", help="List available tasks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        print_tasks()
        return

    if args.workflow:
        if args.task:
            raise ValueError("Use either a workflow or a task, not both.")
        run_workflow(args.workflow)
        return

    if args.task is None:
        print_tasks()
        return

    run_task(args.task, args.task_args)


if __name__ == "__main__":
    main()
