from __future__ import annotations

import argparse
import inspect
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


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
    "build-validation": Task(
        "data/build_validation.py",
        "Build multi-target validation files.",
        "python src/pipeline/run.py build-validation",
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
    "dssm-recall": Task(
        "recall/dssm_recall.py",
        "Generate multi-target DSSM recall predictions.",
        "python src/pipeline/run.py dssm-recall --k 50",
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
    "train-dssm": Task(
        "models/train_dssm.py",
        "Train the multi-target DSSM recall model.",
        "python src/pipeline/run.py train-dssm --max-pairs 1000000 --epochs 5",
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


WORKFLOWS = {
    "validation": Workflow(
        "Build validation recall candidates and analyze candidate oracle.",
        (
            WorkflowStep("build-validation"),
            WorkflowStep("build-covis-matrix", ("--top-k", "50")),
            WorkflowStep("popular-recall"),
            WorkflowStep("covisitation-recall", ("--k", "50")),
            WorkflowStep("dssm-recall", ("--k", "50")),
            WorkflowStep("build-recall-candidates"),
            WorkflowStep("analyze-recall-candidates"),
        ),
        "python src/pipeline/run.py --workflow validation",
    ),
    "ranker": Workflow(
        "Train and evaluate the LightGBM ranker on validation data.",
        (
            WorkflowStep("build-ranker-train-data"),
            WorkflowStep("train-ranker"),
            WorkflowStep("ranker-predict"),
            WorkflowStep("evaluate", ("--pred-file", "ranker_predictions.csv")),
        ),
        "python src/pipeline/run.py --workflow ranker",
    ),
    "test": Workflow(
        "Build test predictions and submission from prepared model artifacts.",
        (
            WorkflowStep("build-test-events"),
            WorkflowStep(
                "popular-recall",
                (
                    "--train-file", "test_events.parquet",
                    "--test-events-file", "test_events.parquet",
                    "--output-file", "test_popular_predictions.csv",
                ),
            ),
            WorkflowStep(
                "covisitation-recall",
                (
                    "--train-file", "test_events.parquet",
                    "--test-events-file", "test_events.parquet",
                    "--output-file", "test_covisitation_predictions.csv",
                    "--k", "50",
                ),
            ),
            WorkflowStep(
                "dssm-recall",
                (
                    "--train-file", "test_events.parquet",
                    "--test-events-file", "test_events.parquet",
                    "--output-file", "test_dssm_predictions.csv",
                    "--k", "50",
                ),
            ),
            WorkflowStep(
                "build-recall-candidates",
                (
                    "--popular-file", "test_popular_predictions.csv",
                    "--covis-file", "test_covisitation_predictions.csv",
                    "--dssm-file", "test_dssm_predictions.csv",
                    "--output-file", "test_recall_candidates.parquet",
                ),
            ),
            WorkflowStep("build-ranker-inference-data"),
            WorkflowStep(
                "ranker-predict",
                (
                    "--candidates-file", "test_ranker_data.parquet",
                    "--test-events-file", "test_events.parquet",
                    "--output-file", "test_ranker_predictions.csv",
                ),
            ),
            WorkflowStep("build-submission", ("--pred-file", "test_ranker_predictions.csv")),
        ),
        "python src/pipeline/run.py --workflow test",
    ),
}
WORKFLOWS["all"] = Workflow(
    "Run validation candidate building, ranker training, and evaluation.",
    WORKFLOWS["validation"].steps + WORKFLOWS["ranker"].steps,
    "python src/pipeline/run.py --workflow all",
)


WORKFLOW_GROUPS = {
    "Main workflows": ("all", "validation", "ranker", "test"),
}

TASK_GROUPS = {
    "Data": (
        "build-validation",
        "build-test-events",
    ),
    "Recall": (
        "popular-recall",
        "build-covis-matrix",
        "covisitation-recall",
        "dssm-recall",
        "fusion-recall",
        "build-recall-candidates",
    ),
    "Ranker": (
        "build-ranker-train-data",
        "build-ranker-inference-data",
        "train-ranker",
        "ranker-predict",
    ),
    "Evaluation": (
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
    spec.loader.exec_module(module)
    return module


def run_task(task_name: str, task_args: list[str] | None = None) -> None:
    task = TASKS[task_name]
    module = load_task_module(task_name, task)
    task_main = getattr(module, "main", None)

    if not callable(task_main):
        raise AttributeError(f"{task.path} does not expose a callable main()")

    task_args = task_args or []
    main_params = inspect.signature(task_main).parameters

    if task_args and not main_params:
        raise ValueError(f"{task_name} does not accept task arguments: {task_args}")

    if main_params:
        task_main(task_args)
    else:
        task_main()


def run_workflow(workflow_name: str) -> None:
    workflow = WORKFLOWS[workflow_name]
    print(f"Running workflow: {workflow_name}")
    print(f"{workflow.description}")

    for index, step in enumerate(workflow.steps, start=1):
        args_text = " ".join(step.args)
        command_text = f"{step.task_name} {args_text}".strip()
        print(f"\n[{index}/{len(workflow.steps)}] {command_text}")
        run_task(step.task_name, list(step.args))


def print_tasks() -> None:
    print("Workflows:")
    workflow_name_width = max(len(workflow_name) for workflow_name in WORKFLOWS)
    for group_name, workflow_names in WORKFLOW_GROUPS.items():
        print(f"  {group_name}:")
        for workflow_name in workflow_names:
            workflow = WORKFLOWS[workflow_name]
            print(f"    {workflow_name:<{workflow_name_width}} {workflow.description} Example: {workflow.example}")

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
