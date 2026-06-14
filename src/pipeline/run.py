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


SRC_DIR = Path(__file__).resolve().parents[1]

TASKS = {
    "eda": Task("data/eda.py", "Run exploratory data analysis."),
    "build-validation": Task("data/build_validation.py", "Build leave-one-out validation files."),
    "build-multi-target-validation": Task(
        "data/build_multi_target_validation.py",
        "Build multi-target validation files.",
    ),
    "build-dssm-dataset": Task("data/build_dssm_dataset.py", "Inspect DSSM training pairs."),
    "popular-recall": Task("recall/popular_recall.py", "Generate popular item recall."),
    "popular-recall-multi-target": Task(
        "recall/popular_recall_multi_target.py",
        "Generate multi-target popular item recall.",
    ),
    "build-covis-matrix": Task("recall/build_covis_matrix.py", "Build co-visitation top-k matrix."),
    "build-covis-matrix-multi-target": Task(
        "recall/build_covis_matrix_multi_target.py",
        "Build multi-target co-visitation top-k matrix.",
    ),
    "covisitation": Task("recall/covisitation.py", "Build full co-visitation recall."),
    "covisitation-recall": Task("recall/covisitation_recall.py", "Generate co-visitation recall from saved matrix."),
    "covisitation-recall-multi-target": Task(
        "recall/covisitation_recall_multi_target.py",
        "Generate multi-target co-visitation recall from saved matrix.",
    ),
    "dssm-recall": Task("recall/generate_dssm_recall.py", "Generate DSSM recall predictions."),
    "fusion-recall": Task("recall/fusion_recall.py", "Fuse multi-source recall predictions."),
    "train-dssm": Task("models/train_dssm.py", "Train the DSSM model."),
    "evaluate": Task("evaluation/evaluate.py", "Evaluate recall predictions."),
    "evaluate-multi-target": Task("evaluation/evaluate_multi_target.py", "Evaluate multi-target recall predictions."),
    "device-check": Task("utils/test.py", "Print the torch device used by this environment."),
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


def print_tasks() -> None:
    print("Available tasks:")
    task_name_width = max(len(task_name) for task_name in TASKS)
    for task_name in sorted(TASKS):
        print(f"  {task_name:<{task_name_width}} {TASKS[task_name].description}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified entrypoint for OTTO recommendation experiments.")
    parser.add_argument("task", nargs="?", choices=sorted(TASKS), help="Task to run.")
    parser.add_argument("task_args", nargs=argparse.REMAINDER, help="Arguments passed to the selected task.")
    parser.add_argument("--list", action="store_true", help="List available tasks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list or args.task is None:
        print_tasks()
        return

    run_task(args.task, args.task_args)


if __name__ == "__main__":
    main()
