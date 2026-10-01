"""CLI for initializing, inspecting and running reproducible OTTO stages."""

# pylint: disable=wrong-import-position

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# This file is intentionally executable by path from the repository root.
# Pylint receives the same source root through pyproject.toml.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import resolve_project_config  # noqa: E402
from utils.experiment import ExperimentRun  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage reproducible OTTO experiments.")
    subparsers = parser.add_subparsers(dest="action", required=True)

    for action in ("init", "status"):
        sub = subparsers.add_parser(action)
        sub.add_argument("--config", type=Path)
        sub.add_argument("--experiment-id", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path)
    run.add_argument("--experiment-id", required=True)
    run.add_argument("--stage", required=True)
    run.add_argument("--input", action="append", default=[])
    run.add_argument("--force", action="store_true")
    run.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def open_experiment(args: argparse.Namespace) -> ExperimentRun:
    config = resolve_project_config(args.config)
    return ExperimentRun.create(ROOT, config, args.experiment_id)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    experiment = open_experiment(args)

    if args.action == "init":
        print(experiment.experiment_dir)
        return
    if args.action == "status":
        print(json.dumps(experiment.manifest(), indent=2, ensure_ascii=False))
        return

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("A command is required after '--'.")
    result = experiment.run_stage(
        args.stage,
        command,
        args.input,
        force=args.force,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
