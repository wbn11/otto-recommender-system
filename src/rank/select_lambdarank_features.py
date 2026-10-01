"""Select LambdaRank feature groups with deterministic backward elimination.

The final validation window is deliberately absent from this stage.  Every
trial uses the same ranker-window candidates, eligible groups, session hash
split, LightGBM seed and hyperparameters.  Starting from all 49 features, the
least-important non-base group is tentatively removed.  The removal is kept
only when internal NDCG@20 stays within a configured tolerance of the full
model.  This produces an auditable feature set without tuning on final_valid.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.registry import (
    FEATURES,
    FEATURE_GROUPS,
    feature_names_for_groups,
)
from utils.config import resolve_project_config


MANDATORY_GROUPS = ("base",)
DEFAULT_CANDIDATE_GROUPS = tuple(
    group for group in FEATURE_GROUPS if group not in MANDATORY_GROUPS
)
FEATURE_TO_GROUP = {feature.name: feature.group for feature in FEATURES}


def _train_lambdarank(*args, **kwargs):
    # Lazy import keeps lightweight registry/selection unit tests independent
    # of the server-only DuckDB and LightGBM runtime.
    from rank.train_lambdarank import train_lambdarank

    return train_lambdarank(*args, **kwargs)


def _best_ndcg(report: Mapping[str, Any], eval_at: int) -> float:
    scores = report["best_score"]["internal_valid"]
    key = f"ndcg@{int(eval_at)}"
    if key not in scores:
        raise KeyError(f"Training report does not contain {key}: {scores}")
    return float(scores[key])


def _group_gain(model_dir: Path) -> dict[str, float]:
    totals = {group: 0.0 for group in FEATURE_GROUPS}
    with (model_dir / "feature_importance.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            totals[FEATURE_TO_GROUP[row["feature"]]] += float(row["gain"])
    return totals


def _copy_selected_model(
    source: Path,
    destination: Path,
    *,
    final_destination: Path,
    selection_origin: Mapping[str, Any],
) -> None:
    destination.mkdir()
    for name in ("model.txt", "feature_schema.json", "feature_importance.csv"):
        shutil.copy2(source / name, destination / name)
    source_report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    source_report["output"] = str(final_destination.resolve())
    source_report["feature_selection"] = dict(selection_origin)
    (destination / "report.json").write_text(
        json.dumps(source_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def select_lambdarank_features(
    features_dir: str | Path,
    output_dir: str | Path,
    config: Mapping[str, Any],
    *,
    max_allowed_ndcg_drop: float = 0.0002,
    candidate_groups: Sequence[str] = DEFAULT_CANDIDATE_GROUPS,
    max_groups: int | None = None,
    num_boost_round: int | None = None,
) -> dict[str, Any]:
    """Run greedy group removal and persist every trial plus selected model."""

    if max_allowed_ndcg_drop < 0:
        raise ValueError("max_allowed_ndcg_drop must be non-negative")
    unknown = set(candidate_groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown candidate feature groups: {sorted(unknown)}")
    if set(candidate_groups) & set(MANDATORY_GROUPS):
        raise ValueError(f"Mandatory groups cannot be removed: {MANDATORY_GROUPS}")
    if len(set(candidate_groups)) != len(tuple(candidate_groups)):
        raise ValueError("candidate_groups must be unique")

    configured_groups = tuple(config["feature_groups"])
    if set(configured_groups) != set(FEATURE_GROUPS):
        raise ValueError(
            "Feature selection requires feature shards containing all canonical groups"
        )

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite feature selection: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    staging = destination.parent / f".{destination.name}.building-{token}"
    runs_dir = staging / "runs"
    runs_dir.mkdir(parents=True)
    started = time.perf_counter()

    try:
        all_groups = tuple(FEATURE_GROUPS)
        full_dir = runs_dir / "00-full"
        print("[feature-selection] training full 49-feature reference", flush=True)
        full_report = _train_lambdarank(
            features_dir,
            full_dir,
            config,
            max_groups=max_groups,
            num_boost_round=num_boost_round,
            feature_names=feature_names_for_groups(all_groups),
        )
        eval_at = int(config["ranker"]["eval_at"])
        full_ndcg = _best_ndcg(full_report, eval_at)
        group_gain = _group_gain(full_dir)
        removable = set(candidate_groups)
        removal_order = sorted(
            removable,
            key=lambda group: (group_gain[group], group),
        )

        current_groups = all_groups
        selected_run = full_dir
        selected_ndcg = full_ndcg
        steps: list[dict[str, Any]] = []
        for index, group in enumerate(removal_order, start=1):
            trial_groups = tuple(item for item in current_groups if item != group)
            trial_dir = runs_dir / f"{index:02d}-minus-{group}"
            print(
                f"[feature-selection] trial={index}/{len(removal_order)} "
                f"remove={group} features={len(feature_names_for_groups(trial_groups))}",
                flush=True,
            )
            trial_report = _train_lambdarank(
                features_dir,
                trial_dir,
                config,
                max_groups=max_groups,
                num_boost_round=num_boost_round,
                feature_names=feature_names_for_groups(trial_groups),
            )
            trial_ndcg = _best_ndcg(trial_report, eval_at)
            delta_vs_full = trial_ndcg - full_ndcg
            delta_vs_current = trial_ndcg - selected_ndcg
            accepted = delta_vs_full >= -max_allowed_ndcg_drop
            if accepted:
                current_groups = trial_groups
                selected_run = trial_dir
                selected_ndcg = trial_ndcg
            step = {
                "step": index,
                "removed_group": group,
                "trial_groups": list(trial_groups),
                "trial_feature_count": len(feature_names_for_groups(trial_groups)),
                "trial_ndcg_at_20": round(trial_ndcg, 10),
                "delta_vs_full": round(delta_vs_full, 10),
                "delta_vs_current": round(delta_vs_current, 10),
                "accepted": accepted,
                "selected_groups_after_step": list(current_groups),
                "trial_model": str(trial_dir.relative_to(staging)),
                "best_iteration": int(trial_report["best_iteration"]),
                "training_seconds": float(trial_report["timing"]["training_seconds"]),
            }
            steps.append(step)
            print(
                f"[feature-selection] remove={group} ndcg={trial_ndcg:.8f} "
                f"delta_full={delta_vs_full:+.8f} "
                f"decision={'remove' if accepted else 'keep'}",
                flush=True,
            )

        selected_features = feature_names_for_groups(current_groups)
        selection_origin = {
            "method": "greedy_backward_group_elimination",
            "reference_metric": f"internal_valid_ndcg@{eval_at}",
            "full_ndcg": round(full_ndcg, 10),
            "selected_ndcg": round(selected_ndcg, 10),
            "max_allowed_ndcg_drop": max_allowed_ndcg_drop,
            "selected_groups": list(current_groups),
            "selected_feature_names": list(selected_features),
            "source_run": str(selected_run.relative_to(staging)),
            "final_valid_used_for_selection": False,
        }
        selected_model_dir = staging / "selected_model"
        _copy_selected_model(
            selected_run,
            selected_model_dir,
            final_destination=destination / "selected_model",
            selection_origin=selection_origin,
        )

        with (staging / "group_importance.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["group", "full_model_gain", "removal_order"])
            order_lookup = {group: index + 1 for index, group in enumerate(removal_order)}
            for group in FEATURE_GROUPS:
                writer.writerow([group, group_gain[group], order_lookup.get(group, "")])
        with (staging / "selection_steps.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            fieldnames = [
                "step",
                "removed_group",
                "trial_feature_count",
                "trial_ndcg_at_20",
                "delta_vs_full",
                "delta_vs_current",
                "accepted",
                "best_iteration",
                "training_seconds",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for step in steps:
                writer.writerow({name: step[name] for name in fieldnames})

        report = {
            "method": "greedy_backward_group_elimination",
            "purpose": "Remove redundant feature groups without consulting final_valid.",
            "features": str(Path(features_dir).resolve()),
            "output": str(destination),
            "configuration": {
                "candidate_groups": list(candidate_groups),
                "mandatory_groups": list(MANDATORY_GROUPS),
                "removal_order": removal_order,
                "max_allowed_ndcg_drop": max_allowed_ndcg_drop,
                "max_groups": max_groups,
                "num_boost_round": num_boost_round,
                "seed": int(config["seed"]),
            },
            "full_model": {
                "feature_count": len(feature_names_for_groups(all_groups)),
                "ndcg_at_20": round(full_ndcg, 10),
                "model": "runs/00-full",
                "group_gain": group_gain,
            },
            "steps": steps,
            "selected_model": {
                "groups": list(current_groups),
                "feature_count": len(selected_features),
                "feature_names": list(selected_features),
                "ndcg_at_20": round(selected_ndcg, 10),
                "delta_vs_full": round(selected_ndcg - full_ndcg, 10),
                "model": "selected_model",
                "source_run": str(selected_run.relative_to(staging)),
            },
            "artifacts": {
                "selected_model": "selected_model",
                "group_importance": "group_importance.csv",
                "selection_steps": "selection_steps.csv",
            },
            "assertions": {
                "passed": (
                    "base" in current_groups
                    and selected_ndcg >= full_ndcg - max_allowed_ndcg_drop
                    and not set(current_groups) - set(FEATURE_GROUPS)
                ),
                "base_group_retained": "base" in current_groups,
                "final_valid_not_used": True,
                "selected_within_tolerance": (
                    selected_ndcg >= full_ndcg - max_allowed_ndcg_drop
                ),
                "same_seed_and_split_for_all_trials": True,
            },
            "timing": {
                "total_seconds": round(time.perf_counter() - started, 6),
            },
        }
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(destination)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select LambdaRank feature groups by backward elimination."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-allowed-ndcg-drop", type=float, default=0.0002)
    parser.add_argument(
        "--candidate-groups",
        nargs="+",
        choices=DEFAULT_CANDIDATE_GROUPS,
        default=DEFAULT_CANDIDATE_GROUPS,
    )
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--num-boost-round", type=int)
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
        output_dir = Path(experiment_dir) / "feature_selection" / "lambdarank"
    else:
        raise ValueError("Provide --output-dir or run through the experiment runner")
    report = select_lambdarank_features(
        args.features_dir,
        output_dir,
        resolve_project_config(config_path),
        max_allowed_ndcg_drop=args.max_allowed_ndcg_drop,
        candidate_groups=args.candidate_groups,
        max_groups=args.max_groups,
        num_boost_round=args.num_boost_round,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
