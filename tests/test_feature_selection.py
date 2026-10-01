from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from features.registry import (
    FEATURES,
    FEATURE_GROUPS,
    FEATURE_NAMES,
    feature_names_for_groups,
    registry_payload_for_features,
)
from rank import select_lambdarank_features as selection


def test_feature_subset_registry_preserves_canonical_order() -> None:
    names = feature_names_for_groups(("base", "item", "temporal"))
    assert names == tuple(
        feature.name
        for feature in FEATURES
        if feature.group in {"base", "item", "temporal"}
    )
    payload = registry_payload_for_features(names)
    assert payload["feature_count"] == len(names)
    assert payload["enabled_groups"] == ["base", "item", "temporal"]
    assert [feature["name"] for feature in payload["features"]] == list(names)

    with pytest.raises(ValueError, match="canonical registry order"):
        registry_payload_for_features(reversed(names))


def test_backward_group_selection_keeps_harmful_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_to_group = {feature.name: feature.group for feature in FEATURES}

    def fake_train(
        features_dir,
        output_dir,
        config,
        *,
        max_groups=None,
        num_boost_round=None,
        feature_names=None,
    ):
        selected = tuple(feature_names or FEATURE_NAMES)
        selected_groups = {
            feature_to_group[name]
            for name in selected
        }
        # Removing session is harmless/slightly positive. Removing item after
        # that is deliberately harmful and must therefore be rejected.
        if "session" not in selected_groups and "item" not in selected_groups:
            score = 0.748
        elif "session" not in selected_groups:
            score = 0.7501
        else:
            score = 0.75

        output = Path(output_dir)
        output.mkdir(parents=True)
        (output / "model.txt").write_text("fake model", encoding="utf-8")
        schema = registry_payload_for_features(selected)
        schema["model_feature_names"] = list(selected)
        (output / "feature_schema.json").write_text(
            json.dumps(schema), encoding="utf-8"
        )
        group_gain = {
            "session": 0.1,
            "item": 2.0,
            "temporal": 3.0,
            "interaction": 4.0,
            "recall": 5.0,
            "base": 6.0,
        }
        with (output / "feature_importance.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["feature", "gain", "split"])
            for name in selected:
                writer.writerow([name, group_gain[feature_to_group[name]], 1])
        report = {
            "output": str(output),
            "best_score": {"internal_valid": {"ndcg@20": score}},
            "best_iteration": 10,
            "timing": {"training_seconds": 0.01},
        }
        (output / "report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        return report

    monkeypatch.setattr(selection, "_train_lambdarank", fake_train)
    config = {
        "seed": 2024,
        "feature_groups": list(FEATURE_GROUPS),
        "ranker": {"eval_at": 20},
    }
    output = tmp_path / "selection"
    report = selection.select_lambdarank_features(
        tmp_path / "features",
        output,
        config,
        max_allowed_ndcg_drop=0.0002,
        candidate_groups=("session", "item"),
    )

    assert report["assertions"]["passed"] is True
    assert report["steps"][0]["removed_group"] == "session"
    assert report["steps"][0]["accepted"] is True
    assert report["steps"][1]["removed_group"] == "item"
    assert report["steps"][1]["accepted"] is False
    assert "session" not in report["selected_model"]["groups"]
    assert "item" in report["selected_model"]["groups"]
    assert (output / "selected_model" / "model.txt").is_file()
    selected_schema = json.loads(
        (output / "selected_model" / "feature_schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert tuple(selected_schema["model_feature_names"]) == feature_names_for_groups(
        report["selected_model"]["groups"]
    )
