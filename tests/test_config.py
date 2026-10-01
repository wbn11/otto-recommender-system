from pathlib import Path

from utils.config import config_hash, deep_merge, resolve_project_config


ROOT = Path(__file__).resolve().parents[1]


def test_deep_merge_replaces_lists_and_preserves_nested_values():
    base = {"seed": 1, "nested": {"keep": True, "replace": 2}, "items": [1, 2]}
    override = {"nested": {"replace": 3}, "items": [9]}

    merged = deep_merge(base, override)

    assert merged == {"seed": 1, "nested": {"keep": True, "replace": 3}, "items": [9]}
    assert base["nested"]["replace"] == 2


def test_smoke_experiment_loads_base_profile_and_override():
    config = resolve_project_config(
        ROOT / "configs" / "experiments" / "pipeline_smoke.yaml"
    )

    assert config["data_mode"] == "debug"
    assert config["data"]["max_sessions_per_window"] == 10_000
    assert config["dssm"]["embedding_dim"] == 128
    assert config["dssm"]["epochs"] == 1
    assert config["runtime"]["workers"] == 2


def test_config_hash_is_order_independent():
    assert config_hash({"a": 1, "b": {"c": 2}}) == config_hash({"b": {"c": 2}, "a": 1})


def test_attention_ablation_matches_baseline_training_settings():
    baseline = resolve_project_config(
        ROOT / "configs" / "experiments" / "dssm_baseline.yaml"
    )
    attention = resolve_project_config(
        ROOT / "configs" / "experiments" / "dssm_attention.yaml"
    )

    for key in (
        "embedding_dim",
        "temperature",
        "max_sequence_length",
        "batch_size",
        "type_loss_weights",
        "epochs",
        "learning_rate",
        "weight_decay",
        "amp",
        "hard_negatives",
        "default_embedding",
        "history_dropout",
    ):
        assert attention["dssm"][key] == baseline["dssm"][key]
    assert baseline["dssm"]["attention"] is False
    assert attention["dssm"]["attention"] is True


def test_attention_debug_disables_unrelated_dssm_modules():
    config = resolve_project_config(
        ROOT / "configs" / "experiments" / "dssm_attention_debug.yaml"
    )

    assert config["data_mode"] == "debug"
    assert config["dssm"]["attention"] is True
    assert config["dssm"]["default_embedding"] is False
    assert config["dssm"]["history_dropout"] == 0.0
    assert config["dssm"]["hard_negatives"] == 0
