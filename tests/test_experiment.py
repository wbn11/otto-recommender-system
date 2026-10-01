import json
import os
import sys

import pytest

from utils.experiment import ExperimentRun, file_fingerprint


def test_experiment_initialization_and_safe_resume(tmp_path):
    config = {"seed": 2024, "runtime": {"artifact_root": "artifacts"}}
    experiment = ExperimentRun.create(tmp_path, config, "unit-test")

    assert (experiment.experiment_dir / "resolved_config.yaml").exists()
    assert (experiment.experiment_dir / "manifest.json").exists()
    assert not any(path.is_dir() for path in experiment.experiment_dir.iterdir())
    assert ExperimentRun.create(tmp_path, config, "unit-test").experiment_dir == experiment.experiment_dir

    with pytest.raises(ValueError, match="different configuration"):
        ExperimentRun.create(tmp_path, {"seed": 7}, "unit-test")


def test_stage_cache_uses_command_config_and_input_fingerprint(tmp_path):
    input_path = tmp_path / "input.txt"
    input_path.write_text("version-one", encoding="utf-8")
    config = {"seed": 2024, "runtime": {"artifact_root": "artifacts"}}
    experiment = ExperimentRun.create(tmp_path, config, "cache-test")
    command = [sys.executable, "-c", "print('stage-ok')"]

    first = experiment.run_stage("sample", command, [input_path])
    second = experiment.run_stage("sample", command, [input_path])

    assert first["status"] == "completed"
    assert second["status"] == "cached"
    assert sorted(
        path.name for path in experiment.experiment_dir.iterdir() if path.is_dir()
    ) == ["logs", "stages"]
    assert "stage-ok" in (experiment.experiment_dir / "logs" / "sample.log").read_text(encoding="utf-8")

    input_path.write_text("version-two", encoding="utf-8")
    third = experiment.run_stage("sample", command, [input_path])
    assert third["status"] == "completed"
    assert third["signature"] != first["signature"]

    manifest = json.loads(experiment.manifest_path.read_text(encoding="utf-8"))
    assert manifest["stages"]["sample"]["status"] == "completed"


def test_stage_name_cannot_escape_stage_directory(tmp_path):
    experiment = ExperimentRun.create(
        tmp_path,
        {"runtime": {"artifact_root": "artifacts"}},
        "safe-name",
    )
    with pytest.raises(ValueError, match="stage name"):
        experiment.run_stage("../escape", [sys.executable, "-c", "pass"])


def test_experiment_id_rejects_milestone_prefix(tmp_path):
    config = {"runtime": {"artifact_root": "artifacts"}}

    with pytest.raises(ValueError, match="method name"):
        ExperimentRun.create(tmp_path, config, "m3-time-covis-smoke")

    experiment = ExperimentRun.create(tmp_path, config, "time-covis-smoke")
    assert experiment.experiment_dir.name == "time-covis-smoke"


def test_directory_fingerprint_covers_nested_file_contents(tmp_path):
    dataset = tmp_path / "events"
    nested = dataset / "partition"
    nested.mkdir(parents=True)
    first_file = dataset / "part-00000.parquet"
    second_file = nested / "part-00001.parquet"
    first_file.write_bytes(b"first-version")
    second_file.write_bytes(b"second-file")

    first = file_fingerprint(dataset)
    assert first["type"] == "directory"
    assert first["file_count"] == 2
    assert first["total_size"] == len(b"first-version") + len(b"second-file")

    original_stat = first_file.stat()
    first_file.write_bytes(b"other-version")
    os.utime(first_file, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    second = file_fingerprint(dataset)

    assert second["file_count"] == first["file_count"]
    assert second["total_size"] == first["total_size"]
    assert second["tree_sha256"] != first["tree_sha256"]


def test_directory_fingerprint_changes_when_a_shard_is_added(tmp_path):
    dataset = tmp_path / "sessions"
    dataset.mkdir()
    (dataset / "part-00000.parquet").write_bytes(b"one")
    first = file_fingerprint(dataset)

    (dataset / "part-00001.parquet").write_bytes(b"two")
    second = file_fingerprint(dataset)

    assert second["file_count"] == first["file_count"] + 1
    assert second["tree_sha256"] != first["tree_sha256"]
