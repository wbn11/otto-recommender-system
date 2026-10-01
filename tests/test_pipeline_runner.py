from pathlib import Path

from pipeline.run import Task, load_task_module


def test_dynamic_task_loader_supports_dataclasses(tmp_path, monkeypatch):
    module_path = tmp_path / "task_with_dataclass.py"
    module_path.write_text(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Value:\n"
        "    number: int\n"
        "def main(argv=None):\n"
        "    return Value(1)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("pipeline.run.SRC_DIR", Path(tmp_path))

    module = load_task_module("dataclass-test", Task(module_path.name, "test", "test"))

    assert module.main().number == 1
