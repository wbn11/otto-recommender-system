"""Experiment manifests, stage logging and safe cache/resume support."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from utils.config import config_hash, dump_yaml


MANIFEST_SCHEMA_VERSION = 1
TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "polars",
    "pyarrow",
    "duckdb",
    "torch",
    "lightgbm",
    "faiss-cpu",
    "faiss-gpu",
    "PyYAML",
    "psutil",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_text(command: Sequence[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def git_metadata(root: Path) -> dict[str, Any]:
    commit = _run_text(("git", "rev-parse", "HEAD"), root)
    status = _run_text(("git", "status", "--porcelain"), root)
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def hardware_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil

        metadata["memory_total_bytes"] = psutil.virtual_memory().total
    except ImportError:
        metadata["memory_total_bytes"] = None

    try:
        import torch

        metadata["torch_cuda_available"] = torch.cuda.is_available()
        metadata["torch_cuda_version"] = torch.version.cuda
        metadata["gpu_count"] = torch.cuda.device_count()
        metadata["gpus"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except (ImportError, RuntimeError):
        metadata.update({
            "torch_cuda_available": False,
            "torch_cuda_version": None,
            "gpu_count": 0,
            "gpus": [],
        })
    return metadata


def _content_digest(path: Path, full_hash_limit_bytes: int) -> tuple[str, str]:
    """Hash a whole small file or the head and tail of a large file."""

    stat = path.stat()
    digest = hashlib.sha256()
    if stat.st_size <= full_hash_limit_bytes:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return "full", digest.hexdigest()

    sample_bytes = 64 * 1024
    with path.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        handle.seek(max(0, stat.st_size - sample_bytes))
        digest.update(handle.read(sample_bytes))
    return "head_tail_64k", digest.hexdigest()


def file_fingerprint(
    path: str | Path,
    hash_limit_bytes: int = 64 * 1024 * 1024,
) -> dict[str, Any]:
    """Return a stable content-aware fingerprint for one file or directory.

    Directory fingerprints recursively cover every file's relative path,
    size, modification time and content digest. Large files use a bounded
    head/tail digest so fingerprinting multi-gigabyte Parquet datasets remains
    cheap; Parquet footer changes are covered by the tail sample.
    """

    input_path = Path(path).resolve()
    stat = input_path.stat()
    if input_path.is_file():
        hash_mode, content_digest = _content_digest(input_path, hash_limit_bytes)
        fingerprint: dict[str, Any] = {
            "path": str(input_path),
            "type": "file",
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "hash_mode": hash_mode,
        }
        fingerprint["sha256" if hash_mode == "full" else "sample_sha256"] = content_digest
        return fingerprint

    if not input_path.is_dir():
        raise ValueError(f"Input is neither a file nor a directory: {input_path}")

    dataset_digest = hashlib.sha256()
    file_count = 0
    total_size = 0
    max_mtime_ns = stat.st_mtime_ns
    directory_full_hash_limit = min(hash_limit_bytes, 1024 * 1024)
    for child in sorted(
        item for item in input_path.rglob("*")
        if item.is_file() and not item.is_symlink()
    ):
        child_stat = child.stat()
        hash_mode, content_digest = _content_digest(child, directory_full_hash_limit)
        entry = {
            "path": child.relative_to(input_path).as_posix(),
            "size": child_stat.st_size,
            "mtime_ns": child_stat.st_mtime_ns,
            "hash_mode": hash_mode,
            "content_digest": content_digest,
        }
        dataset_digest.update(
            json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        dataset_digest.update(b"\n")
        file_count += 1
        total_size += child_stat.st_size
        max_mtime_ns = max(max_mtime_ns, child_stat.st_mtime_ns)

    return {
        "path": str(input_path),
        "type": "directory",
        "file_count": file_count,
        "total_size": total_size,
        "max_mtime_ns": max_mtime_ns,
        "tree_sha256": dataset_digest.hexdigest(),
        "fingerprint_mode": "relative_path,size,mtime_ns,content_digest",
        "full_hash_limit_bytes_per_file": directory_full_hash_limit,
    }


def _signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


class _PeakMemoryMonitor:
    def __init__(self, process: subprocess.Popen[Any]):
        self.process = process
        self.peak_rss_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            import psutil
        except ImportError:
            return

        def sample() -> None:
            root_process = psutil.Process(self.process.pid)
            while not self._stop.wait(0.2):
                try:
                    processes = [root_process, *root_process.children(recursive=True)]
                    rss = sum(item.memory_info().rss for item in processes if item.is_running())
                    self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
                except (psutil.Error, OSError):
                    break

        self._thread = threading.Thread(target=sample, name="peak-memory-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)


@dataclass(frozen=True)
class ExperimentRun:
    root: Path
    experiment_dir: Path
    config: dict[str, Any]
    config_digest: str

    @classmethod
    def create(
        cls,
        root: str | Path,
        config: dict[str, Any],
        experiment_id: str | None = None,
    ) -> "ExperimentRun":
        project_root = Path(root).resolve()
        digest = config_hash(config)
        if not experiment_id:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            experiment_id = f"{stamp}-{digest}"
        if not experiment_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError("experiment_id may contain only letters, numbers, '-' and '_'.")
        first_token = experiment_id.replace("_", "-").split("-", maxsplit=1)[0]
        if (
            len(first_token) > 1
            and first_token[0].lower() == "m"
            and first_token[1:].isdigit()
        ):
            raise ValueError(
                "experiment_id must use a method name, not a milestone prefix "
                "such as 'm3-' or 'm4-'."
            )

        artifact_root = Path(config.get("runtime", {}).get("artifact_root", "artifacts"))
        if not artifact_root.is_absolute():
            artifact_root = project_root / artifact_root
        experiment_dir = artifact_root / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = experiment_dir / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("config_hash") != digest:
                raise ValueError(
                    f"Experiment {experiment_id!r} already exists with a different configuration."
                )
        else:
            manifest = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "experiment_id": experiment_id,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "config_hash": digest,
                "git": git_metadata(project_root),
                "hardware": hardware_metadata(),
                "packages": package_versions(),
                "stages": {},
            }
            _write_json(manifest_path, manifest)
            dump_yaml(config, experiment_dir / "resolved_config.yaml")

        return cls(project_root, experiment_dir, config, digest)

    @property
    def manifest_path(self) -> Path:
        return self.experiment_dir / "manifest.json"

    def manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def stage_payload(
        self,
        name: str,
        command: Sequence[str],
        inputs: Iterable[str | Path] = (),
    ) -> dict[str, Any]:
        return {
            "name": name,
            "command": list(command),
            "config_hash": self.config_digest,
            "inputs": [file_fingerprint(path) for path in inputs],
        }

    def stage_is_current(self, name: str, signature: str) -> bool:
        record = self.manifest().get("stages", {}).get(name, {})
        return record.get("status") == "completed" and record.get("signature") == signature

    def _update_stage(self, name: str, record: dict[str, Any]) -> None:
        manifest = self.manifest()
        manifest.setdefault("stages", {})[name] = record
        manifest["updated_at"] = utc_now()
        _write_json(self.manifest_path, manifest)
        stages_dir = self.experiment_dir / "stages"
        stages_dir.mkdir(exist_ok=True)
        _write_json(stages_dir / f"{name}.json", record)

    def run_stage(
        self,
        name: str,
        command: Sequence[str],
        inputs: Iterable[str | Path] = (),
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        if not name.replace("-", "").replace("_", "").isalnum():
            raise ValueError("stage name may contain only letters, numbers, '-' and '_'.")
        payload = self.stage_payload(name, command, inputs)
        signature = _signature(payload)
        if not force and self.stage_is_current(name, signature):
            return {"name": name, "status": "cached", "signature": signature}

        started_at = utc_now()
        start_time = time.perf_counter()
        log_path = self.experiment_dir / "logs" / f"{name}.log"
        log_path.parent.mkdir(exist_ok=True)
        environment = os.environ.copy()
        environment["OTTO_EXPERIMENT_DIR"] = str(self.experiment_dir)
        environment["OTTO_CONFIG_PATH"] = str(self.experiment_dir / "resolved_config.yaml")

        running_record = {
            **payload,
            "signature": signature,
            "status": "running",
            "started_at": started_at,
            "log": str(log_path.relative_to(self.experiment_dir)),
        }
        self._update_stage(name, running_record)

        monitor = None
        try:
            with log_path.open("w", encoding="utf-8", newline="\n") as log_handle:
                process = subprocess.Popen(
                    list(command),
                    cwd=self.root,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                monitor = _PeakMemoryMonitor(process)
                monitor.start()
                return_code = process.wait()
                monitor.stop()
        except OSError as exc:
            failed_record = {
                **running_record,
                "status": "failed",
                "finished_at": utc_now(),
                "runtime_seconds": round(time.perf_counter() - start_time, 6),
                "peak_rss_bytes": monitor.peak_rss_bytes if monitor else None,
                "return_code": None,
                "error": str(exc),
            }
            self._update_stage(name, failed_record)
            raise

        completed_record = {
            **running_record,
            "status": "completed" if return_code == 0 else "failed",
            "finished_at": utc_now(),
            "runtime_seconds": round(time.perf_counter() - start_time, 6),
            "peak_rss_bytes": monitor.peak_rss_bytes or None,
            "return_code": return_code,
        }
        self._update_stage(name, completed_record)
        if return_code:
            raise subprocess.CalledProcessError(return_code, list(command))
        return completed_record
