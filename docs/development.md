# Development workflow

## Repository layout

The repository is upgraded in place. Git history is the archive of the old
internship demo; the final tree does not keep a second `legacy/` implementation.

```text
configs/                 layered runtime and experiment configuration
data/                    local raw OTTO files (ignored by Git)
artifacts/{experiment}/  isolated generated data, models, metrics and logs
scripts/                 environment/bootstrap commands
src/data/                ingestion, schemas and temporal splitting
src/recall/              Popular, Revisit, Multi-CoVis and DSSM retrieval
src/models/              trainable model definitions and training code
src/candidate/           source-balanced candidate union
src/features/            point-in-time feature builders and registry
src/rank/                LightGBM training and partitioned inference
src/evaluation/          offline metrics and experiment analyses
src/pipeline/             task orchestration and experiment lifecycle
src/utils/               shared configuration and manifest utilities
tests/                   unit and debug integration tests
```

Existing modules are replaced milestone by milestone. An old implementation is
removed only after its replacement passes the debug smoke test. The old
`outputs/` directory is never consumed by the upgraded pipeline and may be
deleted after the new debug end-to-end workflow passes.

Canonical full-data Parquet uses 5,000,000 event rows and 2,000,000 session
rows per physical file. Each file contains 250,000-row groups, giving
DuckDB/Polars scan granularity without creating many small files.

## A6000 virtual environment

On Ubuntu with a compatible NVIDIA driver and Python 3.10-3.12:

```bash
bash scripts/setup_a6000.sh
source .venv/bin/activate
python src/pipeline/run.py check-environment --require-gpu
python -m pytest
```

The default PyTorch wheel uses CUDA 12.6. Override its wheel index when the
host driver requires another supported runtime:

```bash
OTTO_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
  bash scripts/setup_a6000.sh
```

The default GPU FAISS wheel is `faiss-gpu-cu12`. Both package choices can be
overridden without editing the script:

```bash
OTTO_PYTHON=python3.11 \
OTTO_VENV_DIR=.venv \
OTTO_FAISS_PACKAGE=faiss-gpu-cu12 \
  bash scripts/setup_a6000.sh
```

If the server cannot reach the official PyPI endpoint, select an accessible
mirror explicitly. This also bypasses broken global pip index configuration:

```bash
OTTO_PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  bash scripts/setup_a6000.sh
```

PyTorch and FAISS are deliberately installed before `requirements.txt`, where
they are omitted to prevent pip from silently selecting CPU-only wheels.

## Copy from Windows to the Ubuntu server

Use WSL `rsync` when available. It supports incremental transfer and resumes
large dataset copies with `--partial`. Replace the SSH placeholders first.

Create the destination:

```powershell
ssh -p PORT USER@SERVER_IP "mkdir -p /home/USER/wbn/OTTO/data"
```

Copy source code and configuration without local caches or generated files:

```powershell
wsl rsync -avh --progress `
  --exclude='.git/' `
  --exclude='.venv/' `
  --exclude='data/' `
  --exclude='outputs/' `
  --exclude='artifacts/' `
  /mnt/e/OTTO/ `
  USER@SERVER_IP:/home/USER/wbn/OTTO/ `
  -e "ssh -p PORT"
```

Copy the labeled training data separately with resumable partial files:

```powershell
wsl rsync -avh --partial --info=progress2 `
  /mnt/e/OTTO/data/otto-recsys-train.jsonl `
  USER@SERVER_IP:/home/USER/wbn/OTTO/data/ `
  -e "ssh -p PORT"
```

The competition test JSONL has no labels and is not used for model selection or
offline metrics. Kaggle currently exposes a Late Submission action for OTTO, so
keep the file locally and upload it when the selected final model is ready for
an optional post-competition score.

Optional late-submission test upload:

```powershell
wsl rsync -avh --partial --info=progress2 `
  /mnt/e/OTTO/data/otto-recsys-test.jsonl `
  USER@SERVER_IP:/home/USER/wbn/OTTO/data/ `
  -e "ssh -p PORT"
```

Running either command again transfers only changed or incomplete content. Do
not add `--delete`: the upload command must not remove remote artifacts.

If WSL/rsync is unavailable, package code only with Windows `tar.exe`, upload
the small archive, then extract it remotely:

```powershell
tar.exe -czf E:\otto-code.tar.gz `
  --exclude=OTTO/.git `
  --exclude=OTTO/.venv `
  --exclude=OTTO/data `
  --exclude=OTTO/outputs `
  --exclude=OTTO/artifacts `
  -C E:\ OTTO

scp -P PORT E:\otto-code.tar.gz USER@SERVER_IP:/tmp/otto-code.tar.gz

ssh -p PORT USER@SERVER_IP `
  "mkdir -p /home/USER/wbn && tar -xzf /tmp/otto-code.tar.gz -C /home/USER/wbn"
```

Verify the upload before installing dependencies:

```powershell
ssh -p PORT USER@SERVER_IP `
  "cd /home/USER/wbn/OTTO && pwd && find configs src tests -type f | wc -l && ls -lh data/otto-recsys-train.jsonl"
```

## Experiment lifecycle

Initialize a debug experiment:

```bash
python src/pipeline/experiment.py init \
  --config configs/experiments/debug.yaml \
  --experiment-id debug-m0
```

Run a tracked stage:

```bash
python src/pipeline/experiment.py run \
  --config configs/experiments/debug.yaml \
  --experiment-id debug-m0 \
  --stage tests \
  --input configs/base.yaml \
  -- python -m pytest
```

Inspect the resolved environment, configuration and stage state:

```bash
python src/pipeline/experiment.py status \
  --config configs/experiments/debug.yaml \
  --experiment-id debug-m0
```

A completed stage is reused only when the resolved configuration, command and
all declared input fingerprints match. Pass `--force` to rerun deliberately.
