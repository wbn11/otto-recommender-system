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
src/features/            point-in-time feature builders and registry
src/rank/                LightGBM training and partitioned inference
src/evaluation/          offline metrics and experiment analyses
src/pipeline/             task orchestration and experiment lifecycle
src/utils/               shared configuration and manifest utilities
tests/                   unit and debug integration tests
```

Git history retains the former internship implementation; the working tree
contains only the final strict pipeline and its maintained comparison paths.

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

After removing obsolete tracked files, mirror only the maintained source
directories with scoped deletion. This removes stale code on the server while
leaving `data/`, `artifacts/`, `outputs/`, `.venv/` and `.git/` untouched:

```powershell
wsl rsync -avh --delete /mnt/e/OTTO/configs/ USER@SERVER_IP:/home/USER/wbn/OTTO/configs/ -e "ssh -p PORT"
wsl rsync -avh --delete /mnt/e/OTTO/docs/   USER@SERVER_IP:/home/USER/wbn/OTTO/docs/   -e "ssh -p PORT"
wsl rsync -avh --delete /mnt/e/OTTO/scripts/ USER@SERVER_IP:/home/USER/wbn/OTTO/scripts/ -e "ssh -p PORT"
wsl rsync -avh --delete /mnt/e/OTTO/src/     USER@SERVER_IP:/home/USER/wbn/OTTO/src/     -e "ssh -p PORT"
wsl rsync -avh --delete /mnt/e/OTTO/tests/   USER@SERVER_IP:/home/USER/wbn/OTTO/tests/   -e "ssh -p PORT"
wsl rsync -avh /mnt/e/OTTO/README.md /mnt/e/OTTO/pyproject.toml /mnt/e/OTTO/requirements.txt /mnt/e/OTTO/requirements-dev.txt USER@SERVER_IP:/home/USER/wbn/OTTO/ -e "ssh -p PORT"
```

Copy the labeled training data separately with resumable partial files:

```powershell
wsl rsync -avh --partial --info=progress2 `
  /mnt/e/OTTO/data/otto-recsys-train.jsonl `
  USER@SERVER_IP:/home/USER/wbn/OTTO/data/ `
  -e "ssh -p PORT"
```

The competition test JSONL has no labels and is not required by the maintained
offline pipeline. Copy only `otto-recsys-train.jsonl` unless a separate
submission workflow is added later. Running the command again transfers only
changed or incomplete content. Do not add `--delete`: the upload command must
not remove remote artifacts.

If WSL/rsync is unavailable, use the Windows built-in `tar.exe`, `scp` and
`ssh`. The server validates the archive, moves the previous source tree to a
timestamped backup, and then extracts the clean tree. Generated data and the
virtual environment are outside the replacement list:

```powershell
tar.exe -czf E:\OTTO-source-clean.tar.gz -C E:\OTTO `
  configs docs scripts src tests `
  README.md pyproject.toml requirements.txt requirements-dev.txt

scp -P PORT E:\OTTO-source-clean.tar.gz USER@SERVER_IP:/tmp/OTTO-source-clean.tar.gz

ssh -p PORT USER@SERVER_IP 'set -e; PROJECT=/home/USER/wbn/OTTO; ARCHIVE=/tmp/OTTO-source-clean.tar.gz; tar -tzf "$ARCHIVE" >/dev/null; BACKUP="/home/USER/wbn/OTTO-source-backup-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$BACKUP"; for name in configs docs scripts src tests reports README.md pyproject.toml requirements.txt requirements-dev.txt; do if [ -e "$PROJECT/$name" ]; then mv "$PROJECT/$name" "$BACKUP/"; fi; done; tar -xzf "$ARCHIVE" -C "$PROJECT"; echo "backup=$BACKUP"'
```

Verify the upload before installing dependencies:

```powershell
ssh -p PORT USER@SERVER_IP `
  "cd /home/USER/wbn/OTTO && pwd && find configs src tests -type f | wc -l && ls -lh data/otto-recsys-train.jsonl"
```

## Experiment lifecycle

Initialize a smoke experiment. Experiment identifiers use method names rather
than development milestone numbers:

```bash
python src/pipeline/experiment.py init \
  --config configs/experiments/pipeline_smoke.yaml \
  --experiment-id environment-smoke
```

Run a tracked stage:

```bash
python src/pipeline/experiment.py run \
  --config configs/experiments/pipeline_smoke.yaml \
  --experiment-id environment-smoke \
  --stage tests \
  --input configs/base.yaml \
  -- python -m pytest
```

Inspect the resolved environment, configuration and stage state:

```bash
python src/pipeline/experiment.py status \
  --config configs/experiments/pipeline_smoke.yaml \
  --experiment-id environment-smoke
```

A completed stage is reused only when the resolved configuration, command and
all declared input fingerprints match. Pass `--force` to rerun deliberately.
Fresh experiments initially contain only `manifest.json` and
`resolved_config.yaml`. `logs/`, `stages/` and method output directories are
created lazily only when a stage actually writes them.

Use method names for experiment directories; milestone prefixes such as `m3-`
or `m4-` are not part of the maintained naming convention:

| Purpose | Recommended experiment id |
| :--- | :--- |
| Full Parquet ingestion | `data-full` |
| Point-in-time split | `time-split` |
| Popular and Revisit | `popular-revisit` |
| CoVis matrices and recall | `type-covis`, `buy2buy`, `time-covis` |
| Fixed-position DSSM comparison | `dssm-baseline` |
| Final Attention DSSM | `dssm-attention` |
| Final fused candidates | `candidates-attention` |
| Final features | `features-attention` |
| Final ranker | `lambdarank-attention` |
| Feature-group ablation | `feature-selection` |

An experiment directory therefore contains only its metadata, stage logs and
the outputs written by that method. For example, `type-covis/` creates
`recall/`, `logs/` and `stages/`; it does not create empty `models/`,
`features/`, `candidates/` or `snapshots/` directories.
