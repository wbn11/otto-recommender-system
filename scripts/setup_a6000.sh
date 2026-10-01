#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${OTTO_PYTHON:-python3}"
VENV_DIR="${OTTO_VENV_DIR:-.venv}"
PYPI_INDEX_URL="${OTTO_PYPI_INDEX_URL:-https://pypi.org/simple}"
TORCH_INDEX_URL="${OTTO_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
FAISS_PACKAGE="${OTTO_FAISS_PACKAGE:-faiss-gpu-cu12}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable '${PYTHON_BIN}' was not found." >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import sys

if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(
        f"Python 3.10-3.12 is required; found {sys.version.split()[0]}. "
        "Install python3.11 and rerun with OTTO_PYTHON=python3.11."
    )
PY

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
"${VENV_PYTHON}" -m ensurepip --upgrade
"${VENV_PYTHON}" -m pip install \
  --index-url "${PYPI_INDEX_URL}" \
  --upgrade pip setuptools wheel
"${VENV_PYTHON}" -m pip install torch --index-url "${TORCH_INDEX_URL}"
"${VENV_PYTHON}" -m pip install \
  --index-url "${PYPI_INDEX_URL}" \
  "${FAISS_PACKAGE}"
"${VENV_PYTHON}" -m pip install \
  --index-url "${PYPI_INDEX_URL}" \
  -r requirements.txt -r requirements-dev.txt
"${VENV_PYTHON}" src/tools/check_environment.py --require-gpu

echo "Virtual environment '${VENV_DIR}' is ready."
echo "Activate it with: source ${VENV_DIR}/bin/activate"
