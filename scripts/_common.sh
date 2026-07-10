#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"

require_python() {
    if [[ ! -x "${PYTHON_BIN}" ]]; then
        cat >&2 <<EOF
Linux/WSL virtual environment not found at .venv/bin/python.
Run:
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt
EOF
        exit 1
    fi
}

require_project_package() {
    if ! "${PYTHON_BIN}" -c "import sms_spam_ham_analysis" >/dev/null 2>&1; then
        cat >&2 <<EOF
Project package is not installed in the virtual environment.
Run:
  source .venv/bin/activate
  python -m pip install -r requirements.txt
EOF
        exit 1
    fi
}

run_python_module() {
    require_python
    require_project_package
    "${PYTHON_BIN}" -m "$@"
}

require_azure_config() {
    require_python
    require_project_package
    "${PYTHON_BIN}" -m sms_spam_ham_analysis.azure_config
}
