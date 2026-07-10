#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

args=(sms_spam_ham_analysis.download_data)

if [[ "${1:-}" == "--force" ]]; then
    args+=(--force)
fi

run_python_module "${args[@]}"
