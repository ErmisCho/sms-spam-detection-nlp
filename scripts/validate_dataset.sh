#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

run_python_module sms_spam_ham_analysis.data \
    --input "${PROJECT_ROOT}/data/raw" \
    --output "${PROJECT_ROOT}/outputs/validated_sms_dataset.csv"
