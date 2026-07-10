#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

run_python_module sms_spam_ham_analysis.modeling \
    --dataset "${PROJECT_ROOT}/outputs/validated_sms_dataset.csv" \
    --model-out "${PROJECT_ROOT}/outputs/models/tfidf_classifier.joblib"
