#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

PROVIDER="sklearn-svd"
EMBEDDING_MODEL=""
CLUSTERS="auto"
SAMPLE_SIZE=""

usage() {
    cat <<EOF
Usage: bash scripts/cluster_sms.sh [options]

Options:
  --provider VALUE          Embedding provider: sklearn-svd or azure-openai.
  --embedding-model VALUE   Optional embedding deployment/model name.
  --clusters VALUE          Number of clusters or auto. Default: auto.
  --sample-size VALUE       Optional number of messages to sample.
  -h, --help                Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --provider)
            PROVIDER="${2:?Missing value for --provider}"
            shift 2
            ;;
        --embedding-model)
            EMBEDDING_MODEL="${2:?Missing value for --embedding-model}"
            shift 2
            ;;
        --clusters)
            CLUSTERS="${2:?Missing value for --clusters}"
            shift 2
            ;;
        --sample-size)
            SAMPLE_SIZE="${2:?Missing value for --sample-size}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "${PROVIDER}" == "azure-openai" ]]; then
    require_azure_config
fi

args=(
    sms_spam_ham_analysis.clustering
    --dataset "${PROJECT_ROOT}/outputs/validated_sms_dataset.csv"
    --clusters "${CLUSTERS}"
    --provider "${PROVIDER}"
)

if [[ -n "${EMBEDDING_MODEL}" ]]; then
    args+=(--embedding-model "${EMBEDDING_MODEL}")
fi

if [[ -n "${SAMPLE_SIZE}" ]]; then
    args+=(--sample-size "${SAMPLE_SIZE}")
fi

run_python_module "${args[@]}"
