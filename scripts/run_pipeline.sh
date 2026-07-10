#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

USE_AZURE=0
FULL_AZURE=0
AZURE_SAMPLE_SIZE=100
AZURE_CLUSTERS=5

usage() {
    cat <<EOF
Usage: bash scripts/run_pipeline.sh [options]

Options:
  --use-azure                 Run a sampled Azure OpenAI embedding clustering step.
  --full-azure                Run full-dataset Azure OpenAI embedding clustering.
  --azure-sample-size VALUE   Sample size for --use-azure. Default: 100.
  --azure-clusters VALUE      Cluster count for Azure runs. Default: 5.
  -h, --help                  Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --use-azure)
            USE_AZURE=1
            shift
            ;;
        --full-azure)
            FULL_AZURE=1
            USE_AZURE=1
            shift
            ;;
        --azure-sample-size)
            AZURE_SAMPLE_SIZE="${2:?Missing value for --azure-sample-size}"
            shift 2
            ;;
        --azure-clusters)
            AZURE_CLUSTERS="${2:?Missing value for --azure-clusters}"
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

if [[ "${USE_AZURE}" -eq 1 ]]; then
    require_azure_config
fi

STARTED_AT="$(date '+%Y-%m-%d %H:%M:%S')"
START_SECONDS="${SECONDS}"

format_elapsed() {
    local elapsed="$1"
    printf '%02d:%02d:%02d' "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))"
}

print_timing() {
    local status="$1"
    local elapsed="$((SECONDS - START_SECONDS))"
    printf '\n%s\n' "${status}"
    printf 'Started: %s\n' "${STARTED_AT}"
    printf 'Finished: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
    printf 'Elapsed: %s\n' "$(format_elapsed "${elapsed}")"
}

run_step() {
    local name="$1"
    shift
    printf '\n==> %s\n' "${name}"
    if ! "$@"; then
        print_timing "Pipeline failed during: ${name}"
        exit 1
    fi
}

run_step "Validate dataset" bash "${SCRIPT_DIR}/validate_dataset.sh"
run_step "Analyze text patterns" bash "${SCRIPT_DIR}/analyze_text.sh"
run_step "Train and evaluate TF-IDF model" bash "${SCRIPT_DIR}/model_sms.sh"

if [[ "${USE_AZURE}" -eq 1 ]]; then
    if [[ "${FULL_AZURE}" -eq 1 ]]; then
        run_step "Cluster full dataset with Azure OpenAI" \
            bash "${SCRIPT_DIR}/cluster_sms.sh" --provider azure-openai --clusters "${AZURE_CLUSTERS}"
    else
        run_step "Cluster Azure OpenAI sample" \
            bash "${SCRIPT_DIR}/cluster_sms.sh" --provider azure-openai --sample-size "${AZURE_SAMPLE_SIZE}" --clusters "${AZURE_CLUSTERS}"
    fi
else
    run_step "Cluster full dataset locally" bash "${SCRIPT_DIR}/cluster_sms.sh"
fi

run_step "Generate figures and artifact index" bash "${SCRIPT_DIR}/generate_outputs.sh"

print_timing "Pipeline complete."
printf 'Cluster summary: outputs/cluster_summary.md\n'
printf 'Artifact index: outputs/artifact_index.md\n'
