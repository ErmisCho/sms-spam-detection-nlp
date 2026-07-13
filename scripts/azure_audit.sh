#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <resource-group> [subscription-id]" >&2
  exit 2
fi

resource_group=$1
subscription_id=${2:-$(az account show --query id --output tsv)}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
scope="/subscriptions/${subscription_id}/resourceGroups/${resource_group}"

az account show --output table

if [[ $(az group exists --name "$resource_group") != "true" ]]; then
  echo "Resource group '$resource_group' does not exist." >&2
  exit 1
fi

echo "Resources in ${resource_group}:"
az resource list --resource-group "$resource_group" \
  --query "[].{name:name,type:type,location:location}" --output table

echo "Resources outside the free-tier-oriented baseline (expected: no rows):"
az resource list --resource-group "$resource_group" \
  --query "[?type!='Microsoft.App/managedEnvironments' && type!='Microsoft.App/containerApps' && type!='Microsoft.Consumption/budgets'].{name:name,type:type}" \
  --output table

echo "Month-to-date actual cost by resource type (billing data can be delayed):"
az rest --method post \
  --url "https://management.azure.com${scope}/providers/Microsoft.CostManagement/query?api-version=2025-03-01" \
  --body "@${repo_root}/infra/azure/cost-query.json" \
  --query "properties.{columns:columns[].name,rows:rows}" --output json

echo "Configured resource-group budget alerts (availability depends on subscription type):"
az consumption budget list \
  --query "[].{name:name,amount:amount,timeGrain:timeGrain}" --output table || true
