#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 || $2 != "--confirm" ]]; then
  echo "Usage: $0 <resource-group> --confirm <same-resource-group>" >&2
  exit 2
fi

resource_group=$1
confirmation=$3

if [[ $resource_group != "$confirmation" ]]; then
  echo "Confirmation must exactly match resource group '$resource_group'." >&2
  exit 2
fi

if [[ $(az group exists --name "$resource_group") != "true" ]]; then
  echo "Resource group '$resource_group' is already absent."
  exit 0
fi

echo "Deleting dedicated resource group '$resource_group' and every resource inside it."
az group delete --name "$resource_group" --yes

if [[ $(az group exists --name "$resource_group") == "true" ]]; then
  echo "Resource group '$resource_group' still exists after deletion returned." >&2
  exit 1
fi

echo "Verified: resource group '$resource_group' no longer exists."
