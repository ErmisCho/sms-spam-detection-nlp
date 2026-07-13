param(
    [Parameter(Mandatory = $true)]
    [string]$ResourceGroup,
    [string]$SubscriptionId
)

$ErrorActionPreference = "Stop"

if (-not $SubscriptionId) {
    $SubscriptionId = az account show --query id --output tsv
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Scope = "/subscriptions/$SubscriptionId/resourceGroups/$ResourceGroup"

az account show --output table
$Exists = az group exists --name $ResourceGroup
if ($Exists -ne "true") {
    throw "Resource group '$ResourceGroup' does not exist."
}

Write-Host "Resources in ${ResourceGroup}:"
az resource list --resource-group $ResourceGroup `
    --query "[].{name:name,type:type,location:location}" --output table

Write-Host "Resources outside the free-tier-oriented baseline (expected: no rows):"
az resource list --resource-group $ResourceGroup `
    --query "[?type!='Microsoft.App/managedEnvironments' && type!='Microsoft.App/containerApps' && type!='Microsoft.Consumption/budgets'].{name:name,type:type}" `
    --output table

Write-Host "Month-to-date actual cost by resource type (billing data can be delayed):"
az rest --method post `
    --url "https://management.azure.com$Scope/providers/Microsoft.CostManagement/query?api-version=2025-03-01" `
    --body "@$RepoRoot/infra/azure/cost-query.json" `
    --query "properties.{columns:columns[].name,rows:rows}" --output json

Write-Host "Configured resource-group budget alerts (availability depends on subscription type):"
az consumption budget list `
    --query "[].{name:name,amount:amount,timeGrain:timeGrain}" --output table
