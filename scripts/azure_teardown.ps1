param(
    [Parameter(Mandatory = $true)]
    [string]$ResourceGroup,
    [Parameter(Mandatory = $true)]
    [string]$Confirm
)

$ErrorActionPreference = "Stop"

if ($ResourceGroup -ne $Confirm) {
    throw "-Confirm must exactly match resource group '$ResourceGroup'."
}

$Exists = az group exists --name $ResourceGroup
if ($Exists -ne "true") {
    Write-Host "Resource group '$ResourceGroup' is already absent."
    exit 0
}

Write-Host "Deleting dedicated resource group '$ResourceGroup' and every resource inside it."
az group delete --name $ResourceGroup --yes

$StillExists = az group exists --name $ResourceGroup
if ($StillExists -eq "true") {
    throw "Resource group '$ResourceGroup' still exists after deletion returned."
}

Write-Host "Verified: resource group '$ResourceGroup' no longer exists."
