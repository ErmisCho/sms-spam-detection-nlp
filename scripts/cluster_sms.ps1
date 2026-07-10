param(
    [ValidateSet("sklearn-svd", "azure-openai")]
    [string]$Provider = "sklearn-svd",
    [string]$EmbeddingModel = "",
    [string]$Clusters = "auto",
    [int]$SampleSize = 0
)

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "_common.ps1")

$projectRoot = Split-Path -Parent $PSScriptRoot
$datasetPath = Join-Path $projectRoot "outputs\validated_sms_dataset.csv"
$pythonPath = Get-ProjectPythonPath

Require-Python | Out-Null

if ($Provider -eq "azure-openai") {
    Require-AzureConfig
}

$argsList = @(
    "-m", "sms_spam_ham_analysis.clustering",
    "--dataset", $datasetPath,
    "--clusters", $Clusters,
    "--provider", $Provider
)

if ($EmbeddingModel -ne "") {
    $argsList += @("--embedding-model", $EmbeddingModel)
}

if ($SampleSize -gt 0) {
    $argsList += @("--sample-size", $SampleSize)
}

& $pythonPath @argsList
exit $LASTEXITCODE
