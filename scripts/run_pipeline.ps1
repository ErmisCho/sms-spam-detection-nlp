param(
    [switch]$UseAzure,
    [switch]$FullAzure,
    [int]$AzureSampleSize = 100,
    [int]$AzureClusters = 5
)

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "_common.ps1")

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Get-ProjectPythonPath

Require-Python | Out-Null

if ($FullAzure) {
    $UseAzure = $true
}

if ($UseAzure) {
    Require-AzureConfig
}

$pipelineStartedAt = Get-Date
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

function Format-Elapsed {
    param([TimeSpan]$Elapsed)
    return "{0:00}:{1:00}:{2:00}" -f [Math]::Floor($Elapsed.TotalHours), $Elapsed.Minutes, $Elapsed.Seconds
}

function Write-PipelineTiming {
    param([string]$Status)
    if ($stopwatch.IsRunning) {
        $stopwatch.Stop()
    }
    Write-Host ""
    Write-Host $Status
    Write-Host "Started: $($pipelineStartedAt.ToString('yyyy-MM-dd HH:mm:ss'))"
    Write-Host "Finished: $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))"
    Write-Host "Elapsed: $(Format-Elapsed $stopwatch.Elapsed)"
}

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )
    Write-Host ""
    Write-Host "==> $Name"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-PipelineTiming "Pipeline failed during: $Name"
        exit $LASTEXITCODE
    }
}

Invoke-Step "Validate dataset" {
    & (Join-Path $PSScriptRoot "validate_dataset.ps1")
}
Invoke-Step "Analyze text patterns" {
    & (Join-Path $PSScriptRoot "analyze_text.ps1")
}
Invoke-Step "Train and evaluate TF-IDF model" {
    & (Join-Path $PSScriptRoot "model_sms.ps1")
}

if ($UseAzure) {
    if ($FullAzure) {
        Invoke-Step "Cluster full dataset with Azure OpenAI" {
            & (Join-Path $PSScriptRoot "cluster_sms.ps1") -Provider azure-openai -Clusters $AzureClusters
        }
    }
    else {
        Invoke-Step "Cluster Azure OpenAI sample" {
            & (Join-Path $PSScriptRoot "cluster_sms.ps1") -Provider azure-openai -SampleSize $AzureSampleSize -Clusters $AzureClusters
        }
    }
}
else {
    Invoke-Step "Cluster full dataset locally" {
        & (Join-Path $PSScriptRoot "cluster_sms.ps1")
    }
}

Invoke-Step "Generate figures and artifact index" {
    & (Join-Path $PSScriptRoot "generate_outputs.ps1")
}

Write-PipelineTiming "Pipeline complete."
Write-Host "Cluster summary: outputs\cluster_summary.md"
Write-Host "Artifact index: outputs\artifact_index.md"
