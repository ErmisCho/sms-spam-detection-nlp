param(
    [switch]$CreateZip,
    [string]$PackageName = "sms-spam-detection-nlp-portfolio"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$releaseRoot = Join-Path $projectRoot "submissions"
$stagingRoot = Join-Path $releaseRoot $PackageName
$zipPath = Join-Path $releaseRoot "$PackageName.zip"

$entries = @(
    "README.md",
    "setup.py",
    "requirements.txt",
    ".env.example",
    "data\raw\.gitkeep",
    "src",
    "scripts",
    "tests",
    "docs\project_notes.md",
    "docs\publishing_checklist.md",
    "outputs\artifact_index.md",
    "outputs\classification_report.md",
    "outputs\cluster_summary.md",
    "outputs\clustering",
    "outputs\dataset_validation.json",
    "outputs\frequent_words.md",
    "outputs\vocabulary_findings.md",
    "outputs\ngram_findings.md",
    "outputs\model_metrics.json",
    "outputs\confusion_matrix.csv",
    "outputs\error_examples.csv",
    "outputs\figures"
)

function Convert-ToArtifactPath {
    param([string]$Path)
    return $Path.Replace("\", "/")
}

function Add-ArtifactLines {
    param(
        [System.Collections.Generic.List[string]]$Lines,
        [string]$Root,
        [string[]]$Artifacts
    )

    foreach ($artifact in $Artifacts) {
        $path = Join-Path $Root $artifact
        $status = if (Test-Path $path) { "present" } else { "missing" }
        $displayPath = Convert-ToArtifactPath $artifact
        Write-Host "[$status] $displayPath"
        $Lines.Add("- ``$displayPath`` - $status")
    }
}

function Write-PortfolioArtifactIndex {
    param([string]$Root)

    $tablesAndReports = [System.Collections.Generic.List[string]]::new()
    @(
        "outputs\dataset_validation.json",
        "outputs\frequent_words.md",
        "outputs\vocabulary_findings.md",
        "outputs\ngram_findings.md",
        "outputs\classification_report.md",
        "outputs\model_metrics.json",
        "outputs\confusion_matrix.csv",
        "outputs\error_examples.csv",
        "outputs\cluster_summary.md"
    ) | ForEach-Object { $tablesAndReports.Add($_) }

    foreach ($provider in @("local", "azure")) {
        $providerRoot = Join-Path $Root "outputs\clustering\$provider"
        if (Test-Path $providerRoot) {
            $comparisonPath = Join-Path $Root "outputs\clustering\provider_comparison.md"
            if ((Test-Path $comparisonPath) -and (-not $tablesAndReports.Contains("outputs\clustering\provider_comparison.md"))) {
                $tablesAndReports.Add("outputs\clustering\provider_comparison.md")
            }
            $tablesAndReports.Add("outputs\clustering\$provider\cluster_summary.md")
            $tablesAndReports.Add("outputs\clustering\$provider\embeddings\metadata.json")
        }
    }

    $figures = @(
        "outputs\figures\top_words.png",
        "outputs\figures\vocabulary_comparison.png",
        "outputs\figures\ngrams.png",
        "outputs\figures\confusion_matrix.png",
        "outputs\figures\semantic_clusters.png"
    )

    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add("# Artifact Index")
    $lines.Add("")
    $lines.Add("This index reflects the compact portfolio archive. Intermediate CSVs, model binaries, embedding arrays, raw data, secrets, and local draft materials are intentionally excluded.")
    $lines.Add("")
    $lines.Add("## Included Tables And Reports")
    $lines.Add("")
    Write-Host ""
    Write-Host "Indexing included tables and reports:"
    Add-ArtifactLines -Lines $lines -Root $Root -Artifacts $tablesAndReports.ToArray()
    $lines.Add("")
    $lines.Add("## Included Figures")
    $lines.Add("")
    Write-Host ""
    Write-Host "Indexing included figures:"
    Add-ArtifactLines -Lines $lines -Root $Root -Artifacts $figures
    $lines.Add("")

    $indexPath = Join-Path $Root "outputs\artifact_index.md"
    $indexParent = Split-Path -Parent $indexPath
    if (-not (Test-Path $indexParent)) {
        New-Item -ItemType Directory -Path $indexParent -Force | Out-Null
    }
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllLines($indexPath, $lines, $utf8NoBom)
}

Write-Host "Portfolio archive entries:"
foreach ($entry in $entries) {
    $source = Join-Path $projectRoot $entry
    $status = if (Test-Path $source) { "present" } else { "missing" }
    Write-Host "[$status] $entry"
}

if (-not $CreateZip) {
    Write-Host ""
    Write-Host "Dry run only. Re-run with -CreateZip to create submissions\$PackageName.zip."
    exit 0
}

if (Test-Path $stagingRoot) {
    Remove-Item $stagingRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $stagingRoot | Out-Null

foreach ($entry in $entries) {
    $source = Join-Path $projectRoot $entry
    if (-not (Test-Path $source)) {
        continue
    }

    $destination = Join-Path $stagingRoot $entry
    $destinationParent = Split-Path -Parent $destination
    if (-not (Test-Path $destinationParent)) {
        New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
    }
    Copy-Item $source $destination -Recurse -Force
}

Get-ChildItem $stagingRoot -Directory -Filter "__pycache__" -Recurse | Remove-Item -Recurse -Force
Get-ChildItem $stagingRoot -File -Include "*.pyc", "*.pyo" -Recurse | Remove-Item -Force
Get-ChildItem $stagingRoot -File -Include "*.npy" -Recurse | Remove-Item -Force
Get-ChildItem $stagingRoot -File -Filter "semantic_clusters.csv" -Recurse |
    Where-Object { $_.FullName -like "*\outputs\clustering\*" } |
    Remove-Item -Force

Write-PortfolioArtifactIndex -Root $stagingRoot

if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::Open($zipPath, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    Get-ChildItem -Path $stagingRoot -File -Recurse | ForEach-Object {
        $relativePath = $_.FullName.Substring($stagingRoot.Length).TrimStart("\", "/")
        $entryPath = $relativePath.Replace("\", "/")
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $zip,
            $_.FullName,
            $entryPath,
            [System.IO.Compression.CompressionLevel]::Optimal
        ) | Out-Null
    }
}
finally {
    $zip.Dispose()
}
Write-Host "Created $zipPath"
