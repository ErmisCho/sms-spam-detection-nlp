$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$inputPath = Join-Path $projectRoot "data\raw"
$outputPath = Join-Path $projectRoot "outputs\validated_sms_dataset.csv"
$pythonPath = Join-Path $projectRoot ".venv-win\Scripts\python.exe"

if (-not (Test-Path $pythonPath)) {
    Write-Error "Native Windows virtual environment not found at .venv-win\Scripts\python.exe. Run: python -m venv .venv-win; .\.venv-win\Scripts\Activate.ps1; python -m pip install -r requirements.txt"
}

& $pythonPath -m sms_spam_ham_analysis.data --input $inputPath --output $outputPath
exit $LASTEXITCODE
