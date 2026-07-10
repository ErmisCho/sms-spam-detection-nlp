$script:ProjectRoot = Split-Path -Parent $PSScriptRoot

function Get-ProjectPythonPath {
    return (Join-Path $script:ProjectRoot ".venv-win\Scripts\python.exe")
}

function Require-Python {
    $pythonPath = Get-ProjectPythonPath
    if (-not (Test-Path $pythonPath)) {
        Write-Error "Native Windows virtual environment not found at .venv-win\Scripts\python.exe. Run: python -m venv .venv-win; .\.venv-win\Scripts\Activate.ps1; python -m pip install -r requirements.txt"
        exit 1
    }
    return $pythonPath
}

function Require-ProjectPackage {
    $pythonPath = Require-Python
    & $pythonPath -c "import sms_spam_ham_analysis" *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Project package is not installed in the virtual environment. Run: .\.venv-win\Scripts\Activate.ps1; python -m pip install -r requirements.txt"
        exit 1
    }
}

function Require-AzureConfig {
    $pythonPath = Require-Python
    Require-ProjectPackage
    & $pythonPath -m sms_spam_ham_analysis.azure_config
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
