param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "_common.ps1")

$pythonPath = Get-ProjectPythonPath
Require-Python | Out-Null
Require-ProjectPackage

$argsList = @("-m", "sms_spam_ham_analysis.download_data")
if ($Force) {
    $argsList += "--force"
}

& $pythonPath @argsList
exit $LASTEXITCODE
