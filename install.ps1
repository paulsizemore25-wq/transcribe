#Requires -Version 5.1
<#
    One-shot Windows setup: creates .venv, installs the dependencies into it,
    and verifies the result.  Nothing is installed system-wide.

    Run it with:  powershell -ExecutionPolicy Bypass -File .\install.ps1
    (or just double-click install.bat)
#>

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Find-Python {
    # The py launcher is the reliable way to get a specific version on Windows.
    foreach ($candidate in @(@("py", "-3"), @("python"), @("python3"))) {
        $exe = $candidate[0]
        $prefix = @($candidate | Select-Object -Skip 1)
        try {
            $version = & $exe @prefix -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        } catch {
            continue
        }
        if ($LASTEXITCODE -ne 0 -or -not $version) { continue }
        $parts = $version.Trim().Split(".")
        if ([int]$parts[0] -gt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 9)) {
            return ,@($exe) + $prefix
        }
        Write-Host "Skipping $exe (Python $version, need 3.9+)"
    }
    return $null
}

$python = Find-Python
if (-not $python) {
    Write-Host ""
    Write-Host "Python 3.9 or newer was not found." -ForegroundColor Red
    Write-Host "Install it from https://www.python.org/downloads/windows/ and"
    Write-Host "tick 'Add python.exe to PATH' in the installer, then run this again."
    exit 1
}

$exe = $python[0]
$prefix = @($python | Select-Object -Skip 1)

Write-Host "==> Creating virtual environment in .venv"
& $exe @prefix -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "failed to create the virtual environment" }

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Write-Host "==> Installing dependencies (this downloads ~1 GB, give it a few minutes)"
& $venvPython -m pip install --upgrade pip | Out-Null
& $venvPython -m pip install -e ".[dev]"
if ($LASTEXITCODE -ne 0) { throw "dependency installation failed" }

Write-Host "==> Verifying"
& $venvPython -m transcribe --check
if ($LASTEXITCODE -ne 0) { throw "the install did not verify cleanly" }

Write-Host ""
Write-Host "Done. Transcribe a file with:" -ForegroundColor Green
Write-Host "    .\.venv\Scripts\transcribe.exe C:\path\to\recording.m4a"
Write-Host ""
Write-Host "The first run downloads the model (~1.6 GB); after that it works offline."
