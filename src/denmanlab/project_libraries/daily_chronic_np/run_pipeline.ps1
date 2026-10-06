param(
    [string]$Config = "pipeline/config.yaml"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvName = "spikeinterface"
$CondaExe = @(
    $env:CONDA_EXE
    (Get-Command conda.exe -ErrorAction SilentlyContinue).Source
    (Join-Path $env:LOCALAPPDATA "anaconda3\Scripts\conda.exe")
    (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe")
    (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe")
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1

if (-not $CondaExe) {
    throw "Could not find conda.exe. Activate Conda or install Anaconda/Miniconda."
}

function Invoke-Pipeline {
    param([string[]]$PythonArgs)
    Push-Location $RepoRoot
    try {
        & $CondaExe run --no-capture-output -n $EnvName python @PythonArgs
    }
    finally {
        Pop-Location
    }
}

function Show-Status {
    Invoke-Pipeline @("-m", "pipeline.status", "--config", $Config)
}

function Test-RunningRows {
    Push-Location $RepoRoot
    try {
        & $CondaExe run -n $EnvName python -m pipeline.status --config $Config --running-exit-code | Out-Host
        return ($LASTEXITCODE -eq 2)
    }
    finally {
        Pop-Location
    }
}

function Guard-Running {
    if (Test-RunningRows) {
        Write-Host ""
        Write-Host "A probe is already running. Use status/GPU monitoring; do not start another run yet." -ForegroundColor Yellow
        return $false
    }
    return $true
}

function Run-AllPending {
    if (-not (Guard-Running)) { return }
    Invoke-Pipeline @("-m", "pipeline.run_pending", "--config", $Config)
}

function Run-OneProbe {
    if (-not (Guard-Running)) { return }
    Show-Status
    $sessionId = Read-Host "Session ID to run"
    if ([string]::IsNullOrWhiteSpace($sessionId)) { return }
    Invoke-Pipeline @("-m", "pipeline.run_one", "--config", $Config, "--session-id", $sessionId)
}

function Run-SmokeTest {
    if (-not (Guard-Running)) { return }
    Show-Status
    $sessionId = Read-Host "Session ID for smoke test"
    if ([string]::IsNullOrWhiteSpace($sessionId)) { return }
    $duration = Read-Host "Duration seconds [default 10]"
    if ([string]::IsNullOrWhiteSpace($duration)) { $duration = "10" }
    Invoke-Pipeline @("-m", "pipeline.run_one", "--config", $Config, "--session-id", $sessionId, "--duration-seconds", $duration)
}

function Show-Reports {
    Invoke-Pipeline @("-m", "pipeline.reports", "--config", $Config)
}

function Open-Report {
    Invoke-Pipeline @("-m", "pipeline.reports", "--config", $Config, "--open")
}

function Show-Gpu {
    & nvidia-smi
}

function Verify-Backup {
    Invoke-Pipeline @("-m", "pipeline.backup", "--config", $Config, "--all", "--update-registry")
}

function Show-Troubleshooting {
    Write-Host ""
    Write-Host "Troubleshooting"
    Write-Host "---------------"
    Write-Host "1. If the prompt looks frozen, check status and nvidia-smi. Kilosort can run quietly."
    Write-Host "2. Active rows usually show status=sorting or status=qc_running."
    Write-Host "3. Output is beside each probe's continuous.dat in spikeinterface_output/."
    Write-Host "4. Do not start a second run while a row is running."
    Write-Host "5. If a run fails, inspect error_message in status before retrying."
    Write-Host ""
    Show-Status
}

function Show-Menu {
    Clear-Host
    Write-Host "SpikeInterface Pipeline Runner"
    Write-Host "=============================="
    Write-Host "Repo:   $RepoRoot"
    Write-Host "Config: $Config"
    Write-Host ""
    Write-Host "1. Show registry status"
    Write-Host "2. Run all pending probes"
    Write-Host "3. Run one probe by session ID"
    Write-Host "4. Run short smoke test"
    Write-Host "5. List summary reports"
    Write-Host "6. Open newest summary report"
    Write-Host "7. Show GPU status"
    Write-Host "8. Verify raw backup"
    Write-Host "9. Troubleshooting / stuck job info"
    Write-Host "Q. Quit"
    Write-Host ""
}

while ($true) {
    Show-Menu
    $choice = Read-Host "Choose an option"
    switch ($choice.ToUpperInvariant()) {
        "1" { Show-Status; Pause }
        "2" { Run-AllPending; Pause }
        "3" { Run-OneProbe; Pause }
        "4" { Run-SmokeTest; Pause }
        "5" { Show-Reports; Pause }
        "6" { Open-Report; Pause }
        "7" { Show-Gpu; Pause }
        "8" { Verify-Backup; Pause }
        "9" { Show-Troubleshooting; Pause }
        "Q" { break }
        default { Write-Host "Unknown option: $choice"; Pause }
    }
}
