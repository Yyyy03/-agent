# Windows launcher for the Mint Finance Agent.
# Mirrors start_latest_finagent.sh (macOS). Starts the Python bridge (:4182) and
# the vinext frontend (:3000). All paths are relative to this script.
#
#   Usage:  .\start_latest_finagent.ps1
#
# First run creates a Windows venv at backend\.venv-win and installs
# backend\requirements.txt (torch/docling/etc., can take several minutes).

# "Continue" so native stderr (pip progress/notices) is not treated as a
# terminating error; each critical step checks $LASTEXITCODE / Test-Path and
# explicitly aborts via Write-Error + exit 1 on failure.
$ErrorActionPreference = "Continue"

$AppDir       = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir   = Join-Path $AppDir "backend"
$EnvFile      = Join-Path $AppDir ".env.local_mintcu"
$VenvDir      = Join-Path $BackendDir ".venv-win"
$VenvPy       = Join-Path $VenvDir "Scripts\python.exe"
$FrontendPort = "3000"
$BridgePort   = "4182"
$BridgeLog    = Join-Path $AppDir "latest-finagent-bridge.log"
$DevLog       = Join-Path $AppDir "latest-finagent-dev.log"

function Read-EnvValue([string]$Key) {
    if (-not (Test-Path $EnvFile)) { return "" }
    $line = Get-Content -LiteralPath $EnvFile |
        Where-Object { $_ -match "^\s*$Key=" } | Select-Object -First 1
    if (-not $line) { return "" }
    return ($line -split "=", 2)[1].Trim()
}

function Find-Python {
    # Prefer 3.12, then the default py, then python/python3 on PATH.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $result = $null
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 --version *> $null
        if ($LASTEXITCODE -eq 0) { $result = @("py", "-3.12") }
        else {
            & py --version *> $null
            if ($LASTEXITCODE -eq 0) { $result = @("py") }
        }
    }
    if (-not $result) {
        foreach ($exe in @("python", "python3")) {
            if (Get-Command $exe -ErrorAction SilentlyContinue) {
                & $exe --version *> $null
                if ($LASTEXITCODE -eq 0) { $result = @($exe); break }
            }
        }
    }
    $ErrorActionPreference = $prev
    return $result
}

function Stop-PortListener([string]$Port) {
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($conns) {
        $conns | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object {
            try { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue } catch {}
        }
        Write-Host "Stopped process listening on :$Port"
    }
}

# --- sanity checks -----------------------------------------------------------
if (-not (Test-Path $EnvFile)) { Write-Error "Missing env file: $EnvFile"; exit 1 }

$BridgeToken = Read-EnvValue "FIRE_AGENT_BRIDGE_TOKEN"
if (-not $BridgeToken) { Write-Error "Missing FIRE_AGENT_BRIDGE_TOKEN in $EnvFile"; exit 1 }

if (-not (Test-Path "node_modules\vinext\dist\cli.js")) {
    Write-Host "node_modules not found - running npm install ..."
    Push-Location $AppDir
    & npm install 2>&1 | Out-Host
    Pop-Location
}

# --- backend venv (auto-create + install on first run) -----------------------
if (-not (Test-Path $VenvPy)) {
    $pyArgs = Find-Python
    if (-not $pyArgs) { Write-Error "No Python runtime found. Install Python 3.12 (recommended) or 3.13."; exit 1 }
    Write-Host "Creating Windows venv at $VenvDir using $($pyArgs -join ' ')"
    $create = @($pyArgs) + @("-m", "venv", $VenvDir)
    & $create[0] @($create[1..($create.Length - 1)]) 2>&1 | Out-Host
    if (-not (Test-Path $VenvPy)) { Write-Error "venv creation failed."; exit 1 }

    # Web-app only dependency set (skips the offline torch/sentence-transformers
    # ML stack). Uses the aliyun mirror because pypi.org is unreachable here;
    # the mirror is a full PyPI mirror, so it also works with normal internet.
    Write-Host "Installing web-app dependencies from requirements-web.txt (this can take several minutes) ..."
    $index = @("--index-url", "https://mirrors.aliyun.com/pypi/simple/", "--trusted-host", "mirrors.aliyun.com")
    & $VenvPy -m pip install @index -r (Join-Path $BackendDir "requirements-web.txt") 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) { Write-Error "pip install failed."; exit 1 }
}

# --- free the ports ----------------------------------------------------------
Stop-PortListener $BridgePort
Stop-PortListener $FrontendPort
Start-Sleep -Seconds 1

# --- start bridge (:4182) ----------------------------------------------------
Write-Host "Starting bridge on :$BridgePort ..."
Start-Process -FilePath $VenvPy `
    -ArgumentList @("bridge\fire_agent_bridge.py", "--host", "127.0.0.1", "--port", $BridgePort, "--env-file", $EnvFile) `
    -WorkingDirectory $BackendDir `
    -RedirectStandardOutput $BridgeLog `
    -RedirectStandardError (Join-Path $AppDir "latest-finagent-bridge.err") `
    -WindowStyle Hidden
Start-Sleep -Seconds 3

# --- start frontend (:3000) --------------------------------------------------
# Call the vinext CLI directly with --env-file so the frontend picks up
# FIRE_AGENT_BRIDGE_URL (a plain `npm run dev` would not load this file).
Write-Host "Starting frontend on :$FrontendPort ..."
Start-Process -FilePath "node" `
    -ArgumentList @("--env-file=$EnvFile", "node_modules\vinext\dist\cli.js", "dev") `
    -WorkingDirectory $AppDir `
    -RedirectStandardOutput $DevLog `
    -RedirectStandardError (Join-Path $AppDir "latest-finagent-dev.err") `
    -WindowStyle Hidden
Start-Sleep -Seconds 3

# --- status ------------------------------------------------------------------
$bridgeUp = [bool](Get-NetTCPConnection -LocalPort $BridgePort -State Listen -ErrorAction SilentlyContinue)
$frontUp  = [bool](Get-NetTCPConnection -LocalPort $FrontendPort -State Listen -ErrorAction SilentlyContinue)
Write-Host ""
Write-Host "bridge  :$BridgePort  listening: $bridgeUp"
Write-Host "frontend:$FrontendPort  listening: $frontUp"
if ($bridgeUp) {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$BridgePort/health" -TimeoutSec 5
        Write-Host "bridge health: $($health | ConvertTo-Json -Compress)"
    } catch { Write-Host "bridge health check failed: $_" }
}
Write-Host ""
Write-Host "Open http://localhost:$FrontendPort/?view=workbench"
Write-Host "Logs: $(Split-Path $BridgeLog) (bridge) / $(Split-Path $DevLog) (frontend)"
if (-not ($bridgeUp -and $frontUp)) { Write-Warning "Not both ports are up yet - check the log files." }