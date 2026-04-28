param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$NoReload
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Repo Python interpreter not found at $python"
}

Set-Location $repoRoot

$args = @(
    "-m",
    "uvicorn",
    "app.main:app",
    "--host",
    $BindHost,
    "--port",
    "$Port"
)

if (-not $NoReload) {
    $args += "--reload"
}

Write-Host "[backend] python=$python"
Write-Host "[backend] cwd=$repoRoot"
Write-Host "[backend] args=$($args -join ' ')"

& $python @args
exit $LASTEXITCODE
