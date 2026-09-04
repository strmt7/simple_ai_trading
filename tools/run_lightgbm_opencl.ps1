param(
    [string[]]$PythonArgs = @("-m", "simple_ai_trading", "--help"),
    [string]$Manifest = "docs/review/2026-09-04/gpu-runtime-manifest.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $RepoRoot
try {
    $Runtime = Get-Content -Raw -LiteralPath $Manifest | ConvertFrom-Json
    $Wheel = $Runtime.wheel_path
    if (-not (Test-Path -LiteralPath $Wheel -PathType Leaf)) {
        throw "Reviewed local GPU wheel is unavailable; use the documented rebuild recipe"
    }
    $ActualHash = (Get-FileHash -LiteralPath $Wheel -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualHash -ne $Runtime.wheel_sha256) {
        throw "GPU wheel identity mismatch; refusing an unreviewed replacement"
    }
    # uv overlays are cached. This keeps the locked base environment intact and
    # prevents an ordinary dependency sync from erasing the local GPU runtime.
    & uv run --no-sync --with $Wheel python -m tools.verify_lightgbm_opencl_runtime --manifest $Manifest
    if ($LASTEXITCODE -ne 0) { throw "GPU runtime preflight failed" }
    & uv run --no-sync --with $Wheel python @PythonArgs
    if ($LASTEXITCODE -ne 0) { throw "GPU Python command failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
