param(
    [string]$RocmRoot = $env:ROCM_PATH,
    [string]$OllamaHost = "",
    [ValidateRange(1, 3600)][int]$SessionSeconds = 300,
    [switch]$NoStop,
    [switch]$DebugLog
)

$ErrorActionPreference = "Stop"
if (-not $RocmRoot) { $RocmRoot = $env:HIP_PATH }
if (-not $RocmRoot -or -not $OllamaHost) {
    throw "Provide an installed ROCm root and a dedicated loopback endpoint explicitly."
}
if ($DebugLog) {
    throw "Raw provider logging is disabled; use redacted operator diagnostics."
}

# NoStop remains accepted for compatibility. All invocations leave existing
# providers untouched and terminate only the newly owned session's children.
$runtime = Get-Command uv -ErrorAction Stop
$repository = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repository
try {
    & $runtime.Source run --no-sync python -m tools.run_ollama_rocm_session `
        --rocm-root $RocmRoot --host $OllamaHost --seconds $SessionSeconds
    if ($LASTEXITCODE -ne 0) { throw "Bounded provider session failed." }
} finally {
    Pop-Location
}
