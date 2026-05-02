param(
    [string]$RocmRoot = "C:\Program Files\AMD\ROCm\7.1",
    [string]$OllamaHost = "127.0.0.1:11434",
    [switch]$NoStop,
    [switch]$DebugLog
)

$ErrorActionPreference = "Stop"

$tensilePath = Join-Path $RocmRoot "bin\rocblas\library"
if (-not (Test-Path $tensilePath)) {
    throw "ROCm rocBLAS library path not found: $tensilePath"
}

$gfx1201Kernel = Join-Path $tensilePath "TensileLibrary_lazy_gfx1201.dat"
if (-not (Test-Path $gfx1201Kernel)) {
    throw "ROCm library path exists but does not include gfx1201 kernels for RX 9070 XT: $tensilePath"
}

if (-not $NoStop) {
    Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 2
}

$env:OLLAMA_LLM_LIBRARY = "rocm"
$env:ROCBLAS_TENSILE_LIBPATH = $tensilePath
$env:HIP_PATH = $RocmRoot
$env:ROCM_PATH = $RocmRoot
$env:HIP_VISIBLE_DEVICES = "0"
$env:OLLAMA_HOST = "http://$OllamaHost"
Remove-Item Env:OLLAMA_VULKAN -ErrorAction SilentlyContinue
Remove-Item Env:ROCR_VISIBLE_DEVICES -ErrorAction SilentlyContinue

$ollama = Get-Command ollama -ErrorAction Stop
if ($DebugLog) {
    $logRoot = Join-Path (Get-Location) "data\logs"
    New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
    $stdout = Join-Path $logRoot "ollama-rocm-stdout.log"
    $stderr = Join-Path $logRoot "ollama-rocm-stderr.log"
    Start-Process -FilePath $ollama.Source -ArgumentList "serve" -WindowStyle Hidden `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
} else {
    Start-Process -FilePath $ollama.Source -ArgumentList "serve" -WindowStyle Hidden
}

Start-Sleep -Seconds 8
$version = Invoke-RestMethod -Uri "http://$OllamaHost/api/version" -TimeoutSec 10

[pscustomobject]@{
    status = "started"
    version = $version.version
    host = "http://$OllamaHost"
    llm_library = $env:OLLAMA_LLM_LIBRARY
    rocm_root = $RocmRoot
    rocblas_tensile_libpath = $env:ROCBLAS_TENSILE_LIBPATH
}
