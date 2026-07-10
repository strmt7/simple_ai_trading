param(
    [string]$Exe = "",
    [string]$Out = "",
    [int]$MinWidth = 1200,
    [int]$MinHeight = 720,
    [switch]$RealStatus
)

$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
if (-not $Exe) {
    $Exe = Join-Path $Repo "build\windows\SimpleAITrading.exe"
}
if (-not $Out) {
    $Out = Join-Path $Repo "artifacts\native-windows-dashboard.png"
}
if (-not (Test-Path -LiteralPath $Exe)) {
    throw "Native app executable not found: $Exe"
}

Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class SatNativeCapture {
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;
    }

    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr hWnd, IntPtr hdcBlt, uint nFlags);
}
"@

[void][SatNativeCapture]::SetProcessDPIAware()
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Out) | Out-Null

function Wait-Until([scriptblock]$Predicate, [string]$Description, [int]$TimeoutMs = 15000) {
    $deadline = [DateTime]::UtcNow.AddMilliseconds($TimeoutMs)
    do {
        if (& $Predicate) {
            return
        }
        Start-Sleep -Milliseconds 100
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "Timed out waiting for $Description"
}

$oldRepoRoot = $env:SIMPLE_AI_TRADING_REPO_ROOT
$oldDryRun = $env:SIMPLE_AI_TRADING_GUI_DRY_RUN
$oldSmoke = $env:SIMPLE_AI_TRADING_GUI_SMOKE
$oldSmokeLog = $env:SIMPLE_AI_TRADING_GUI_SMOKE_LOG
$process = $null
$bitmap = $null
$graphics = $null
try {
    $env:SIMPLE_AI_TRADING_REPO_ROOT = $Repo
    if ($RealStatus.IsPresent) {
        Remove-Item Env:SIMPLE_AI_TRADING_GUI_DRY_RUN -ErrorAction SilentlyContinue
    } else {
        $env:SIMPLE_AI_TRADING_GUI_DRY_RUN = "1"
    }
    Remove-Item Env:SIMPLE_AI_TRADING_GUI_SMOKE -ErrorAction SilentlyContinue
    Remove-Item Env:SIMPLE_AI_TRADING_GUI_SMOKE_LOG -ErrorAction SilentlyContinue

    $process = Start-Process -FilePath $Exe -PassThru -WindowStyle Normal
    Wait-Until {
        $process.Refresh()
        $process.MainWindowHandle -ne [IntPtr]::Zero
    } "native app window handle" 10000
    Start-Sleep -Milliseconds $(if ($RealStatus.IsPresent) { 5000 } else { 600 })

    $rect = New-Object SatNativeCapture+RECT
    if (-not [SatNativeCapture]::GetWindowRect($process.MainWindowHandle, [ref]$rect)) {
        throw "GetWindowRect failed"
    }
    $width = $rect.Right - $rect.Left
    $height = $rect.Bottom - $rect.Top
    if ($width -lt $MinWidth -or $height -lt $MinHeight) {
        throw "Captured window is too small: ${width}x${height}; expected at least ${MinWidth}x${MinHeight}"
    }

    $bitmap = [System.Drawing.Bitmap]::new($width, $height)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $hdc = $graphics.GetHdc()
    try {
        if (-not [SatNativeCapture]::PrintWindow($process.MainWindowHandle, $hdc, 2)) {
            throw "PrintWindow failed"
        }
    } finally {
        $graphics.ReleaseHdc($hdc)
    }
    $bitmap.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
    Write-Output "captured native Windows app: $Out (${width}x${height})"
} finally {
    if ($graphics -ne $null) { $graphics.Dispose() }
    if ($bitmap -ne $null) { $bitmap.Dispose() }
    if ($process -ne $null -and -not $process.HasExited) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    }
    if ($null -eq $oldRepoRoot) { Remove-Item Env:SIMPLE_AI_TRADING_REPO_ROOT -ErrorAction SilentlyContinue } else { $env:SIMPLE_AI_TRADING_REPO_ROOT = $oldRepoRoot }
    if ($null -eq $oldDryRun) { Remove-Item Env:SIMPLE_AI_TRADING_GUI_DRY_RUN -ErrorAction SilentlyContinue } else { $env:SIMPLE_AI_TRADING_GUI_DRY_RUN = $oldDryRun }
    if ($null -eq $oldSmoke) { Remove-Item Env:SIMPLE_AI_TRADING_GUI_SMOKE -ErrorAction SilentlyContinue } else { $env:SIMPLE_AI_TRADING_GUI_SMOKE = $oldSmoke }
    if ($null -eq $oldSmokeLog) { Remove-Item Env:SIMPLE_AI_TRADING_GUI_SMOKE_LOG -ErrorAction SilentlyContinue } else { $env:SIMPLE_AI_TRADING_GUI_SMOKE_LOG = $oldSmokeLog }
}
