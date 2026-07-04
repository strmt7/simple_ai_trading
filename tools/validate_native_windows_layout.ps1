param(
    [string]$Exe = "",
    [string]$Out = "",
    [int]$MinWidth = 1200,
    [int]$MinHeight = 720
)

$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
if (-not $Exe) {
    $Exe = Join-Path $Repo "build\windows\SimpleAITrading.exe"
}
if (-not $Out) {
    $Out = Join-Path $Repo "artifacts\native-windows-layout-audit.png"
}
if (-not (Test-Path -LiteralPath $Exe)) {
    throw "Native app executable not found: $Exe"
}

Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Text;

public static class SatNativeLayoutAudit {
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;
    }

    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);
    [DllImport("user32.dll")] public static extern IntPtr GetDlgItem(IntPtr hWnd, int nIDDlgItem);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool MoveWindow(IntPtr hWnd, int X, int Y, int nWidth, int nHeight, bool bRepaint);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr hWnd, IntPtr hdcBlt, uint nFlags);
    [DllImport("user32.dll")] public static extern uint GetDpiForSystem();
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern IntPtr SendMessage(IntPtr hWnd, int Msg, IntPtr wParam, IntPtr lParam);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern IntPtr SendMessage(IntPtr hWnd, int Msg, IntPtr wParam, StringBuilder lParam);
}
"@

[void][SatNativeLayoutAudit]::SetProcessDPIAware()
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Out) | Out-Null

$WM_GETTEXT = 0x000D
$WM_GETTEXTLENGTH = 0x000E
$LB_GETCOUNT = 0x018B
$PageListId = 100
$CommandComboId = 101
$ArgsEditId = 102
$OutputEditId = 103
$RunSelectedId = 104
$SelectedHelpId = 105
$StopAllId = 106
$AiPreflightId = 107
$RiskReportId = 108
$ModelLabId = 109
$BacktestChartId = 110
$StatusBarId = 111
$QuickBaseId = 200

function Assert-True([bool]$Condition, [string]$Message) {
    if (-not $Condition) {
        throw $Message
    }
}

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

function Get-Rect([IntPtr]$Handle) {
    $rect = New-Object SatNativeLayoutAudit+RECT
    Assert-True ([SatNativeLayoutAudit]::GetWindowRect($Handle, [ref]$rect)) "GetWindowRect failed"
    [pscustomobject]@{
        Left = [int]$rect.Left
        Top = [int]$rect.Top
        Right = [int]$rect.Right
        Bottom = [int]$rect.Bottom
        Width = [int]($rect.Right - $rect.Left)
        Height = [int]($rect.Bottom - $rect.Top)
    }
}

function Get-Control([IntPtr]$Window, [int]$Id, [string]$Name) {
    $control = [SatNativeLayoutAudit]::GetDlgItem($Window, $Id)
    Assert-True ($control -ne [IntPtr]::Zero) "$Name control id $Id was not found"
    return $control
}

function Get-ControlText([IntPtr]$Control) {
    $length = [SatNativeLayoutAudit]::SendMessage($Control, $WM_GETTEXTLENGTH, [IntPtr]::Zero, [IntPtr]::Zero).ToInt32()
    $builder = [System.Text.StringBuilder]::new([Math]::Max(1, $length + 1))
    [void][SatNativeLayoutAudit]::SendMessage($Control, $WM_GETTEXT, [IntPtr]($length + 1), $builder)
    return $builder.ToString()
}

function Assert-VisibleControl([IntPtr]$Window, [int]$Id, [string]$Name, [int]$MinW, [int]$MinH) {
    $control = Get-Control $Window $Id $Name
    Assert-True ([SatNativeLayoutAudit]::IsWindowVisible($control)) "$Name is not visible"
    $rect = Get-Rect $control
    Assert-True ($rect.Width -ge $MinW) "$Name width $($rect.Width) is smaller than $MinW"
    Assert-True ($rect.Height -ge $MinH) "$Name height $($rect.Height) is smaller than $MinH"
    return $rect
}

function Test-Overlap($A, $B) {
    $left = [Math]::Max($A.Left, $B.Left)
    $top = [Math]::Max($A.Top, $B.Top)
    $right = [Math]::Min($A.Right, $B.Right)
    $bottom = [Math]::Min($A.Bottom, $B.Bottom)
    return ($right -gt $left) -and ($bottom -gt $top)
}

function Assert-NoOverlap($AName, $A, $BName, $B) {
    Assert-True (-not (Test-Overlap $A $B)) "$AName overlaps $BName"
}

function Assert-InsideWindow($Name, $Rect, $WindowRect) {
    Assert-True ($Rect.Left -ge $WindowRect.Left) "$Name starts left of the window"
    Assert-True ($Rect.Top -ge $WindowRect.Top) "$Name starts above the window"
    Assert-True ($Rect.Right -le $WindowRect.Right) "$Name extends past the right window edge"
    Assert-True ($Rect.Bottom -le $WindowRect.Bottom) "$Name extends past the bottom window edge"
}

function Capture-Window([IntPtr]$Window, [string]$Path, $WindowRect) {
    $bitmap = $null
    $graphics = $null
    try {
        $bitmap = [System.Drawing.Bitmap]::new($WindowRect.Width, $WindowRect.Height)
        $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
        $hdc = $graphics.GetHdc()
        try {
            Assert-True ([SatNativeLayoutAudit]::PrintWindow($Window, $hdc, 2)) "PrintWindow failed"
        } finally {
            $graphics.ReleaseHdc($hdc)
        }
        $bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    } finally {
        if ($graphics -ne $null) { $graphics.Dispose() }
        if ($bitmap -ne $null) { $bitmap.Dispose() }
    }
}

function Assert-PixelHealth([string]$Path) {
    $bitmap = [System.Drawing.Bitmap]::FromFile($Path)
    try {
        $unique = [System.Collections.Generic.HashSet[string]]::new()
        $samples = 0
        $visible = 0
        $accent = 0
        $stepX = [Math]::Max(1, [int]($bitmap.Width / 120))
        $stepY = [Math]::Max(1, [int]($bitmap.Height / 80))
        for ($y = 0; $y -lt $bitmap.Height; $y += $stepY) {
            for ($x = 0; $x -lt $bitmap.Width; $x += $stepX) {
                $color = $bitmap.GetPixel($x, $y)
                [void]$unique.Add("$($color.R),$($color.G),$($color.B)")
                $samples += 1
                if (($color.R + $color.G + $color.B) -gt 80) {
                    $visible += 1
                }
                if ($color.G -gt 110 -and $color.B -gt 110 -and $color.R -lt 140) {
                    $accent += 1
                }
            }
        }
        Assert-True ($samples -gt 0) "screenshot sampling produced no pixels"
        Assert-True ($unique.Count -ge 24) "screenshot appears too flat: only $($unique.Count) sampled colors"
        Assert-True (($visible / $samples) -gt 0.25) "screenshot appears blank or under-rendered"
        Assert-True ($accent -ge 4) "screenshot is missing expected accent pixels"
    } finally {
        $bitmap.Dispose()
    }
}

$oldRepoRoot = $env:SIMPLE_AI_TRADING_REPO_ROOT
$oldDryRun = $env:SIMPLE_AI_TRADING_GUI_DRY_RUN
$oldSmoke = $env:SIMPLE_AI_TRADING_GUI_SMOKE
$oldSmokeLog = $env:SIMPLE_AI_TRADING_GUI_SMOKE_LOG
$process = $null
try {
    $env:SIMPLE_AI_TRADING_REPO_ROOT = $Repo
    $env:SIMPLE_AI_TRADING_GUI_DRY_RUN = "1"
    Remove-Item Env:SIMPLE_AI_TRADING_GUI_SMOKE -ErrorAction SilentlyContinue
    Remove-Item Env:SIMPLE_AI_TRADING_GUI_SMOKE_LOG -ErrorAction SilentlyContinue

    $process = Start-Process -FilePath $Exe -PassThru -WindowStyle Normal
    Wait-Until {
        $process.Refresh()
        $process.MainWindowHandle -ne [IntPtr]::Zero
    } "native app window handle" 10000

    $window = $process.MainWindowHandle
    $dpi = [double][SatNativeLayoutAudit]::GetDpiForSystem()
    $targetWidth = [Math]::Max($MinWidth, [int][Math]::Ceiling(1500.0 * $dpi / 96.0))
    $targetHeight = [Math]::Max($MinHeight, [int][Math]::Ceiling(980.0 * $dpi / 96.0))
    [void][SatNativeLayoutAudit]::MoveWindow($window, 80, 80, $targetWidth, $targetHeight, $true)
    [void][SatNativeLayoutAudit]::SetForegroundWindow($window)
    Start-Sleep -Milliseconds 800

    $windowRect = Get-Rect $window
    Assert-True ($windowRect.Width -ge $MinWidth) "window width $($windowRect.Width) is smaller than $MinWidth"
    Assert-True ($windowRect.Height -ge $MinHeight) "window height $($windowRect.Height) is smaller than $MinHeight"

    $page = Assert-VisibleControl $window $PageListId "workflow navigation" 180 250
    $combo = Assert-VisibleControl $window $CommandComboId "command picker" 260 28
    $args = Assert-VisibleControl $window $ArgsEditId "command options" 220 28
    $output = Assert-VisibleControl $window $OutputEditId "activity log" 700 150
    $run = Assert-VisibleControl $window $RunSelectedId "run selected" 120 34
    $help = Assert-VisibleControl $window $SelectedHelpId "show help" 120 30
    $stop = Assert-VisibleControl $window $StopAllId "stop and close" 140 58
    $ai = Assert-VisibleControl $window $AiPreflightId "pause" 120 58
    $risk = Assert-VisibleControl $window $RiskReportId "risk review" 140 58
    $model = Assert-VisibleControl $window $ModelLabId "positions" 140 54
    $chart = Assert-VisibleControl $window $BacktestChartId "reconcile" 140 54
    $status = Assert-VisibleControl $window $StatusBarId "API budget footer" 900 20

    foreach ($pair in @(
        @{ Name = "workflow navigation"; Rect = $page },
        @{ Name = "command picker"; Rect = $combo },
        @{ Name = "command options"; Rect = $args },
        @{ Name = "activity log"; Rect = $output },
        @{ Name = "run selected"; Rect = $run },
        @{ Name = "show help"; Rect = $help },
        @{ Name = "stop and close"; Rect = $stop },
        @{ Name = "pause"; Rect = $ai },
        @{ Name = "risk review"; Rect = $risk },
        @{ Name = "positions"; Rect = $model },
        @{ Name = "reconcile"; Rect = $chart },
        @{ Name = "API budget footer"; Rect = $status }
    )) {
        Assert-InsideWindow $pair.Name $pair.Rect $windowRect
    }

    Assert-True ($page.Right -lt $combo.Left) "workflow navigation overlaps the command/work area"
    Assert-NoOverlap "command options" $args "run selected" $run
    Assert-NoOverlap "run selected" $run "show help" $help
    Assert-NoOverlap "stop trading" $stop "pause bot" $ai
    Assert-NoOverlap "pause bot" $ai "risk check" $risk
    Assert-NoOverlap "risk check" $risk "positions" $model
    Assert-NoOverlap "positions" $model "reconcile" $chart
    Assert-True ($run.Bottom -lt $stop.Top) "top command controls overlap safety tools"
    Assert-True ($stop.Bottom -lt $output.Top) "safety tools overlap activity log"
    Assert-True ($output.Bottom -lt $status.Top) "activity log overlaps API budget footer"

    $quickRects = @()
    for ($i = 0; $i -lt 4; $i++) {
        $quickRects += Assert-VisibleControl $window ($QuickBaseId + $i) "dashboard workflow card $i" 190 58
    }
    for ($i = 0; $i -lt $quickRects.Count; $i++) {
        Assert-True ($quickRects[$i].Bottom -lt $stop.Top) "dashboard workflow card $i overlaps safety tools"
        for ($j = $i + 1; $j -lt $quickRects.Count; $j++) {
            Assert-NoOverlap "dashboard workflow card $i" $quickRects[$i] "dashboard workflow card $j" $quickRects[$j]
        }
    }
    $hiddenQuick = Get-Control $window ($QuickBaseId + 4) "unused dashboard workflow card"
    Assert-True (-not [SatNativeLayoutAudit]::IsWindowVisible($hiddenQuick)) "unused dashboard workflow card is visible"

    $pageCount = [SatNativeLayoutAudit]::SendMessage((Get-Control $window $PageListId "workflow navigation"), $LB_GETCOUNT, [IntPtr]::Zero, [IntPtr]::Zero).ToInt32()
    Assert-True ($pageCount -eq 7) "expected 7 workflow pages, found $pageCount"
    Assert-True ((Get-ControlText (Get-Control $window $StatusBarId "API budget footer")).Contains("API budget")) "API budget footer text is missing"

    Capture-Window $window $Out $windowRect
    Assert-PixelHealth $Out

    Write-Output "native Windows layout audit passed: $Out ($($windowRect.Width)x$($windowRect.Height))"
} finally {
    if ($process -ne $null -and -not $process.HasExited) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    }
    if ($null -eq $oldRepoRoot) { Remove-Item Env:SIMPLE_AI_TRADING_REPO_ROOT -ErrorAction SilentlyContinue } else { $env:SIMPLE_AI_TRADING_REPO_ROOT = $oldRepoRoot }
    if ($null -eq $oldDryRun) { Remove-Item Env:SIMPLE_AI_TRADING_GUI_DRY_RUN -ErrorAction SilentlyContinue } else { $env:SIMPLE_AI_TRADING_GUI_DRY_RUN = $oldDryRun }
    if ($null -eq $oldSmoke) { Remove-Item Env:SIMPLE_AI_TRADING_GUI_SMOKE -ErrorAction SilentlyContinue } else { $env:SIMPLE_AI_TRADING_GUI_SMOKE = $oldSmoke }
    if ($null -eq $oldSmokeLog) { Remove-Item Env:SIMPLE_AI_TRADING_GUI_SMOKE_LOG -ErrorAction SilentlyContinue } else { $env:SIMPLE_AI_TRADING_GUI_SMOKE_LOG = $oldSmokeLog }
}
