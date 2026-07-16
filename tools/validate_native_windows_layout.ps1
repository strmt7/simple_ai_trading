param(
    [string]$Exe = "",
    [string]$Out = "",
    [int]$MinWidth = 1200,
    [int]$MinHeight = 720
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
if (-not $Exe) { $Exe = Join-Path $Repo "build\windows\SimpleAITrading.exe" }
if (-not $Out) { $Out = Join-Path $Repo "artifacts\native-windows-layout-audit.png" }
if (-not (Test-Path -LiteralPath $Exe)) { throw "Native app executable not found: $Exe" }

Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class SatNativeLayoutAudit {
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);
    [DllImport("user32.dll")] public static extern IntPtr GetDlgItem(IntPtr hWnd, int id);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool MoveWindow(IntPtr hWnd, int x, int y, int width, int height, bool repaint);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr hWnd, IntPtr hdc, uint flags);
    [DllImport("user32.dll")] public static extern bool RedrawWindow(IntPtr hWnd, IntPtr updateRect, IntPtr updateRegion, uint flags);
    [DllImport("user32.dll")] public static extern bool UpdateWindow(IntPtr hWnd);
    [DllImport("dwmapi.dll")] public static extern int DwmFlush();
    [DllImport("user32.dll")] public static extern uint GetDpiForSystem();
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern IntPtr SendMessage(IntPtr hWnd, int msg, IntPtr wParam, IntPtr lParam);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern IntPtr SendMessage(IntPtr hWnd, int msg, IntPtr wParam, StringBuilder lParam);
}
"@

[void][SatNativeLayoutAudit]::SetProcessDPIAware()
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Out) | Out-Null

$WM_COMMAND = 0x0111
$WM_GETTEXT = 0x000D
$WM_GETTEXTLENGTH = 0x000E
$LB_GETCOUNT = 0x018B
$LB_SETCURSEL = 0x0186
$CB_GETCOUNT = 0x0146
$LBN_SELCHANGE = 1
$PageListId = 100
$CommandComboId = 101
$ArgsEditId = 102
$OutputEditId = 103
$RunId = 104
$HelpId = 105
$StopId = 106
$PauseId = 107
$StatusId = 111
$ProfileId = 112
$LeverageId = 113
$AiId = 114
$ReinvestId = 115
$ModeId = 116

function Assert-True([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}

function Wait-Until([scriptblock]$Predicate, [string]$Description, [int]$TimeoutMs = 15000) {
    $deadline = [DateTime]::UtcNow.AddMilliseconds($TimeoutMs)
    do {
        if (& $Predicate) { return }
        Start-Sleep -Milliseconds 100
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "Timed out waiting for $Description"
}

function Get-Control([IntPtr]$Window, [int]$Id, [string]$Name) {
    $control = [SatNativeLayoutAudit]::GetDlgItem($Window, $Id)
    Assert-True ($control -ne [IntPtr]::Zero) "$Name control id $Id was not found"
    return $control
}

function Get-Rect([IntPtr]$Handle) {
    $raw = New-Object SatNativeLayoutAudit+RECT
    Assert-True ([SatNativeLayoutAudit]::GetWindowRect($Handle, [ref]$raw)) "GetWindowRect failed"
    return [pscustomobject]@{
        Left = [int]$raw.Left; Top = [int]$raw.Top; Right = [int]$raw.Right; Bottom = [int]$raw.Bottom
        Width = [int]($raw.Right - $raw.Left); Height = [int]($raw.Bottom - $raw.Top)
    }
}

function Get-ControlText([IntPtr]$Control) {
    $length = [SatNativeLayoutAudit]::SendMessage($Control, $WM_GETTEXTLENGTH, [IntPtr]::Zero, [IntPtr]::Zero).ToInt32()
    $builder = [Text.StringBuilder]::new([Math]::Max(1, $length + 1))
    [void][SatNativeLayoutAudit]::SendMessage($Control, $WM_GETTEXT, [IntPtr]($length + 1), $builder)
    return $builder.ToString()
}

function Assert-Visible([IntPtr]$Window, [int]$Id, [string]$Name, [int]$MinW, [int]$MinH) {
    $control = Get-Control $Window $Id $Name
    Assert-True ([SatNativeLayoutAudit]::IsWindowVisible($control)) "$Name is not visible"
    $rect = Get-Rect $control
    Assert-True ($rect.Width -ge $MinW) "$Name width $($rect.Width) is smaller than $MinW"
    Assert-True ($rect.Height -ge $MinH) "$Name height $($rect.Height) is smaller than $MinH"
    return $rect
}

function Assert-Hidden([IntPtr]$Window, [int]$Id, [string]$Name) {
    $control = Get-Control $Window $Id $Name
    Assert-True (-not [SatNativeLayoutAudit]::IsWindowVisible($control)) "$Name should be hidden"
}

function Test-Overlap($A, $B) {
    return ([Math]::Min($A.Right, $B.Right) -gt [Math]::Max($A.Left, $B.Left)) -and
           ([Math]::Min($A.Bottom, $B.Bottom) -gt [Math]::Max($A.Top, $B.Top))
}

function Assert-NoOverlap([string]$AName, $A, [string]$BName, $B) {
    Assert-True (-not (Test-Overlap $A $B)) "$AName overlaps $BName"
}

function Assert-Inside([string]$Name, $Rect, $Window) {
    Assert-True ($Rect.Left -ge $Window.Left -and $Rect.Top -ge $Window.Top) "$Name starts outside the window"
    Assert-True ($Rect.Right -le $Window.Right -and $Rect.Bottom -le $Window.Bottom) "$Name extends outside the window"
}

function Select-Page([IntPtr]$Window, [IntPtr]$List, [int]$Index) {
    [void][SatNativeLayoutAudit]::SendMessage($List, $LB_SETCURSEL, [IntPtr]$Index, [IntPtr]::Zero)
    $wParam = [IntPtr](($LBN_SELCHANGE -shl 16) -bor ($PageListId -band 0xffff))
    [void][SatNativeLayoutAudit]::SendMessage($Window, $WM_COMMAND, $wParam, $List)
    Start-Sleep -Milliseconds 100
}

function Capture-Window([IntPtr]$Window, [string]$Path, $Rect) {
    for ($attempt = 1; $attempt -le 8; $attempt++) {
        [void][SatNativeLayoutAudit]::RedrawWindow($Window, [IntPtr]::Zero, [IntPtr]::Zero, 0x0181)
        [void][SatNativeLayoutAudit]::UpdateWindow($Window)
        [void][SatNativeLayoutAudit]::DwmFlush()
        Start-Sleep -Milliseconds 120
        $bitmap = [Drawing.Bitmap]::new($Rect.Width, $Rect.Height)
        $graphics = [Drawing.Graphics]::FromImage($bitmap)
        try {
            $hdc = $graphics.GetHdc()
            try { Assert-True ([SatNativeLayoutAudit]::PrintWindow($Window, $hdc, 2)) "PrintWindow failed" }
            finally { $graphics.ReleaseHdc($hdc) }
            $bitmap.Save($Path, [Drawing.Imaging.ImageFormat]::Png)
        } finally {
            $graphics.Dispose(); $bitmap.Dispose()
        }
        try {
            Assert-PixelHealth $Path
            return
        } catch {
            if ($attempt -eq 8) { throw }
        }
    }
}

function Assert-PixelHealth([string]$Path) {
    $bitmap = [Drawing.Bitmap]::FromFile($Path)
    try {
        $colors = [Collections.Generic.HashSet[string]]::new()
        $visible = 0; $accent = 0; $samples = 0
        $stepX = [Math]::Max(1, [int]($bitmap.Width / 120)); $stepY = [Math]::Max(1, [int]($bitmap.Height / 80))
        for ($y = 0; $y -lt $bitmap.Height; $y += $stepY) {
            for ($x = 0; $x -lt $bitmap.Width; $x += $stepX) {
                $c = $bitmap.GetPixel($x, $y); [void]$colors.Add("$($c.R),$($c.G),$($c.B)"); $samples++
                if (($c.R + $c.G + $c.B) -gt 80) { $visible++ }
                if ($c.G -gt 110 -and $c.B -gt 100 -and $c.R -lt 145) { $accent++ }
            }
        }
        Assert-True ($colors.Count -ge 24) "screenshot appears flat: only $($colors.Count) sampled colors"
        Assert-True (($visible / $samples) -gt 0.25) "screenshot appears blank or under-rendered"
        Assert-True ($accent -ge 4) "screenshot is missing accent pixels"
    } finally { $bitmap.Dispose() }
}

$oldRepoRoot = $env:SIMPLE_AI_TRADING_REPO_ROOT
$oldDryRun = $env:SIMPLE_AI_TRADING_GUI_DRY_RUN
$process = $null
try {
    $env:SIMPLE_AI_TRADING_REPO_ROOT = $Repo
    $env:SIMPLE_AI_TRADING_GUI_DRY_RUN = "1"
    $process = Start-Process -FilePath $Exe -PassThru -WindowStyle Normal
    Wait-Until { $process.Refresh(); $process.MainWindowHandle -ne [IntPtr]::Zero } "native app window handle" 10000
    $window = $process.MainWindowHandle
    $dpi = [double][SatNativeLayoutAudit]::GetDpiForSystem()
    $targetWidth = [Math]::Max($MinWidth, [int][Math]::Ceiling(1500 * $dpi / 96))
    $targetHeight = [Math]::Max($MinHeight, [int][Math]::Ceiling(980 * $dpi / 96))
    [void][SatNativeLayoutAudit]::MoveWindow($window, 60, 60, $targetWidth, $targetHeight, $true)
    [void][SatNativeLayoutAudit]::SetForegroundWindow($window)
    Start-Sleep -Milliseconds 700

    $windowRect = Get-Rect $window
    Assert-True ($windowRect.Width -ge $MinWidth -and $windowRect.Height -ge $MinHeight) "window is below minimum size"
    $page = Assert-Visible $window $PageListId "navigation" 180 250
    $start = Assert-Visible $window $RunId "start" 80 36
    $pause = Assert-Visible $window $PauseId "pause" 70 36
    $stop = Assert-Visible $window $StopId "stop and close" 120 36
    $mode = Assert-Visible $window $ModeId "execution mode" 90 26
    $profile = Assert-Visible $window $ProfileId "risk profile" 120 26
    $leverage = Assert-Visible $window $LeverageId "leverage" 90 26
    $ai = Assert-Visible $window $AiId "AI toggle" 120 32
    $reinvest = Assert-Visible $window $ReinvestId "reinvest toggle" 140 32
    $status = Assert-Visible $window $StatusId "API budget value" 100 20

    Assert-Hidden $window $CommandComboId "expert command picker on overview"
    Assert-Hidden $window $ArgsEditId "expert flags on overview"
    Assert-Hidden $window $OutputEditId "activity log on overview"
    foreach ($item in @(
        @{N="navigation";R=$page}, @{N="start";R=$start}, @{N="pause";R=$pause}, @{N="stop";R=$stop},
        @{N="mode";R=$mode}, @{N="profile";R=$profile}, @{N="leverage";R=$leverage}, @{N="AI";R=$ai},
        @{N="reinvest";R=$reinvest}, @{N="API budget";R=$status}
    )) { Assert-Inside $item.N $item.R $windowRect }
    Assert-NoOverlap "start" $start "pause" $pause
    Assert-NoOverlap "pause" $pause "stop" $stop
    Assert-NoOverlap "mode" $mode "profile" $profile
    Assert-NoOverlap "profile" $profile "leverage" $leverage
    Assert-NoOverlap "leverage" $leverage "AI" $ai
    Assert-NoOverlap "AI" $ai "reinvest" $reinvest
    Assert-True ($page.Right -lt $mode.Left) "navigation overlaps overview controls"
    Assert-True ($mode.Bottom -lt $status.Top) "overview settings overlap telemetry footer"
    Assert-True ((Get-ControlText (Get-Control $window $RunId "start")) -eq "Start") "overview start label is wrong"
    Assert-True ((Get-ControlText (Get-Control $window $StopId "stop")) -eq "Stop + Close") "overview stop label is wrong"

    $pageList = Get-Control $window $PageListId "navigation"
    $pageCount = [SatNativeLayoutAudit]::SendMessage($pageList, $LB_GETCOUNT, [IntPtr]::Zero, [IntPtr]::Zero).ToInt32()
    Assert-True ($pageCount -eq 7) "expected 7 workflow pages, found $pageCount"
    for ($index = 1; $index -lt $pageCount; $index++) {
        Select-Page $window $pageList $index
        $combo = Get-Control $window $CommandComboId "command picker"
        Assert-True ([SatNativeLayoutAudit]::IsWindowVisible($combo)) "page $index command picker is hidden"
        $count = [SatNativeLayoutAudit]::SendMessage($combo, $CB_GETCOUNT, [IntPtr]::Zero, [IntPtr]::Zero).ToInt32()
        Assert-True ($count -gt 0) "page $index has no generated CLI commands"
        Assert-Hidden $window $ModeId "overview mode on page $index"
        Assert-Hidden $window $ProfileId "overview profile on page $index"
    }
    Select-Page $window $pageList 0
    Capture-Window $window $Out $windowRect
    Write-Output "native Windows layout audit passed: $Out ($($windowRect.Width)x$($windowRect.Height))"
} finally {
    if ($process -ne $null -and -not $process.HasExited) { Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue }
    if ($null -eq $oldRepoRoot) { Remove-Item Env:SIMPLE_AI_TRADING_REPO_ROOT -ErrorAction SilentlyContinue } else { $env:SIMPLE_AI_TRADING_REPO_ROOT = $oldRepoRoot }
    if ($null -eq $oldDryRun) { Remove-Item Env:SIMPLE_AI_TRADING_GUI_DRY_RUN -ErrorAction SilentlyContinue } else { $env:SIMPLE_AI_TRADING_GUI_DRY_RUN = $oldDryRun }
}
