param(
    [string]$Exe = "",
    [switch]$SkipRealCompute
)

$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
if (-not $Exe) {
    $Exe = Join-Path $Repo "build\windows\SimpleAITrading.exe"
}
if (-not (Test-Path -LiteralPath $Exe)) {
    throw "Native app executable not found: $Exe"
}

Add-Type @"
using System;
using System.Text;
using System.Runtime.InteropServices;

public static class SatNativeUi {
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] public static extern IntPtr GetDlgItem(IntPtr hWnd, int nIDDlgItem);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern IntPtr SendMessage(IntPtr hWnd, int Msg, IntPtr wParam, IntPtr lParam);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern IntPtr SendMessage(IntPtr hWnd, int Msg, IntPtr wParam, StringBuilder lParam);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern IntPtr SendMessage(IntPtr hWnd, int Msg, IntPtr wParam, string lParam);
}
"@

[void][SatNativeUi]::SetProcessDPIAware()

$WM_COMMAND = 0x0111
$WM_GETTEXT = 0x000D
$WM_GETTEXTLENGTH = 0x000E
$BM_CLICK = 0x00F5
$LB_GETCOUNT = 0x018B
$LB_SETCURSEL = 0x0186
$CB_GETCOUNT = 0x0146
$CB_SETCURSEL = 0x014E
$CB_FINDSTRINGEXACT = 0x0158
$LBN_SELCHANGE = 1
$CBN_SELCHANGE = 1

$PageListId = 100
$CommandComboId = 101
$OutputEditId = 103
$RunSelectedId = 104
$SelectedHelpId = 105
$StopAllId = 106
$AiPreflightId = 107
$RiskReportId = 108
$ModelLabId = 109
$BacktestChartId = 110
$QuickBaseId = 200

function New-WParam([int]$id, [int]$notification) {
    return [IntPtr](($id -band 0xffff) -bor (($notification -band 0xffff) -shl 16))
}

function Send-MessageInt([IntPtr]$Handle, [int]$Message, [int]$WParam = 0, [int]$LParam = 0) {
    return [SatNativeUi]::SendMessage($Handle, $Message, [IntPtr]$WParam, [IntPtr]$LParam).ToInt32()
}

function Get-Control([IntPtr]$Window, [int]$Id) {
    $control = [SatNativeUi]::GetDlgItem($Window, $Id)
    if ($control -eq [IntPtr]::Zero) {
        throw "Control $Id was not found"
    }
    return $control
}

function Get-ControlText([IntPtr]$Control) {
    $length = [SatNativeUi]::SendMessage($Control, $WM_GETTEXTLENGTH, [IntPtr]::Zero, [IntPtr]::Zero).ToInt32()
    $builder = [System.Text.StringBuilder]::new([Math]::Max(1, $length + 1))
    [void][SatNativeUi]::SendMessage($Control, $WM_GETTEXT, [IntPtr]($length + 1), $builder)
    return $builder.ToString()
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

function Select-Page([IntPtr]$Window, [IntPtr]$PageList, [int]$Index) {
    [void]([SatNativeUi]::SendMessage($PageList, $LB_SETCURSEL, [IntPtr]$Index, [IntPtr]::Zero))
    [void]([SatNativeUi]::SendMessage($Window, $WM_COMMAND, (New-WParam $PageListId $LBN_SELCHANGE), $PageList))
    Start-Sleep -Milliseconds 150
}

function Select-Command([IntPtr]$Window, [IntPtr]$Combo, [string]$DisplayText) {
    $index = [SatNativeUi]::SendMessage($Combo, $CB_FINDSTRINGEXACT, [IntPtr](-1), $DisplayText).ToInt32()
    if ($index -lt 0) {
        throw "Command picker item not found: $DisplayText"
    }
    [void]([SatNativeUi]::SendMessage($Combo, $CB_SETCURSEL, [IntPtr]$index, [IntPtr]::Zero))
    [void]([SatNativeUi]::SendMessage($Window, $WM_COMMAND, (New-WParam $CommandComboId $CBN_SELCHANGE), $Combo))
    Start-Sleep -Milliseconds 150
}

function Click-Control([IntPtr]$Control) {
    [void]([SatNativeUi]::SendMessage($Control, $BM_CLICK, [IntPtr]::Zero, [IntPtr]::Zero))
}

function Assert-OutputContains([IntPtr]$Output, [string]$Needle, [int]$TimeoutMs = 15000) {
    Wait-Until { (Get-ControlText $Output).Contains($Needle) } "output containing '$Needle'" $TimeoutMs
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
    [void][SatNativeUi]::SetForegroundWindow($process.MainWindowHandle)

    $window = $process.MainWindowHandle
    $pageList = Get-Control $window $PageListId
    $combo = Get-Control $window $CommandComboId
    $output = Get-Control $window $OutputEditId

    $pageCount = Send-MessageInt $pageList $LB_GETCOUNT
    if ($pageCount -ne 7) {
        throw "Expected 7 workflow pages, found $pageCount"
    }

    for ($page = 0; $page -lt $pageCount; $page++) {
        Select-Page $window $pageList $page
        $comboCount = Send-MessageInt $combo $CB_GETCOUNT
        if ($comboCount -le 0) {
            throw "Workflow page $page has no command picker entries"
        }
    }

    Select-Page $window $pageList 0
    Select-Command $window $combo "Dashboard / compute"
    Click-Control (Get-Control $window $RunSelectedId)
    Assert-OutputContains $output "dry-run: simple-ai-trading compute"

    Click-Control (Get-Control $window $SelectedHelpId)
    Assert-OutputContains $output "dry-run: simple-ai-trading compute --help"

    foreach ($expected in @(
        @{ Id = $QuickBaseId + 0; Text = "Health Check"; Needle = "dry-run: simple-ai-trading doctor" },
        @{ Id = $QuickBaseId + 1; Text = "Paper Status"; Needle = "dry-run: simple-ai-trading positions" },
        @{ Id = $QuickBaseId + 2; Text = "Risk Snapshot"; Needle = "dry-run: simple-ai-trading risk --paper" },
        @{ Id = $QuickBaseId + 3; Text = "Backtest Chart"; Needle = "dry-run: simple-ai-trading backtest-chart" },
        @{ Id = $QuickBaseId + 4; Text = "Model Lab Smoke"; Needle = "dry-run: simple-ai-trading model-lab --objective conservative" }
    )) {
        $button = Get-Control $window $expected.Id
        $text = Get-ControlText $button
        if ($text -ne $expected.Text) {
            throw "Quick button $($expected.Id) expected '$($expected.Text)', found '$text'"
        }
        if (-not [SatNativeUi]::IsWindowVisible($button)) {
            throw "Quick button $($expected.Id) is not visible"
        }
        Click-Control $button
        Assert-OutputContains $output $expected.Needle
    }

    foreach ($expected in @(
        @{ Id = $StopAllId; Text = "Stop Trading"; Needle = "dry-run: simple-ai-trading close all" },
        @{ Id = $AiPreflightId; Text = "AI Check"; Needle = "dry-run: simple-ai-trading ai" },
        @{ Id = $RiskReportId; Text = "Risk Check"; Needle = "dry-run: simple-ai-trading risk --paper" },
        @{ Id = $ModelLabId; Text = "Model Lab"; Needle = "dry-run: simple-ai-trading model-lab --objective conservative" },
        @{ Id = $BacktestChartId; Text = "Backtest Chart"; Needle = "dry-run: simple-ai-trading backtest-chart" }
    )) {
        $button = Get-Control $window $expected.Id
        $text = Get-ControlText $button
        if ($text -ne $expected.Text) {
            throw "Safety button $($expected.Id) expected '$($expected.Text)', found '$text'"
        }
        Click-Control $button
        Assert-OutputContains $output $expected.Needle
    }

    Stop-Process -Id $process.Id -Force
    $process = $null

    if (-not $SkipRealCompute.IsPresent) {
        Remove-Item Env:SIMPLE_AI_TRADING_GUI_DRY_RUN -ErrorAction SilentlyContinue
        $log = Join-Path $env:TEMP "SimpleAITradingNativeRealComputeSmoke.log"
        Remove-Item -LiteralPath $log -ErrorAction SilentlyContinue
        $env:SIMPLE_AI_TRADING_GUI_SMOKE = "1"
        $env:SIMPLE_AI_TRADING_GUI_SMOKE_LOG = $log
        $process = Start-Process -FilePath $Exe -PassThru -WindowStyle Normal
        Wait-Until { Test-Path -LiteralPath $log } "real compute smoke log" 30000
        $content = Get-Content -Path $log -Raw
        if ($content -notmatch "compute=" -or $content -notmatch "\(exit 0\)") {
            throw "Real compute smoke did not finish successfully:`n$content"
        }
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        $process = $null
    }

    Write-Output "native Windows UI smoke passed"
} finally {
    if ($process -ne $null -and -not $process.HasExited) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    }
    if ($null -eq $oldRepoRoot) { Remove-Item Env:SIMPLE_AI_TRADING_REPO_ROOT -ErrorAction SilentlyContinue } else { $env:SIMPLE_AI_TRADING_REPO_ROOT = $oldRepoRoot }
    if ($null -eq $oldDryRun) { Remove-Item Env:SIMPLE_AI_TRADING_GUI_DRY_RUN -ErrorAction SilentlyContinue } else { $env:SIMPLE_AI_TRADING_GUI_DRY_RUN = $oldDryRun }
    if ($null -eq $oldSmoke) { Remove-Item Env:SIMPLE_AI_TRADING_GUI_SMOKE -ErrorAction SilentlyContinue } else { $env:SIMPLE_AI_TRADING_GUI_SMOKE = $oldSmoke }
    if ($null -eq $oldSmokeLog) { Remove-Item Env:SIMPLE_AI_TRADING_GUI_SMOKE_LOG -ErrorAction SilentlyContinue } else { $env:SIMPLE_AI_TRADING_GUI_SMOKE_LOG = $oldSmokeLog }
}
