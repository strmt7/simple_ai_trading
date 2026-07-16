param(
    [string]$Exe = "",
    [switch]$SkipRealCompute
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
if (-not $Exe) { $Exe = Join-Path $Repo "build\windows\SimpleAITrading.exe" }
if (-not (Test-Path -LiteralPath $Exe)) { throw "Native app executable not found: $Exe" }

Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class SatNativeUi {
    [DllImport("user32.dll")] public static extern IntPtr GetDlgItem(IntPtr hWnd, int id);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern IntPtr SendMessage(IntPtr hWnd, int msg, IntPtr wParam, IntPtr lParam);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern IntPtr SendMessage(IntPtr hWnd, int msg, IntPtr wParam, string lParam);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern IntPtr SendMessage(IntPtr hWnd, int msg, IntPtr wParam, StringBuilder lParam);
}
"@

$WM_COMMAND = 0x0111
$WM_GETTEXT = 0x000D
$WM_GETTEXTLENGTH = 0x000E
$BM_CLICK = 0x00F5
$LB_GETCOUNT = 0x018B
$LB_SETCURSEL = 0x0186
$CB_GETCOUNT = 0x0146
$CB_FINDSTRINGEXACT = 0x0158
$CB_SETCURSEL = 0x014E
$LBN_SELCHANGE = 1
$CBN_SELCHANGE = 1
$PageListId = 100
$CommandComboId = 101
$OutputEditId = 103
$RunId = 104
$StopId = 106
$PauseId = 107
$ProfileId = 112
$LeverageId = 113
$AiId = 114
$ReinvestId = 115
$ModeId = 116

function Wait-Until([scriptblock]$Predicate, [string]$Description, [int]$TimeoutMs = 15000) {
    $deadline = [DateTime]::UtcNow.AddMilliseconds($TimeoutMs)
    do {
        if (& $Predicate) { return }
        Start-Sleep -Milliseconds 50
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "Timed out waiting for $Description"
}

function Get-Control([IntPtr]$Window, [int]$Id) {
    $control = [SatNativeUi]::GetDlgItem($Window, $Id)
    if ($control -eq [IntPtr]::Zero) { throw "Control id $Id was not found" }
    return $control
}

function Get-ControlText([IntPtr]$Control) {
    $length = [SatNativeUi]::SendMessage($Control, $WM_GETTEXTLENGTH, [IntPtr]::Zero, [IntPtr]::Zero).ToInt32()
    $builder = [Text.StringBuilder]::new([Math]::Max(1, $length + 1))
    [void][SatNativeUi]::SendMessage($Control, $WM_GETTEXT, [IntPtr]($length + 1), $builder)
    return $builder.ToString()
}

function New-WParam([int]$Id, [int]$Notification) {
    return [IntPtr](($Notification -shl 16) -bor ($Id -band 0xffff))
}

function Select-Page([IntPtr]$Window, [IntPtr]$List, [int]$Index) {
    [void][SatNativeUi]::SendMessage($List, $LB_SETCURSEL, [IntPtr]$Index, [IntPtr]::Zero)
    [void][SatNativeUi]::SendMessage($Window, $WM_COMMAND, (New-WParam $PageListId $LBN_SELCHANGE), $List)
    Start-Sleep -Milliseconds 100
}

function Select-Combo([IntPtr]$Window, [IntPtr]$Combo, [int]$Id, [string]$Text) {
    $index = [SatNativeUi]::SendMessage($Combo, $CB_FINDSTRINGEXACT, [IntPtr](-1), $Text).ToInt32()
    if ($index -lt 0) { throw "Combo $Id does not contain '$Text'" }
    [void][SatNativeUi]::SendMessage($Combo, $CB_SETCURSEL, [IntPtr]$index, [IntPtr]::Zero)
    [void][SatNativeUi]::SendMessage($Window, $WM_COMMAND, (New-WParam $Id $CBN_SELCHANGE), $Combo)
    Start-Sleep -Milliseconds 75
}

function Click-Control([IntPtr]$Control) {
    [void][SatNativeUi]::SendMessage($Control, $BM_CLICK, [IntPtr]::Zero, [IntPtr]::Zero)
}

function Assert-Text([IntPtr]$Control, [string]$Expected, [string]$Name) {
    $actual = Get-ControlText $Control
    if ($actual -ne $Expected) { throw "$Name expected '$Expected', found '$actual'" }
}

function Assert-OutputContains([IntPtr]$Output, [string]$Needle, [int]$TimeoutMs = 10000) {
    Wait-Until { (Get-ControlText $Output).Contains($Needle) } "output containing '$Needle'" $TimeoutMs
}

$oldRepoRoot = $env:SIMPLE_AI_TRADING_REPO_ROOT
$oldDryRun = $env:SIMPLE_AI_TRADING_GUI_DRY_RUN
$oldDelay = $env:SIMPLE_AI_TRADING_GUI_DRY_RUN_DELAY_MS
$oldDelayCommand = $env:SIMPLE_AI_TRADING_GUI_DRY_RUN_DELAY_COMMAND
$oldFailCommand = $env:SIMPLE_AI_TRADING_GUI_DRY_RUN_FAIL_COMMAND
$oldContractSha = $env:SIMPLE_AI_TRADING_GUI_DRY_RUN_CONTRACT_SHA256
$oldSmoke = $env:SIMPLE_AI_TRADING_GUI_SMOKE
$oldSmokeLog = $env:SIMPLE_AI_TRADING_GUI_SMOKE_LOG
$process = $null
try {
    $env:SIMPLE_AI_TRADING_REPO_ROOT = $Repo
    $env:SIMPLE_AI_TRADING_GUI_DRY_RUN = "1"
    $env:SIMPLE_AI_TRADING_GUI_DRY_RUN_DELAY_MS = "2500"
    Remove-Item Env:SIMPLE_AI_TRADING_GUI_SMOKE -ErrorAction SilentlyContinue
    Remove-Item Env:SIMPLE_AI_TRADING_GUI_SMOKE_LOG -ErrorAction SilentlyContinue

    $process = Start-Process -FilePath $Exe -PassThru -WindowStyle Normal
    Wait-Until { $process.Refresh(); $process.MainWindowHandle -ne [IntPtr]::Zero } "native app window handle" 10000
    [void][SatNativeUi]::SetForegroundWindow($process.MainWindowHandle)
    $window = $process.MainWindowHandle
    $pageList = Get-Control $window $PageListId
    $output = Get-Control $window $OutputEditId
    $profile = Get-Control $window $ProfileId
    $leverage = Get-Control $window $LeverageId
    $mode = Get-Control $window $ModeId
    $ai = Get-Control $window $AiId
    $reinvest = Get-Control $window $ReinvestId

    Wait-Until { (Get-ControlText $profile) -eq "Conservative" } "initial operator status" 10000
    Assert-Text $mode "Paper" "execution mode"
    Assert-Text $leverage "5x" "conservative leverage"
    Assert-Text $ai "AI on (gated)" "AI toggle"
    Assert-Text $reinvest "Reinvest off" "reinvestment toggle"
    Assert-Text (Get-Control $window $StopId) "Stop + Close" "stop control"

    Select-Combo $window $profile $ProfileId "Regular"
    Assert-Text $leverage "10x" "regular profile leverage"
    Select-Combo $window $mode $ModeId "Testnet live"
    Click-Control $ai
    Click-Control $reinvest
    Assert-Text $ai "AI off" "AI toggle after click"
    Assert-Text $reinvest "Reinvest on" "reinvestment toggle after click"

    Click-Control (Get-Control $window $RunId)
    $startCommand = "simple-ai-trading autonomous start --objective regular --live"
    Assert-OutputContains $output $startCommand 5000
    Assert-OutputContains $output "simple-ai-trading strategy --profile regular --leverage 10 --reinvest-profits" 5000
    Assert-OutputContains $output "simple-ai-trading ai --disable" 5000

    Click-Control (Get-Control $window $PauseId)
    Assert-OutputContains $output "simple-ai-trading autonomous pause" 3000
    Wait-Until {
        $text = Get-ControlText $output
        $pause = $text.IndexOf("> simple-ai-trading autonomous pause")
        $pauseResult = $text.IndexOf("dry-run: simple-ai-trading autonomous pause", $pause + 1)
        $pause -ge 0 -and $pauseResult -gt $pause
    } "pause control completion" 3000
    Click-Control (Get-Control $window $StopId)
    Assert-OutputContains $output "simple-ai-trading autonomous stop" 3000
    Wait-Until {
        $text = Get-ControlText $output
        $start = $text.IndexOf("> simple-ai-trading autonomous start --objective regular --live")
        $pause = $text.IndexOf("> simple-ai-trading autonomous pause", $start + 1)
        $stop = $text.IndexOf("> simple-ai-trading autonomous stop", $start + 1)
        $startResult = $text.IndexOf("dry-run: simple-ai-trading autonomous start --objective regular --live", $start + 1)
        $start -ge 0 -and $pause -gt $start -and $stop -gt $pause -and $startResult -gt $stop
    } "independent pause/stop completion before blocking start returns" 7000
    if ((Get-ControlText $output).Contains("simple-ai-trading close all")) {
        throw "Stop control invoked unsafe ledger-only close all"
    }

    $pageCount = [SatNativeUi]::SendMessage($pageList, $LB_GETCOUNT, [IntPtr]::Zero, [IntPtr]::Zero).ToInt32()
    if ($pageCount -ne 7) { throw "Expected 7 workflow pages, found $pageCount" }
    for ($page = 1; $page -lt $pageCount; $page++) {
        Select-Page $window $pageList $page
        $combo = Get-Control $window $CommandComboId
        if (-not [SatNativeUi]::IsWindowVisible($combo)) { throw "Page $page command picker is hidden" }
        $count = [SatNativeUi]::SendMessage($combo, $CB_GETCOUNT, [IntPtr]::Zero, [IntPtr]::Zero).ToInt32()
        if ($count -le 0) { throw "Page $page has no generated command entries" }
    }

    Select-Page $window $pageList 2
    $researchCombo = Get-Control $window $CommandComboId
    foreach ($modelCommand in @(
        "Polymarket models / polymarket-ridge",
        "Polymarket models / polymarket-mlp",
        "AI validation / ai-benchmark"
    )) {
        Select-Combo $window $researchCombo $CommandComboId $modelCommand
    }

    Select-Page $window $pageList 5
    $commandCombo = Get-Control $window $CommandComboId
    Select-Combo $window $commandCombo $CommandComboId "Runtime health / status"
    Click-Control (Get-Control $window $RunId)
    Assert-OutputContains $output "dry-run: simple-ai-trading status" 5000

    Stop-Process -Id $process.Id -Force
    $process = $null

    $env:SIMPLE_AI_TRADING_GUI_DRY_RUN_DELAY_MS = "0"
    $env:SIMPLE_AI_TRADING_GUI_DRY_RUN_FAIL_COMMAND = "ai --enable"
    $process = Start-Process -FilePath $Exe -PassThru -WindowStyle Normal
    Wait-Until { $process.Refresh(); $process.MainWindowHandle -ne [IntPtr]::Zero } "fail-closed app window handle" 10000
    $window = $process.MainWindowHandle
    $output = Get-Control $window $OutputEditId
    Wait-Until { (Get-ControlText (Get-Control $window $ProfileId)) -eq "Conservative" } "fail-closed operator status" 10000
    Click-Control (Get-Control $window $RunId)
    Assert-OutputContains $output "simple-ai-trading strategy --profile conservative --leverage 5 --no-reinvest-profits" 5000
    Assert-OutputContains $output "dry-run: simple-ai-trading ai --enable" 5000
    Assert-OutputContains $output "Workflow stopped after failed command (exit 2)" 5000
    Start-Sleep -Milliseconds 500
    if ((Get-ControlText $output).Contains("> simple-ai-trading autonomous start")) {
        throw "Failed configuration was followed by autonomous start"
    }
    Stop-Process -Id $process.Id -Force
    $process = $null
    Remove-Item Env:SIMPLE_AI_TRADING_GUI_DRY_RUN_FAIL_COMMAND -ErrorAction SilentlyContinue

    $env:SIMPLE_AI_TRADING_GUI_DRY_RUN_DELAY_MS = "2000"
    $env:SIMPLE_AI_TRADING_GUI_DRY_RUN_DELAY_COMMAND = "ai --enable"
    $process = Start-Process -FilePath $Exe -PassThru -WindowStyle Normal
    Wait-Until { $process.Refresh(); $process.MainWindowHandle -ne [IntPtr]::Zero } "cancellation app window handle" 10000
    $window = $process.MainWindowHandle
    $output = Get-Control $window $OutputEditId
    Wait-Until { (Get-ControlText (Get-Control $window $ProfileId)) -eq "Conservative" } "cancellation operator status" 10000
    Click-Control (Get-Control $window $RunId)
    Assert-OutputContains $output "> simple-ai-trading ai --enable" 5000
    Click-Control (Get-Control $window $StopId)
    Assert-OutputContains $output "dry-run: simple-ai-trading autonomous stop" 5000
    Assert-OutputContains $output "Workflow cancelled by a safety control" 5000
    if ((Get-ControlText $output).Contains("> simple-ai-trading autonomous start")) {
        throw "Cancelled configuration was followed by autonomous start"
    }
    Stop-Process -Id $process.Id -Force
    $process = $null
    Remove-Item Env:SIMPLE_AI_TRADING_GUI_DRY_RUN_DELAY_COMMAND -ErrorAction SilentlyContinue

    $env:SIMPLE_AI_TRADING_GUI_DRY_RUN_DELAY_MS = "0"
    $env:SIMPLE_AI_TRADING_GUI_DRY_RUN_CONTRACT_SHA256 = "mismatched-test-contract"
    $process = Start-Process -FilePath $Exe -PassThru -WindowStyle Normal
    Wait-Until { $process.Refresh(); $process.MainWindowHandle -ne [IntPtr]::Zero } "contract mismatch app window handle" 10000
    $window = $process.MainWindowHandle
    $output = Get-Control $window $OutputEditId
    Wait-Until { (Get-ControlText (Get-Control $window $ProfileId)) -eq "Conservative" } "contract mismatch operator status" 10000
    Click-Control (Get-Control $window $RunId)
    Assert-OutputContains $output "Workflow blocked: the native app and Python backend command contracts are not verified as identical" 5000
    if ((Get-ControlText $output).Contains("> simple-ai-trading strategy")) {
        throw "Contract mismatch was followed by strategy mutation"
    }
    Click-Control (Get-Control $window $StopId)
    Assert-OutputContains $output "dry-run: simple-ai-trading autonomous stop" 5000
    Stop-Process -Id $process.Id -Force
    $process = $null
    Remove-Item Env:SIMPLE_AI_TRADING_GUI_DRY_RUN_CONTRACT_SHA256 -ErrorAction SilentlyContinue

    if (-not $SkipRealCompute.IsPresent) {
        Remove-Item Env:SIMPLE_AI_TRADING_GUI_DRY_RUN -ErrorAction SilentlyContinue
        Remove-Item Env:SIMPLE_AI_TRADING_GUI_DRY_RUN_DELAY_MS -ErrorAction SilentlyContinue
        Remove-Item Env:SIMPLE_AI_TRADING_GUI_DRY_RUN_DELAY_COMMAND -ErrorAction SilentlyContinue
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
        if ($process -ne $null -and -not $process.HasExited) { Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue }
        $process = $null
    }

    Write-Output "native Windows UI smoke passed"
} finally {
    if ($process -ne $null -and -not $process.HasExited) { Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue }
    if ($null -eq $oldRepoRoot) { Remove-Item Env:SIMPLE_AI_TRADING_REPO_ROOT -ErrorAction SilentlyContinue } else { $env:SIMPLE_AI_TRADING_REPO_ROOT = $oldRepoRoot }
    if ($null -eq $oldDryRun) { Remove-Item Env:SIMPLE_AI_TRADING_GUI_DRY_RUN -ErrorAction SilentlyContinue } else { $env:SIMPLE_AI_TRADING_GUI_DRY_RUN = $oldDryRun }
    if ($null -eq $oldDelay) { Remove-Item Env:SIMPLE_AI_TRADING_GUI_DRY_RUN_DELAY_MS -ErrorAction SilentlyContinue } else { $env:SIMPLE_AI_TRADING_GUI_DRY_RUN_DELAY_MS = $oldDelay }
    if ($null -eq $oldDelayCommand) { Remove-Item Env:SIMPLE_AI_TRADING_GUI_DRY_RUN_DELAY_COMMAND -ErrorAction SilentlyContinue } else { $env:SIMPLE_AI_TRADING_GUI_DRY_RUN_DELAY_COMMAND = $oldDelayCommand }
    if ($null -eq $oldFailCommand) { Remove-Item Env:SIMPLE_AI_TRADING_GUI_DRY_RUN_FAIL_COMMAND -ErrorAction SilentlyContinue } else { $env:SIMPLE_AI_TRADING_GUI_DRY_RUN_FAIL_COMMAND = $oldFailCommand }
    if ($null -eq $oldContractSha) { Remove-Item Env:SIMPLE_AI_TRADING_GUI_DRY_RUN_CONTRACT_SHA256 -ErrorAction SilentlyContinue } else { $env:SIMPLE_AI_TRADING_GUI_DRY_RUN_CONTRACT_SHA256 = $oldContractSha }
    if ($null -eq $oldSmoke) { Remove-Item Env:SIMPLE_AI_TRADING_GUI_SMOKE -ErrorAction SilentlyContinue } else { $env:SIMPLE_AI_TRADING_GUI_SMOKE = $oldSmoke }
    if ($null -eq $oldSmokeLog) { Remove-Item Env:SIMPLE_AI_TRADING_GUI_SMOKE_LOG -ErrorAction SilentlyContinue } else { $env:SIMPLE_AI_TRADING_GUI_SMOKE_LOG = $oldSmokeLog }
}
