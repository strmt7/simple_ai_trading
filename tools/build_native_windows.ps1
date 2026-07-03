$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
$VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $VsWhere)) {
    throw "vswhere.exe not found; install Visual Studio Build Tools with C++ desktop workload."
}

$VsRoot = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $VsRoot) {
    throw "Visual Studio C++ tools not found."
}

$CMake = Join-Path $VsRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$Ninja = Join-Path $VsRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
$VcVars = Join-Path $VsRoot "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path -LiteralPath $CMake)) { throw "CMake not found at $CMake" }
if (-not (Test-Path -LiteralPath $Ninja)) { throw "Ninja not found at $Ninja" }
if (-not (Test-Path -LiteralPath $VcVars)) { throw "vcvars64.bat not found at $VcVars" }

$Python = Join-Path $Repo ".venv311\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCommand) {
        $Python = $PythonCommand.Source
    } else {
        $PyCommand = Get-Command py -ErrorAction SilentlyContinue
        if (-not $PyCommand) {
            throw "Python 3.11+ was not found. Install Python or create .venv311 before building the native app."
        }
        $Python = $PyCommand.Source
    }
}

if ((Split-Path -Leaf $Python) -ieq "py.exe") {
    & $Python -3.11 "$Repo\tools\generate_windows_contract.py"
} else {
    & $Python "$Repo\tools\generate_windows_contract.py"
}

$BuildDir = Join-Path $Repo "build\windows"
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
$Command = "`"$VcVars`" >nul && `"$CMake`" -S `"$Repo\native\windows`" -B `"$BuildDir`" -G Ninja -DCMAKE_MAKE_PROGRAM=`"$Ninja`" -DCMAKE_BUILD_TYPE=Release && `"$CMake`" --build `"$BuildDir`" --config Release"
cmd /c $Command
if ($LASTEXITCODE -ne 0) {
    throw "Native Windows build failed with exit code $LASTEXITCODE"
}
