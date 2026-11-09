param(
    [string]$PythonExecutable,
    [string]$CMakeExecutable,
    [string]$NinjaExecutable,
    [string]$VsWhereExecutable,
    [string]$BuildDirectory,
    [ValidateRange(1, 256)][int]$ParallelJobs = 2
)

$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
function Resolve-Executable([string]$Explicit, [string]$Name) {
    if ($Explicit) {
        if (Test-Path -LiteralPath $Explicit -PathType Leaf) {
            return (Resolve-Path -LiteralPath $Explicit).Path
        }
        $found = Get-Command $Explicit -CommandType Application -ErrorAction SilentlyContinue
        if ($found) { return $found.Source }
        throw "Configured executable '$Explicit' was not found."
    }
    $found = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue
    if ($found) { return $found.Source }
    return $null
}

$VsWhere = Resolve-Executable $VsWhereExecutable "vswhere.exe"
if (-not $VsWhere -and ${env:ProgramFiles(x86)}) {
    $candidate = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $candidate -PathType Leaf) { $VsWhere = $candidate }
}
if (-not $VsWhere) {
    throw "Install Visual Studio C++ Build Tools or pass -VsWhereExecutable; no system installation is performed automatically."
}
$VsRoot = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if ($LASTEXITCODE -ne 0) { throw "Visual Studio discovery failed with exit code $LASTEXITCODE" }
if (-not $VsRoot) {
    throw "Visual Studio C++ tools not found."
}

$CMake = Resolve-Executable $CMakeExecutable "cmake.exe"
$Ninja = Resolve-Executable $NinjaExecutable "ninja.exe"
$VcVars = Join-Path $VsRoot "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path -LiteralPath $VcVars)) { throw "vcvars64.bat not found at $VcVars" }

$Python = Resolve-Executable $PythonExecutable "python.exe"
if (-not $Python) {
    throw "Python 3.11+ is the bootstrap prerequisite. Install it or pass -PythonExecutable."
}
if (-not $PythonExecutable -or -not $CMake -or -not $Ninja) {
    $installation = & $Python "$Repo\tools\install_environment.py"
    if ($LASTEXITCODE -ne 0) { throw "Isolated dependency installation failed." }
    $runtime = $installation | ConvertFrom-Json
    if ($runtime.ready -ne $true) { throw "The isolated environment is not ready." }
    if (-not $PythonExecutable) { $Python = [string]$runtime.python }
    $toolDirectory = Split-Path -Parent ([string]$runtime.python)
    if (-not $CMakeExecutable) { $CMake = Join-Path $toolDirectory "cmake.exe" }
    if (-not $NinjaExecutable) { $Ninja = Join-Path $toolDirectory "ninja.exe" }
}
if (-not (Test-Path -LiteralPath $CMake -PathType Leaf)) { throw "CMake is missing from the selected environment." }
if (-not (Test-Path -LiteralPath $Ninja -PathType Leaf)) { throw "Ninja is missing from the selected environment." }

& $Python "$Repo\tools\verify_runtime_entrypoint.py" --repository $Repo
if ($LASTEXITCODE -ne 0) {
    throw "Runtime entry-point verification failed with exit code $LASTEXITCODE"
}

& $Python "$Repo\tools\generate_windows_contract.py"
if ($LASTEXITCODE -ne 0) {
    throw "Native command-contract generation failed with exit code $LASTEXITCODE"
}

$BuildDir = if ($BuildDirectory) { [System.IO.Path]::GetFullPath($BuildDirectory) } else { Join-Path $Repo "build\windows" }
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
$Command = "`"$VcVars`" >nul && `"$CMake`" -S `"$Repo\native\windows`" -B `"$BuildDir`" -G Ninja -DCMAKE_MAKE_PROGRAM=`"$Ninja`" -DCMAKE_BUILD_TYPE=Release && `"$CMake`" --build `"$BuildDir`" --config Release --parallel $ParallelJobs"
cmd /c $Command
if ($LASTEXITCODE -ne 0) {
    throw "Native Windows build failed with exit code $LASTEXITCODE"
}
