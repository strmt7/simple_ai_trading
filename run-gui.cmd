@echo off
setlocal
set "APP_DIR=%~dp0"

if "%~1"=="--help" goto help
if "%~1"=="-h" goto help

cd /d "%APP_DIR%" || (
    echo [simple-ai-trading] Failed to enter "%APP_DIR%".
    exit /b 1
)

if defined PYTHONPATH (
    set "PYTHONPATH=%APP_DIR%src;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%APP_DIR%src"
)

set "NATIVE_EXE=%APP_DIR%build\windows\SimpleAITrading.exe"
if exist "%NATIVE_EXE%" (
    if not defined SIMPLE_AI_TRADING_PYTHON if exist "%APP_DIR%.venv311\Scripts\python.exe" set "SIMPLE_AI_TRADING_PYTHON=%APP_DIR%.venv311\Scripts\python.exe"
    if not defined SIMPLE_AI_TRADING_PYTHON if exist "%APP_DIR%.venv\Scripts\python.exe" set "SIMPLE_AI_TRADING_PYTHON=%APP_DIR%.venv\Scripts\python.exe"
    echo [simple-ai-trading] using native Windows app: "%NATIVE_EXE%"
    "%NATIVE_EXE%" %*
    exit /b %ERRORLEVEL%
)

set "TRADER_EXE=%APP_DIR%.venv311\Scripts\simple-ai-trading-windows.exe"
if exist "%TRADER_EXE%" goto run_local_exe

set "TRADER_EXE=%APP_DIR%.venv\Scripts\simple-ai-trading-windows.exe"
if exist "%TRADER_EXE%" goto run_local_exe

where.exe simple-ai-trading-windows >nul 2>nul
if not errorlevel 1 goto run_path_exe

set "TRADER_PY=%APP_DIR%.venv311\Scripts\python.exe"
if exist "%TRADER_PY%" (
    "%TRADER_PY%" -c "import simple_ai_trading.windows_app" >nul 2>nul
    if not errorlevel 1 goto run_local_module
)

set "TRADER_PY=%APP_DIR%.venv\Scripts\python.exe"
if exist "%TRADER_PY%" (
    "%TRADER_PY%" -c "import simple_ai_trading.windows_app" >nul 2>nul
    if not errorlevel 1 goto run_local_module
)

py -3.11 -c "import simple_ai_trading.windows_app" >nul 2>nul
if not errorlevel 1 goto run_py311_module

python -c "import sys; sys.exit(1) if sys.version_info < (3, 11) else None; import simple_ai_trading.windows_app" >nul 2>nul
if not errorlevel 1 goto run_python_module

echo [simple-ai-trading] Unable to launch the native Windows app.
echo Checked:
echo   "%APP_DIR%build\windows\SimpleAITrading.exe"
echo   "%APP_DIR%.venv311\Scripts\simple-ai-trading-windows.exe"
echo   "%APP_DIR%.venv\Scripts\simple-ai-trading-windows.exe"
echo   simple-ai-trading-windows on PATH
echo   "%APP_DIR%.venv311\Scripts\python.exe" -m simple_ai_trading.windows_app
echo   "%APP_DIR%.venv\Scripts\python.exe" -m simple_ai_trading.windows_app
echo   py -3.11 -m simple_ai_trading.windows_app
echo   python -m simple_ai_trading.windows_app with Python 3.11+
echo.
echo Build the native app with:
echo   powershell -ExecutionPolicy Bypass -File tools\build_native_windows.ps1
exit /b 1

:help
echo usage: run-gui.cmd [--help]
echo.
echo Launches the native Simple AI Trading Windows app. Build it with:
echo   powershell -ExecutionPolicy Bypass -File tools\build_native_windows.ps1
exit /b 0

:run_local_exe
echo [simple-ai-trading] using local Windows app launcher: "%TRADER_EXE%"
"%TRADER_EXE%" %*
exit /b %ERRORLEVEL%

:run_path_exe
for /f "delims=" %%I in ('where.exe simple-ai-trading-windows 2^>nul') do (
    set "TRADER_EXE=%%I"
    goto run_path_exe_resolved
)

:run_path_exe_resolved
echo [simple-ai-trading] using PATH Windows app launcher: "%TRADER_EXE%"
simple-ai-trading-windows %*
exit /b %ERRORLEVEL%

:run_local_module
echo [simple-ai-trading] using Python module: "%TRADER_PY%" -m simple_ai_trading.windows_app
"%TRADER_PY%" -m simple_ai_trading.windows_app %*
exit /b %ERRORLEVEL%

:run_py311_module
echo [simple-ai-trading] using Python launcher: py -3.11 -m simple_ai_trading.windows_app
py -3.11 -m simple_ai_trading.windows_app %*
exit /b %ERRORLEVEL%

:run_python_module
echo [simple-ai-trading] using PATH Python: python -m simple_ai_trading.windows_app
python -m simple_ai_trading.windows_app %*
exit /b %ERRORLEVEL%
