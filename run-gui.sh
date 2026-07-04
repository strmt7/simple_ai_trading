#!/usr/bin/env sh
set -u

APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P) || exit 1

case "${1:-}" in
    --help|-h)
        cat <<EOF
usage: run-gui.sh [--help]

Launches the native Simple AI Trading Windows app. On Windows, build it with:
  powershell -ExecutionPolicy Bypass -File tools\\build_native_windows.ps1
EOF
        exit 0
        ;;
esac

cd "$APP_DIR" || {
    echo "[simple-ai-trading] Failed to enter $APP_DIR." >&2
    exit 1
}

if [ -n "${PYTHONPATH:-}" ]; then
    PYTHONPATH="$APP_DIR/src:$PYTHONPATH"
else
    PYTHONPATH="$APP_DIR/src"
fi
export PYTHONPATH
SIMPLE_AI_TRADING_REPO_ROOT="$APP_DIR"
export SIMPLE_AI_TRADING_REPO_ROOT

if [ -x "$APP_DIR/build/windows/SimpleAITrading.exe" ]; then
    if [ -z "${SIMPLE_AI_TRADING_PYTHON:-}" ] && [ -x "$APP_DIR/.venv311/Scripts/python.exe" ]; then
        SIMPLE_AI_TRADING_PYTHON="$APP_DIR/.venv311/Scripts/python.exe"
        export SIMPLE_AI_TRADING_PYTHON
    fi
    if [ -z "${SIMPLE_AI_TRADING_PYTHON:-}" ] && [ -x "$APP_DIR/.venv311/bin/python" ]; then
        SIMPLE_AI_TRADING_PYTHON="$APP_DIR/.venv311/bin/python"
        export SIMPLE_AI_TRADING_PYTHON
    fi
    echo "[simple-ai-trading] using native Windows app: $APP_DIR/build/windows/SimpleAITrading.exe"
    exec "$APP_DIR/build/windows/SimpleAITrading.exe" "$@"
fi

try_exe() {
    candidate=$1
    label=$2
    shift 2
    if [ -x "$candidate" ]; then
        echo "[simple-ai-trading] using $label: $candidate"
        exec "$candidate" "$@"
    fi
}

try_python() {
    candidate=$1
    label=$2
    shift 2
    if [ -x "$candidate" ] && "$candidate" -c 'import simple_ai_trading.windows_app' >/dev/null 2>&1; then
        echo "[simple-ai-trading] using $label: $candidate -m simple_ai_trading.windows_app"
        exec "$candidate" -m simple_ai_trading.windows_app "$@"
    fi
}

try_exe "$APP_DIR/.venv311/bin/simple-ai-trading-windows" "local .venv311 Windows app launcher" "$@"
try_exe "$APP_DIR/.venv/bin/simple-ai-trading-windows" "local .venv Windows app launcher" "$@"
try_exe "$APP_DIR/.venv311/Scripts/simple-ai-trading-windows.exe" "local .venv311 Windows app launcher" "$@"
try_exe "$APP_DIR/.venv/Scripts/simple-ai-trading-windows.exe" "local .venv Windows app launcher" "$@"

if command -v simple-ai-trading-windows >/dev/null 2>&1; then
    TRADER_EXE=$(command -v simple-ai-trading-windows)
    echo "[simple-ai-trading] using PATH Windows app launcher: $TRADER_EXE"
    exec simple-ai-trading-windows "$@"
fi

try_python "$APP_DIR/.venv311/bin/python" "local .venv311 Python module" "$@"
try_python "$APP_DIR/.venv/bin/python" "local .venv Python module" "$@"
try_python "$APP_DIR/.venv311/Scripts/python.exe" "local .venv311 Windows Python module" "$@"
try_python "$APP_DIR/.venv/Scripts/python.exe" "local .venv Windows Python module" "$@"

if command -v python3.11 >/dev/null 2>&1 && python3.11 -c 'import simple_ai_trading.windows_app' >/dev/null 2>&1; then
    echo "[simple-ai-trading] using PATH Python: python3.11 -m simple_ai_trading.windows_app"
    exec python3.11 -m simple_ai_trading.windows_app "$@"
fi

if command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; assert sys.version_info >= (3, 11); import simple_ai_trading.windows_app' >/dev/null 2>&1; then
    echo "[simple-ai-trading] using PATH Python: python3 -m simple_ai_trading.windows_app"
    exec python3 -m simple_ai_trading.windows_app "$@"
fi

cat >&2 <<EOF
[simple-ai-trading] Unable to launch the native Windows app.
Checked local/PATH simple-ai-trading-windows launchers and Python 3.11+ module entrypoints.

On Windows, build the native app with:
  powershell -ExecutionPolicy Bypass -File tools\\build_native_windows.ps1
EOF
exit 1
