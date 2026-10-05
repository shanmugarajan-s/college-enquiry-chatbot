#!/usr/bin/env bash
# Run Interactive CLI in Git Bash (with winpty support for MinTTY)

if command -v python3 &>/dev/null; then
    PY_CMD="python3"
elif command -v python &>/dev/null; then
    PY_CMD="python"
elif command -v py &>/dev/null; then
    PY_CMD="py"
else
    echo "Error: Python executable not found in PATH."
    exit 1
fi

if command -v winpty &>/dev/null; then
    winpty $PY_CMD src/cli.py
else
    $PY_CMD src/cli.py
fi
