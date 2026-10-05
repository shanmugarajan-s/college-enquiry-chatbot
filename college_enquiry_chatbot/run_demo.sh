#!/usr/bin/env bash
# Run Demo Script for Git Bash / Linux / macOS

# Detect Python command
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

echo "Using Python: $PY_CMD"
$PY_CMD run_demo.py
