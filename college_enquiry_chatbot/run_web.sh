#!/usr/bin/env bash
# Run Web Server in Git Bash

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

echo "Starting College Chatbot Web Server using: $PY_CMD"
$PY_CMD web/app.py
