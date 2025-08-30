#!/usr/bin/env bash

# Exit on error
set -e

# Function to detect OS
detect_os() {
    case "$OSTYPE" in
        msys*|cygwin*|win*)     echo "windows" ;;
        darwin*)                echo "macos" ;;
        linux*)                 echo "linux" ;;
        *)                      echo "unknown" ;;
    esac
}

OS=$(detect_os)

# 1. Ensure uv is installed
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Installing..."
    if command -v pip >/dev/null 2>&1; then
        pip install uv
    elif command -v python3 >/dev/null 2>&1; then
        python3 -m pip install uv
    elif command -v python >/dev/null 2>&1; then
        python -m pip install uv
    else
        echo "Error: Python/pip not found. Please install Python first."
        exit 1
    fi
fi

# 2. Run uv sync
echo "Running uv sync..."
uv sync

# 3. Setup CLI
uv pip install -e .

echo "Setup complete!"

echo "To run the CLI, use the command: unitcraft"
