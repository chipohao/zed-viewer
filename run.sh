#!/bin/bash
# ZED 2i Motion Tracker launcher
# Automatically creates venv and installs dependencies if needed

cd "$(dirname "$0")"

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "Installing dependencies..."
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r requirements.txt
    echo ""
fi

source "$VENV_DIR/bin/activate"
python3 zed_motion_tracker.py
