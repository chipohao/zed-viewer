#!/bin/bash
# ZED 2i Motion Tracker launcher
# Automatically creates venv and installs dependencies if needed
# Usage: ./run.sh [options]
#   --no-osc        Disable OSC output
#   --no-syphon     Disable Syphon output
#   --no-vcam       Disable virtual webcam output
#   --osc-port 8000 Change OSC port
#   --camera 0      Change camera index

cd "$(dirname "$0")"

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Always check dependencies are up to date
"$VENV_DIR/bin/pip" install -q -r requirements.txt

"$VENV_DIR/bin/python3" zed_motion_tracker.py --no-syphon "$@"
