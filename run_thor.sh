#!/bin/bash
# run_thor.sh - Start a THOR Worker Node

# Ensure we are in the project root
cd "$(dirname "$0")"

# Activate venv if present
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Configuration
export API_TOKEN="${API_TOKEN:-supersecret}"
export ODIN_URL="${ODIN_URL:-http://localhost:8080}"
export PYTHONPATH=$PYTHONPATH:.

echo "Starting THOR Worker..."
echo "Connecting to ODIN at $ODIN_URL"

python triforce/thor/main.py
