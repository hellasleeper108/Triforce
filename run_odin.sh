#!/bin/bash
# run_odin.sh - Start the ODIN Controller

# Ensure we are in the project root
cd "$(dirname "$0")"

# Activate venv if present
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Load .env if present
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Set default token if not provided
export API_TOKEN="${API_TOKEN:-supersecret}"
export PYTHONPATH=$PYTHONPATH:.

echo "Starting ODIN Controller..."
echo "Access Dashboard at http://localhost:8080/dashboard"
echo "API Token: $API_TOKEN"

python triforce/odin/main.py
