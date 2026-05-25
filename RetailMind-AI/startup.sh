#!/bin/bash
set -e

echo "=== RetailMind AI Startup ==="
echo "Python: $(python --version)"
echo "Working dir: $(pwd)"
echo "Contents: $(ls /home/site/wwwroot/)"

# Find the actual app directory
if [ -d "/home/site/wwwroot/RetailMind-AI" ]; then
    APP_DIR="/home/site/wwwroot/RetailMind-AI"
    echo "Found app at: $APP_DIR"
elif [ -f "/home/site/wwwroot/backend/main.py" ]; then
    APP_DIR="/home/site/wwwroot"
    echo "Found app at: $APP_DIR"
else
    echo "ERROR: Cannot find app directory"
    ls -la /home/site/wwwroot/
    exit 1
fi

cd $APP_DIR
echo "Changed to: $(pwd)"

# Install dependencies
echo "=== Installing dependencies ==="
pip install --quiet -r requirements.txt

# Set Python path
export PYTHONPATH=$APP_DIR
export APP_ROOT=$APP_DIR

echo "=== Starting gunicorn ==="
gunicorn -w 2 -k uvicorn.workers.UvicornWorker backend.main:app --bind 0.0.0.0:8000 --timeout 120 --access-logfile '-' --error-logfile '-'
