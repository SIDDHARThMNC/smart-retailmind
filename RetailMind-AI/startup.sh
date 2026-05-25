#!/bin/bash
set -e

echo "=== RetailMind AI Startup ==="
echo "Python: $(python --version)"
echo "Working dir: $(pwd)"

# Install dependencies if not already installed
pip install --quiet -r /home/site/wwwroot/requirements.txt

echo "=== Starting gunicorn ==="
gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.main:app --bind 0.0.0.0:8000 --timeout 120 --access-logfile '-' --error-logfile '-'
