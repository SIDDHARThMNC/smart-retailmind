#!/bin/bash

echo ">>> Python: $(python --version)"
echo ">>> PWD: $(pwd)"
echo ">>> wwwroot contents:"
ls /home/site/wwwroot/

# Detect app root
if [ -d "/home/site/wwwroot/RetailMind-AI" ]; then
    export APP_DIR="/home/site/wwwroot/RetailMind-AI"
else
    export APP_DIR="/home/site/wwwroot"
fi

echo ">>> APP_DIR: $APP_DIR"
cd $APP_DIR

# Force install all dependencies into system python
echo ">>> Installing dependencies..."
pip install --upgrade pip --quiet
pip install -r $APP_DIR/requirements.txt --quiet

echo ">>> Verifying uvicorn..."
python -c "import uvicorn; print('uvicorn OK:', uvicorn.__version__)"
python -c "import gunicorn; print('gunicorn OK:', gunicorn.__version__)"

export PYTHONPATH=$APP_DIR
export APP_ROOT=$APP_DIR

echo ">>> Starting app..."
exec gunicorn -w 2 -k uvicorn.workers.UvicornWorker backend.main:app \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -
