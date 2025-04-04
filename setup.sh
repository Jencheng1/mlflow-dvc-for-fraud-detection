#!/bin/bash

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt

# Initialize DVC
dvc init

# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/models
mkdir -p mlruns

# Configure DVC remote storage (optional)
# dvc remote add -d myremote /path/to/remote/storage

# Configure MLflow tracking
export MLFLOW_TRACKING_URI="http://localhost:5000"

# Start MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000 &

# Wait for MLflow server to start
sleep 5

# Run the pipeline
dvc run -n generate_data
dvc run -n train_model

echo "Setup completed! You can now access:"
echo "MLflow UI: http://localhost:5000"
echo "Frontend: http://localhost:8501"
echo "Backend API: http://localhost:8000" 