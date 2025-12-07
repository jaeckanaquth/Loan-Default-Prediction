# Loan Default Prediction - MLOps Project

A complete end-to-end machine learning project for predicting loan defaults, built with production-ready MLOps practices in mind.

## What This Project Does

This project predicts whether a borrower will default on their loan using machine learning. But more importantly, it demonstrates how to build, deploy, and maintain ML models in production - covering everything from data preprocessing to model monitoring.

## Project Structure

```
├── src/
│   ├── data/          # Data preparation and preprocessing
│   ├── features/       # Feature engineering transformers
│   ├── models/         # Model training scripts
│   ├── inference/     # FastAPI inference server
│   └── monitoring/    # Drift detection and monitoring
├── notebooks/          # Jupyter notebook for EDA and experimentation
├── tests/             # Unit tests
└── scripts/           # Utility scripts for model management
```

## Getting Started

### Prerequisites

- Python 3.10+
- Conda (I use an environment called `snow`)
- Docker Desktop (for containerized services)

### Setup

1. **Clone and navigate to the project:**
   ```bash
   cd Loan-Default-Prediction
   conda activate snow
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start supporting services (optional):**
   ```bash
   docker-compose up -d
   ```
   This starts MLflow, MinIO, and Postgres locally.

4. **Prepare your data:**
   - Place your raw dataset in `src/data/raw/`
   - Run the data preparation script:
     ```bash
     python src/data/prepare.py
     ```
   This will create processed data in `data/processed/loan.csv`

## Training a Model

### Option 1: Using the Notebook (Recommended for exploration)

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Open `notebooks/loan_default_pipeline.ipynb`

3. Run the cells - it will:
   - Load and explore the data
   - Preprocess features
   - Train a RandomForest model
   - Evaluate performance
   - Save the model and log to MLflow

### Option 2: Using the Training Script

```bash
python src/models/train.py
```

The script creates a sklearn Pipeline (preprocessing + model) and saves it to `src/models/model_rf.pkl`. It also logs everything to MLflow if you have it running.

## Running the Inference API

The inference server is a FastAPI application that serves predictions via REST API.

### Quick Start

**Windows:**
```bash
start_app.bat
```

**Or manually:**
```bash
conda activate snow
uvicorn src.inference.app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Endpoints

- **GET `/health`** - Health check
- **POST `/predict`** - Make predictions
- **GET `/metrics`** - Prometheus metrics (for monitoring)

### Making Predictions

The API expects a JSON payload with all features used during training:

```python
import requests

sample = {
    "features": {
        "Car_Owned": 0.0,
        "Bike_Owned": 0.0,
        "Active_Loan": 1.0,
        # ... include all features from training
    }
}

response = requests.post('http://localhost:8000/predict', json=sample)
print(response.json())
# {'predictions': [0], 'probabilities': [[0.85, 0.15]]}
```

**Note:** The notebook includes a cell that automatically generates a sample payload with median values for all features, which is handy for testing.

## Key Features

### Input Validation
- Strict Pydantic schemas validate all inputs
- Feature type checking and range validation
- Automatic feature ordering based on model requirements

### Model Pipeline
- Preprocessing (imputation, feature selection) wrapped in sklearn Pipeline
- Ensures consistent transformations between training and inference
- Easy to version and deploy

### Monitoring & Drift Detection
- Prometheus metrics exposed at `/metrics`
- Automated drift detection comparing current data to reference
- Can send alerts via Slack webhooks when drift is detected

### Security
- Optional API key authentication (set `API_KEY_ENABLED=true` and `API_KEY=your-key`)
- Input validation prevents malicious or malformed requests

### CI/CD
- GitHub Actions workflows for:
  - Code quality checks (linting, formatting)
  - Unit tests with coverage
  - Docker image building
  - Model training and promotion workflows

## Docker Deployment

Build and run the API as a container:

```bash
# Build
docker build -t loan-prediction-api:latest .

# Run
docker run -p 8000:8000 \
  -e MODEL_PATH=src/models/model_rf.pkl \
  loan-prediction-api:latest
```

The Dockerfile includes health checks and is optimized for production use.

## Monitoring Drift

To check for data or prediction drift:

```bash
python -m src.monitoring.monitor data/processed/loan.csv
```

This compares recent predictions (from `predictions.log`) against your reference dataset and generates a drift report. You can optionally configure Slack webhooks for alerts.

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Tests cover API endpoints, input validation, and error handling.

## MLflow Integration

If you have MLflow running (via docker-compose or standalone), the training process automatically:
- Logs all experiments
- Tracks metrics and parameters
- Stores model artifacts
- Supports model versioning and promotion

Access the MLflow UI at `http://localhost:5000`

## What Makes This Production-Ready

1. **Modular Code** - Clean separation of concerns (data, features, models, inference)
2. **Testing** - Unit tests for critical components
3. **Validation** - Strict input validation prevents bad data from breaking things
4. **Monitoring** - Metrics and drift detection help catch issues early
5. **Containerization** - Docker makes deployment consistent
6. **CI/CD** - Automated testing and deployment workflows
7. **Documentation** - Clear structure and comments

## Notes

- The model expects all features that were used during training. Missing features will cause an error.
- Prediction logs are written to `predictions.log` for monitoring purposes
- The project uses sklearn Pipelines to ensure preprocessing consistency
- MLflow integration is optional but recommended for experiment tracking
