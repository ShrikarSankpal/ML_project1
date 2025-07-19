# House Price Prediction ML Pipeline

This project demonstrates a full machine learning workflow for predicting house prices using the California housing dataset. It includes data processing, model training, selection, serving via API, and automated testing.

---

## Features of the project

- ETL pipeline to fetch and store raw data
- Train multiple models using scikit-learn and log with MLflow
- Automatically select and save the best model
- Serve predictions using FastAPI
- Test the API with pytest

## Project Structure
project1/
│
├── etl/ # Data loading and preprocessing
├── train/ # Model training
├── select_model/ # Model selection using MLflow
├── serve/ # FastAPI app to serve predictions
├── tests/ # Tests using pytest
├── models/ # Saved models (gitignored)
├── data/ # Data storage (gitignored)
├── paths.py # Centralized path management
└── requirements.txt

## Setup

# Create and activate virtual environment
python3.10 -m venv project1-env
source project1-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Tests
PYTHONPATH=. pytest

# Serve app
python3.10 -m serve.app

# Sample Input
{
  "features": [8.32, 41.0, 6.9841, 1.0238, 322.0, 2.5556, 37.88, -122.23]
}
