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
├── __init__.py
├── data
│   ├── data_for_serving
│   └── data_for_training
│       ├── features.csv
│       └── target.csv
├── dockerfile
├── etl
│   ├── process_servering_data.py
│   └── process_training_data.py
├── models
│   └── best
│       ├── model
│       │   ├── MLmodel
│       │   ├── conda.yaml
│       │   ├── model.pkl
│       │   ├── python_env.yaml
│       │   └── requirements.txt
│       └── run_id.txt
├── paths.py
├── requirements.txt
├── select_model
│   └── select.py
├── serve
│   ├── __init__.py
│   └── app.py
├── tests
│   └── test_serving.py
├── train
│   └── train.py
└── utils
    └── __init__.py

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
