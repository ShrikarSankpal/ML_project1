from pathlib import Path

# Base directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
TRAINING_DATA_DIR = DATA_DIR / "data_for_training"
SERVING_DATA_DIR = DATA_DIR / "data_for_serving"

# Model directory
BEST_MODEL_DIR = PROJECT_ROOT / "models" / "best"

# Experiment name
EXPERIMENT_NAME = "house_price_prediction"
