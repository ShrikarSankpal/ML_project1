# serve/app.py

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os


from paths import BEST_MODEL_DIR

MODEL_PATH = os.path.join(BEST_MODEL_DIR, "model", "model.pkl")

# Load model
model = joblib.load(MODEL_PATH)

app = FastAPI()

def make_prediction(model, features: list[float]) -> float:
    return model.predict([features])[0]

# Input schema
class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(input_data: InputData):
    prediction = make_prediction(model, input_data.features)
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    # This block runs the FastAPI app using uvicorn when this file is executed directly.
    # "serve.app:app" points to the 'app' instance inside the 'serve/app.py' module.
    # 'reload=True' enables auto-reload on code changes (useful for development).
    uvicorn.run("serve.app:app", host="127.0.0.1", port=8000, reload=True)
