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

# Input schema
class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict(np.array(data.features).reshape(1, -1))
    return {"prediction": prediction.tolist()[0]}

if __name__ == "__main__":
    import uvicorn
    # This block runs the FastAPI app using uvicorn when this file is executed directly.
    # "serve.app:app" points to the 'app' instance inside the 'serve/app.py' module.
    # 'reload=True' enables auto-reload on code changes (useful for development).
    uvicorn.run("serve.app:app", host="127.0.0.1", port=8000, reload=True)
