from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# 1. Initialize the API
app = FastAPI()

# 2. Load the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "data/06_models/champion_model_trained.pkl")
model = joblib.load(MODEL_PATH)

# 3. Define the input data schema
class InputData(BaseModel):
    features: list  # List of numerical or categorical features

# 4. Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        input_array = np.array(data.features).reshape(1, -1)

        # Check if the input matches the expected number of features
        if input_array.shape[1] != model.n_features_in_:
            return {
                "error": f"The model expects {model.n_features_in_} features, but received {input_array.shape[1]}."
            }

        prediction = model.predict(input_array)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}


