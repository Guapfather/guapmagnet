from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("guapmagnet_model.joblib")

# Input format
class PredictInput(BaseModel):
    bid: float
    ask: float

# Start FastAPI app
app = FastAPI()

@app.post("/predict")
def predict(input: PredictInput):
    features = np.array([[input.bid, input.ask]])
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0].max()

    label_map = {1: "BUY", 0: "SELL", 2: "HOLD"}
    signal = label_map.get(prediction, "UNKNOWN")

    return {
        "signal": signal,
        "confidence": round(float(confidence), 4)
    }


    return {"signal": signal, "confidence": confidence}
