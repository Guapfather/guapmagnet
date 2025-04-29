# magnet.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI()

# Hugging Face API
HF_TOKEN = "hf_bEabSREPJUFVfkQq1CcrylmrxWaenzQbC"  # Your real token
HF_MODEL_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"  # Elite Model

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

class TickData(BaseModel):
    bid: float
    ask: float
    spread: float
    volume: float
    time: str

@app.get("/")
async def root():
    return {"status": "GuapMagnet server online"}

@app.post("/predict/")
async def predict(tick: TickData):
    payload = {
        "inputs": {
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": tick.spread,
            "volume": tick.volume,
            "timestamp": tick.time
        }
    }
    response = requests.post(HF_MODEL_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Model Inference Failed")

    output = response.json()

    try:
        signal = output.get("signal", "HOLD")
        confidence = output.get("confidence", 0.5)
    except Exception:
        signal = "HOLD"
        confidence = 0.5

    return {"signal": signal, "confidence": confidence}
