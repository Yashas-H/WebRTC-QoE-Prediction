# src/server.py

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import xgboost as xgb
import joblib


# ============================================================
# Load Model + Scaler
# ============================================================

booster = xgb.Booster()
booster.load_model("models/xgb_session_model.json")

scaler = joblib.load("models/xgb_session_scaler.pkl")

# Must match your training script
FEATURE_COLS = [
    "sendBitrateKbps", "recvBitrateKbps", "fpsRecv",
    "jitterMs", "rttMs", "packetsLost", "packetsReceived",
    "width", "height", "pixelArea",
    "framesDecoded", "framesDropped", "jitterBufferMs",
    "bitrate_per_pixel",
]


# ============================================================
# FastAPI Setup
# ============================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QoERequest(BaseModel):
    features: dict   # dictionary of WebRTC stats


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post("/predict")
def predict(payload: QoERequest):
    feats = payload.features

    # Convert dict → ordered feature vector
    x = np.array([[feats[col] for col in FEATURE_COLS]], dtype=np.float32)

    # Scale features
    xs = scaler.transform(x)

    # Predict
    dmat = xgb.DMatrix(xs)
    mos = float(booster.predict(dmat)[0])

    return {"mos": mos}
