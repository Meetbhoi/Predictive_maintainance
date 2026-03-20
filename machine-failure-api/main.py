import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ── Load models & preprocessor ──────────────────────────────────────────────
rf_high_recall = joblib.load("models/rf_high_recall.pkl")
rf_balanced    = joblib.load("models/rf_balanced.pkl")
ohe            = joblib.load("models/preprocessor.pkl")

# Thresholds (same as notebook)
THRESHOLD_HIGH_RECALL = 0.2
THRESHOLD_BALANCED    = 0.3

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Machine Failure Prediction API",
    description="Predicts machine failure from sensor readings. Returns RED / ORANGE / GREEN alert.",
    version="1.0.0",
)

# ── Request schema ───────────────────────────────────────────────────────────
class SensorReading(BaseModel):
    Type: str               # "L", "M", or "H"
    air_temperature: float  # [K]
    process_temperature: float  # [K]
    rotational_speed: float     # [rpm]
    torque: float               # [Nm]
    tool_wear: float            # [min]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Type": "L",
                    "air_temperature": 298.1,
                    "process_temperature": 308.6,
                    "rotational_speed": 1551.0,
                    "torque": 42.8,
                    "tool_wear": 0.0,
                }
            ]
        }
    }

# ── Response schema ──────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    alert: str              # "RED", "ORANGE", "GREEN"
    failure_predicted: bool
    confidence_balanced: float
    confidence_high_recall: float
    message: str

# ── Feature engineering (mirrors notebook exactly) ───────────────────────────
def build_features(reading: SensorReading) -> pd.DataFrame:
    raw = pd.DataFrame([{
        "Type":                    reading.Type,
        "Air temperature [K]":     reading.air_temperature,
        "Process temperature [K]": reading.process_temperature,
        "Rotational speed [rpm]":  reading.rotational_speed,
        "Torque [Nm]":             reading.torque,
        "Tool wear [min]":         reading.tool_wear,
    }])

    type_encoded = pd.DataFrame(
        ohe.transform(raw[["Type"]]),
        columns=ohe.get_feature_names_out(["Type"]),
    )
    features = pd.concat([type_encoded, raw.drop(columns="Type")], axis=1)
    return features

# ── Prediction logic ─────────────────────────────────────────────────────────
def get_alert(prob_balanced: float, prob_high_recall: float) -> tuple[str, str]:
    flag_high_recall = prob_high_recall >= THRESHOLD_HIGH_RECALL
    flag_balanced    = prob_balanced    >= THRESHOLD_BALANCED

    if flag_high_recall and flag_balanced:
        return "RED",    "Both models predict failure. Immediate inspection required."
    elif flag_high_recall and not flag_balanced:
        return "ORANGE", "Sensitive model flagged a potential failure. Schedule inspection soon."
    else:
        return "GREEN",  "No failure predicted. Machine operating normally."

# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    features = build_features(reading)

    prob_balanced    = float(rf_balanced.predict_proba(features)[:, 1][0])
    prob_high_recall = float(rf_high_recall.predict_proba(features)[:, 1][0])

    alert, message = get_alert(prob_balanced, prob_high_recall)

    return PredictionResponse(
        alert=alert,
        failure_predicted=alert != "GREEN",
        confidence_balanced=round(prob_balanced, 4),
        confidence_high_recall=round(prob_high_recall, 4),
        message=message,
    )