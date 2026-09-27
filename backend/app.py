from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

from battery_data import (
    available_batteries,
    forecast_response,
    history_response,
    save_uploaded_battery,
)
from explainability import explain_soh_prediction

app = FastAPI(
    title="EV Battery Intelligence API",
    description="Battery SOH, RUL and anomaly prediction API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

soh_model = joblib.load("soh_model.pkl")
rul_model = joblib.load("rul_model.pkl")
anomaly_model = joblib.load("anomaly_model.pkl")
features = joblib.load("feature_columns.pkl")
features = list(features)


class BatteryInput(BaseModel):
    avg_voltage: float
    min_voltage: float
    max_voltage: float
    voltage_range: float
    voltage_std: float
    avg_current: float
    current_std: float
    avg_temperature: float
    max_temperature: float
    temperature_rise: float
    discharge_time: float
    discharge_energy_Wh: float


@app.get("/")
def home():
    return {
        "message": "EV Battery Intelligence API",
        "status": "running"
    }


@app.post("/predict")
def predict(data: BatteryInput):
    input_data = data.model_dump()

    X = pd.DataFrame([input_data], columns=features)

    soh = soh_model.predict(X)[0]
    rul = rul_model.predict(X)[0]
    anomaly = anomaly_model.predict(X)[0]

    return {
        "SOH": round(float(soh), 2),
        "RUL_cycles": max(0, round(float(rul), 0)),
        "anomaly_status": "Anomaly" if anomaly == -1 else "Normal"
    }


@app.get("/batteries")
def batteries():
    return {
        "batteries": available_batteries(),
        "detail": "Upload or add processed NASA battery files if this list is empty.",
    }


@app.get("/battery-history/{battery_id}")
def battery_history(battery_id: str):
    return history_response(battery_id)


@app.get("/battery-forecast/{battery_id}")
def battery_forecast(battery_id: str):
    return forecast_response(battery_id)


@app.post("/explain")
def explain(data: BatteryInput):
    input_data = data.model_dump()
    return explain_soh_prediction(soh_model, features, input_data)


@app.post("/upload-battery")
async def upload_battery(file: UploadFile = File(...)):
    return await save_uploaded_battery(file)
