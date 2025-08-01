from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

# Load model
with open("energy_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define input schema
class EnergyInput(BaseModel):
    Global_reactive_power: float
    Voltage: float
    Global_intensity: float
    Sub_metering_1: int
    Sub_metering_2: int
    Sub_metering_3: int
    hour: int

@app.post("/predict")
def predict_energy(data: EnergyInput):
    input_array = np.array([[data.Global_reactive_power, data.Voltage, data.Global_intensity,
                             data.Sub_metering_1, data.Sub_metering_2, data.Sub_metering_3, data.hour]])
    prediction = model.predict(input_array)[0]
    return {"predicted_global_active_power_kW": round(prediction, 3)}
@app.get("/")
def read_root():
    return {
        "message": "👋 Welcome to the Energy Prediction API!",
        "usage": "Use POST /predict with your input data or visit /docs for Swagger UI."
    }