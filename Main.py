from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
import pickle

app = FastAPI()

# Load your trained model
with open("energy_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define input schema for single prediction
class EnergyInput(BaseModel):
    Global_reactive_power: float
    Voltage: float
    Global_intensity: float
    Sub_metering_1: int
    Sub_metering_2: int
    Sub_metering_3: int
    hour: int

@app.get("/")
def read_root():
    return {
        "message": "👋 Welcome to the Energy Prediction API!",
        "usage": "POST /predict for single input or /upload_csv to batch predict from a file."
    }

# Predict from JSON input
@app.post("/predict")
def predict_energy(data: EnergyInput):
    input_array = np.array([[
        data.Global_reactive_power, data.Voltage, data.Global_intensity,
        data.Sub_metering_1, data.Sub_metering_2, data.Sub_metering_3, data.hour
    ]])
    prediction = model.predict(input_array)[0]
    return {"predicted_global_active_power_kW": round(prediction, 3)}

# Upload CSV for batch predictions
@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        return JSONResponse(content={"error": "Please upload a CSV file"}, status_code=400)

    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        return JSONResponse(content={"error": f"Could not read CSV: {str(e)}"}, status_code=400)

    # Normalize column names (handle lowercase, spacing issues)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Rename normalized columns to match expected format
    column_mapping = {
        "global_reactive_power": "Global_reactive_power",
        "voltage": "Voltage",
        "global_intensity": "Global_intensity",
        "sub_metering_1": "Sub_metering_1",
        "sub_metering_2": "Sub_metering_2",
        "sub_metering_3": "Sub_metering_3",
        "time": "Time"
    }
    df.rename(columns=column_mapping, inplace=True)

    # Extract hour from Time column if needed
    if "hour" not in df.columns and "Time" in df.columns:
        try:
            df["hour"] = pd.to_datetime(df["Time"], format="%I:%M:%S %p").dt.hour
        except Exception as e:
            return JSONResponse(content={"error": f"Could not parse 'Time' column: {str(e)}"}, status_code=400)

    # Final required columns
    required_columns = [
        "Global_reactive_power", "Voltage", "Global_intensity",
        "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "hour"
    ]

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return JSONResponse(
            content={"error": f"Missing required columns: {missing_cols}"},
            status_code=422
        )

    # Prediction
    inputs = df[required_columns].values
    predictions = model.predict(inputs)

    df["Predicted_global_active_power_kW"] = predictions.round(3)
    results = df[["Predicted_global_active_power_kW"]].to_dict(orient="records")

    return {
        "predictions": results,
        "message": f" Processed {len(results)} rows from CSV"
    }
