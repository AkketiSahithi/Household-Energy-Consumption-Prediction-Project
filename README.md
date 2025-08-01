# Residential Energy Consumption Prediction Platform

This project is a FastAPI-based web API that predicts residential energy consumption using a trained machine learning model. It supports both single predictions via JSON and batch predictions through CSV file upload.

## Features
- Predicts global active power (kW) based on input features
- JSON input for single predictions
- CSV upload for batch predictions
- Swagger UI for easy testing

## Technologies Used
- FastAPI
- Uvicorn
- Scikit-learn
- Pandas
- NumPy
- Pydantic
- Python-Multipart

## Installation

```bash
git clone https://github.com/AkketiSahithi/Residential-Energy-Consumption-Prediction-Platform.git
cd Residential-Energy-Consumption-Prediction-Platform
pip install -r requirements.txt
````

## Run the API

```bash
uvicorn Main:app --reload
```

Then open your browser and visit:

* Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* Redoc UI (alternative): [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

## API Endpoints

* `GET /` - Welcome message
* `POST /predict` - Single prediction via JSON
* `POST /upload_csv` - Batch predictions via CSV


