from fastapi import FastAPI
import joblib
import pandas as pd

from app.schema import CustomerData

app = FastAPI()

# Load model + threshold
pipeline = joblib.load("model/pipeline.pkl")
threshold = joblib.load("model/threshold.pkl")


@app.get("/")
def home():
    return {"message": "Churn Prediction API Running"}


@app.post("/predict")
def predict(data: CustomerData):

    # Convert input → dataframe
    df = pd.DataFrame([data.dict()])

    # Prediction
    proba = pipeline.predict_proba(df)[0][1]
    prediction = int(proba >= threshold)

    return {
        "churn_probability": float(proba),
        "prediction": prediction
    }