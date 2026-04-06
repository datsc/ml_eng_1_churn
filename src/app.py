"""app.py — Minimal FastAPI app to learn the basics
First written: 30.3.2026

 

 


First written: 30.3.2026
 
Usage:
    uvicorn src.app:app --reload
"""
 
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
 
app = FastAPI()
 
  
@app.get("/")
def root():
    return {"message": "Telco Churn Predictor API", "docs": "/docs"}
 
 
# Load model once at startup
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)
 
 
class CustomerFeatures(BaseModel):
    SeniorCitizen: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    gender: int
    Partner: int
    Dependents: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
 
 
@app.get("/health")
def health():
    return {"status": "ok"}
 
 
@app.post("/predict")
def predict(customer: CustomerFeatures):
    import pandas as pd
    input_df = pd.DataFrame([customer.dict()])
    prediction = int(model.predict(input_df)[0])
    probability = round(float(model.predict_proba(input_df)[0][1]), 4)
    label = "churn" if prediction == 1 else "no churn"
    return {
        "churn_prediction": prediction,
        "churn_probability": probability,
        "label": label,
    }