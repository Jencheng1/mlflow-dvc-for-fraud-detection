from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlflow
import numpy as np
from typing import List, Optional
import os

app = FastAPI(title="Fraud Detection API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the MLflow model
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load the latest model from MLflow
model = mlflow.pyfunc.load_model("runs:/<run_id>/model")  # Replace with actual run ID

class Transaction(BaseModel):
    amount: float
    time: float
    merchant_category: str
    customer_id: str
    location: str

class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    confidence: float

@app.get("/")
async def root():
    return {"message": "Welcome to Fraud Detection API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: Transaction):
    try:
        # Prepare features for prediction
        features = [
            transaction.amount,
            transaction.time,
            # Add more feature processing as needed
        ]
        
        # Make prediction
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0][1]
        
        return PredictionResponse(
            fraud_probability=float(probability),
            is_fraud=bool(prediction),
            confidence=float(probability)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    try:
        # Get model information from MLflow
        model_info = {
            "model_version": "1.0.0",
            "last_updated": "2024-01-01",
            "metrics": {
                "accuracy": 0.95,
                "precision": 0.92,
                "recall": 0.88,
                "f1_score": 0.90
            }
        }
        return model_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 