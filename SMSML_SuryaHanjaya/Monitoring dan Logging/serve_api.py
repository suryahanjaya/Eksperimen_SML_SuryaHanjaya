"""
serve_api.py
FastAPI Web Server for serving Adult Income Classifier model predictions.
Author: Surya Hanjaya
"""

import os
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

app = FastAPI(
    title="Adult Income Classifier API",
    description="API to serve predictions using the trained RandomForest model logged in local MLflow.",
    version="1.0.0"
)

# Configuration
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "adult-income-baseline"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Global model container
model = None

def get_latest_model():
    """Find the latest run in the baseline experiment and load its model."""
    global model
    try:
        print(f"[INFO] Connecting to MLflow at {MLFLOW_TRACKING_URI}...")
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            print(f"[WARN] Experiment '{EXPERIMENT_NAME}' not found. Ensure the MLflow server is running and modelling.py has been executed.")
            return None
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attribute.start_time DESC"]
        )
        
        if runs.empty:
            print("[WARN] No runs found in the experiment.")
            return None
        
        latest_run_id = runs.iloc[0]["run_id"]
        model_uri = f"runs:/{latest_run_id}/model"
        print(f"[INFO] Loading latest model from run {latest_run_id} ({model_uri})...")
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        print("[SUCCESS] Model loaded successfully!")
        return loaded_model
    except Exception as e:
        print(f"[ERROR] Failed to load model from MLflow: {e}")
        return None


@app.on_event("startup")
def startup_event():
    global model
    model = get_latest_model()


@app.get("/")
def read_root():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "message": "Adult Income Classifier API is running. Use POST /predict to get predictions."
    }


class PredictRequest(BaseModel):
    # Expects a list of dictionaries matching the feature names and values
    data: List[Dict[str, Any]]


@app.post("/predict")
def predict(request: PredictRequest):
    global model
    if model is None:
        model = get_latest_model()
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model is not loaded. Ensure local MLflow server is running and a run is logged."
            )
    
    try:
        # Convert incoming JSON data into a pandas DataFrame
        df = pd.DataFrame(request.data)
        
        # Get predictions (0 or 1)
        predictions = model.predict(df)
        
        # Return predicted labels
        return {
            "predictions": predictions.tolist(),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction error: {str(e)}"
        )


if __name__ == "__main__":
    print("[START] Starting API server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
