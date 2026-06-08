"""
inference.py
Flask REST API for Adult Income Classifier with full Prometheus monitoring.
Endpoints:
  GET  /health    — Health check
  POST /predict   — Run model inference
  GET  /metrics   — Prometheus metrics (served by prometheus_client)
Author: Surya Hanjaya
"""

import os
import time
import json
import logging
import joblib
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify

# Import prometheus exporter and metrics
from prometheus_exporter import (
    start_exporter,
    record_request,
    record_prediction,
    record_error,
    set_model_load_time,
    active_requests,
    latency_seconds,
)

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH  = os.environ.get("MODEL_PATH",  "model/random_forest_model.pkl")
SCALER_PATH = os.environ.get("SCALER_PATH", "model/scaler.pkl")
PORT        = int(os.environ.get("PORT", 5000))
METRICS_PORT= int(os.environ.get("METRICS_PORT", 8002))

NUMERIC_FEATURES = [
    "age", "fnlwgt", "education_num",
    "capital_gain", "capital_loss", "hours_per_week",
]

# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)

# ── Global model/scaler ──────────────────────────────────────
_model  = None
_scaler = None


def load_model_and_scaler():
    """Load model and scaler from disk. Returns load time in seconds."""
    global _model, _scaler
    t_start = time.time()

    try:
        logger.info(f"Loading model from: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded: {type(_model).__name__}")
    except FileNotFoundError:
        logger.warning(f"Model file not found at {MODEL_PATH}. Using dummy model.")
        _model = None

    try:
        logger.info(f"Loading scaler from: {SCALER_PATH}")
        _scaler = joblib.load(SCALER_PATH)
        logger.info("Scaler loaded.")
    except FileNotFoundError:
        logger.warning(f"Scaler file not found at {SCALER_PATH}. Skipping scaler.")
        _scaler = None

    load_time = time.time() - t_start
    set_model_load_time(load_time)
    logger.info(f"Model load time: {load_time:.4f}s")
    return load_time


def preprocess_input(data: dict) -> np.ndarray:
    """
    Preprocess raw JSON input into model-ready feature array.
    Expects a dictionary with feature names as keys.
    """
    df = pd.DataFrame([data])

    # Apply scaler to numeric features if available
    if _scaler is not None:
        existing_numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
        if existing_numeric:
            df[existing_numeric] = _scaler.transform(df[existing_numeric])

    return df.values


# ============================================================
# ENDPOINTS
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    t_start = time.time()
    active_requests.inc()
    try:
        status = {
            "status" : "healthy",
            "model"  : type(_model).__name__ if _model else "not_loaded",
            "scaler" : "loaded" if _scaler else "not_loaded",
        }
        record_request("GET", "/health", 200)
        latency_seconds.labels(endpoint="/health").observe(time.time() - t_start)
        return jsonify(status), 200
    finally:
        active_requests.dec()


@app.route("/predict", methods=["POST"])
def predict():
    """
    Run inference on input features.
    
    Request body (JSON):
    {
        "features": {
            "age": 39,
            "fnlwgt": 77516,
            "education_num": 13,
            "capital_gain": 2174,
            "capital_loss": 0,
            "hours_per_week": 40,
            ... (other one-hot encoded columns)
        }
    }
    
    Response:
    {
        "prediction": 0,
        "prediction_label": "<=50K",
        "probability": [0.82, 0.18],
        "latency_ms": 12.5
    }
    """
    t_start = time.time()
    active_requests.inc()

    try:
        # Parse request body
        body = request.get_json(force=True)
        if not body or "features" not in body:
            record_error("invalid_input")
            record_request("POST", "/predict", 400)
            return jsonify({"error": "Missing 'features' key in request body"}), 400

        features = body["features"]

        # Model check
        if _model is None:
            record_error("model_not_loaded")
            record_request("POST", "/predict", 503)
            return jsonify({"error": "Model not loaded"}), 503

        # Preprocess
        try:
            X = preprocess_input(features)
        except Exception as e:
            record_error("preprocessing_error")
            record_request("POST", "/predict", 422)
            logger.error(f"Preprocessing error: {e}")
            return jsonify({"error": f"Preprocessing failed: {str(e)}"}), 422

        # Inference
        try:
            prediction  = int(_model.predict(X)[0])
            probability = _model.predict_proba(X)[0].tolist()
        except Exception as e:
            record_error("inference_error")
            record_request("POST", "/predict", 500)
            logger.error(f"Inference error: {e}")
            return jsonify({"error": f"Inference failed: {str(e)}"}), 500

        label = ">50K" if prediction == 1 else "<=50K"
        latency_ms = (time.time() - t_start) * 1000

        # Record metrics
        record_prediction(prediction, success=True)
        record_request("POST", "/predict", 200)
        latency_seconds.labels(endpoint="/predict").observe(time.time() - t_start)

        logger.info(
            f"Prediction: {label} | Proba: {probability} | Latency: {latency_ms:.2f}ms"
        )

        return jsonify({
            "prediction"       : prediction,
            "prediction_label" : label,
            "probability"      : probability,
            "latency_ms"       : round(latency_ms, 3),
        }), 200

    except Exception as e:
        record_error("unexpected_error")
        record_request("POST", "/predict", 500)
        logger.exception(f"Unexpected error: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

    finally:
        active_requests.dec()


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Run batch inference on multiple samples.
    
    Request body (JSON):
    {
        "batch": [
            {"features": {...}},
            {"features": {...}}
        ]
    }
    """
    t_start = time.time()
    active_requests.inc()

    try:
        body = request.get_json(force=True)
        if not body or "batch" not in body:
            record_error("invalid_input")
            record_request("POST", "/predict/batch", 400)
            return jsonify({"error": "Missing 'batch' key"}), 400

        batch = body["batch"]
        if not isinstance(batch, list) or len(batch) == 0:
            record_error("invalid_input")
            record_request("POST", "/predict/batch", 400)
            return jsonify({"error": "'batch' must be a non-empty list"}), 400

        if _model is None:
            record_error("model_not_loaded")
            record_request("POST", "/predict/batch", 503)
            return jsonify({"error": "Model not loaded"}), 503

        results = []
        for item in batch:
            features = item.get("features", {})
            X = preprocess_input(features)
            pred   = int(_model.predict(X)[0])
            proba  = _model.predict_proba(X)[0].tolist()
            label  = ">50K" if pred == 1 else "<=50K"
            results.append({
                "prediction"      : pred,
                "prediction_label": label,
                "probability"     : proba,
            })
            record_prediction(pred, success=True)

        latency_ms = (time.time() - t_start) * 1000
        record_request("POST", "/predict/batch", 200)
        latency_seconds.labels(endpoint="/predict/batch").observe(time.time() - t_start)

        return jsonify({
            "results"    : results,
            "count"      : len(results),
            "latency_ms" : round(latency_ms, 3),
        }), 200

    except Exception as e:
        record_error("batch_inference_error")
        record_request("POST", "/predict/batch", 500)
        logger.exception(f"Batch inference error: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        active_requests.dec()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  Adult Income Classifier — Inference API")
    logger.info("=" * 60)

    # Start Prometheus metrics exporter on port 8000
    logger.info(f"Starting Prometheus exporter on port {METRICS_PORT}...")
    start_exporter(port=METRICS_PORT)

    # Load model and scaler
    load_time = load_model_and_scaler()

    logger.info(f"Starting Flask API on port {PORT}...")
    logger.info(f"  /health          → Health check")
    logger.info(f"  /predict         → Single inference (POST)")
    logger.info(f"  /predict/batch   → Batch inference (POST)")
    logger.info(f"  :8000/metrics    → Prometheus metrics (GET)")

    app.run(host="0.0.0.0", port=PORT, debug=False)
