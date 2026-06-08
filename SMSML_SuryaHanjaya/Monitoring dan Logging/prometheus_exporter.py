"""
prometheus_exporter.py
Custom Prometheus metrics exporter for Adult Income Classifier inference system.
Exposes 10+ metrics on port 8000 at /metrics endpoint.
Author: Surya Hanjaya
"""

import time
import threading
import psutil
import os

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server,
    REGISTRY,
)

# ============================================================
# METRIC DEFINITIONS (10 metrics)
# ============================================================

# 1. Total HTTP requests received
request_total = Counter(
    "request_total",
    "Total number of HTTP requests received by the inference API",
    ["method", "endpoint", "status_code"],
)

# 2. Total predictions made
prediction_total = Counter(
    "prediction_total",
    "Total number of predictions made by the model",
    ["predicted_class"],
)

# 3. Total errors encountered
error_total = Counter(
    "error_total",
    "Total number of errors encountered during inference",
    ["error_type"],
)

# 4. Request latency histogram
latency_seconds = Histogram(
    "latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

# 5. CPU usage
cpu_usage_percent = Gauge(
    "cpu_usage_percent",
    "Current CPU usage percentage of the inference server process",
)

# 6. Memory usage
memory_usage_percent = Gauge(
    "memory_usage_percent",
    "Current memory (RAM) usage percentage",
)

# 7. Disk usage
disk_usage_percent = Gauge(
    "disk_usage_percent",
    "Current disk usage percentage of the root filesystem",
)

# 8. Model load time
model_load_time_seconds = Gauge(
    "model_load_time_seconds",
    "Time taken (seconds) to load the ML model at startup",
)

# 9. Prediction success rate (sliding window — updated periodically)
prediction_success_rate = Gauge(
    "prediction_success_rate",
    "Ratio of successful predictions to total requests (0.0 – 1.0)",
)

# 10. Active (in-flight) requests
active_requests = Gauge(
    "active_requests",
    "Number of requests currently being processed",
)

# ============================================================
# SYSTEM METRICS COLLECTOR (background thread)
# ============================================================

_total_requests   = 0
_success_requests = 0
_lock = threading.Lock()


def update_system_metrics(interval: float = 5.0) -> None:
    """Background thread: update CPU, memory, disk metrics every `interval` seconds."""
    while True:
        try:
            # CPU usage (non-blocking, uses 1-second interval internally)
            cpu_usage_percent.set(psutil.cpu_percent(interval=1))

            # Memory usage
            mem = psutil.virtual_memory()
            memory_usage_percent.set(mem.percent)

            # Disk usage (root partition)
            disk = psutil.disk_usage("/")
            disk_usage_percent.set(disk.percent)

            # Prediction success rate
            with _lock:
                if _total_requests > 0:
                    rate = _success_requests / _total_requests
                else:
                    rate = 1.0
            prediction_success_rate.set(rate)

        except Exception as e:
            print(f"[WARN] Failed to update system metrics: {e}")

        time.sleep(interval)


def record_request(method: str, endpoint: str, status_code: int) -> None:
    """Increment request_total counter."""
    global _total_requests
    request_total.labels(
        method=method,
        endpoint=endpoint,
        status_code=str(status_code),
    ).inc()
    with _lock:
        _total_requests += 1


def record_prediction(predicted_class: int, success: bool = True) -> None:
    """Increment prediction_total counter and update success tracking."""
    global _success_requests
    prediction_total.labels(
        predicted_class=str(predicted_class)
    ).inc()
    if success:
        with _lock:
            _success_requests += 1


def record_error(error_type: str) -> None:
    """Increment error_total counter."""
    error_total.labels(error_type=error_type).inc()


def set_model_load_time(seconds: float) -> None:
    """Set model load time gauge."""
    model_load_time_seconds.set(seconds)


def start_exporter(port: int = 8000) -> None:
    """
    Start the Prometheus HTTP metrics server.
    Must be called once before starting the inference API.
    """
    # Start background system metrics collector
    t = threading.Thread(target=update_system_metrics, args=(5.0,), daemon=True)
    t.start()
    print(f"[INFO] System metrics collector started (interval=5s)")

    # Start Prometheus HTTP server
    start_http_server(port)
    print(f"[INFO] Prometheus exporter running at http://localhost:{port}/metrics")


# ── Standalone mode ──────────────────────────────────────────
if __name__ == "__main__":
    print("[INFO] Starting Prometheus exporter in standalone mode...")
    start_exporter(port=8000)
    print("[INFO] Exporter ready. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[INFO] Exporter stopped.")
