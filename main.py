"""
main.py — IDRMS FastAPI Entry Point
=====================================
Run:  python -m uvicorn main:app --reload --host 0.0.0.0 --port 8080
Docs: http://127.0.0.1:8080/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import create_tables
from ml.model_loader import load_models
from routers import (
    auth_router, incidents_router, alerts_router,
    evacuation_centers_router, residents_router,
    resources_router, users_router, activity_log_router,
    dashboard_router, reports_router, map_router,
    risk_router, predict_router,
)

app = FastAPI(
    title="IDRMS API",
    description=(
        "Incident and Disaster Risk Management System — Barangay Kauswagan, CDO.\n\n"
        "**ML Prediction** endpoints are under `/api/predict/` — classifies residents "
        "as LOW / MEDIUM / HIGH flood risk using Decision Tree, Random Forest, "
        "and Logistic Regression trained on IDRMS resident data.\n\n"
        "Run `python ml/train_model.py` once before starting the server."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    create_tables()
    load_models()

PREFIX = "/api"
app.include_router(auth_router,               prefix=PREFIX)
app.include_router(incidents_router,          prefix=PREFIX)
app.include_router(alerts_router,             prefix=PREFIX)
app.include_router(evacuation_centers_router, prefix=PREFIX)
app.include_router(residents_router,          prefix=PREFIX)
app.include_router(resources_router,          prefix=PREFIX)
app.include_router(users_router,              prefix=PREFIX)
app.include_router(activity_log_router,       prefix=PREFIX)
app.include_router(dashboard_router,          prefix=PREFIX)
app.include_router(reports_router,            prefix=PREFIX)
app.include_router(map_router,                prefix=PREFIX)
app.include_router(risk_router,               prefix=PREFIX)
app.include_router(predict_router,            prefix=PREFIX)

@app.get("/")
def root():
    return {
        "message": "IDRMS API is running.",
        "docs":    "http://127.0.0.1:8080/docs",
        "ml":      "POST /api/predict/resident/ to classify flood risk",
    }