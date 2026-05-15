"""
routers/predict_router.py
=========================
ML flood-risk prediction endpoints.
All routes under /api/predict/
"""

import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from ml.model_loader import (
    predict_resident_risk,
    get_model_summary,
    get_feature_importances,
    is_models_loaded,
    compute_risk_score,
    RAINY_MONTHS,
)

router = APIRouter(
    prefix="/predict",
    tags=["ML Predict – RiskIntelligencePage / RiskScreen"],
)

VALID_ZONES   = ["Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5", "Zone 6"]
VALID_EVAC    = ["Safe", "Evacuated", "Unaccounted"]
VALID_WEATHER = ["None", "Medium", "High"]
VALID_MODELS  = ["decision_tree", "random_forest", "logistic_regression"]


# ── Schemas ──────────────────────────────────────────────────────────────────

class ResidentPredictInput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # ← fixes "model_name" warning

    zone:                 str
    evacuation_status:    str
    household_members:    int               = 1
    vulnerability_tags:   List[str]         = []
    rainy_season:         Optional[bool]    = None
    zone_incident_count:  int               = 0
    weather_risk:         str               = "None"
    risk_score:           Optional[float]   = None
    model_name:           str               = "random_forest"
    # optional metadata — returned as-is
    resident_name:        Optional[str]     = None
    resident_id:          Optional[int]     = None


class ResidentPredictOutput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # ← fixes "model_used" warning

    resident_name:    Optional[str]
    resident_id:      Optional[int]
    zone:             str
    evacuation_status: str
    household_members: int
    vulnerability_tags: List[str]
    risk_score:       float
    risk_label:       str
    confidence_pct:   float
    probabilities:    dict
    model_used:       str
    predicted_at:     str


class BatchPredictInput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # ← fixes "model_name" warning

    residents:  List[ResidentPredictInput]
    model_name: str = "random_forest"


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/resident/", response_model=ResidentPredictOutput)
def predict_one(data: ResidentPredictInput):
    """Predict flood risk (LOW / MEDIUM / HIGH) for a single resident."""
    if data.zone not in VALID_ZONES:
        raise HTTPException(400, f"Invalid zone. Choose from: {VALID_ZONES}")
    if data.evacuation_status not in VALID_EVAC:
        raise HTTPException(400, f"Invalid evacuation_status. Choose from: {VALID_EVAC}")
    if data.weather_risk not in VALID_WEATHER:
        raise HTTPException(400, f"Invalid weather_risk. Choose from: {VALID_WEATHER}")
    if data.model_name not in VALID_MODELS:
        raise HTTPException(400, f"Invalid model_name. Choose from: {VALID_MODELS}")

    rainy = data.rainy_season
    if rainy is None:
        rainy = (datetime.datetime.now().month in RAINY_MONTHS)

    result = predict_resident_risk(
        zone=data.zone,
        evac_status=data.evacuation_status,
        household_members=data.household_members,
        vuln_tags=data.vulnerability_tags,
        rainy_season=rainy,
        zone_incident_count=data.zone_incident_count,
        weather_risk=data.weather_risk,
        risk_score=data.risk_score,
        model_name=data.model_name,
    )

    if "error" in result:
        raise HTTPException(503, result["error"])

    return ResidentPredictOutput(
        resident_name=data.resident_name,
        resident_id=data.resident_id,
        zone=data.zone,
        evacuation_status=data.evacuation_status,
        household_members=data.household_members,
        vulnerability_tags=data.vulnerability_tags,
        risk_score=result["risk_score"],
        risk_label=result["risk_label"],
        confidence_pct=result["confidence_pct"],
        probabilities=result["probabilities"],
        model_used=result["model_used"],
        predicted_at=datetime.datetime.now().isoformat(),
    )


@router.post("/batch/")
def predict_batch(data: BatchPredictInput):
    """Predict flood risk for up to 500 residents. Results sorted HIGH → MEDIUM → LOW."""
    if len(data.residents) > 500:
        raise HTTPException(400, "Maximum 500 residents per batch request.")

    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    rainy = (datetime.datetime.now().month in RAINY_MONTHS)
    results = []

    for r in data.residents:
        res = predict_resident_risk(
            zone=r.zone,
            evac_status=r.evacuation_status,
            household_members=r.household_members,
            vuln_tags=r.vulnerability_tags,
            rainy_season=rainy,
            zone_incident_count=r.zone_incident_count,
            weather_risk=r.weather_risk,
            risk_score=r.risk_score,
            model_name=data.model_name,
        )
        results.append({
            "resident_name":     r.resident_name,
            "resident_id":       r.resident_id,
            "zone":              r.zone,
            "evacuation_status": r.evacuation_status,
            "vulnerability_tags": r.vulnerability_tags,
            "risk_score":        res.get("risk_score"),
            "risk_label":        res.get("risk_label", "UNKNOWN"),
            "confidence_pct":    res.get("confidence_pct"),
            "model_used":        res.get("model_used"),
        })

    results.sort(key=lambda x: order.get(x["risk_label"], 3))
    high   = sum(1 for r in results if r["risk_label"] == "HIGH")
    medium = sum(1 for r in results if r["risk_label"] == "MEDIUM")
    low    = sum(1 for r in results if r["risk_label"] == "LOW")

    return {
        "total":        len(results),
        "high_count":   high,
        "medium_count": medium,
        "low_count":    low,
        "model_used":   data.model_name,
        "predicted_at": datetime.datetime.now().isoformat(),
        "results":      results,
    }


@router.get("/model-info/")
def model_info():
    """Returns training accuracy and F1 scores for all 3 models."""
    return get_model_summary()


@router.get("/feature-importance/")
def feature_importance(top_n: int = 10):
    """Returns top N most important features from the Random Forest."""
    fi = get_feature_importances(top_n)
    if not fi:
        raise HTTPException(503, "Feature importance data not found. Run train_model.py first.")
    return {"top_n": top_n, "features": fi}


@router.get("/health/")
def health():
    """Confirms all ML models are loaded and ready."""
    loaded = is_models_loaded()
    return {
        "status":  "ready" if loaded else "not_ready",
        "models":  ["decision_tree", "random_forest", "logistic_regression"] if loaded else [],
        "message": "All models loaded." if loaded else "Run python ml/train_model.py first.",
    }