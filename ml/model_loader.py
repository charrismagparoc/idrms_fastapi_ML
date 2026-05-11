import os
import json
import datetime
import numpy as np
import pandas as pd
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")


ZONE_BASE = {
    "Zone 1": 25, "Zone 2": 42, "Zone 3": 78,
    "Zone 4": 18, "Zone 5": 82, "Zone 6": 48,
}

VULN_WEIGHTS = {
    "Bedridden": 12, "PWD": 10, "Senior Citizen": 8,
    "Pregnant": 8,  "Infant": 7,
}

EVAC_SCORE = {"Safe": 0, "Evacuated": -15, "Unaccounted": 18}
RAINY_MONTHS = set(range(6, 12))


_models  = {}
_scaler  = None
_columns = []
_loaded  = False


def load_models():
    global _models, _scaler, _columns, _loaded
    try:
        _scaler  = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
            _columns = json.load(f)
        for name, fname in [
            ("decision_tree",       "decision_tree.pkl"),
            ("random_forest",       "random_forest.pkl"),
            ("logistic_regression", "logistic_regression.pkl"),
        ]:
            _models[name] = joblib.load(os.path.join(MODEL_DIR, fname))
        _loaded = True
        print(f"[ML] Models loaded successfully from {MODEL_DIR}")
    except FileNotFoundError as e:
        print(f"[ML] WARNING: Model files not found — {e}")
        print("[ML] Run 'python ml/train_model.py' first.")


def is_models_loaded() -> bool:
    return _loaded


def compute_risk_score(zone, evac_status, household_members,
                       vuln_tags, rainy_season,
                       zone_incident_count=0, weather_risk="None") -> int:
    """Mirror of scoreResident() in useRiskEngine.js."""
    score = ZONE_BASE.get(zone, 30)
    vuln_score = sum(VULN_WEIGHTS.get(t, 5) for t in (vuln_tags or []))
    score += min(vuln_score, 40)
    score += EVAC_SCORE.get(evac_status, 0)
    members = max(int(household_members or 1), 1)
    score += min((members - 1) * 1.8, 12)
    score += min(int(zone_incident_count or 0) * 6, 20)
    if weather_risk == "High":   score += 15
    elif weather_risk == "Medium": score += 7
    if rainy_season:             score += 8
    return int(min(max(round(score), 0), 100))


def _build_feature_row(zone, evac_status, household_members,
                       vuln_tags, rainy_season,
                       zone_incident_count, weather_risk,
                       risk_score) -> pd.DataFrame:
    """Convert raw resident inputs into the encoded feature row the model expects."""
    row = {col: 0 for col in _columns}

    
    zone_col = f"zone_{zone}"
    if zone_col in row:
        row[zone_col] = 1

    
    if evac_status == "Evacuated"   and "evac_Evacuated"   in row: row["evac_Evacuated"]   = 1
    if evac_status == "Unaccounted" and "evac_Unaccounted" in row: row["evac_Unaccounted"] = 1

    
    if weather_risk == "Medium" and "weather_Medium" in row: row["weather_Medium"] = 1
    if weather_risk == "High"   and "weather_High"   in row: row["weather_High"]   = 1


    tag_map = {
        "Bedridden":      "tag_bedridden",
        "PWD":            "tag_pwd",
        "Senior Citizen": "tag_senior_citizen",
        "Pregnant":       "tag_pregnant",
        "Infant":         "tag_infant",
    }
    for tag, col in tag_map.items():
        if tag in (vuln_tags or []) and col in row:
            row[col] = 1

    
    if "household_members"   in row: row["household_members"]   = int(household_members or 1)
    if "rainy_season"        in row: row["rainy_season"]        = int(bool(rainy_season))
    if "zone_incident_count" in row: row["zone_incident_count"] = int(zone_incident_count or 0)
    if "risk_score"          in row: row["risk_score"]          = float(risk_score)

    return pd.DataFrame([row])[_columns]


def predict_resident_risk(zone, evac_status, household_members,
                          vuln_tags, rainy_season=None,
                          zone_incident_count=0, weather_risk="None",
                          risk_score=None, model_name="random_forest") -> dict:
    if not _loaded:
        return {"error": "Models not loaded. Run python ml/train_model.py first."}

    
    if rainy_season is None:
        rainy_season = (datetime.datetime.now().month in RAINY_MONTHS)

   
    if risk_score is None:
        risk_score = compute_risk_score(
            zone, evac_status, household_members,
            vuln_tags, rainy_season, zone_incident_count, weather_risk
        )

    X = _build_feature_row(zone, evac_status, household_members,
                           vuln_tags, rainy_season,
                           zone_incident_count, weather_risk, risk_score)

    model_key = model_name.lower().replace(" ", "_")
    model = _models.get(model_key)
    if model is None:
        return {"error": f"Unknown model '{model_name}'. Choose: decision_tree, random_forest, logistic_regression"}

    if model_key == "logistic_regression":
        X_input = _scaler.transform(X)
    else:
        X_input = X.values

    label       = model.predict(X_input)[0]
    proba_raw   = model.predict_proba(X_input)[0]
    classes     = list(model.classes_)
    proba_dict  = {c: round(float(p), 4) for c, p in zip(classes, proba_raw)}
    confidence  = round(float(max(proba_raw)) * 100, 1)

    return {
        "risk_label":    label,
        "risk_score":    risk_score,
        "confidence_pct": confidence,
        "probabilities": proba_dict,
        "model_used":    model_key,
    }


def get_model_summary() -> dict:
    path = os.path.join(MODEL_DIR, "model_summary.json")
    if not os.path.exists(path):
        return {"error": "model_summary.json not found. Run train_model.py first."}
    with open(path) as f:
        return json.load(f)


def get_feature_importances(top_n=10) -> list:
    path = os.path.join(MODEL_DIR, "feature_importances.csv")
    if not os.path.exists(path):
        return []
    import pandas as pd
    df = pd.read_csv(path).head(top_n)
    return df.to_dict(orient="records")
