import os, json, datetime, warnings, random
import numpy as np
import pandas as pd
import joblib

from sklearn.tree          import DecisionTreeClassifier, export_text
from sklearn.ensemble      import RandomForestClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics       import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

MODEL_DIR    = os.path.join(os.path.dirname(__file__), "model")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "cchain_idrms_dataset.csv")
os.makedirs(MODEL_DIR, exist_ok=True)

ZONE_BASE  = {"Zone 1":25,"Zone 2":42,"Zone 3":78,"Zone 4":18,"Zone 5":82,"Zone 6":48}
EVAC_SCORE = {"Safe":0,"Evacuated":-15,"Unaccounted":18}


def main():
    sep = "="*60

    print(f"\n{sep}")
    print("STEP 1 — Load Project CCHAIN Dataset (CDO Barangay Kauswagan)")
    print(sep)
    df = pd.read_csv(DATASET_PATH)
    print(f"  Source: Project CCHAIN climate_atmosphere (Thinking Machines, 2024)")
    print(f"  City:   Cagayan de Oro  |  Barangay: Kauswagan")
    print(f"  Coverage: 20 years daily climate data (2003-2022)")
    print(f"  Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")
    print(f"\n  CCHAIN climate columns included:")
    print(f"    precipitation, temp_mean, temp_min, temp_max,")
    print(f"    heat_index, wind_speed, relative_humidity")
    print(f"\n  Combined IDRMS resident fields:")
    print(f"    zone, evacuation_status, household_members,")
    print(f"    zone_incident_count, vulnerability tags")
    print(f"\n  Label distribution:")
    for label, count in df['risk_label'].value_counts().items():
        print(f"    {label:8}: {count:,} ({count/len(df)*100:.1f}%)")

    df.to_csv(os.path.join(MODEL_DIR,"training_data.csv"), index=False)
    print(f"\n  training_data.csv saved.")

    print(f"\n{sep}")
    print("STEP 2 — Feature Engineering & Encoding")
    print(sep)

    
    zone_d = pd.get_dummies(df['zone'],             prefix='zone')
    evac_d = pd.get_dummies(df['evacuation_status'],prefix='evac')
    wx_d   = pd.get_dummies(df['weather_risk'],     prefix='weather')
    zone_d.drop(columns=['zone_Zone 1'], inplace=True, errors='ignore')
    evac_d.drop(columns=['evac_Safe'],   inplace=True, errors='ignore')
    wx_d.drop(  columns=['weather_None'],inplace=True, errors='ignore')

    tag_cols = ['tag_bedridden','tag_pwd','tag_senior_citizen','tag_pregnant','tag_infant']
    
    climate_cols = ['precipitation','temp_mean','heat_index','wind_speed','relative_humidity']
    num_cols = ['household_members','rainy_season','zone_incident_count','risk_score']

    X = pd.concat([zone_d, evac_d, wx_d, df[tag_cols], df[num_cols], df[climate_cols]], axis=1)
    y = df['risk_label']

    feat_cols = list(X.columns)
    print(f"  Total features: {len(feat_cols)}")
    for i, col in enumerate(feat_cols, 1):
        print(f"    {i:2}. {col}")

    with open(os.path.join(MODEL_DIR,"feature_columns.json"),"w") as f:
        json.dump(feat_cols, f, indent=2)

    print(f"\n{sep}")
    print("STEP 3 — 80/20 Stratified Train-Test Split")
    print(sep)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    for label, count in y_train.value_counts().items():
        print(f"    {label:8}: {count:,} ({count/len(y_train)*100:.1f}%)")

    print(f"\n{sep}")
    print("STEP 4 — StandardScaler")
    print(sep)
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR,"scaler.pkl"))
    print("  Fitted on training set only. Saved.")

    print(f"\n{sep}")
    print("STEP 5 — Train Decision Tree, Random Forest, Logistic Regression")
    print(sep)

    models_cfg = {
        "Decision Tree": DecisionTreeClassifier(
            criterion="entropy", max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=8, max_features="sqrt",
            class_weight="balanced", random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced",
            solver="lbfgs", random_state=42),
    }

    results = {}
    for name, model in models_cfg.items():
        print(f"\n{'─'*55}")
        print(f"  {name}")
        Xi_tr = X_train_s if name=="Logistic Regression" else X_train
        Xi_te = X_test_s  if name=="Logistic Regression" else X_test
        model.fit(Xi_tr, y_train)
        y_pred = model.predict(Xi_te)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test,y_pred,labels=["HIGH"],average="macro",zero_division=0)
        rec  = recall_score(  y_test,y_pred,labels=["HIGH"],average="macro",zero_division=0)
        f1h  = f1_score(      y_test,y_pred,labels=["HIGH"],average="macro",zero_division=0)
        f1w  = f1_score(      y_test,y_pred,average="weighted")

        print(f"  Accuracy:            {acc:.2%}")
        print(f"  Precision (HIGH):    {prec:.2%}")
        print(f"  Recall (HIGH):       {rec:.2%}")
        print(f"  F1-Score (HIGH):     {f1h:.2%}")
        print(f"  F1-Score (weighted): {f1w:.2%}")
        print()
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred, labels=["HIGH","MEDIUM","LOW"])
        print("  Confusion Matrix (HIGH/MEDIUM/LOW):")
        print(cm)

        results[name] = {
            "accuracy":round(float(acc),4),
            "precision_high":round(float(prec),4),
            "recall_high":round(float(rec),4),
            "f1_high":round(float(f1h),4),
            "f1_weighted":round(float(f1w),4),
        }
        fname = name.lower().replace(" ","_")+".pkl"
        joblib.dump(model, os.path.join(MODEL_DIR,fname))
        print(f"\n  Saved → {fname}")


    rf = models_cfg["Random Forest"]
    fi = pd.DataFrame({"feature":feat_cols,"importance":rf.feature_importances_}
                     ).sort_values("importance",ascending=False)
    fi.to_csv(os.path.join(MODEL_DIR,"feature_importances.csv"),index=False)
    print(f"\n{'─'*55}")
    print("  Top 10 Features (Random Forest):")
    print(fi.head(10).to_string(index=False))

    rules = export_text(models_cfg["Decision Tree"], feature_names=feat_cols)
    with open(os.path.join(MODEL_DIR,"decision_tree_rules.txt"),"w") as f:
        f.write(rules)

    mapping = {
        "dataset_source": "Project CCHAIN climate_atmosphere — Thinking Machines Data Science (2024)",
        "dataset_url": "https://www.kaggle.com/datasets/thinkdatasci/project-cchain/",
        "city": "Cagayan de Oro", "barangay": "Kauswagan",
        "coverage": "20 years daily climate (2003-2022) at barangay level",
        "license": "CC BY 4.0",
        "cchain_columns_used": {
            "precipitation":     "→ rainy_season (>=5mm/day = rainy)",
            "temp_mean":         "→ weather_risk (combined with humidity + wind)",
            "heat_index":        "→ weather_risk (high heat index = danger)",
            "wind_speed":        "→ weather_risk (>=5m/s = High, >=3m/s = Medium)",
            "relative_humidity": "→ weather_risk (used with temp for heat stress)",
        },
        "idrms_features": {
            "zone":              "Geographic zone (Zone 1-6) from ZONE_BASE in constants.js",
            "evacuation_status": "Safe/Evacuated/Unaccounted from useRiskEngine.js",
            "household_members": "Number of people in household from survey form",
            "zone_incident_count":"Past flood incidents in zone",
            "vulnerability_tags":"Bedridden/PWD/Senior/Pregnant/Infant from survey",
            "risk_score":        "Computed by IDRMS scoreResident() formula",
            "risk_label":        "HIGH>=70 / MEDIUM 40-69 / LOW<40",
        }
    }
    with open(os.path.join(MODEL_DIR,"feature_mapping.json"),"w") as f:
        json.dump(mapping, f, indent=2)

    summary = {
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_source": "Project CCHAIN climate_atmosphere (Thinking Machines, 2024) — Cagayan de Oro barangay level",
        "dataset_url": "https://www.kaggle.com/datasets/thinkdatasci/project-cchain/",
        "dataset_size": 5000,
        "feature_columns": feat_cols,
        "models": results,
    }
    with open(os.path.join(MODEL_DIR,"model_summary.json"),"w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{sep}")
    print("ALL DONE — files saved to ml/model/")
    print(sep)
    for fname in sorted(os.listdir(MODEL_DIR)):
        sz = os.path.getsize(os.path.join(MODEL_DIR,fname))
        print(f"  {fname:<45} {sz/1024:.1f} KB")
    print(f"\nDataset: Project CCHAIN climate_atmosphere (CDO barangay level)")
    print(f"URL: https://www.kaggle.com/datasets/thinkdatasci/project-cchain/")
    print(f"Next: python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()