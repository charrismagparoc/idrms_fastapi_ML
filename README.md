# IDRMS
### Integrated Disaster Risk Management System
**Barangay Kauswagan, Cagayan de Oro City**

https://idrms-fastapi-ml.onrender.com (FastAPI)

https://idrms-lab5.vercel.app/  (web)

https://drive.google.com/file/d/1ysWtv-iQ3L218nssrkQOotvtiZmh6_xX/view?fbclid=IwY2xjawSCU6BleHRuA2FlbQIxMABicmlkETFIVTVVUlRWaWNvNzc4MXRDc3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHo6KEE00nQcl2a-lSPMCEX2VNLRJZEuaK_hiBKR9w0oZKXTlx2-YDOcoRZWV_aem_NFPjSZrvFw8q8vTGujXIiQ (mobapp)


---

## 1. Project Description

The Integrated Disaster Risk Management System (IDRMS) is a system developed for the Barangay Disaster Risk Reduction and Management Council (BDRRMC) of Barangay Kauswagan, Cagayan de Oro City. The system provides real-time monitoring, incident reporting, evacuation management, resident tracking, resource management, and machine learning-powered flood risk prediction to support the barangay's disaster preparedness and response operations across geographic zones. It consists of three interconnected components: a React web application, an Expo mobile application, and a FastAPI backend server that serves both the core REST API and the ML prediction layer.

---

## 2. Features

- **Dashboard** — real-time summary cards for total incidents, active alerts, open evacuation centers, resident count, and zone-level risk status
- **Incident management** — report, verify, and resolve incidents (Flood, Fire, Earthquake, Landslide, Storm) across zones with status tracking (Pending → Verified → Active → Responded → Resolved)
- **Alert system** — create and broadcast advisories, warnings, danger alerts, and resolved notices to specific zones with recipient count tracking
- **Evacuation center management** — monitor capacity, current occupancy, available facilities, contact persons, and status (Open / Full / Closed)
- **Resident registry** — manage resident records including zone assignment, household members, vulnerability tags, evacuation status, contact info, and GPS location
- **Resource management** — track equipment, medical supplies, food supply, vehicles, and safety gear with quantity, availability, unit, and status (Available / Low / Depleted)
- **Activity log** — append-only audit trail of all user actions across the system with urgency flags
- **Map view** — geo-visualize incidents, evacuation centers, and residents by zone
- **Reports** — aggregated statistics on incidents by type/severity/zone, alert distribution, evacuation occupancy, and resource levels
- **Risk intelligence** — per-resident and per-zone flood risk scoring using the IDRMS rule-based engine (useRiskEngine.js / useRisk.js)
- **ML flood risk prediction** — classify residents as LOW / MEDIUM / HIGH flood risk using three scikit-learn classifiers trained on a hybrid CCHAIN + synthetic dataset
- **User management** — manage system accounts with role-based access (Admin, Staff, Viewer)
- **Light / dark theme toggle** — system-wide theme support in the mobile app via ThemeContext.js

---

## 3. Technology Stack

### 3.1 Web Application

| Field | Detail |
|-------|--------|
| Project name | idrms |
| Framework | React (Vite, JavaScript) |
| Build tool | Vite — `npm run dev` / `npm run build` |
| Key pages | Dashboard, Incidents, Alerts, Evacuation, Residents, Resources, Activity Log, Map, Reports, Risk Intelligence, Users, Login |
| Risk engine | useRiskEngine.js — `scoreResident()`, `scoreZone()`, `getRiskLabel()` |
| State / context | AppContext.jsx, WeatherContext.jsx |
| API calls | useLocalData.js — fetches all data from FastAPI at `API_BASE_URL` |

### 3.2 Mobile Application

| Field | Detail |
|-------|--------|
| Project name | idrms-mobile |
| Framework | React Native — Expo SDK ~54.0.0 |
| Run commands | `npx expo start` / `npx expo start --tunnel` |
| Key screens | DashboardScreen, IncidentsScreen, AlertsScreen, EvacuationScreen, ResidentsScreen, ResourcesScreen, ActivityLogScreen, MapScreen, ReportsScreen, RiskScreen, UsersScreen, LoginScreen |
| Risk engine | useRisk.js — mirrors useRiskEngine.js logic for resident and zone scoring |
| State / context | AuthContext.js, AppContext.js, ThemeContext.js (light/dark theme) |

### 3.3 Backend — FastAPI

| Field | Detail |
|-------|--------|
| Framework | FastAPI — Python + Uvicorn |
| Run command | `python -m uvicorn main:app --reload --host 0.0.0.0 --port 8080` |
| API prefix | `/api` |
| Swagger docs | http://127.0.0.1:8080/docs |
| Core routers | `/auth`, `/incidents`, `/alerts`, `/evacuation-centers`, `/residents`, `/resources`, `/users`, `/activity-log`, `/dashboard`, `/reports`, `/map`, `/risk` |
| ML router | `/predict` — see Section 3.4 |

### 3.4 ML Prediction Layer

The ML layer is embedded inside the FastAPI backend. Models are loaded into memory at server startup via `ml/model_loader.py`. Run `train_model.py` once before starting the server.

| Field | Detail |
|-------|--------|
| Router | `/api/predict/` |
| Endpoints | `/predict/resident`, `/predict/batch`, `/predict/model-info`, `/predict/health` |
| Preprocessing | StandardScaler — fitted on training set only, saved as `scaler.pkl` |
| Class weighting | `class_weight=balanced` — handles imbalanced HIGH/MEDIUM/LOW distribution |
| Train-test split | 80/20 stratified split (random_state=42) |
| Output labels | LOW (score < 40), MEDIUM (40–69), HIGH (≥ 70) |
| Model files | `ml/model/decision_tree.pkl`, `random_forest.pkl`, `logistic_regression.pkl`, `scaler.pkl` |

### 3.5 Database

| Field | Detail |
|-------|--------|
| Engine | SQLite — file: `idrms.db` |
| Tables | users, incidents, alerts, evacuation_centers, residents, resources, activity_log |

---

## 4. Dataset

### 4.1 Overview

The ML model was trained on a hybrid dataset (`cchain_idrms_dataset.csv`) combining real climate data from Project CCHAIN with synthesized resident survey fields.

| Field | Detail |
|-------|--------|
| Dataset name | cchain_idrms_dataset.csv |
| Total rows | 5,000 |
| Total columns | 24 |
| Kaggle source | https://www.kaggle.com/datasets/thinkdatasci/project-cchain/ |
| HDX source | https://data.humdata.org/dataset/project-cchain |
| CCHAIN table used | climate_atmosphere.csv — filtered to Cagayan de Oro / Kauswagan |
| Climate coverage | 20 years daily barangay-level data — 2003-01-01 to 2022-12-31 |

### 4.2 Real CCHAIN Columns

These columns contain actual recorded climate data for Barangay Kauswagan, Cagayan de Oro.

| Column | Description |
|--------|-------------|
| city | Cagayan de Oro (fixed) |
| barangay | Kauswagan (fixed) |
| date | Daily date (2003-01-01 to 2022-12-31) |
| month | Month number (1–12) |
| precipitation | Rainfall in mm/day |
| temp_mean | Mean temperature in °C |
| temp_min | Minimum temperature in °C |
| temp_max | Maximum temperature in °C |
| heat_index | Perceived temperature (°C) — used to determine weather_risk |
| wind_speed | Wind speed in m/s — used to determine weather_risk |
| relative_humidity | Relative humidity in % — used to determine weather_risk |

### 4.3 Synthesized Resident Fields

These columns were generated algorithmically using the IDRMS resident scoring formula from useRiskEngine.js and useRisk.js, paired with random CCHAIN climate rows.

| Column | Description |
|--------|-------------|
| zone | Randomly assigned Zone 1–6 (Zone 5=1,471 / Zone 3=1,300 / Zone 6=844 / Zone 2=686 / Zone 1=413 / Zone 4=286) |
| evacuation_status | Safe (2,957) / Unaccounted (1,040) / Evacuated (1,003) — randomly assigned |
| household_members | Randomly generated integer — number of people in the household |
| zone_incident_count | Randomly generated — number of past incidents in the resident's zone |
| tag_bedridden | Binary (0/1) — resident is bedridden |
| tag_pwd | Binary (0/1) — resident is a Person with Disability |
| tag_senior_citizen | Binary (0/1) — resident is a Senior Citizen |
| tag_pregnant | Binary (0/1) — resident is pregnant |
| tag_infant | Binary (0/1) — household has an infant |

### 4.4 Derived / Computed Columns

| Column | Description |
|--------|-------------|
| rainy_season | 1 if month is June–November (precipitation ≥ 5 mm/day), else 0 |
| weather_risk | None / Medium / High — derived from heat_index, wind_speed, and relative_humidity |
| risk_score | Integer 0–100 — computed by `scoreResident()` formula from useRiskEngine.js |
| risk_label | HIGH (≥70) / MEDIUM (40–69) / LOW (<40) — target label. Distribution: HIGH=4,019 (80.4%), MEDIUM=873 (17.5%), LOW=108 (2.2%) |

### 4.5 Feature Engineering

The 13 raw input columns are one-hot encoded into 23 model features before training:

| Raw Input | Encoded Features |
|-----------|-----------------|
| zone (6 categories) | zone_Zone2, zone_Zone3, zone_Zone4, zone_Zone5, zone_Zone6 — 5 columns (Zone 1 is the dropped baseline) |
| evacuation_status (3 categories) | evac_Evacuated, evac_Unaccounted — 2 columns (Safe is the dropped baseline) |
| weather_risk (3 categories) | weather_High, weather_Medium — 2 columns (None is the dropped baseline) |
| vulnerability tags (5 binary) | tag_bedridden, tag_pwd, tag_senior_citizen, tag_pregnant, tag_infant — 5 columns (already binary) |
| household_members | 1 column (numeric) |
| rainy_season | 1 column (binary) |
| zone_incident_count | 1 column (numeric) |
| risk_score | 1 column (numeric) |
| precipitation | 1 column (numeric — real CCHAIN) |
| temp_mean | 1 column (numeric — real CCHAIN) |
| heat_index | 1 column (numeric — real CCHAIN) |
| wind_speed | 1 column (numeric — real CCHAIN) |
| relative_humidity | 1 column (numeric — real CCHAIN) |

**Total: 23 model features after one-hot encoding.**

### 4.6 Risk Scoring Formula

The `risk_score` and `risk_label` target values are computed by `scoreResident()` in useRiskEngine.js:

| Rule | Points |
|------|--------|
| Zone base score | Zone 5=82, Zone 3=78, Zone 6=48, Zone 2=42, Zone 1=25, Zone 4=18 |
| Vulnerability tags | Bedridden +12, PWD +10, Senior Citizen +8, Pregnant +8, Infant +7 (capped at +40 total) |
| Evacuation status | Unaccounted +18, Evacuated −15, Safe ±0 |
| Household members | +1.8 pts per extra member above 1 (capped at +12) |
| Zone incident count | +6 pts per incident (capped at +20) |
| Weather risk | High +15, Medium +7, None ±0 |
| Rainy season | +8 pts if current month is June–November |
| Final score range | Clamped to 0–100 |

### 4.7 Model Performance

| Model | Accuracy | F1 Weighted | Precision HIGH | Recall HIGH |
|-------|----------|-------------|----------------|-------------|
| Decision Tree | 100.00% | 100.00% | 100.00% | 100.00% |
| Random Forest | 99.90% | 99.90% | 99.88% | 100.00% |
| Logistic Regression | 98.00% | 98.05% | 100.00% | 98.26% |

> **Note:** High accuracy is expected because `risk_label` is derived directly from `risk_score`, which is itself a training feature. The classifiers are essentially learning the rule engine formula.

---

## 5. System Architecture
<img width="760" height="874" alt="image" src="https://github.com/user-attachments/assets/c5fe43b8-a80f-4e02-a63a-0898e7db4206" />


---

## 6. Installation & Setup

### 6.1 Requirements
- Python 3.11 (exactly — 3.12+ is incompatible)
- Node.js
- Expo CLI

### 6.2 Backend Setup (FastAPI)

```bash
git clone https://github.com/charrismagparoc/idrms_fastapi_ML.git
cd idrms_fastapi_ML
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python ml/train_model.py           # run once to generate .pkl files
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

### 6.3 Web App Setup

```bash
git clone https://github.com/charrismagparoc/idrms-lab5.git
cd idrms-lab5
npm install
npm run dev
```

### 6.4 Mobile App Setup

```bash
git clone https://github.com/charrismagparoc/idrms_MobApp.git
cd idrms_MobApp
npm install
# Update API_BASE_URL in src/data/constants.js to your machine's LAN IP
npx expo start
```

---

## 7. Deployment Links

| Service | URL |
|---------|-----|
| Web app (dev) | http://localhost:5173 |
| Mobile app | Scan the Expo QR code — device must be on the same LAN as the server |
| FastAPI REST | http://192.168.1.21:8080/api/ |
| Swagger UI | http://192.168.1.21:8080/docs |
| CCHAIN dataset (Kaggle) | https://www.kaggle.com/datasets/thinkdatasci/project-cchain/ |
| CCHAIN dataset (HDX) | https://data.humdata.org/dataset/project-cchain |

---

## 8. Test Account

A default admin account is seeded on first startup:

| Field | Value |
|-------|-------|
| Email | admin@kauswagan.gov.ph |
| Password | admin123 |
| Role | Admin |

Additional roles available: Staff, Viewer. All accounts can be managed from the Users screen (Admin only).

> **Note:** Passwords are stored as plain text in this demo build. Hash with bcrypt before any production deployment.

---

## 9. Team Members and Roles

| Name | Role / Modules |
|------|----------------|
| Echavia, Aldren | IncidentsScreen, AlertsScreen, EvacuationScreen, RiskScreen; Risk Engines (web & mobile); dataset research; FastAPI |
| Gorra, Razelle Eve Blanch | ResidentsScreen, ResourcesScreen, MapScreen; dataset extraction; state/context (web & mobile); documentation; FastAPI |
| Guangco, Trisha | ReportsScreen, UsersScreen (web & mobile); documentation; system architecture; dataset extraction; FastAPI |
| Magparoc, Charris | ActivityLogScreen, DashboardScreen, LoginScreen (web & mobile); dataset synthesis; ERD; FastAPI; deployment |

---

## 10. Known Limitations

- **Python 3.12+ incompatible** — pydantic-core fails to build; use Python 3.11 exactly
- **Plain-text passwords** — must be hashed with bcrypt before production use
- **No JWT / token expiry** — authentication is not production-secure
- **Hardcoded API_BASE_URL** — must be updated manually in constants.js per device/network
- **No live weather API** — weather risk is manually entered
- **Synthesized training data** — retraining on real barangay survey data will improve model accuracy
- **risk_label derived from risk_score** — explains near-perfect model accuracy; this is by design
- **No offline support** — mobile app requires active LAN connection to the FastAPI server

---

## 11. ERD
<img width="581" height="1350" alt="image" src="https://github.com/user-attachments/assets/d6bab261-534a-431d-b713-f59e8d546e08" />


---

## 12. Screenshots

Login

<img width="676" height="349" alt="image" src="https://github.com/user-attachments/assets/0056afa9-61d9-4183-8e31-54ba53681ff3" />

Dashboard
<img width="687" height="356" alt="image" src="https://github.com/user-attachments/assets/acb107f6-474f-4eb9-97bb-953ea3d7dbcf" />

<img width="718" height="404" alt="image" src="https://github.com/user-attachments/assets/a940740b-8811-45f8-bb94-388952ad870f" />


Map

<img width="716" height="368" alt="image" src="https://github.com/user-attachments/assets/741b1f90-ea62-4bea-8f50-2204cc26dd05" />

Incident

<img width="716" height="344" alt="image" src="https://github.com/user-attachments/assets/4cc33e12-5399-449e-a579-08953c504368" />


Alert

<img width="772" height="399" alt="image" src="https://github.com/user-attachments/assets/665b9cd0-fbd1-4d9a-996d-d0a2c2b91af1" />

Evacuation

<img width="763" height="395" alt="image" src="https://github.com/user-attachments/assets/0f810d6c-238f-4380-b058-b1d7206806d2" />


Resident

<img width="741" height="387" alt="image" src="https://github.com/user-attachments/assets/2e4cfbac-8f7b-4bc3-b648-3d6dd1625431" />

Resources


<img width="735" height="382" alt="image" src="https://github.com/user-attachments/assets/85b8a0c9-08d0-48f3-9a52-bf062f2fa8e3" />


Report

<img width="747" height="380" alt="image" src="https://github.com/user-attachments/assets/02ec07f6-1605-4193-9834-c1de4376d9c2" />


Risk Intelligence

<img width="755" height="426" alt="image" src="https://github.com/user-attachments/assets/a9d1b678-9277-486b-aa5d-24131dff72a1" />

User

<img width="840" height="432" alt="image" src="https://github.com/user-attachments/assets/10c9f1dc-877f-4114-9c4a-0326980d6aed" />

Activity Log

<img width="848" height="444" alt="image" src="https://github.com/user-attachments/assets/bd807c5b-bc5d-4e19-ba9b-697f6d7a9bef" />




### Web App

- Dashboard — stat cards, zone risk summary, and active alert banner
- Incidents page — list view, Add Incident form, and status timeline
- Alerts page — alert list, add form, and zone filter
- Evacuation page — center list, capacity status, and occupancy tracker
- Residents page — resident list, vulnerability tags, and evacuation status
- Resources page — resource list, availability status, and stock level indicator
- Activity Log page — audit trail with urgency flags and action filter
- Risk Intelligence — zone scores, per-resident risk table, and batch ML prediction
- Map page — geo-pins for incidents, evacuation centers, and residents with zone overlays
- Reports page — aggregated charts, statistics, and export to CSV
- Users page — account management and activity summary per user


Login:	


<img width="233" height="517" alt="image" src="https://github.com/user-attachments/assets/58f0c454-e522-403c-9bdd-78fe320fbf67" />


Dashboard:   


<img width="219" height="517" alt="image" src="https://github.com/user-attachments/assets/0fe77de5-de85-4af7-b82e-bb0d280e0f5c" />



<img width="226" height="517" alt="image" src="https://github.com/user-attachments/assets/e7ed0f32-6f8c-4a48-8c97-3078377cb6b6" />


ALERT:


<img width="219" height="511" alt="image" src="https://github.com/user-attachments/assets/8e099e40-235b-4094-8419-674879a8938d" />



<img width="236" height="524" alt="image" src="https://github.com/user-attachments/assets/0b8bfa4f-8d71-4921-bddd-6eb27ef7724b" />



<img width="239" height="531" alt="image" src="https://github.com/user-attachments/assets/f3ef4626-5172-495d-8b30-1e247a180d3b" />



<img width="228" height="522" alt="image" src="https://github.com/user-attachments/assets/466265e2-9317-41b0-aa66-af622456a8bd" />

Evacuation:

<img width="208" height="516" alt="image" src="https://github.com/user-attachments/assets/54159626-35ec-4293-95f7-c810c2a8ce1a" />



<img width="225" height="500" alt="image" src="https://github.com/user-attachments/assets/05e029ea-3b12-4b3a-8180-b67718ad6777" />


<img width="248" height="507" alt="image" src="https://github.com/user-attachments/assets/c7a95cd4-e241-4a21-ba1e-beceb906074d" />



<img width="239" height="506" alt="image" src="https://github.com/user-attachments/assets/319962ca-0327-44aa-bf7d-b3579d0b554d" />




Resident Management:


<img width="236" height="500" alt="image" src="https://github.com/user-attachments/assets/724fca43-dc3b-4ed4-be47-70256fc27884" />







Resource

<img width="275" height="602" alt="image" src="https://github.com/user-attachments/assets/00fa5281-860c-4c25-95e3-01984a34c15c" />


<img width="276" height="612" alt="image" src="https://github.com/user-attachments/assets/d01abb23-721e-4b3e-ad86-327c29c88e32" />






Incidents



<img width="289" height="606" alt="image" src="https://github.com/user-attachments/assets/27c92cba-2c86-4507-8722-ad6b2fc29661" />



Risk Assessment


<img width="277" height="544" alt="image" src="https://github.com/user-attachments/assets/c12d2e84-3428-40a1-8db9-0fc81e2c80ec" />


<img width="260" height="538" alt="image" src="https://github.com/user-attachments/assets/c406032f-583b-4164-a32f-76e4aaf905d2" />

Reports


<img width="225" height="530" alt="image" src="https://github.com/user-attachments/assets/dacc88a5-33aa-43fa-aea0-f731ef82e18c" />



<img width="250" height="554" alt="image" src="https://github.com/user-attachments/assets/95bea7d0-ec7f-4da7-a181-3e0b590e7f4c" />

Map


<img width="255" height="565" alt="image" src="https://github.com/user-attachments/assets/3a191e64-eb27-4ea6-afed-fcff670e4953" />

Users


<img width="256" height="569" alt="image" src="https://github.com/user-attachments/assets/41363c0e-d8a8-4670-ac21-071c4826a728" />


Activity log


<img width="275" height="569" alt="image" src="https://github.com/user-attachments/assets/84153225-74bb-426e-af6e-0e161ac2c646" />









### Mobile App

- DashboardScreen — summary cards, zone risk summary, and active alert banner
- RiskScreen — zone risk list, ML prediction result, and score breakdown
- MapScreen — resident, incident, and evacuation center pins with zone overlays
- IncidentsScreen — list, add form, status timeline, and status filter
- AlertsScreen — alert list, add form, and zone filter
- EvacuationScreen — center list, capacity status, and occupancy tracker
- ResidentsScreen — resident list, vulnerability tags, and evacuation status
- ResourcesScreen — resource list, availability status, and stock level indicator
- ActivityLogScreen — audit trail with urgency flags and action filter
- ReportsScreen — aggregated charts and statistics
- UsersScreen — account management and role assignment

### FastAPI


<img width="974" height="481" alt="image" src="https://github.com/user-attachments/assets/813f0092-1a93-4619-a375-cab1b073527a" />


<img width="975" height="485" alt="image" src="https://github.com/user-attachments/assets/f529c582-542b-43a4-8a78-95630394195a" />


<img width="975" height="485" alt="image" src="https://github.com/user-attachments/assets/7facd39f-a217-478c-88de-feaa22247738" />



<img width="975" height="480" alt="image" src="https://github.com/user-attachments/assets/ccd8ece5-28d6-49a8-a3a4-d9b9b4f41e78" />



<img width="975" height="497" alt="image" src="https://github.com/user-attachments/assets/94c4638f-690e-430c-bf3a-8dc704718794" />




