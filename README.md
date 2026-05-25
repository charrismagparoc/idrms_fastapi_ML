# IDRMS FastAPI Backend

## Setup

```bash
pip install -r requirements.txt
python ml/train_model.py      # train ML models once
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Docs
- Swagger UI: http://localhost:8000/docs
- ReDoc:       http://localhost:8000/redoc

## Endpoints
- `POST   /api/auth/login/`
- `GET    /api/incidents/`
- `GET    /api/alerts/`
- `GET    /api/evacuation-centers/`
- `GET    /api/residents/`
- `GET    /api/resources/`
- `GET    /api/users/`
- `GET    /api/activity-log/`
- `GET    /api/dashboard/summary/`
- `GET    /api/reports/summary/`
- `GET    /api/map/`
- `GET    /api/risk/zones/`
- `POST   /api/predict/resident/`

## Connection
Web App and Mobile App both connect to this backend at:
`http://localhost:8000/api`

For mobile on a physical device or emulator, replace `localhost` with your machine's local IP (e.g. `http://192.168.1.x:8000/api`).
