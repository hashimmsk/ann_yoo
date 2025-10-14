# ANN-Based Survival Prediction Using RCT Data

## Overview
This repository now focuses on the **datanuri v7a** pipeline—an ANN-based survival prediction system trained on historical single-arm RCT data. Earlier experimental scripts (v1–v6) were removed to streamline the project around the production-ready v7a model and supporting services.

## Repository Layout
- `frontend/`: public-facing pages (`index.html`, `about.html`, `contact.html`) plus static assets
- `backend/`: FastAPI + Flask hybrid backend, retraining script, and requirements files
- `models/`: v7a training script (`ajdANN_v7a.py`) alongside the exported weights under `models/saved_models_v7a/`
- `rct_data/`: historical single-arm trial dataset used for training/evaluation
- `docs/`: project overview (`overview.md`) and calibration notes (`calibration.md`)
- `Prof Notes/`: reference images from collaboration with Dr. Yoo

## Requirements
Make sure you have the following Python libraries installed:

- **tensorflow**
- **numpy**
- **pandas**
- **scikit-learn**
- **matplotlib**
- **prettytable**

Install them using:

```
pip install -r backend/requirements.txt
```

## How to Run
1. **Retrain or confirm the v7a model** (optional):
   ```
   cd backend
   python retrain_v7a.py
   ```

2. **Start the application**:
   ```
   python start_app.py
   ```
   - FastAPI backend: `http://localhost:8000`
   - Flask-served frontend: `http://localhost:5000`

3. **Prediction API**: Send POST requests to `/predict` with patient data to receive mPFS and PFS6 estimates.

## Outputs
- Frontend visualizations and CSV exports under `frontend/`
- Model artifacts stored in `models/saved_models_v7a`
- Historical CSV inputs stored under `rct_data/`