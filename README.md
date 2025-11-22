# Dr Yoo Research – datanuri Survival Prediction Platform

## Overview
This repository packages the datanuri (ADJANN v7a) research stack used by Dr. Yoo’s team to model mean progression-free survival (mPFS) and six-month PFS (PFS6). The project combines a multitask neural network, a FastAPI prediction service with fallbacks, a modern browser frontend, and utilities for training and reporting.

## Key Capabilities
- Multitask ANN (`models/ajdANN_v7a.py`) that fits mPFS regression and calibrated PFS6 classification in one pass.
- FastAPI backend (`backend/api.py`) that loads v7a artifacts, applies clinical calibration heuristics, and falls back to legacy or heuristic estimates when needed.
- Flask wrapper (`backend/start_app.py`) that serves the static frontend along with the API for a cohesive local experience.
- Responsive frontend (`frontend/index.html`) built with Tailwind and Chart.js for clinician-friendly inputs and visualizations.
- Offline report generators in `reports/` (`v1/generate_report.py` for the technical narrative and `v3/generate_report.py` for a plain-language summary).

## Repository Layout
```text
.
├── backend/                 # API, startup script, dataset, and dependency pins
│   ├── api.py               # FastAPI app, model loading, calibration logic
│   ├── start_app.py         # Boots FastAPI + Flask, checks dependencies/artifacts
│   ├── retrain_v7a.py       # One-shot retraining helper with backups
│   ├── dat_hc_simul.csv     # Curated clinical dataset (required for training)
│   └── requirements*.txt    # Full/minimal dependency sets
├── frontend/                # Static web client served by Flask or opened directly
│   ├── index.html           # Primary UI (Tailwind, fetches `/predict`)
│   ├── about.html, contact.html
│   └── logo.png             # Branding asset
├── models/                  # Model code and saved artifacts
│   ├── ajdANN_v7a.py        # Training pipeline + model definition
│   └── saved_models_v7a/    # Default trained weights and scaler
├── reports/
│   ├── v1/
│   │   ├── generate_report.py  # Technical methods/results script
│   │   └── report.txt          # Sample export (technical language)
│   └── v3/
│       ├── generate_report.py  # Plain-language story generator
│       └── report.txt          # Sample export (everyday language)
├── requirements.txt         # Delegates to backend requirements
└── rct_data/                # (If present) supplemental trial data
```

## Prerequisites
- Python 3.10 or newer (TensorFlow 2.20 CPU build is pinned).
- Git, virtualenv, and a modern browser for the frontend.
- For GPU acceleration, install the appropriate TensorFlow build separately.

## Setup
```powershell
git clone <repo-url>
cd Dr\ Yoo\ Research
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
On Unix shells replace the activation command with `source .venv/bin/activate`.

## Running the Application
- **Full stack (recommended):**
  ```powershell
  cd backend
  python start_app.py
  ```
  - Flask serves the static UI at `http://localhost:5000`.
  - FastAPI serves JSON endpoints and interactive docs at `http://localhost:8000/docs`.
  - The script auto-checks dependencies, ensures v7a artifacts exist, and retrains if necessary.
- **API only (development):**
  ```powershell
  uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
  ```
  Serve `frontend/index.html` with any static server (or open it directly) while the API runs.

## Data & Artifacts
- Training requires `backend/dat_hc_simul.csv`. Provide an identical schema if replacing the dataset.
- Saved models live in `models/saved_models_v7a/` (default) and backups e.g. `saved_models_v7a_backup/`.
- The API automatically retrains v7a when artifacts are missing or corrupted; manual retraining is available.

## Training & Evaluation
- **One-shot retraining with automatic backup:**
  ```powershell
  cd backend
  python retrain_v7a.py
  ```
- **Custom training run:**
  ```powershell
  python -m models.ajdANN_v7a --data path/to/dat_hc_simul.csv --output models/saved_models_v7a --epochs 200 --batch 32 --temperature 2.0
  ```
  Key CLI flags mirror the defaults used in production. The script prints validation RMSE/MAE/AUC/ACC and stores model/scaler artifacts.

## API Reference
- `GET /api/status` – Health check with active model version.
- `POST /predict` – Primary endpoint. Request body:
  ```json
  {
    "age": 55,
    "male": 1,
    "resec": 55,
    "k_score": 60,
    "methyl": 50,
    "pre_trt_history": 3
  }
  ```
  Response includes mPFS (months), calibrated PFS6 (%), raw probability, and metadata.
- `POST /debug-predict` – Returns raw vs calibrated outputs plus calibration factors for QA.
- Static pages (`/`, `/about`, `/contact`) are served by Flask when `start_app.py` is used.

Interactive docs are available at `http://localhost:8000/docs` when running via Uvicorn/Start App.

## Frontend Usage
- Open `http://localhost:5000` after launching `start_app.py`, or open `frontend/index.html` directly when the API is running.
- Use sample case buttons for quick validation; export predictions as CSV with the built-in tool.
- Dark/light theme is persisted per browser via local storage.

## Reporting Workflow
- Generate a refreshed narrative with:
```powershell
python reports/v1/generate_report.py --output reports/v1/report.txt
python reports/v3/generate_report.py --output reports/v3/report.txt
```
- Each script prints its narrative and writes the same content to the chosen file (defaults shown above).

## Troubleshooting
- **Missing dependencies** – Rerun `pip install -r requirements.txt`. `start_app.py` surfaces missing packages.
- **Model artifacts missing** – The backend will try to retrain automatically; ensure the dataset path is valid.
- **TensorFlow errors on Windows** – Confirm that the installed TensorFlow wheel matches your Python version; upgrade pip before installing.
- **CORS issues** – The API allows `*` origins for development. If serving externally, tighten the CORS configuration in `backend/api.py`.

## Development Tips
- Keep Python scripts formatted with `black` and type-checked with `mypy` (not enforced yet, but recommended).
- Consider adding automated tests for calibration logic and API responses (`pytest` works well with FastAPI).
- Large model artifacts should remain in Git LFS or releases if size becomes an issue.

## Status & Next Steps
- ✅ Core training, inference, UI, and reporting flows are implemented.
- 🔜 Suggested improvements include automated CI, packaging model artifacts, and adding monitoring for calibration drifts.

## License
No license has been declared. Add a `LICENSE` file before publishing the project publicly.

---
_For research support or clinical validation questions, coordinate with Dr. Yoo’s research group before deploying outside controlled environments._

