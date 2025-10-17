# AJDANN Frontend Application

## 🎯 Overview
This is the modern web UI for the ADJANN survival prediction platform. It lets clinicians and researchers enter patient parameters, request predictions from the FastAPI backend, and review calibrated survival estimates.

## ✨ Key Features
- **Responsive Tailwind design** with dark-mode support
- **Prediction form** wired to `http://localhost:8000/predict`
- **Debug mode** hitting `/debug-predict`
- **Sample case buttons** for quick testing
- **Inline calibration insights** and survival curve rendering with Chart.js
- **CSV export** and form reset controls

## 🚀 Local Usage
1. Install backend requirements and start the stack:
   ```bash
   cd backend
   pip install -r requirements.txt
   python start_app.py
   ```
   - Flask serves the static files on `http://localhost:5000`
   - FastAPI lives at `http://localhost:8000`

2. Open the UI:
   - `http://localhost:5000/` for the calculator
   - `http://localhost:5000/about.html` for project context
   - `http://localhost:5000/contact.html` for contact form demo

## 🧩 Frontend Stack
- HTML + Tailwind CDN
- Vanilla JavaScript for API calls and chart logic
- Chart.js for survival curve visualization

## 📝 Notes
- The contact form is a client-side demo; no emails are sent.
- Example cases align with the backend calibration scenarios.
- Add additional static assets under `frontend/static/` if needed.

