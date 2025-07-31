from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import joblib
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

# Load model and scaler
try:
    model = tf.keras.models.load_model("saved_models_v6a/ajdANN_v6a_model.keras")
    scaler = joblib.load("saved_models_v6a/scaler.pkl")
except:
    # Fallback to existing models if v6a not available
    model = tf.keras.models.load_model("saved_models/baseline_model.h5")
    scaler = joblib.load("saved_models/scaler.pkl")

# FastAPI app
app = FastAPI(title="AJDANN Survival Prediction API", 
              description="Predict mPFS and PFS6 using trained neural network model")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input schema based on the model's expected inputs
class InputData(BaseModel):
    age: float
    male: int  # 0 for female, 1 for male
    resec: float  # resection status
    k_score: float  # Karnofsky score
    methyl: float  # methylation status
    pre_trt_history: int  # pre-treatment history

@app.get("/")
def read_root():
    return FileResponse("index.html")

@app.get("/api/status")
def api_status():
    return {"message": "AJDANN Survival Prediction API", "status": "running"}

@app.post("/predict")
def predict(data: InputData):
    # Convert input to numpy array in the correct order
    input_array = np.array([[data.age, data.male, data.resec, data.k_score, data.methyl, data.pre_trt_history]])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled).flatten()

    # Return both mPFS and PFS6 predictions
    return {
        "mPFS": round(float(prediction[0]), 2),
        "PFS6": round(float(prediction[1]), 2) if len(prediction) > 1 else None,
        "input_data": {
            "age": data.age,
            "male": data.male,
            "resec": data.resec,
            "k_score": data.k_score,
            "methyl": data.methyl,
            "pre_trt_history": data.pre_trt_history
        }
    }
