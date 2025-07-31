#!/usr/bin/env python3
"""
AJDANN Application Startup Script
This script trains the model and starts the FastAPI server
"""

import os
import sys
import subprocess
import time

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import tensorflow
        import numpy
        import pandas
        import sklearn
        import joblib
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with one of these commands:")
        print("  pip install -r requirements.txt")
        print("  OR for minimal versions: pip install -r requirements_minimal.txt")
        print("  OR install manually: pip install fastapi uvicorn tensorflow numpy pandas scikit-learn joblib matplotlib pydantic")
        return False

def train_model():
    """Train the AJDANN model"""
    print("🚀 Training AJDANN model...")
    try:
        # Import and run the training script
        import ajdANN_v6a
        print("✅ Model training completed successfully")
        return True
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        print("Note: Make sure the dataset file exists at the specified path")
        return False

def start_server():
    """Start the FastAPI server"""
    print("🌐 Starting FastAPI server...")
    try:
        import uvicorn
        uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")

def main():
    print("=" * 60)
    print("🧠 AJDANN - Advanced Neural Network for Survival Prediction")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Train model (only if saved_models_v6a directory doesn't exist)
    if not os.path.exists("saved_models_v6a"):
        if not train_model():
            print("⚠️  Using existing models from saved_models directory")
    else:
        print("✅ Model already trained, skipping training step")
    
    # Start server
    print("\n" + "=" * 60)
    print("🎯 Application is ready!")
    print("📱 Open your browser and go to: http://localhost:8000")
    print("🌐 API documentation: http://localhost:8000/docs")
    print("=" * 60)
    
    start_server()

if __name__ == "__main__":
    main() 