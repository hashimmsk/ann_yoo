#!/usr/bin/env python3
"""
Retrain AJDANN v7a with improved PFS6 calibration
This script retrains the model using temperature scaling and better target processing
"""

import os
import sys
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"
V7_MODEL_DIR = MODEL_DIR / "saved_models_v7a"
BACKUP_DIR = MODEL_DIR / "saved_models_v7a_backup"

def backup_existing_model():
    """Backup existing v7a model if it exists"""
    if V7_MODEL_DIR.exists():
        if BACKUP_DIR.exists():
            shutil.rmtree(BACKUP_DIR)
        shutil.move(V7_MODEL_DIR, BACKUP_DIR)
        print(f"✅ Backed up existing model to: {BACKUP_DIR}")
        return True
    return False

def retrain_with_calibration():
    """Retrain the v7a model with improved calibration"""
    print("🚀 Retraining AJDANN v7a with improved PFS6 calibration...")
    
    try:
        # Import the training module
        sys.path.insert(0, str(ROOT_DIR / "models"))
        import ajdANN_v7a
        
        # Actually run the training by calling the main function
        print("🔥 Starting actual model training...")
        ajdANN_v7a.main()
        sys.path.pop(0)
        
        print("✅ Model retraining completed successfully!")
        print("🌡️  Temperature scaling applied for better PFS6 calibration")
        return True
        
    except Exception as e:
        print(f"❌ Model retraining failed: {e}")
        return False

def main():
    print("=" * 60)
    print("🔄 AJDANN v7a Retraining with PFS6 Calibration")
    print("=" * 60)
    
    # Check if we need to backup
    had_existing = backup_existing_model()
    
    # Retrain the model
    if retrain_with_calibration():
        print("\n" + "=" * 60)
        print("🎯 Retraining completed successfully!")
        print("📱 The new model should now produce more realistic PFS6 values")
        print("🌐 You can now restart your application to use the new model")
        print("=" * 60)
        
        if had_existing:
            print("\n💡 Note: Your old model was backed up to 'models/saved_models_v7a_backup'")
            print("   You can restore it if needed by renaming the backup folder")
    else:
        print("\n" + "=" * 60)
        print("❌ Retraining failed!")
        if had_existing:
            print("💡 Your old model is still available in 'models/saved_models_v7a_backup'")
        print("=" * 60)

if __name__ == "__main__":
    main()
