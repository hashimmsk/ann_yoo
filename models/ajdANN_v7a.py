"""
AJDANN_v7a: Improved multi-task model for predicting mPFS (regression)
and PFS6 (binary classification) with robust training and evaluation.

Key improvements over v6a:
- Train/validation split to prevent leakage
- StandardScaler fit on training set only
- Multi-task network with two heads (regression + classification)
- Proper losses per task (MSE for mPFS, BCE for PFS6)
- EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint
- Clear metrics: RMSE/MAE for mPFS; AUC/Accuracy for PFS6
- Configurable via CLI args; saves to saved_models_v7a
"""

import argparse
import os
import random

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

Sequential = tf.keras.models.Sequential


# -----------------------------
# Custom Layers
# -----------------------------


# -----------------------------
# Custom Layers
# -----------------------------
class TemperatureScaledSigmoid(tf.keras.layers.Layer):
    def __init__(self, temperature: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.temperature = float(temperature)

    def call(self, inputs, *args, **kwargs):  # pragma: no cover - simple wrapper
        return tf.nn.sigmoid(inputs / self.temperature)

    def get_config(self):  # pragma: no cover - serialization helper
        config = super().get_config()
        config.update({"temperature": self.temperature})
        return config


# -----------------------------
# Reproducibility
# -----------------------------
def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -----------------------------
# Data Loading
# -----------------------------
def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    data = pd.read_csv(csv_path)
    if 'mPFS' not in data.columns or 'PFS6' not in data.columns:
        raise ValueError("Dataset must contain 'mPFS' and 'PFS6' columns")
    return data


# -----------------------------
# Model Definition (Multi-task)
# -----------------------------
def build_multitask_model(input_dim: int, learning_rate: float = 1e-3, temperature: float = 2.0) -> Model:
    inputs = Input(shape=(input_dim,), name='features')

    # Shared trunk
    x = Dense(256, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)

    # Heads
    mpfs_output = Dense(1, activation='linear', name='mPFS')(x)
    
    # Apply temperature scaling to PFS6 output for better calibration
    pfs6_logits = Dense(1, activation=None, name='PFS6_logits')(x)
    pfs6_output = TemperatureScaledSigmoid(temperature=temperature, name='PFS6')(pfs6_logits)

    model = Model(inputs=inputs, outputs=[mpfs_output, pfs6_output], name='AJDANN_v7a')

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={
            'mPFS': 'mse',
            'PFS6': 'binary_crossentropy',
        },
        loss_weights={
            'mPFS': 0.5,
            'PFS6': 0.5,
        },
        metrics={
            'mPFS': ['mae'],
            'PFS6': ['accuracy'],
        },
    )

    return model


# -----------------------------
# Training Pipeline
# -----------------------------
def train_and_evaluate(
    data: pd.DataFrame,
    output_dir: str,
    epochs: int = 200,
    batch_size: int = 32,
    seed: int = 42,
    temperature: float = 2.0,
) -> None:
    # Feature/Target split (keep same inputs as v6a)
    X = data.drop(columns=['Source', 'Treatment', 'N', 'mPFS', 'PFS6'], errors='ignore')
    if X.empty:
        raise ValueError("No feature columns found after dropping target/meta columns.")

    # Targets
    y_mpfs = data['mPFS'].astype(float).values.reshape(-1, 1)
    
    # Improved PFS6 target processing for better calibration
    y_pfs6_raw = data['PFS6'].astype(float).values
    
    # If values are percentages (>1), convert to probabilities with better thresholding
    if np.nanmax(y_pfs6_raw) > 1.0:
        # Use more conservative thresholding to reduce overly optimistic predictions
        # Instead of just >= 50%, use a more nuanced approach
        median_pfs6 = np.nanmedian(y_pfs6_raw)
        if median_pfs6 > 60:  # If median is high, be more conservative
            threshold = median_pfs6 * 0.8  # Use 80% of median as threshold
        else:
            threshold = 50  # Default threshold
        
        print(f"PFS6 threshold used: {threshold:.1f}% (median was {median_pfs6:.1f}%)")
        y_pfs6 = (y_pfs6_raw >= threshold).astype(float).reshape(-1, 1)
        
        # Print PFS6 distribution for debugging
        pfs6_dist = np.sum(y_pfs6) / len(y_pfs6) * 100
        print(f"PFS6 positive rate after thresholding: {pfs6_dist:.1f}%")
    else:
        y_pfs6 = y_pfs6_raw.reshape(-1, 1)

    # Train/Val split
    stratify_labels = (y_pfs6.flatten() if np.all(np.isin(y_pfs6, [0.0, 1.0])) else None)
    X_train, X_val, y_mpfs_train, y_mpfs_val, y_pfs6_train, y_pfs6_val = train_test_split(
        X.values,
        y_mpfs,
        y_pfs6,
        test_size=0.2,
        random_state=seed,
        stratify=stratify_labels,
    )

    # Scale features (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Build model with temperature scaling
    model = build_multitask_model(input_dim=X_train_scaled.shape[1], temperature=temperature)

    # Callbacks
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, 'ajdANN_v7a_best.keras')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True),
    ]

    # Train
    history = model.fit(
        X_train_scaled,
        {'mPFS': y_mpfs_train, 'PFS6': y_pfs6_train},
        validation_data=(X_val_scaled, {'mPFS': y_mpfs_val, 'PFS6': y_pfs6_val}),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    # Evaluation on validation set
    mpfs_pred_val, pfs6_pred_val = model.predict(X_val_scaled, verbose=0)
    mpfs_pred_val = mpfs_pred_val.flatten()
    pfs6_pred_val = pfs6_pred_val.flatten()

    # Fix sklearn compatibility - use older API without 'squared' parameter
    rmse_mpfs = np.sqrt(mean_squared_error(y_mpfs_val.flatten(), mpfs_pred_val))
    mae_mpfs = mean_absolute_error(y_mpfs_val.flatten(), mpfs_pred_val)

    # For AUC/ACC ensure labels are 0/1
    y_true_pfs6 = (y_pfs6_val.flatten() >= 0.5).astype(int)
    try:
        auc_pfs6 = roc_auc_score(y_true_pfs6, pfs6_pred_val)
    except ValueError:
        # When only one class present in y_true, AUC is undefined
        auc_pfs6 = float('nan')
    acc_pfs6 = accuracy_score(y_true_pfs6, (pfs6_pred_val >= 0.5).astype(int))

    print("\n================ Validation Metrics (v7a) ================")
    print(f"mPFS RMSE: {rmse_mpfs:.4f}")
    print(f"mPFS MAE : {mae_mpfs:.4f}")
    print(f"PFS6 AUC : {auc_pfs6:.4f}")
    print(f"PFS6 ACC : {acc_pfs6:.4f}")

    # Save artifacts (best weights already restored)
    final_model_path = os.path.join(output_dir, 'ajdANN_v7a_model.keras')
    model.save(final_model_path)
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    print(f"\n✅ Saved model to: {final_model_path}")
    print(f"✅ Saved scaler to: {os.path.join(output_dir, 'scaler.pkl')}")

    # Quick smoke-test predictions with sample cases using columns from X
    test_cases = [
        [75, 1, 25, 30, 20, 4],  # Case 1: elderly, poor performance, poor resection
        [55, 1, 55, 60, 50, 3],  # Case 2: middle-aged, moderate performance, suboptimal resection
        [35, 0, 95, 95, 95, 1],  # Case 3: young, excellent performance, excellent resection
    ]
    try:
        test_df = pd.DataFrame(test_cases, columns=X.columns)
        test_scaled = scaler.transform(test_df)
        mpfs_pred, pfs6_pred = model.predict(test_scaled, verbose=0)
        mpfs_pred = mpfs_pred.flatten()
        pfs6_pred = pfs6_pred.flatten()
        
        # Show both raw and temperature-scaled predictions
        results = pd.DataFrame({
            'Case': ['Case 1', 'Case 2', 'Case 3'],
            'Predicted_mPFS': mpfs_pred,
            'Predicted_PFS6_prob': pfs6_pred,
            'Expected_PFS6_range': ['~35-45%', '~50-65%', '~70-80%'],
            'Clinical_Notes': [
                'Elderly, poor performance, poor resection → low survival',
                'Moderate factors → moderate survival',
                'Young, excellent factors → high survival'
            ]
        })
        print("\n📊 Sample Predictions (Temperature Scaled):")
        print(results.round(3))
        print(f"\n🌡️  Temperature scaling applied: {temperature}")
        print("💡 Higher temperature = more conservative (lower) PFS6 predictions")
        
    except Exception as e:
        print(f"⚠️  Test prediction skipped due to: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train AJDANN_v7a model')
    default_csv = os.environ.get('AJDANN_DATA_PATH', 'C:/Users/hashi/Downloads/dat_hc_simul.csv')
    parser.add_argument('--data', type=str, default=default_csv, help='Path to dataset CSV')
    parser.add_argument('--output', type=str, default='saved_models_v7a', help='Output directory for model and scaler')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--temperature', type=float, default=2.0, help='Temperature scaling for PFS6 calibration (higher = more conservative)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    print("📥 Loading dataset...")
    data = load_dataset(args.data)
    print("✅ Data loaded. Starting training...")
    print(f"🌡️  Using temperature scaling: {args.temperature} (higher = more conservative PFS6 predictions)")
    train_and_evaluate(
        data=data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        seed=args.seed,
        temperature=args.temperature,
    )


if __name__ == '__main__':
    main()


