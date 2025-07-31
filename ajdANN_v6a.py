# ----------------------------------------
# 📆 AJDANN_v6a: Enhanced Predictive Modeling
# ----------------------------------------
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random
import joblib
from keras.src.layers import BatchNormalization
from keras.src.optimizers import AdamW

from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scikeras.wrappers import KerasRegressor

from tensorflow.python.keras import Input
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Adam = tf.keras.optimizers.Adam

# ----------------------------------------
# 🧪 Reproducibility
# ----------------------------------------
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# ----------------------------------------
# 📅 Load Dataset
# ----------------------------------------
csv_file = "C:/Users/hashi/Downloads/dat_hc_simul.csv"
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"File not found: {csv_file}")

data = pd.read_csv(csv_file)
print("✅ Data loaded successfully.")

X = data.drop(columns=['Source', 'Treatment', 'N', 'mPFS', 'PFS6'])
y = data[['mPFS', 'PFS6']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------
# 🧠 Enhanced ANN Model Definition
# ----------------------------------------
def build_enhanced_ann(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(2)  # Output for mPFS and PFS6
    ])
    model.compile(optimizer=AdamW(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# ----------------------------------------
# 🚀 Train Model
# ----------------------------------------
input_dim = X.shape[1]
epochs = 300

model = build_enhanced_ann(input_dim)
model.fit(X_scaled, y, epochs=epochs, batch_size=32, verbose=1)

# ----------------------------------------
# 💾 Save Model & Scaler
# ----------------------------------------
output_dir = "saved_models_v6a"
os.makedirs(output_dir, exist_ok=True)

model.save(os.path.join(output_dir, "ajdANN_v6a_model.keras"))
joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

print("\n✅ Model and scaler saved.")

# ----------------------------------------
# 🧪 Test Predictions
# ----------------------------------------
test_cases = [
    [75, 1, 25, 30, 20, 4],
    [55, 1, 55, 60, 50, 3],
    [35, 0, 95, 95, 95, 1]
]
test_df = pd.DataFrame(test_cases, columns=X.columns)
test_scaled = scaler.transform(test_df)
preds = model.predict(test_scaled)

results = pd.DataFrame({
    'Case': ['Case 1', 'Case 2', 'Case 3'],
    'Predicted_mPFS': preds[:, 0],
    'Predicted_PFS6': preds[:, 1]
})
print("\n📊 Predictions:")
print(results.round(2))