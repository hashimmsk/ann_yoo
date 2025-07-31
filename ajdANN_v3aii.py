# ----------------------------------------
# 📆 AJDANN_v3aii: Test i = 1000, 2000 with Fixed e=100, n=200
# ----------------------------------------
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random

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
# 📅 Load and Prepare Dataset
# ----------------------------------------
csv_file = "C:/Users/hashi/Downloads/dat_hc_simul.csv"

if not os.path.exists(csv_file):
    raise FileNotFoundError(f"Error: The file '{csv_file}' was not found. Please check the path and try again.")

data = pd.read_csv(csv_file)
print("✅ Historical RCT data loaded successfully!")

# Drop unnecessary columns and prepare input/output
data = data.drop(columns=['Source', 'Treatment'])
X = data.drop(columns=['mPFS'])
y = data['mPFS']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------
# 🧐 Build ANN
# ----------------------------------------
def build_ann(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# ----------------------------------------
# 🔢 RMSE Calculation
# ----------------------------------------
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ----------------------------------------
# 🤖 Model Execution (Fixed e=100, n=200, varying i)
# ----------------------------------------
def run_models_i(X_data, y_data, n_estimators=100):
    input_dim = X_data.shape[1]
    results = {}

    # Bagging ANN (i = n_estimators)
    bagging_model = BaggingRegressor(
        estimator=KerasRegressor(build_fn=build_ann, input_dim=input_dim, epochs=100, batch_size=32, verbose=0),
        n_estimators=n_estimators
    )
    bagging_model.fit(X_data, y_data)
    y_pred_bagging = bagging_model.predict(X_data)
    results['Bagging ANN'] = calculate_rmse(y_data, y_pred_bagging)

    # Random Patches ANN (Fixed n = 200)
    subset_size = 200
    idx = np.random.choice(len(X_data), subset_size, replace=False)
    model = build_ann(input_dim)
    model.fit(X_data[idx], y_data.iloc[idx], epochs=100, batch_size=32, verbose=0)
    y_pred_patch = model.predict(X_data).flatten()
    results['Random Patches ANN'] = calculate_rmse(y_data, y_pred_patch)

    # Random Subspaces ANN (Fixed n = 200)
    feature_subset = np.random.choice(X.columns, size=int(0.6 * X.shape[1]), replace=False)
    mask = np.isin(X.columns, feature_subset)
    X_subspace = X_data[:, mask]
    model = build_ann(X_subspace.shape[1])
    model.fit(X_subspace, y_data, epochs=100, batch_size=32, verbose=0)
    y_pred_subspace = model.predict(X_subspace).flatten()
    results['Random Subspaces ANN'] = calculate_rmse(y_data, y_pred_subspace)

    return results

# ----------------------------------------
# 📊 Run for i = 1000, 2000
# ----------------------------------------
i_values = [1000, 2000]
optimal_rmse_i = []

output_dir = "rmse_tables"
os.makedirs(output_dir, exist_ok=True)

print("\n🔄 Running with fixed e=100 and n=200 for varying i...")
for i in i_values:
    print(f"\u27a1️ i = {i}")
    result = run_models_i(X_scaled, y, n_estimators=i)
    df = pd.DataFrame({"Model": result.keys(), "RMSE": result.values()})
    df.to_csv(os.path.join(output_dir, f"rmse_i_{i}.csv"), index=False)
    print(df.round(3))

print("\n📄 Additional i-based RMSE tables saved in 'rmse_tables'")
