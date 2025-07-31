# ----------------------------------------
# 📦 Import Necessary Libraries
# ----------------------------------------
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random

from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scikeras.wrappers import KerasRegressor

from tensorflow.python.keras import Input
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Adam = tf.keras.optimizers.Adam

# ----------------------------------------
# 🧪 Set Random Seed for Reproducibility
# ----------------------------------------
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# ----------------------------------------
# 📥 Load Dataset
# ----------------------------------------
csv_file = "C:/Users/hashi/Downloads/dat_hc_simul.csv"

if not os.path.exists(csv_file):
    raise FileNotFoundError(f"Error: The file '{csv_file}' was not found. Please check the path and try again.")

data = pd.read_csv(csv_file)
print("✅ Historical RCT data loaded successfully!")

# Drop non-numeric columns
data = data.drop(columns=['Source', 'Treatment'])

X = data.drop(columns=['mPFS'])
y = data['mPFS']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------
# 🧠 Build ANN Model Function
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
# 🧮 RMSE Calculation Helper
# ----------------------------------------
def calculate_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse

# ----------------------------------------
# 🤖 Run Model Variants
# ----------------------------------------
def run_models(X_data, y_data, n_estimators=10, epochs=100, use_subset=False, use_subspace=False):
    input_dim = X_data.shape[1]
    results = {}

    # Baseline ANN
    model = build_ann(input_dim)
    model.fit(X_data, y_data, epochs=epochs, batch_size=32, verbose=0)
    y_pred = model.predict(X_data).flatten()
    results['Baseline ANN'] = calculate_rmse(y_data, y_pred)

    # Bagging ANN
    bagging_model = BaggingRegressor(
        estimator=KerasRegressor(build_fn=build_ann, input_dim=input_dim, epochs=epochs, batch_size=32, verbose=0),
        n_estimators=n_estimators
    )
    bagging_model.fit(X_data, y_data)
    y_pred_bagging = bagging_model.predict(X_data)
    results['Bagging ANN'] = calculate_rmse(y_data, y_pred_bagging)

    # Random Patches
    if use_subset:
        subset_size = int(0.7 * len(X_data))
        idx = np.random.choice(len(X_data), subset_size, replace=False)
        model = build_ann(input_dim)
        model.fit(X_data[idx], y_data.iloc[idx], epochs=epochs, batch_size=32, verbose=0)
        y_pred_patch = model.predict(X_data).flatten()
        results['Random Patches ANN'] = calculate_rmse(y_data, y_pred_patch)

    # Random Subspaces
    if use_subspace:
        feature_subset = np.random.choice(X.columns, size=int(0.6 * X.shape[1]), replace=False)
        mask = np.isin(X.columns, feature_subset)
        X_subspace = X_data[:, mask]
        model = build_ann(X_subspace.shape[1])
        model.fit(X_subspace, y_data, epochs=epochs, batch_size=32, verbose=0)
        y_pred_subspace = model.predict(X_subspace).flatten()
        results['Random Subspaces ANN'] = calculate_rmse(y_data, y_pred_subspace)

    return results

# ----------------------------------------
# 📊 Stage 1: Varying n (Resampling Size)
# ----------------------------------------
n_values = [10, 50, 100, 200]
optimal_rmse_n = []

ios_path = "rmse_tables"
os.makedirs(ios_path, exist_ok=True)

for n in n_values:
    result = run_models(X_scaled, y, n_estimators=n, epochs=100, use_subset=True, use_subspace=True)
    df = pd.DataFrame({"Model": result.keys(), "RMSE": result.values()})
    df.to_csv(f"rmse_tables/rmse_n_{n}.csv", index=False)
    min_rmse = df['RMSE'].min()
    optimal_rmse_n.append((n, min_rmse))

# ----------------------------------------
# 📊 Plot RMSE vs n
# ----------------------------------------
plt.figure()
plt.plot([x[0] for x in optimal_rmse_n], [x[1] for x in optimal_rmse_n], marker='o')
plt.title("Optimal RMSE vs Resampling Size (n)")
plt.xlabel("n")
plt.ylabel("Best RMSE")
plt.grid(True)
plt.savefig("rmse_vs_n.png")

# ----------------------------------------
# 📊 Stage 2: Varying e (Epochs)
# ----------------------------------------
e_values = [10, 20, 50, 100, 200]
optimal_rmse_e = []

for e in e_values:
    result = run_models(X_scaled, y, n_estimators=100, epochs=e, use_subset=True, use_subspace=True)
    df = pd.DataFrame({"Model": result.keys(), "RMSE": result.values()})
    df.to_csv(f"rmse_tables/rmse_e_{e}.csv", index=False)
    min_rmse = df['RMSE'].min()
    optimal_rmse_e.append((e, min_rmse))

# ----------------------------------------
# 📊 Plot RMSE vs e
# ----------------------------------------
plt.figure()
plt.plot([x[0] for x in optimal_rmse_e], [x[1] for x in optimal_rmse_e], marker='o')
plt.title("Optimal RMSE vs Epochs (e)")
plt.xlabel("e")
plt.ylabel("Best RMSE")
plt.grid(True)
plt.savefig("rmse_vs_e.png")

# ----------------------------------------
# 📊 Stage 3: Varying i (Iterations)
# ----------------------------------------
i_values = [100, 200, 500, 1000]
optimal_rmse_i = []

for i in i_values:
    result = run_models(X_scaled, y, n_estimators=i, epochs=100, use_subset=True, use_subspace=True)
    df = pd.DataFrame({"Model": result.keys(), "RMSE": result.values()})
    df.to_csv(f"rmse_tables/rmse_i_{i}.csv", index=False)
    min_rmse = df['RMSE'].min()
    optimal_rmse_i.append((i, min_rmse))

# ----------------------------------------
# 📊 Plot RMSE vs i
# ----------------------------------------
plt.figure()
plt.plot([x[0] for x in optimal_rmse_i], [x[1] for x in optimal_rmse_i], marker='o')
plt.title("Optimal RMSE vs Iterations (i)")
plt.xlabel("i")
plt.ylabel("Best RMSE")
plt.grid(True)
plt.savefig("rmse_vs_i.png")

print("\n✅ All tables saved in 'rmse_tables' folder.")
print("📈 Graphs saved as 'rmse_vs_n.png', 'rmse_vs_e.png', 'rmse_vs_i.png'.")
