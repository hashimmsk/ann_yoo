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
# 🚀 Train Final Models with Optimal Parameters
# ----------------------------------------
optimal_n = 200
optimal_e = 200
optimal_i = 200

input_dim = X_scaled.shape[1]

# Baseline ANN
baseline_model = build_ann(input_dim)
baseline_model.fit(X_scaled, y, epochs=optimal_e, batch_size=32, verbose=0)

# Bagging ANN
bagging_model = BaggingRegressor(
    estimator=KerasRegressor(build_fn=build_ann, input_dim=input_dim, epochs=optimal_e, batch_size=32, verbose=0),
    n_estimators=optimal_i
)
bagging_model.fit(X_scaled, y)

# Random Patches ANN
subset_size = int(0.7 * len(X_scaled))
idx = np.random.choice(len(X_scaled), subset_size, replace=False)
patches_model = build_ann(input_dim)
patches_model.fit(X_scaled[idx], y.iloc[idx], epochs=optimal_e, batch_size=32, verbose=0)

# Random Subspaces ANN
feature_subset = np.random.choice(X.columns, size=int(0.6 * X.shape[1]), replace=False)
mask = np.isin(X.columns, feature_subset)
X_subspace = X_scaled[:, mask]
subspace_model = build_ann(X_subspace.shape[1])
subspace_model.fit(X_subspace, y, epochs=optimal_e, batch_size=32, verbose=0)

print("\n✅ All models trained successfully with optimal hyperparameters.")

# ----------------------------------------
# 📈 Predictive Modelling (User Input)
# ----------------------------------------
print("\n🔍 Enter patient features to predict mPFS:")
user_input = []

for feature in X.columns:
    val = float(input(f"{feature}: "))
    user_input.append(val)

user_array = np.array(user_input).reshape(1, -1)
user_scaled = scaler.transform(user_array)
user_subspace = user_scaled[:, mask]

print("\n🧠 Predicted mPFS for user input:")
print(f"Baseline ANN: {baseline_model.predict(user_scaled)[0][0]:.2f}")
print(f"Bagging ANN: {bagging_model.predict(user_scaled)[0]:.2f}")
print(f"Random Patches ANN: {patches_model.predict(user_scaled)[0][0]:.2f}")
print(f"Random Subspaces ANN: {subspace_model.predict(user_subspace)[0][0]:.2f}")
