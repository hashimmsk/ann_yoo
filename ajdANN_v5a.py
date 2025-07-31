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
# 🧚 Set Random Seed for Reproducibility
# ----------------------------------------
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# ----------------------------------------
# 📅 Load Dataset
# ----------------------------------------
csv_file = "C:/Users/hashi/Downloads/dat_hc_simul.csv"

if not os.path.exists(csv_file):
    raise FileNotFoundError(f"Error: The file '{csv_file}' was not found. Please check the path and try again.")

data = pd.read_csv(csv_file)
print("✅ Historical RCT data loaded successfully!")

# Drop non-numeric columns
data = data.drop(columns=['Source', 'Treatment'])

X = data.drop(columns=['mPFS', 'PFS6'])
y = data[['mPFS', 'PFS6']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------
# 🧠 Build ANN Model for Multi-Output
# ----------------------------------------
def build_multi_output_ann(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(2)  # Two outputs: mPFS and pfs6
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# ----------------------------------------
# 🚀 Train Models with Optimal Parameters
# ----------------------------------------
optimal_n = 200
optimal_e = 200
optimal_i = 200
input_dim = X_scaled.shape[1]

# Baseline ANN
baseline_model = build_multi_output_ann(input_dim)
baseline_model.fit(X_scaled, y, epochs=optimal_e, batch_size=32, verbose=0)

# Bagging ANN
bagging_model = BaggingRegressor(
    estimator=KerasRegressor(build_fn=build_multi_output_ann, input_dim=input_dim, epochs=optimal_e, batch_size=32, verbose=0),
    n_estimators=optimal_i
)
bagging_model.fit(X_scaled, y)

# Random Patches ANN
subset_size = int(0.7 * len(X_scaled))
idx = np.random.choice(len(X_scaled), subset_size, replace=False)
patches_model = build_multi_output_ann(input_dim)
patches_model.fit(X_scaled[idx], y.iloc[idx], epochs=optimal_e, batch_size=32, verbose=0)

# Random Subspaces ANN
feature_subset = np.random.choice(X.columns, size=int(0.6 * X.shape[1]), replace=False)
mask = np.isin(X.columns, feature_subset)
X_subspace = X_scaled[:, mask]
subspace_model = build_multi_output_ann(X_subspace.shape[1])
subspace_model.fit(X_subspace, y, epochs=optimal_e, batch_size=32, verbose=0)

print("\n✅ All models trained successfully with optimal hyperparameters.")

# ----------------------------------------
# 📊 Predicting mPFS and pfs6 for Example Cases
# ----------------------------------------
test_cases = [
    [75, 1, 25, 30, 20, 4],
    [55, 1, 55, 60, 50, 3],
    [35, 0, 95, 95, 95, 1]
]
test_df = pd.DataFrame(test_cases, columns=X.columns)
test_scaled = scaler.transform(test_df)
test_subspace = test_scaled[:, mask]

# Predict with each model
baseline_preds = baseline_model.predict(test_scaled)
bagging_preds = bagging_model.predict(test_scaled)
patches_preds = patches_model.predict(test_scaled)
subspace_preds = subspace_model.predict(test_subspace)

# Compile predictions
final_results = pd.DataFrame({
    "Case": ["Case 1", "Case 2", "Case 3"],
    "Baseline_mPFS": baseline_preds[:, 0],
    "Baseline_PFS6": baseline_preds[:, 1],
    "Bagging_mPFS": bagging_preds[:, 0],
    "Bagging_PFS6": bagging_preds[:, 1],
    "Patches_mPFS": patches_preds[:, 0],
    "Patches_PFS6": patches_preds[:, 1],
    "Subspace_mPFS": subspace_preds[:, 0],
    "Subspace_PFS6": subspace_preds[:, 1]
})

# Round for display
print("\n📊 Predicted mPFS and PFS6 for Provided Cases:")
print(final_results.round(2))
