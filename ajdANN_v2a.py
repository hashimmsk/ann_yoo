#import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from tensorflow.python.keras import Input
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Adam = tf.keras.optimizers.Adam
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

from prettytable import PrettyTable

# --------------------------------
# 📌 Load Historical Dataset
# --------------------------------
csv_file = "C:/Users/hashi/Downloads/dat_hc_simul.csv"

# Ensure the file exists
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"Error: The file '{csv_file}' was not found. Please check the path and try again.")

# Load historical RCT data
data = pd.read_csv(csv_file)
print("✅ Historical RCT data loaded successfully!")

# Drop non-numeric columns
data = data.drop(columns=['Source', 'Treatment'])

# Define input features (X) and target variable (y)
X = data.drop(columns=['mPFS'])
y = data['mPFS']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------
# 📌 Function to Build ANN Model
# --------------------------------
def build_ann(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer for predicting survival time (mPFS)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mae'])
    return model

# --------------------------------
# 📌 Train Multiple Models
# --------------------------------

# Dictionary to store RMSE values and calculations
rmse_results = {}

# Function to compute RMSE breakdown
def calculate_rmse(y_true, y_pred):
    errors = y_true - y_pred
    squared_errors = errors ** 2
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)
    return rmse, mse, y_true[:5], y_pred[:5], squared_errors[:5]  # Show first 5 calculations

# 1️⃣ Standard ANN (Baseline)
baseline_ann = build_ann(input_dim=X_scaled.shape[1])
baseline_ann.fit(X_scaled, y, epochs=100, batch_size=32, verbose=1)
y_pred_baseline = baseline_ann.predict(X_scaled).flatten()
rmse_results['Baseline ANN'] = calculate_rmse(y, y_pred_baseline)

# 2️⃣ Bagging ANN (Multiple Models Averaging)
bagging_model = BaggingRegressor(
    estimator=KerasRegressor(build_fn=build_ann, input_dim=X_scaled.shape[1], epochs=100, batch_size=32, verbose=0),
    n_estimators=5
)
bagging_model.fit(X_scaled, y)
y_pred_bagging = bagging_model.predict(X_scaled)
rmse_results['Bagging ANN'] = calculate_rmse(y, y_pred_bagging)

# 3️⃣ Random Patches (Training on Different Subsets of Patients)
subset_size = int(0.7 * len(X))  # Use 70% of patients per model
random_patches_ann = build_ann(input_dim=X_scaled.shape[1])
idx = np.random.choice(len(X), subset_size, replace=False)
random_patches_ann.fit(X_scaled[idx], y.iloc[idx], epochs=100, batch_size=32, verbose=1)
y_pred_patches = random_patches_ann.predict(X_scaled).flatten()
rmse_results['Random Patches ANN'] = calculate_rmse(y, y_pred_patches)

# 4️⃣ Random Subspaces (Training on Different Feature Subsets)
feature_subset = np.random.choice(X.columns, size=int(0.6 * X.shape[1]), replace=False)
X_subspace = X_scaled[:, np.isin(X.columns, feature_subset)]
random_subspace_ann = build_ann(input_dim=X_subspace.shape[1])
random_subspace_ann.fit(X_subspace, y, epochs=100, batch_size=32, verbose=1)
y_pred_subspace = random_subspace_ann.predict(X_subspace).flatten()
rmse_results['Random Subspaces ANN'] = calculate_rmse(y, y_pred_subspace)

# 5️⃣ Linear Regression (Baseline Comparison)
linear_reg = LinearRegression()
linear_reg.fit(X_scaled, y)
y_pred_linear = linear_reg.predict(X_scaled)
rmse_results['Linear Regression'] = calculate_rmse(y, y_pred_linear)

# --------------------------------
# 📌 Compare Model Performance
# --------------------------------

# Convert RMSE results to a DataFrame for better visualization
rmse_df = pd.DataFrame({
    "Model": rmse_results.keys(),
    "RMSE": [round(v[0], 6) for v in rmse_results.values()],
    "MSE (RME²)": [round(v[1], 6) for v in rmse_results.values()]
})

# Save RMSE results as CSV
rmse_csv_path = "C:/Users/hashi/Downloads/rmse_comparison.csv"
rmse_df.to_csv(rmse_csv_path, index=False)

# --------------------------------
# 📌 Display RMSE Table with Formula Breakdown
# --------------------------------
table = PrettyTable()
table.field_names = ["Model", "RMSE", "MSE (RMSE²)", "True Values (First 5)", "Predicted (First 5)", "Squared Errors (First 5)"]

for model, values in rmse_results.items():
    rmse, mse, true_vals, pred_vals, sq_errors = values
    table.add_row([
        model,
        round(rmse, 6),
        round(mse, 6),
        ", ".join(map(lambda x: f"{x:.2f}", true_vals)),
        ", ".join(map(lambda x: f"{x:.2f}", pred_vals)),
        ", ".join(map(lambda x: f"{x:.2f}", sq_errors))
    ])

print("\n📊 Model RMSE Breakdown (Formula Applied):")
print(table)
print(f"\n✅ Comparison saved at: {rmse_csv_path}")

# --------------------------------
# 📌 Plot RMSE Comparison
# --------------------------------
plt.figure(figsize=(8, 5))
plt.bar(rmse_results.keys(), [v[0] for v in rmse_results.values()], color=['blue', 'green', 'red', 'purple', 'orange'])
plt.xlabel("Model Type")
plt.ylabel("RMSE")
plt.title("RMSE Comparison of ANN & Linear Models")
plt.xticks(rotation=45)
plt.show()
