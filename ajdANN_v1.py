# --------------------------------
# 📌 Import Necessary Libraries
# --------------------------------
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
# 📌 Train Models & Predict Survival Times
# --------------------------------

# 1️⃣ Standard ANN (Baseline)
baseline_ann = build_ann(input_dim=X_scaled.shape[1])
baseline_ann.fit(X_scaled, y, epochs=100, batch_size=32, verbose=1)
y_pred_baseline = baseline_ann.predict(X_scaled)

# 2️⃣ Bagging ANN (Multiple Models Averaging)
bagging_model = BaggingRegressor(
    estimator=KerasRegressor(build_fn=build_ann, input_dim=X_scaled.shape[1], epochs=100, batch_size=32, verbose=0),
    n_estimators=5
)
bagging_model.fit(X_scaled, y)
y_pred_bagging = bagging_model.predict(X_scaled)

# 3️⃣ Random Patches ANN (Subset of Patients)
subset_size = int(0.7 * len(X))  # Use 70% of patients per model
random_patches_ann = build_ann(input_dim=X_scaled.shape[1])
idx = np.random.choice(len(X), subset_size, replace=False)
random_patches_ann.fit(X_scaled[idx], y.iloc[idx], epochs=100, batch_size=32, verbose=1)
y_pred_patches = random_patches_ann.predict(X_scaled)

# 4️⃣ Random Subspaces ANN (Subset of Features)
feature_subset = np.random.choice(X.columns, size=int(0.6 * X.shape[1]), replace=False)
X_subspace = X_scaled[:, np.isin(X.columns, feature_subset)]
random_subspace_ann = build_ann(input_dim=X_subspace.shape[1])
random_subspace_ann.fit(X_subspace, y, epochs=100, batch_size=32, verbose=1)
y_pred_subspace = random_subspace_ann.predict(X_subspace)

# 5️⃣ Linear Regression (Baseline Comparison)
linear_reg = LinearRegression()
linear_reg.fit(X_scaled, y)
y_pred_linear = linear_reg.predict(X_scaled)

# --------------------------------
# 📌 Save Predictions to CSV
# --------------------------------
predictions_df = pd.DataFrame({
    "Actual mPFS": y.values,
    "Baseline ANN Prediction": y_pred_baseline.flatten(),
    "Bagging ANN Prediction": y_pred_bagging.flatten(),
    "Random Patches ANN Prediction": y_pred_patches.flatten(),
    "Random Subspaces ANN Prediction": y_pred_subspace.flatten(),
    "Linear Regression Prediction": y_pred_linear
})

predictions_csv_path = "C:/Users/hashi/Downloads/survival_time_predictions.csv"
predictions_df.to_csv(predictions_csv_path, index=False)

print("\n✅ Predictions saved at:", predictions_csv_path)

# --------------------------------
# 📌 Plot Model Predictions
# --------------------------------
plt.figure(figsize=(10, 6))
plt.plot(y.values, label="Actual mPFS", marker='o', linestyle='dashed')
plt.plot(y_pred_baseline, label="Baseline ANN Prediction", marker='x', linestyle='dashed')
plt.plot(y_pred_bagging, label="Bagging ANN Prediction", marker='s', linestyle='dashed')
plt.plot(y_pred_patches, label="Random Patches ANN Prediction", marker='d', linestyle='dashed')
plt.plot(y_pred_subspace, label="Random Subspaces ANN Prediction", marker='^', linestyle='dashed')
plt.plot(y_pred_linear, label="Linear Regression Prediction", marker='*', linestyle='dashed')

plt.xlabel("Patients")
plt.ylabel("mPFS (Predicted vs Actual)")
plt.title("Comparison of Model Predictions")
plt.legend()
plt.show()
