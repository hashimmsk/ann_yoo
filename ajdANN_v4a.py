# ----------------------------------------
# 📦 Import Required Libraries
# ----------------------------------------
import pandas as pd
import numpy as np
import tensorflow as tf
import os

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
import random
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# ----------------------------------------
# 📥 Load Historical Dataset
# ----------------------------------------
data_path = "C:/Users/hashi/Downloads/dat_hc_simul.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found at: {data_path}")

# Load and clean
historical_data = pd.read_csv(data_path)
historical_data = historical_data.drop(columns=['Source', 'Treatment'])

X = historical_data.drop(columns=['mPFS'])
y = historical_data['mPFS']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------
# 🧠 ANN Model Architecture
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
# 🧮 RMSE Calculator
# ----------------------------------------
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ----------------------------------------
# 🤖 Train All Models with Optimal Params
# ----------------------------------------
def train_all_models(X_data, y_data, n_estimators=200, epochs=200):
    input_dim = X_data.shape[1]
    results = {}

    # Baseline ANN
    baseline = build_ann(input_dim)
    baseline.fit(X_data, y_data, epochs=epochs, batch_size=32, verbose=1)
    y_pred_baseline = baseline.predict(X_data).flatten()
    results['Baseline ANN'] = calculate_rmse(y_data, y_pred_baseline)

    # Bagging ANN
    bagging = BaggingRegressor(
        estimator=KerasRegressor(build_fn=build_ann, input_dim=input_dim, epochs=epochs, batch_size=32, verbose=0),
        n_estimators=n_estimators
    )
    bagging.fit(X_data, y_data)
    y_pred_bagging = bagging.predict(X_data)
    results['Bagging ANN'] = calculate_rmse(y_data, y_pred_bagging)

    # Random Patches ANN
    subset_size = int(0.7 * len(X_data))
    idx = np.random.choice(len(X_data), subset_size, replace=False)
    patches_ann = build_ann(input_dim)
    patches_ann.fit(X_data[idx], y_data.iloc[idx], epochs=epochs, batch_size=32, verbose=1)
    y_pred_patch = patches_ann.predict(X_data).flatten()
    results['Random Patches ANN'] = calculate_rmse(y_data, y_pred_patch)

    # Random Subspaces ANN
    feature_subset = np.random.choice(X.columns, size=int(0.6 * X.shape[1]), replace=False)
    mask = np.isin(X.columns, feature_subset)
    X_subspace = X_data[:, mask]
    subspace_ann = build_ann(X_subspace.shape[1])
    subspace_ann.fit(X_subspace, y_data, epochs=epochs, batch_size=32, verbose=1)
    y_pred_subspace = subspace_ann.predict(X_subspace).flatten()
    results['Random Subspaces ANN'] = calculate_rmse(y_data, y_pred_subspace)

    return results, baseline, bagging, patches_ann, subspace_ann

# ----------------------------------------
# 🚀 Execute Training
# ----------------------------------------
rmse_results, model_baseline, model_bagging, model_patches, model_subspace = train_all_models(X_scaled, y, n_estimators=200, epochs=200)

# Display RMSE Results
print("\n📊 RMSE Results with Optimal Parameters:")
for model, rmse in rmse_results.items():
    print(f"{model}: {rmse:.6f}")

# ----------------------------------------
# 🔮 Predictive Modeling (Example Use)
# ----------------------------------------
# Suppose we want to predict on the same dataset again:
print("\n🔮 Example Predictions from Baseline ANN:")
predictions = model_baseline.predict(X_scaled).flatten()
print(predictions[:10])  # Display first 10 predictions

# Optional: Save predictions
pred_df = pd.DataFrame({"True_mPFS": y, "Predicted_mPFS": predictions})
pred_df.to_csv("final_predictions_v4.csv", index=False)
print("\n✅ Predictions saved to 'final_predictions_v4.csv'")
