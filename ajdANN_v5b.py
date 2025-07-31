# ----------------------------------------
# 📦 Import Necessary Libraries
# ----------------------------------------
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random
import joblib

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
# 📂 Model Saving Paths (Defined Early to Prevent Errors)
# ----------------------------------------
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)

model_paths = {
    "baseline": os.path.join(model_dir, "baseline_model.h5"),
    "bagging": os.path.join(model_dir, "bagging_model.pkl"),
    "patches": os.path.join(model_dir, "patches_model.h5"),
    "subspace": os.path.join(model_dir, "subspace_model.h5"),
    "subspace_mask": os.path.join(model_dir, "subspace_mask.pkl"),
    "scaler": os.path.join(model_dir, "scaler.pkl")
}

# ----------------------------------------
# 📅 Load Dataset
# ----------------------------------------
csv_file = "C:/Users/hashi/Downloads/dat_hc_simul.csv"

if not os.path.exists(csv_file):
    raise FileNotFoundError(f"Error: The file '{csv_file}' was not found. Please check the path and try again.")

data = pd.read_csv(csv_file)
print("✅ Historical RCT data loaded successfully!")

X = data.drop(columns=['Source', 'Treatment', 'N', 'mPFS', 'PFS6'])
y = data[['mPFS', 'PFS6']]

# ----------------------------------------
# 🧪 Standardize Features
# ----------------------------------------
if os.path.exists(model_paths["scaler"]):
    scaler = joblib.load(model_paths["scaler"])
else:
    scaler = StandardScaler()
    scaler.fit(X)
    joblib.dump(scaler, model_paths["scaler"])

X_scaled = scaler.transform(X)

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
        Dense(2)  # Output for mPFS and PFS6
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])
    return model

# ----------------------------------------
# 🔁 Load or Train Models
# ----------------------------------------
optimal_n = 200
optimal_e = 200
optimal_i = 200
input_dim = X.shape[1]

# ➤ Baseline ANN
if os.path.exists(model_paths["baseline"]):
    baseline_model = tf.keras.models.load_model(model_paths["baseline"])
else:
    baseline_model = build_multi_output_ann(input_dim)
    baseline_model.fit(X_scaled, y, epochs=optimal_e, batch_size=32, verbose=0)
    baseline_model.save(model_paths["baseline"])

# ➤ Bagging ANN
if os.path.exists(model_paths["bagging"]):
    bagging_model = joblib.load(model_paths["bagging"])
else:
    bagging_model = BaggingRegressor(
        estimator=KerasRegressor(build_fn=build_multi_output_ann, input_dim=input_dim, epochs=optimal_e, batch_size=32, verbose=0),
        n_estimators=optimal_i
    )
    bagging_model.fit(X_scaled, y)
    joblib.dump(bagging_model, model_paths["bagging"])

# ➤ Random Patches ANN
if os.path.exists(model_paths["patches"]):
    patches_model = tf.keras.models.load_model(model_paths["patches"])
else:
    subset_size = int(0.7 * len(X_scaled))
    idx = np.random.choice(len(X_scaled), subset_size, replace=False)
    patches_model = build_multi_output_ann(input_dim)
    patches_model.fit(X_scaled[idx], y.iloc[idx], epochs=optimal_e, batch_size=32, verbose=0)
    patches_model.save(model_paths["patches"])

# ➤ Random Subspaces ANN
if os.path.exists(model_paths["subspace"]):
    subspace_model = tf.keras.models.load_model(model_paths["subspace"])
    mask = joblib.load(model_paths["subspace_mask"])
else:
    feature_subset = np.random.choice(X.columns, size=int(0.6 * X.shape[1]), replace=False)
    mask = np.isin(X.columns, feature_subset)
    joblib.dump(mask, model_paths["subspace_mask"])
    X_subspace = X_scaled[:, mask]
    subspace_model = build_multi_output_ann(X_subspace.shape[1])
    subspace_model.fit(X_subspace, y, epochs=optimal_e, batch_size=32, verbose=0)
    subspace_model.save(model_paths["subspace"])

print("\n✅ All models are trained and saved (or loaded) successfully.")

# ----------------------------------------
# 🧪 Predict Example Cases
# ----------------------------------------
test_cases = [
    [75, 1, 25, 30, 20, 4],
    [55, 1, 55, 60, 50, 3],
    [35, 0, 95, 95, 95, 1]
]

test_df = pd.DataFrame(test_cases, columns=X.columns)
test_scaled = scaler.transform(test_df)
test_subspace = test_scaled[:, mask]

baseline_preds = baseline_model.predict(test_scaled)
bagging_preds = bagging_model.predict(test_scaled)
patches_preds = patches_model.predict(test_scaled)
subspace_preds = subspace_model.predict(test_subspace)

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

print("\n📊 Predicted mPFS and PFS6 for Provided Cases:")
print(final_results.round(2))

# ----------------------------------------
# 💾 Save Final Results to CSV
# ----------------------------------------

output_csv_path = r"C:\Users\hashi\Desktop\Dr Yoo Research\predicted_results.csv"

# Dummy metadata (customize if needed)
sources = ["EX001", "EX002", "EX003"]
treatments = ["Model", "Model", "Model"]
n_values = [0, 0, 0]  # Placeholder

# Average of all model predictions
avg_preds = (baseline_preds + bagging_preds + patches_preds + subspace_preds) / 4

# Construct final DataFrame for export
csv_results = pd.DataFrame({
    "Source": sources,
    "Treatment": treatments,
    "N": n_values,
    "mPFS": avg_preds[:, 0],
    "PFS6": avg_preds[:, 1],
    "age": test_df["age"],
    "male": test_df["male"],
    "resec": test_df["resec"],
    "Kscore": test_df["Kscore"],
    "methyl": test_df["methyl"],
    "pre_trt_history": test_df["pre_trt_history"]
})

csv_results.to_csv(output_csv_path, index=False)
print(f"\n📝 Prediction table saved to: {output_csv_path}")
