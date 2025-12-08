import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib

DATA_PATH = "data/processed/webrtc_qoe_dataset.csv"
MODEL_PATH = "models/xgb_session_model.json"
SCALER_PATH = "models/xgb_session_scaler.pkl"

os.makedirs("models", exist_ok=True)

# ============================================================
# Load dataset
# ============================================================
df = pd.read_csv(DATA_PATH)

feature_cols = [
    "sendBitrateKbps", "recvBitrateKbps", "fpsRecv",
    "jitterMs", "rttMs", "packetsLost", "packetsReceived",
    "width", "height", "pixelArea",
    "framesDecoded", "framesDropped", "jitterBufferMs",
    "bitrate_per_pixel",
]

df = df.dropna(subset=["mos"])

sessions = sorted(df["session"].unique())

# ============================================================
# SESSION-WISE SPLIT
# ============================================================
train_sessions, temp = train_test_split(sessions, test_size=0.40, random_state=42)
val_sessions, test_sessions = train_test_split(temp, test_size=0.50, random_state=42)

print("=== SESSION SPLIT (XGB) ===")
print("Train sessions:", len(train_sessions))
print("Val sessions  :", len(val_sessions))
print("Test sessions :", len(test_sessions))

df_train = df[df["session"].isin(train_sessions)]
df_val   = df[df["session"].isin(val_sessions)]
df_test  = df[df["session"].isin(test_sessions)]

X_train = df_train[feature_cols]
y_train = df_train["mos"]

X_val = df_val[feature_cols]
y_val = df_val["mos"]

X_test = df_test[feature_cols]
y_test = df_test["mos"]

# ============================================================
# Feature Scaling
# ============================================================
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

joblib.dump(scaler, SCALER_PATH)

# ============================================================
# Train XGBRegressor
# ============================================================
model = XGBRegressor(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    reg_alpha=0.0,
    objective="reg:squarederror",
    eval_metric="mae",
    verbosity=0,       # disable training logs
    random_state=42,
)

print("\nTraining XGBRegressor...\n")

model.fit(
    X_train_s, y_train,
    eval_set=[(X_val_s, y_val)],
    verbose=False,     # <-- stops [xxx] logging completely
)

# ============================================================
# Final Test Performance
# ============================================================
y_pred = model.predict(X_test_s)
test_mae = mean_absolute_error(y_test, y_pred)

print("\n========== FINAL TEST RESULTS (XGB) ==========")
print(f"Test MAE = {test_mae:.4f}")

# ============================================================
# Save model
# ============================================================
model.get_booster().save_model(MODEL_PATH)
print(f"\nSaved model to {MODEL_PATH}")

# ============================================================
# Per-session MAE
# ============================================================
print("\n========== PER-SESSION TEST MAE ==========")

for s in test_sessions:
    df_s = df_test[df_test["session"] == s]
    X_s = scaler.transform(df_s[feature_cols])
    pred_s = model.predict(X_s)
    mae_s = mean_absolute_error(df_s["mos"], pred_s)
    print(f"{s:<25} MAE = {mae_s:.4f}")

# ============================================================
# Feature Importance 
# ============================================================

importances = model.feature_importances_
feat_names = feature_cols

# Pair feature names with importance values
feat_imp = list(zip(feat_names, importances))

# Sort by importance descending
feat_imp_sorted = sorted(feat_imp, key=lambda x: x[1], reverse=True)

print("\n========== XGB FEATURE IMPORTANCE ==========")
for name, score in feat_imp_sorted:
    print(f"{name:<25} {score:.5f}")

# ============================================================
# SHAP FEATURE IMPORTANCE
# ============================================================
print("\n========== SHAP FEATURE IMPORTANCE ==========")

import shap

# Create SHAP explainer for XGBoost
explainer = shap.TreeExplainer(model)

# Use a smaller sample for speed
X_sample = X_train_s[:2000] if X_train_s.shape[0] > 2000 else X_train_s

# Compute SHAP values
shap_values = explainer.shap_values(X_sample)

# Mean |SHAP| value per feature
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

# Sort by importance
shap_imp = sorted(zip(feature_cols, mean_abs_shap), key=lambda x: x[1], reverse=True)

print("\n========== SHAP MEAN |ABS| FEATURE IMPORTANCE ==========")
for name, score in shap_imp:
    print(f"{name:<25} {score:.5f}")
