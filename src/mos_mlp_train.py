import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from torch.utils.data import Dataset, DataLoader

# ===============================================================
# 🔒 REPRODUCIBILITY
# ===============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===============================================================
# LOAD DATA
# ===============================================================
DATA_PATH = "data/processed/webrtc_qoe_dataset.csv"
df = pd.read_csv(DATA_PATH).dropna(subset=["mos"])

feature_cols = [
    "sendBitrateKbps", "recvBitrateKbps", "fpsRecv",
    "jitterMs", "rttMs", "packetsLost", "packetsReceived",
    "width", "height", "pixelArea",
    "framesDecoded", "framesDropped", "jitterBufferMs",
    "bitrate_per_pixel",
]

sessions = sorted(df["session"].unique())

# ===============================================================
# TRAIN / VAL / TEST SESSION SPLIT (60/20/20)
# ===============================================================
train_sessions, temp = train_test_split(sessions, test_size=0.40, random_state=SEED)
val_sessions, test_sessions = train_test_split(temp, test_size=0.50, random_state=SEED)

print("=== SESSION SPLIT (FINAL MLP) ===")
print("Train sessions:", len(train_sessions))
print("Val sessions  :", len(val_sessions))
print("Test sessions :", len(test_sessions))

df_train = df[df["session"].isin(train_sessions)]
df_val   = df[df["session"].isin(val_sessions)]
df_test  = df[df["session"].isin(test_sessions)]

# Use DataFrame consistently to avoid warnings
scaler = StandardScaler()
scaler.fit(df_train[feature_cols])

X_train = scaler.transform(df_train[feature_cols])
y_train = df_train["mos"].values

X_val = scaler.transform(df_val[feature_cols])
y_val = df_val["mos"].values

X_test = scaler.transform(df_test[feature_cols])
y_test = df_test["mos"].values

# ===============================================================
# DATASET
# ===============================================================
class MOSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_loader = DataLoader(MOSDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(MOSDataset(X_val, y_val), batch_size=64, shuffle=False)

# ===============================================================
# FINAL OPTIMIZED MLP MODEL
# ===============================================================
class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)

model = MLP(len(feature_cols))

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.L1Loss()

# ===============================================================
# EARLY STOPPING + CHECKPOINT
# ===============================================================
best_val = float("inf")
patience = 8
wait = 0
best_state = None

# ===============================================================
# TRAINING LOOP
# ===============================================================
for epoch in range(1, 101):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            p = model(xb)
            preds.extend(p.numpy())
            targs.extend(yb.numpy())

    val_mae = mean_absolute_error(targs, preds)
    print(f"Epoch {epoch:03d} | Val MAE = {val_mae:.4f}")

    # Early stopping check
    if val_mae < best_val:
        best_val = val_mae
        best_state = model.state_dict()
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered.")
            break

# Load best model
# Load best model
model.load_state_dict(best_state)

# ===============================================================
# SAVE BEST MODEL + SCALER
# ===============================================================
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(SAVE_DIR, "mlp_mos.pt")
SCALER_PATH = os.path.join(SAVE_DIR, "scaler_mos.npy")
META_PATH = os.path.join(SAVE_DIR, "metadata.json")

torch.save(best_state, MODEL_PATH)
np.save(SCALER_PATH, {"mean": scaler.mean_, "scale": scaler.scale_, "features": feature_cols}, allow_pickle=True)

print("\n[+] Saved model, scaler.")

# ===============================================================
# FINAL TEST PERF
# ===============================================================
model.eval()
test_pred = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
test_mae = mean_absolute_error(y_test, test_pred)

print("\n========== FINAL TEST RESULTS (OPTIMIZED MLP) ==========")
print("Test MAE =", test_mae)

# ===============================================================
# PER-SESSION MAE
# ===============================================================
print("\n========== PER-SESSION TEST MAE ==========")
for sess in test_sessions:
    df_s = df_test[df_test["session"] == sess]
    X_s = scaler.transform(df_s[feature_cols])
    y_s = df_s["mos"].values
    p_s = model(torch.tensor(X_s, dtype=torch.float32)).detach().numpy()
    mae_s = mean_absolute_error(y_s, p_s)
    print(f"{sess:30s} MAE = {mae_s:.4f}")

# ===============================================================
# PERMUTATION FEATURE IMPORTANCE (MANUAL)
# ===============================================================
print("\n========== PERMUTATION FEATURE IMPORTANCE ==========")

def pytorch_predict(X):
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return model(X_t).numpy()

baseline_mae = mean_absolute_error(y_test, pytorch_predict(X_test))
importances = []

for i, feat in enumerate(feature_cols):
    X_perm = X_test.copy()
    shuffled = X_perm[:, i].copy()
    np.random.shuffle(shuffled)
    X_perm[:, i] = shuffled

    perm_mae = mean_absolute_error(y_test, pytorch_predict(X_perm))
    importances.append((feat, perm_mae - baseline_mae))

# Print descending order
for f, imp in sorted(importances, key=lambda x: -x[1]):
    print(f"{f:25s} {imp:.6f}")

# ===============================================================
# SHAP FEATURE IMPORTANCE (DeepLiftShap — compatible on all SHAP versions)
# ===============================================================
print("\n========== SHAP FEATURE IMPORTANCE ==========")

import shap

# Small background sample
background_idx = np.random.choice(len(X_train), size=200, replace=False)
background = torch.tensor(X_train[background_idx], dtype=torch.float32)

X_shap = torch.tensor(X_test[:400], dtype=torch.float32)

try:
    explainer = shap.DeepLiftShap(model)
    shap_values = explainer.shap_values(X_shap, baselines=background)
    shap_values = np.array(shap_values)
except Exception as e:
    print("DeepLiftShap failed → using KernelExplainer fallback. Error:", e)
    explainer = shap.KernelExplainer(
        lambda x: model(torch.tensor(x, dtype=torch.float32)).detach().numpy(),
        X_train[background_idx][:50]
    )
    shap_values = explainer.shap_values(X_test[:200])

# Mean |SHAP|
mean_abs = np.mean(np.abs(shap_values), axis=0)

print("\n========== SHAP MEAN |ABS| FEATURE IMPORTANCE ==========")
for f, val in sorted(zip(feature_cols, mean_abs), key=lambda x: -x[1]):
    print(f"{f:25s} {val:.6f}")