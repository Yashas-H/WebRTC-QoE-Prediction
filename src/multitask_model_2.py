import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import pickle


DATA_PATH = "data/processed/webrtc_qoe_dataset.csv"
REPEATS = 5   # Number of repeated session splits


# -----------------------------
# Dataset Wrapper
# -----------------------------
class QoeDataset(Dataset):
    def __init__(self, X, mos, tier, res, impair):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.mos = torch.tensor(mos, dtype=torch.float32)
        self.tier = torch.tensor(tier, dtype=torch.long)
        self.res = torch.tensor(res, dtype=torch.float32)
        self.impair = torch.tensor(impair, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.mos[idx],
            self.tier[idx],
            self.res[idx],
            self.impair[idx]
        )


# -----------------------------
# Model (same as multitask_model_2.py)
# -----------------------------
class MultiTaskQoeNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.mos_head = nn.Linear(64, 1)
        self.tier_head = nn.Linear(64, 3)
        self.res_head = nn.Linear(64, 1)
        self.impair_head = nn.Linear(64, 4)

    def forward(self, x):
        h = self.shared(x)
        mos = self.mos_head(h).squeeze(-1)
        tier = self.tier_head(h)
        res = self.res_head(h).squeeze(-1)
        impair = self.impair_head(h)
        return mos, tier, res, impair


# -----------------------------
# Load Data
# -----------------------------
def load_dataframe():
    df = pd.read_csv(DATA_PATH)

    feature_cols = [
        "sendBitrateKbps","recvBitrateKbps","fpsSend","fpsRecv",
        "jitterMs","rttMs","packetsLost","packetsReceived",
        "width","height","pixelArea",
        "framesDecoded","framesDropped","decodeMs","jitterBufferMs",
        "frameRateDecoded","frameRateOutput","currentDelayMs",
        "nackCount","pliCount","firCount","qpSumInbound",
        "framesEncoded","encodeUsagePercent","frameRateInput",
        "frameRateSent","qpSumOutbound","availableSendBandwidth",
        "availableRecvBandwidth","targetEncBitrate","actualEncBitrate",
        "retransmitBitrate","bucketDelay",
        "recvBitrateDiff","jitterDiff","rttDiff"
    ]

    X = df[feature_cols].fillna(0).values
    mos = df["mos"].values
    tier = df["qoe_tier"].values
    res = df["resolution_drop"].values
    impair = df["impairment_id"].values
    sessions = df["session"].values

    return df, X, mos, tier, res, impair, sessions, feature_cols


# -----------------------------
# Train Once on a Given Split
# -----------------------------
def train_one_split(train_idx, val_idx, X, mos, tier, res, impair, device):

    # scale per split
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])

    train_ds = QoeDataset(X_train, mos[train_idx], tier[train_idx],
                          res[train_idx], impair[train_idx])
    val_ds   = QoeDataset(X_val, mos[val_idx], tier[val_idx],
                          res[val_idx], impair[val_idx])

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=64)

    # model + optimizers
    model = MultiTaskQoeNet(X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    # Compute class weights for impairment
    _, counts = np.unique(impair[train_idx], return_counts=True)
    weights = (1.0 / counts)
    weights = weights / weights.sum()
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    impair_ce = nn.CrossEntropyLoss(weight=weights)

    # --- Train for fixed epochs ---
    for _ in range(20):
        model.train()
        for Xb, mos_b, tier_b, res_b, impair_b in train_dl:
            Xb, mos_b, tier_b, res_b, impair_b = (
                Xb.to(device), mos_b.to(device), tier_b.to(device),
                res_b.to(device), impair_b.to(device)
            )

            opt.zero_grad()
            mos_p, tier_p, res_p, impair_p = model(Xb)

            loss = (
                mse(mos_p, mos_b) +
                ce(tier_p, tier_b) +
                0.5 * bce(res_p, res_b) +
                0.5 * impair_ce(impair_p, impair_b)
            )
            loss.backward()
            opt.step()

    # ---- Evaluate ----
    model.eval()
    mos_true, mos_pred = [], []
    tier_true, tier_pred = [], []
    res_true, res_pred = [], []
    impair_true, impair_pred = [], []

    with torch.no_grad():
        for Xb, mos_b, tier_b, res_b, impair_b in val_dl:
            Xb = Xb.to(device)
            mos_b, tier_b, res_b, impair_b = (
                mos_b.numpy(), tier_b.numpy(), res_b.numpy(), impair_b.numpy()
            )

            mp, tp, rp, ip = model(Xb)
            mos_pred.extend(mp.cpu().numpy())
            mos_true.extend(mos_b)

            tier_pred.extend(np.argmax(tp.cpu().numpy(), axis=1))
            tier_true.extend(tier_b)

            res_pred.extend((torch.sigmoid(rp) > 0.5).cpu().numpy())
            res_true.extend(res_b)

            impair_pred.extend(np.argmax(ip.cpu().numpy(), axis=1))
            impair_true.extend(impair_b)

    return (
        np.sqrt(mean_squared_error(mos_true, mos_pred)),
        accuracy_score(tier_true, tier_pred),
        accuracy_score(res_true, res_pred),
        accuracy_score(impair_true, impair_pred),
    )


# -----------------------------
# Main: Repeated Evaluation
# -----------------------------
def run_repeated_eval():
    df, X, mos, tier, res, impair, sessions, feats = load_dataframe()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Total sessions: {len(np.unique(sessions))}")

    metrics = []

    for r in range(REPEATS):
        print(f"\n=== Split {r+1}/{REPEATS} ===")
        train_sess, val_sess = train_test_split(
            np.unique(sessions),
            test_size=0.2,
            random_state=42 + r,
        )

        train_idx = df["session"].isin(train_sess).values
        val_idx   = df["session"].isin(val_sess).values

        results = train_one_split(train_idx, val_idx, X, mos, tier, res, impair, device)
        metrics.append(results)

        print(f"Split {r+1} results: MOS={results[0]:.3f}, "
              f"Tier={results[1]:.3f}, Res={results[2]:.3f}, Impair={results[3]:.3f}")

    metrics = np.array(metrics)
    print("\n\n===== FINAL AVERAGED METRICS =====")
    print(f"MOS RMSE:   {metrics[:,0].mean():.3f} ± {metrics[:,0].std():.3f}")
    print(f"Tier Acc:   {metrics[:,1].mean():.3f} ± {metrics[:,1].std():.3f}")
    print(f"Res Acc:    {metrics[:,2].mean():.3f} ± {metrics[:,2].std():.3f}")
    print(f"Impair Acc: {metrics[:,3].mean():.3f} ± {metrics[:,3].std():.3f}")


if __name__ == "__main__":
    run_repeated_eval()
