import os
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from torch.utils.data import Dataset, DataLoader

DATA_PATH = "data/processed/webrtc_qoe_dataset.csv"
MODEL_PATH = "models/multitask_qoe.pt"
SCALER_PATH = "models/multitask_scaler.pkl"
os.makedirs("models", exist_ok=True)


# ============================================================
# Dataset
# ============================================================

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


# ============================================================
# Model Architecture
# ============================================================

class MultiTaskQoeNet(nn.Module):
    def __init__(self, input_dim, impairment_classes=4):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 160),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(160, 96),
            nn.ReLU(),
        )

        # Separate heads
        self.mos_head = nn.Linear(96, 1)
        self.tier_head = nn.Linear(96, 3)
        self.res_head = nn.Linear(96, 1)
        self.impair_head = nn.Linear(96, impairment_classes)

    def forward(self, x):
        h = self.shared(x)

        mos = self.mos_head(h).squeeze(-1)
        tier_logits = self.tier_head(h)
        res_logit = self.res_head(h).squeeze(-1)
        impair_logits = self.impair_head(h)

        return mos, tier_logits, res_logit, impair_logits


# ============================================================
# Load data (session-level split + scaler)
# ============================================================

def load_data():
    print("\nLoading dataset:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    print(f"Rows: {len(df)}")
    sessions = df["session"].unique()
    print(f"Sessions: {len(sessions)}")

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

    unique_sessions = df["session"].unique()

    train_sessions, val_sessions = train_test_split(
        unique_sessions,
        test_size=0.2,
        random_state=42,
    )

    train_mask = df["session"].isin(train_sessions)
    val_mask = df["session"].isin(val_sessions)

    X_train, X_val = X[train_mask], X[val_mask]
    mos_train, mos_val = mos[train_mask], mos[val_mask]
    tier_train, tier_val = tier[train_mask], tier[val_mask]
    res_train, res_val = res[train_mask], res[val_mask]
    impair_train, impair_val = impair[train_mask], impair[val_mask]

    print(f"\nTrain sessions: {len(train_sessions)}")
    print(f"Val sessions:   {len(val_sessions)}")

    # Standardize inputs
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Save scaler
    with open(SCALER_PATH, "wb") as f:
        pickle.dump({"scaler": scaler, "features": feature_cols}, f)

    return (
        QoeDataset(X_train, mos_train, tier_train, res_train, impair_train),
        QoeDataset(X_val, mos_val, tier_val, res_val, impair_val),
        len(feature_cols)
    )


# ============================================================
# Training
# ============================================================

def train_model():
    print("\nRunning multitask model training...\n")

    train_ds, val_ds, input_dim = load_data()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nUsing device:", device)

    model = MultiTaskQoeNet(input_dim=input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    mse = nn.MSELoss()
    tier_ce = nn.CrossEntropyLoss()
    res_bce = nn.BCEWithLogitsLoss()

    # -----------------------------------------
    # CLASS-WEIGHTED IMPAIRMENT LOSS
    # -----------------------------------------
    impair_train = np.array([train_ds.impair[i].item() for i in range(len(train_ds))])
    class_counts = np.bincount(impair_train, minlength=4)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print("\nImpairment class counts:", class_counts)
    print("Impairment class weights:", class_weights.cpu().numpy())

    impair_ce = nn.CrossEntropyLoss(weight=class_weights)

    # dataloaders
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    print("\nStarting training...\n")

    EPOCHS = 30

    for epoch in range(1, EPOCHS + 1):
        model.train()

        for X, mos, tier, res, impair in train_loader:
            X, mos, tier, res, impair = (
                X.to(device),
                mos.to(device),
                tier.to(device),
                res.to(device),
                impair.to(device)
            )

            opt.zero_grad()

            mos_p, tier_lg, res_lg, impair_lg = model(X)

            loss = (
                mse(mos_p, mos) +
                tier_ce(tier_lg, tier) +
                0.3 * res_bce(res_lg, res) +     # Lower weight
                0.7 * impair_ce(impair_lg, impair)  # Higher weight because imbalance
            )

            loss.backward()
            opt.step()

        # --------------------------
        # VALIDATION
        # --------------------------
        model.eval()
        mos_true = []
        mos_pred = []
        tier_true = []
        tier_pred = []
        res_true = []
        res_pred = []
        impair_true = []
        impair_pred = []

        with torch.no_grad():
            for X, mos, tier, res, impair in val_loader:
                X, mos, tier, res, impair = (
                    X.to(device),
                    mos.to(device),
                    tier.to(device),
                    res.to(device),
                    impair.to(device)
                )

                mp, tl, rl, il = model(X)

                mos_true.extend(mos.cpu().numpy())
                mos_pred.extend(mp.cpu().numpy())

                tier_true.extend(tier.cpu().numpy())
                tier_pred.extend(torch.argmax(tl, dim=1).cpu().numpy())

                res_true.extend(res.cpu().numpy())
                res_pred.extend((torch.sigmoid(rl) > 0.5).cpu().numpy())

                impair_true.extend(impair.cpu().numpy())
                impair_pred.extend(torch.argmax(il, dim=1).cpu().numpy())

        print(
            f"Epoch {epoch:02d} | "
            f"MOS RMSE={np.sqrt(mean_squared_error(mos_true, mos_pred)):.3f} | "
            f"Tier Acc={accuracy_score(tier_true, tier_pred):.3f} | "
            f"Res Acc={accuracy_score(res_true, res_pred):.3f} | "
            f"Impair Acc={accuracy_score(impair_true, impair_pred):.3f}"
        )

    # Save model
    torch.save({"model": model.state_dict()}, MODEL_PATH)
    print("\nSaved model:", MODEL_PATH)


# ============================================================

if __name__ == "__main__":
    train_model()
