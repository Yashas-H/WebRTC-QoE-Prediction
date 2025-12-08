import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

DATASET_PATH = "data/processed/webrtc_qoe_dataset.csv"


def load_data():
    df = pd.read_csv(DATASET_PATH)

    # --- Feature columns ---
    feature_cols = [
        # Standard metrics
        "sendBitrateKbps", "recvBitrateKbps", "fpsSend", "fpsRecv",
        "jitterMs", "rttMs", "packetsLost", "packetsReceived",
        "width", "height", "pixelArea",

        # Chrome extended metrics
        "framesDecoded", "framesDropped", "decodeMs", "jitterBufferMs",
        "frameRateDecoded", "frameRateOutput", "currentDelayMs",
        "nackCount", "pliCount", "firCount", "qpSumInbound",
        "framesEncoded", "encodeUsagePercent", "frameRateInput",
        "frameRateSent", "qpSumOutbound", "availableSendBandwidth",
        "availableRecvBandwidth", "targetEncBitrate", "actualEncBitrate",
        "retransmitBitrate", "bucketDelay",

        # Diff features
        "recvBitrateDiff", "jitterDiff", "rttDiff"
    ]

    X = df[feature_cols].fillna(0)
    y_tier = df["qoe_tier"]

    return X, y_tier, feature_cols


def train_qoe_classifier(X, y, feature_cols):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining QoE classifier...")

    clf = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3
    )

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    print("\n=== QoE Tier Classifier Results ===")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # Feature importances
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print("\nTop Feature Importances:")
    for i in sorted_idx[:20]:
        print(f"{feature_cols[i]}: {importances[i]:.4f}")

    clf.save_model("models/qoe_classifier_v2.json")
    print("\nSaved classifier to models/qoe_classifier_v2.json")


def main():
    print("Loading dataset...")
    X, tier, feature_cols = load_data()
    print("Dataset loaded. Rows:", len(X))

    train_qoe_classifier(X, tier, feature_cols)


if __name__ == "__main__":
    main()
