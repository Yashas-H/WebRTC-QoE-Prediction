import json
import os
import pandas as pd
import numpy as np

RAW_DIR = "data/raw"
OUT_CSV = "data/processed/webrtc_qoe_dataset.csv"


def safe_get(d, key, default=None):
    return d.get(key, default) if d is not None else default


def load_json_file(path):
    with open(path, "r") as f:
        return json.load(f)


def process_session(json_data, session_name):
    rows = []

    for snap in json_data:
        t = snap.get("timestampSec", None)
        metrics = snap.get("metrics", {})

        row = {
            "session": session_name,
            "timestamp": t,

            # Standard WebRTC Metrics
            "sendBitrateKbps": safe_get(metrics, "sendBitrateKbps"),
            "recvBitrateKbps": safe_get(metrics, "recvBitrateKbps"),
            "fpsSend": safe_get(metrics, "fpsSend"),
            "fpsRecv": safe_get(metrics, "fpsRecv"),
            "jitterMs": safe_get(metrics, "jitterMs"),
            "rttMs": safe_get(metrics, "rttMs"),
            "packetsLost": safe_get(metrics, "packetsLost"),
            "packetsReceived": safe_get(metrics, "packetsReceived"),
            "width": safe_get(metrics, "width"),
            "height": safe_get(metrics, "height"),

            # Chrome-specific extended metrics
            "framesDecoded": safe_get(metrics, "framesDecoded"),
            "framesDropped": safe_get(metrics, "framesDropped"),
            "decodeMs": safe_get(metrics, "decodeMs"),
            "jitterBufferMs": safe_get(metrics, "jitterBufferMs"),
            "frameRateDecoded": safe_get(metrics, "frameRateDecoded"),
            "frameRateOutput": safe_get(metrics, "frameRateOutput"),
            "currentDelayMs": safe_get(metrics, "currentDelayMs"),
            "nackCount": safe_get(metrics, "nackCount"),
            "pliCount": safe_get(metrics, "pliCount"),
            "firCount": safe_get(metrics, "firCount"),
            "qpSumInbound": safe_get(metrics, "qpSumInbound"),
            "framesEncoded": safe_get(metrics, "framesEncoded"),
            "encodeUsagePercent": safe_get(metrics, "encodeUsagePercent"),
            "frameRateInput": safe_get(metrics, "frameRateInput"),
            "frameRateSent": safe_get(metrics, "frameRateSent"),
            "qpSumOutbound": safe_get(metrics, "qpSumOutbound"),

            # Candidate-pair metrics
            "availableSendBandwidth": safe_get(metrics, "availableSendBandwidth"),
            "availableRecvBandwidth": safe_get(metrics, "availableRecvBandwidth"),
            "targetEncBitrate": safe_get(metrics, "targetEncBitrate"),
            "actualEncBitrate": safe_get(metrics, "actualEncBitrate"),
            "retransmitBitrate": safe_get(metrics, "retransmitBitrate"),
            "bucketDelay": safe_get(metrics, "bucketDelay"),
        }

        # Derived metrics
        w = row["width"]
        h = row["height"]
        row["pixelArea"] = w * h if w and h else None

        rows.append(row)

    return rows


def assign_qoe_label(df):
    """
    Assign QoE tier: 0 = Poor, 1 = Fair, 2 = Good
    """
    qoe_labels = []

    for _, r in df.iterrows():
        # Basic heuristics combining multiple symptoms
        if r["recvBitrateKbps"] is not None and r["recvBitrateKbps"] < 150:
            qoe_labels.append(0)
        elif r["fpsRecv"] is not None and r["fpsRecv"] < 15:
            qoe_labels.append(0)
        elif r["jitterBufferMs"] is not None and r["jitterBufferMs"] > 100:
            qoe_labels.append(0)
        elif r["rttMs"] is not None and r["rttMs"] > 300:
            qoe_labels.append(0)
        elif r["recvBitrateKbps"] is not None and r["recvBitrateKbps"] < 400:
            qoe_labels.append(1)
        elif r["fpsRecv"] is not None and r["fpsRecv"] < 24:
            qoe_labels.append(1)
        else:
            qoe_labels.append(2)

    df["qoe_tier"] = qoe_labels
    return df


def preprocess():
    all_rows = []

    print(f"Loading JSON from {RAW_DIR}...")

    for fname in os.listdir(RAW_DIR):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(RAW_DIR, fname)
        print("Processing:", fname)

        try:
            data = load_json_file(path)
            rows = process_session(data, fname)
            all_rows.extend(rows)
        except Exception as e:
            print("Error processing", fname, "->", e)

    df = pd.DataFrame(all_rows)
    print("Loaded rows:", len(df))

    # Drop rows missing essential metrics
    df = df.dropna(subset=["recvBitrateKbps", "fpsRecv", "jitterMs", "rttMs"])

    # Fill non-essential Chrome metrics with 0
    chrome_cols = [
        "framesDecoded", "framesDropped", "decodeMs", "jitterBufferMs",
        "frameRateDecoded", "frameRateOutput", "currentDelayMs",
        "nackCount", "pliCount", "firCount", "qpSumInbound",
        "framesEncoded", "encodeUsagePercent", "frameRateInput",
        "frameRateSent", "qpSumOutbound", "availableSendBandwidth",
        "availableRecvBandwidth", "targetEncBitrate", "actualEncBitrate",
        "retransmitBitrate", "bucketDelay"
    ]
    df[chrome_cols] = df[chrome_cols].fillna(0)

    # Derived diff features
    df["recvBitrateDiff"] = df["recvBitrateKbps"].diff().fillna(0)
    df["jitterDiff"] = df["jitterMs"].diff().fillna(0)
    df["rttDiff"] = df["rttMs"].diff().fillna(0)

    # Assign QoE labels
    df = assign_qoe_label(df)

    # Save
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print("Saved processed dataset to", OUT_CSV)


if __name__ == "__main__":
    preprocess()
