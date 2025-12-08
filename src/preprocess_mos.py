import os
import json
import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
OUT_PATH = "data/processed/webrtc_qoe_dataset.csv"


def safe_get(d, key, default=None):
    return d.get(key, default) if d is not None else default


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def process_session(json_data, session_name):
    rows = []
    for snap in json_data:
        ts = snap.get("timestampSec", None)
        m = snap.get("metrics", {})

        row = {
            "session": session_name,
            "timestamp": ts,

            "sendBitrateKbps": safe_get(m, "sendBitrateKbps"),
            "recvBitrateKbps": safe_get(m, "recvBitrateKbps"),
            "fpsRecv": safe_get(m, "fpsRecv"),
            "jitterMs": safe_get(m, "jitterMs"),
            "rttMs": safe_get(m, "rttMs"),
            "packetsLost": safe_get(m, "packetsLost"),
            "packetsReceived": safe_get(m, "packetsReceived"),
            "width": safe_get(m, "width"),
            "height": safe_get(m, "height"),
            "framesDecoded": safe_get(m, "framesDecoded"),
            "framesDropped": safe_get(m, "framesDropped"),
            "jitterBufferMs": safe_get(m, "jitterBufferMs"),
        }

        w, h = row["width"], row["height"]
        row["pixelArea"] = (w * h) if (w and h) else 0
        rows.append(row)

    return rows


# ============================================================
# MOS HEURISTIC — mos_det
# ============================================================

def mos_det(row):
    # Extract raw metrics with safe defaults
    rb   = row["recvBitrateKbps"] or 0
    fps  = row["fpsRecv"] or 0
    jit  = row["jitterMs"] or 0
    rtt  = row["rttMs"] or 0
    jbuf = row["jitterBufferMs"] or 0
    drops = row["framesDropped"] or 0
    lost  = row["packetsLost"] or 0
    recv  = row["packetsReceived"] or 0

    # Logistic bitrate-to-MOS mapping (saturates at higher bitrates)
    mos_br = 4.8 / (1.0 + np.exp(-(rb - 500.0) / 250.0))

    # Impairment penalties for motion, jitter, RTT, and freeze indicators
    fps_norm = min(fps / 30.0, 1.0)
    fpsPenalty = 0.6 * (1.0 - fps_norm) * 0.8
    jitPenalty = 0.35 * (1.0 - np.exp(-jit / 60.0)) * 0.8
    rttPenalty = 0.35 * (1.0 - np.exp(-rtt / 200.0)) * 0.8
    freezePenalty = min(0.7, 0.01 * drops + 0.002 * jbuf) * 0.8

    # MOS contribution from bitrate after subtracting impairments
    mos_from_bitrate = (
        mos_br
        - fpsPenalty
        - jitPenalty
        - rttPenalty
        - freezePenalty
    )

    # Quality functions for individual dimensions
    if fps <= 10:
        q_fps = 0.0
    elif fps >= 30:
        q_fps = 1.0
    else:
        q_fps = (fps - 10.0) / 20.0

    if jit <= 5:
        q_jitter = 1.0
    elif jit >= 30:
        q_jitter = 0.0
    else:
        q_jitter = 1.0 - (jit - 5.0) / 25.0

    if rtt <= 50:
        q_rtt = 1.0
    elif rtt >= 250:
        q_rtt = 0.0
    else:
        q_rtt = 1.0 - (rtt - 50.0) / 200.0

    # Loss and drop-based smoothness indicators
    loss_rate = lost / (lost + recv + 1e-6)
    q_loss = 1.0 - min(loss_rate / 0.05, 1.0)

    total_frames = (row["framesDecoded"] or 0) + drops
    drop_rate = drops / (total_frames + 1e-6)
    q_drop = 1.0 - min(drop_rate / 0.10, 1.0)

    # Jitter buffer contribution to freeze likelihood
    if jbuf <= 20:
        q_jbuf = 1.0
    elif jbuf >= 150:
        q_jbuf = 0.0
    else:
        q_jbuf = 1.0 - (jbuf - 20.0) / 130.0

    # Aggregate smoothness and network quality sub-scores
    q_smooth  = 0.4 * q_fps + 0.6 * q_drop
    q_network = 0.45 * q_jitter + 0.30 * q_rtt + 0.25 * q_loss
    q_freeze  = 0.5 * q_drop + 0.5 * q_jbuf

    # Combined non-bitrate QoE score
    q_other = 0.4 * q_smooth + 0.4 * q_network + 0.2 * q_freeze
    mos_from_other = 1.0 + 4.0 * q_other

    # Weighted blend between bitrate MOS and non-bitrate MOS
    alpha = 0.50
    mos = alpha * mos_from_bitrate + (1.0 - alpha) * mos_from_other

    # Final bounded MOS score
    return float(np.clip(mos, 1.0, 4.8))

def mos_to_qoe_tier(mos):
    if mos < 2.5:
        return 0
    if mos < 3.5:
        return 1
    return 2


# ============================================================
# PREPROCESS PIPELINE
# ============================================================

def preprocess():
    all_rows = []

    print(f"Loading JSON from: {RAW_DIR}")
    for fname in os.listdir(RAW_DIR):
        if fname.endswith(".json"):
            try:
                data = load_json(os.path.join(RAW_DIR, fname))
                all_rows.extend(process_session(data, fname))
            except Exception as e:
                print("Error:", fname, e)

    df = pd.DataFrame(all_rows)

    metric_cols = [
        "sendBitrateKbps", "recvBitrateKbps", "fpsRecv",
        "jitterMs", "rttMs", "packetsLost", "packetsReceived",
        "width", "height", "pixelArea",
        "framesDecoded", "framesDropped", "jitterBufferMs",
    ]
    df[metric_cols] = df[metric_cols].fillna(0)

    df["bitrate_per_pixel"] = df.apply(
        lambda r: (r["recvBitrateKbps"] / r["pixelArea"]) if r["pixelArea"] > 0 else 0,
        axis=1
    )

    df["recvBitrateDiff"] = df.groupby("session")["recvBitrateKbps"].diff().fillna(0)
    df["jitterDiff"] = df.groupby("session")["jitterMs"].diff().fillna(0)
    df["rttDiff"] = df.groupby("session")["rttMs"].diff().fillna(0)
    df["fpsRecvDiff"] = df.groupby("session")["fpsRecv"].diff().fillna(0)

    df["jitter_roll_std_3"] = df.groupby("session")["jitterMs"].transform(lambda s: s.rolling(3).std()).fillna(0)
    df["bitrate_roll_std_3"] = df.groupby("session")["recvBitrateKbps"].transform(lambda s: s.rolling(3).std()).fillna(0)

    df["mos"] = df.apply(mos_det, axis=1)
    df["qoe_tier"] = df["mos"].apply(mos_to_qoe_tier)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nSaved processed dataset to {OUT_PATH}")
    print("\nMOS stats:")
    print(df["mos"].describe())


    # ============================================================
    # ANALYTICS SUMMARY (MINIMAL)
    # ============================================================

    print("\n================ QoE TIER DISTRIBUTION ================")
    print(df["qoe_tier"].value_counts())

    print("\n================ CORRELATION WITH MOS ================")
    corr = (
        df[
            [
                "recvBitrateKbps", "sendBitrateKbps", "fpsRecv",
                "jitterMs", "rttMs", "packetsLost", "framesDropped",
                "bitrate_per_pixel", "jitterBufferMs"
            ]
        ]
        .corrwith(df["mos"])
        .sort_values(ascending=False)
    )
    print(corr)

    print("\n================ PER-SESSION MOS STATS ================")
    session_stats = df.groupby("session")["mos"].agg(["mean", "min", "max", "std"])
    print(session_stats)


if __name__ == "__main__":
    preprocess()
