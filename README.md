# 🚀 WebRTC QoE Prediction — Execution Instructions

This document provides the **exact, verified commands** required to preprocess data, train models, evaluate performance, generate WebRTC data, and run the real-time MOS/QoE inference server for the WebRTC QoE Prediction project.

All commands reflect your **actual workflow and command history**, ensuring reproducibility.

---

# 1. Environment Setup (Windows)

## Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

## Install dependencies
```bash
pip install -r requirements.txt
```

---

# 2. 🔴 Data Generation — Collecting WebRTC getStats Metrics (In case you want to genereate new session data)

You can generate raw WebRTC metrics using the built-in **WebRTC data collection script**.

## A. Start a local server for the data collection page
```bash
python -m http.server 8000
```

## B. Open the WebRTC collector in your browser
```
http://localhost:8000/index.html
```

## C. Start the WebRTC call
The script (`webrtc_collect.js`) will:

- Create a WebRTC loopback call
- Collect `RTCPeerConnection.getStats()` snapshots
- Record bitrate, jitter, fps, RTT, frame drops, etc.

## D. Download collected session data
Click **Download JSON** (if implemented), or manually save logs.

Place the exported file into:

```
data/raw/your_session_name.json
```

Example record:
```json
{
  "timestampSec": 1712345678,
  "metrics": {
    "sendBitrateKbps": 820,
    "recvBitrateKbps": 750,
    "fpsRecv": 28,
    "jitterMs": 4,
    "rttMs": 32,
    "framesDropped": 1,
    "framesDecoded": 300,
    "jitterBufferMs": 6
  }
}
```

---

# 3. 📊 Data Preprocessing

Convert raw WebRTC logs into engineered features + MOS:

```bash
python src\preprocess_mos.py
```

Outputs:

```
data/processed/webrtc_qoe_dataset.csv
```

This includes:

- Bitrate features  
- Frame stats  
- Loss/jitter/RTT  
- MOS heuristic value  
- QoE tier  

---

# 4. 🎯 Model Training and evaluation

## Train the XGBoost MOS regressor
```bash
python src\mos_xgb_train.py
```

Produces:

- `models/xgb_session_model.json`

## (Optional) Train the MLP model
```bash
python src\mos_mlp_train.py
```

---


# 6. ⚡ Start Real-Time Inference Backend (FastAPI)

```bash
uvicorn src.server:app --reload --port 8000
```

Backend endpoint:

### **POST `/predict`**
Returns:

- MOS  
- QoE tier  
- Engineered features  

---

# 7. 🌐 Frontend Web Application (Live QoE Demo)

Run a static server:

```bash
cd webapp
python -m http.server 8080
```

Open:

```
http://localhost:8080/index.html
```

The frontend will:

- Start a WebRTC loop  
- Call `/predict` continuously  
- Display **live MOS & QoE tier**  

---

# 8. ✅ Command Summary

| Task | Command |
|------|---------|
| Activate environment | `venv\Scripts\activate` |
| Install dependencies | `pip install -r requirements.txt` |
| Generate WebRTC data | `python -m http.server` + browser session |
| Preprocess data | `python src\preprocess_mos.py` |
| Train XGBoost model | `python src\mos_xgb_train.py` |
| Train MLP | `python src\mos_mlp_train.py` |
| Start backend | `uvicorn src.server:app --reload --port 8000` |
| Frontend live QoE | ` cd webapp python -m http.server 8080` |

---

# 9. 📌 Notes

- Generate *multiple* network impairment scenarios for better dataset diversity.
- Ensure backend is running before testing the live frontend.
- `webrtc_collect.js` must compute bitrate using delta bytes to avoid noisy readings.

