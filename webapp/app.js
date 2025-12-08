// =========================================================
// GLOBALS
// =========================================================
let pc1 = null; // sender
let pc2 = null; // receiver
let localStream = null;
let statsInterval = null;

const UPDATE_INTERVAL_MS = 250;

// For bitrate calculation
let prevOutbound = null;
let prevInbound = null;

// =========================================================
// DOM
// =========================================================
const localVideo = document.getElementById("localVideo");
const remoteVideo = document.getElementById("remoteVideo");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const mosValueEl = document.getElementById("mosValue");
const tierValueEl = document.getElementById("tierValue");

// Log panel
let logEl = document.getElementById("log");
if (!logEl) {
    logEl = document.createElement("pre");
    logEl.id = "log";
    logEl.style.textAlign = "left";
    logEl.style.maxHeight = "280px";
    logEl.style.overflowY = "auto";
    logEl.style.background = "#111";
    logEl.style.color = "#0f0";
    logEl.style.padding = "10px";
    logEl.style.fontSize = "12px";
    logEl.style.margin = "20px auto";
    logEl.style.width = "90%";
    document.body.appendChild(logEl);
}

// =========================================================
// LOG FUNCTION
// =========================================================
function log(msg, obj) {
    const time = new Date().toISOString().split("T")[1].split(".")[0];
    const line = `[${time}] ${msg}`;
    console.log(line, obj || "");
    logEl.textContent += line + (obj ? " " + JSON.stringify(obj) : "") + "\n";
    logEl.scrollTop = logEl.scrollHeight;
}

// =========================================================
// QoE TIER
// =========================================================
function computeTier(mos) {
    if (mos >= 3.5) return "Good";
    if (mos >= 2.5) return "Fair";
    return "Poor";
}

function updateMOSUI(mos) {
    mosValueEl.textContent = `MOS: ${mos.toFixed(2)}`;
    const tier = computeTier(mos);
    tierValueEl.textContent = `Tier: ${tier}`;

    mosValueEl.style.color =
        tier === "Good" ? "#27ae60" :
        tier === "Fair" ? "#f1c40f" :
        "#e74c3c";
}

// =========================================================
// BACKEND CALL
// =========================================================
async function sendToModel(features) {
    try {
        log("Sending features:", features);

        const resp = await fetch("http://localhost:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ features })
        });

        const data = await resp.json();
        log("Backend response:", data);

        if (typeof data.mos === "number") {
            updateMOSUI(data.mos);
        }
    } catch (err) {
        log("❌ Backend error", err);
    }
}

// =========================================================
// BITRATE HELPER
// =========================================================
function computeKbps(prev, curBytes, curTsMs) {
    if (!prev || curBytes < prev.bytes || curTsMs <= prev.ts) {
        return { kbps: 0, state: { bytes: curBytes, ts: curTsMs } };
    }
    const deltaBytes = curBytes - prev.bytes;
    const deltaMs = curTsMs - prev.ts;
    return {
        kbps: (8 * deltaBytes) / deltaMs,
        state: { bytes: curBytes, ts: curTsMs }
    };
}

// =========================================================
// MAIN STATS LOOP (pc1 + pc2)
// =========================================================
async function startStatsLoop() {
    if (statsInterval) clearInterval(statsInterval);

    log(`📊 Starting stats loop @ ${UPDATE_INTERVAL_MS}ms`);

    statsInterval = setInterval(async () => {
        if (!pc1 || !pc2) return;

        const sendStats = await pc1.getStats(); // outbound + RTT
        const recvStats = await pc2.getStats(); // inbound + resolution

        let f = {
            sendBitrateKbps: 0,
            recvBitrateKbps: 0,
            fpsRecv: 0,
            jitterMs: 0,
            rttMs: 0,
            packetsLost: 0,
            packetsReceived: 0,
            width: 0,
            height: 0,
            pixelArea: 0,
            framesDecoded: 0,
            framesDropped: 0,
            jitterBufferMs: 0,
            bitrate_per_pixel: 0
        };

        // ===============================
        // OUTBOUND RTP (pc1)
        // ===============================
        sendStats.forEach(report => {
            if (report.type === "outbound-rtp" && report.kind === "video") {
                const { kbps, state } = computeKbps(
                    prevOutbound,
                    report.bytesSent,
                    report.timestamp
                );
                f.sendBitrateKbps = kbps;
                prevOutbound = state;
            }

            if (report.type === "remote-inbound-rtp") {
                f.rttMs = (report.roundTripTime || 0) * 1000;

                if (report.jitterBufferDelay != null &&
                    report.jitterBufferEmittedCount > 0) {
                    f.jitterBufferMs =
                        (report.jitterBufferDelay / report.jitterBufferEmittedCount) * 1000;
                }
            }
        });

        // ===============================
        // INBOUND RTP (pc2)
        // ===============================
        recvStats.forEach(report => {
            if (report.type === "inbound-rtp" && report.kind === "video") {
                f.packetsLost = report.packetsLost || 0;
                f.packetsReceived = report.packetsReceived || 0;
                f.jitterMs = (report.jitter || 0) * 1000;
                f.framesDecoded = report.framesDecoded || 0;
                f.framesDropped = report.framesDropped || 0;
                f.fpsRecv = report.framesPerSecond || 0;

                const { kbps, state } = computeKbps(
                    prevInbound,
                    report.bytesReceived,
                    report.timestamp
                );
                f.recvBitrateKbps = kbps;
                prevInbound = state;
            }

            if (report.type === "track" && report.kind === "video") {
                f.width = report.frameWidth;
                f.height = report.frameHeight;
                f.pixelArea = f.width * f.height;
            }
        });

        f.bitrate_per_pixel =
            f.pixelArea > 0 ? f.recvBitrateKbps / f.pixelArea : 0;

        log("Computed features:", f);
        sendToModel(f);

    }, UPDATE_INTERVAL_MS);
}

// =========================================================
// START CALL
// =========================================================
async function startCall() {
    log("▶️ Start Call clicked");

    startBtn.disabled = true;
    stopBtn.disabled = false;

    try {
        log("Requesting camera...");
        localStream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false
        });

        log("Camera acquired");
        localVideo.srcObject = localStream;

        pc1 = new RTCPeerConnection();
        pc2 = new RTCPeerConnection();

        log("pc1 + pc2 created");

        pc1.onicecandidate = e => e.candidate && pc2.addIceCandidate(e.candidate);
        pc2.onicecandidate = e => e.candidate && pc1.addIceCandidate(e.candidate);

        pc2.ontrack = event => {
            log("🎥 pc2 received track");
            remoteVideo.srcObject = event.streams[0];
        };

        localStream.getTracks().forEach(track => {
            log("Adding track:", track.kind);
            pc1.addTrack(track, localStream);
        });

        const offer = await pc1.createOffer();
        await pc1.setLocalDescription(offer);
        await pc2.setRemoteDescription(offer);

        const answer = await pc2.createAnswer();
        await pc2.setLocalDescription(answer);
        await pc1.setRemoteDescription(answer);

        log("✅ WebRTC connection established");

        prevInbound = null;
        prevOutbound = null;

        startStatsLoop();

    } catch (err) {
        log("❌ Start call error:", err);
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }
}

// =========================================================
// STOP CALL
// =========================================================
function stopCall() {
    log("⏹ Stopping call");

    if (statsInterval) clearInterval(statsInterval);

    if (pc1) pc1.close();
    if (pc2) pc2.close();

    pc1 = pc2 = null;

    if (localStream) {
        localStream.getTracks().forEach(t => t.stop());
        localStream = null;
    }

    localVideo.srcObject = null;
    remoteVideo.srcObject = null;

    mosValueEl.textContent = "MOS: --";
    tierValueEl.textContent = "Tier: --";

    startBtn.disabled = false;
    stopBtn.disabled = true;
}

// =========================================================
// BUTTON HANDLERS
// =========================================================
startBtn.addEventListener("click", startCall);
stopBtn.addEventListener("click", stopCall);
