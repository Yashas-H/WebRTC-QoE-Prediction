// webrtc_collect.js

let pc1, pc2;
let localStream;
let statsInterval = null;
let sessionStats = [];
let sessionStartTime = null;

const logEl = document.getElementById("log");
function log(msg) {
  console.log(msg);
  logEl.textContent += msg + "\n";
}

// Track previous metrics for bitrate calculations
let prevOutbound = null;
let prevInbound = null;

/**************************************************
 * CREATE REMOTE VIDEO ELEMENT EARLY
 **************************************************/
const remoteVideo = document.createElement("video");
remoteVideo.autoplay = true;
remoteVideo.muted = true;
remoteVideo.style.width = "320px";
remoteVideo.style.border = "1px solid #ccc";
remoteVideo.style.display = "block";
remoteVideo.style.marginTop = "10px";
document.body.appendChild(remoteVideo);

/**************************************************
 * START CALL
 **************************************************/
async function startCall() {
  try {
    log("Starting call...");

    document.getElementById("stopBtn").disabled = false;
    document.getElementById("startBtn").disabled = true;

    // Request camera
    localStream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false
    });

    pc1 = new RTCPeerConnection();
    pc2 = new RTCPeerConnection();

    // Send local tracks from pc1 → pc2
    localStream.getTracks().forEach(track => pc1.addTrack(track, localStream));

    pc1.onicecandidate = e => e.candidate && pc2.addIceCandidate(e.candidate);
    pc2.onicecandidate = e => e.candidate && pc1.addIceCandidate(e.candidate);

    pc2.ontrack = event => {
      log("Remote track received, attaching stream...");
      remoteVideo.srcObject = event.streams[0];
    };

    // Offer/Answer exchange
    const offer = await pc1.createOffer();
    await pc1.setLocalDescription(offer);
    await pc2.setRemoteDescription(offer);

    const answer = await pc2.createAnswer();
    await pc2.setLocalDescription(answer);
    await pc1.setRemoteDescription(answer);

    // Reset stats
    sessionStats = [];
    prevOutbound = null;
    prevInbound = null;
    sessionStartTime = performance.now();

    // Delay before polling to ensure stats appear
    setTimeout(() => {
      statsInterval = setInterval(collectStats, 1000);
      log("Started stats polling...");
    }, 300);

    log("Call started, waiting for stats...");
  } catch (err) {
    log("Error starting call: " + err);
    document.getElementById("startBtn").disabled = false;
    document.getElementById("stopBtn").disabled = true;
  }
}

/**************************************************
 * COLLECT METRICS (pc1 = outbound, pc2 = inbound)
 **************************************************/
async function collectStats() {
  if (!pc1 || !pc2) return;

  const now = performance.now();
  const elapsedSec = (now - sessionStartTime) / 1000.0;

  let outbound = null;
  let inbound = null;
  let candidate = null;

  // 1) Get outbound + candidate from pc1 (sender)
  const report1 = await pc1.getStats(null);
  report1.forEach(stat => {
    if (stat.type === "outbound-rtp" && stat.kind === "video") {
      outbound = stat;
    }
    if (stat.type === "candidate-pair" && stat.nominated) {
      candidate = stat;
    }
  });

  // 2) Get inbound + candidate from pc2 (receiver)
  const report2 = await pc2.getStats(null);
  report2.forEach(stat => {
    if (stat.type === "inbound-rtp" && stat.kind === "video") {
      inbound = stat;
    }
    if (!candidate && stat.type === "candidate-pair" && stat.nominated) {
      candidate = stat;
    }
  });

  if (!inbound || !candidate) {
    // outbound is optional; inbound + candidate are required
    log(
      "Missing stats: inbound=" +
        !!inbound +
        " candidate=" +
        !!candidate +
        " (outbound=" +
        !!outbound +
        ")"
    );
    return;
  }

  // BITRATE CALCULATIONS
  let sendBitrate = null;
  let recvBitrate = null;

  if (outbound && prevOutbound) {
    const dt = elapsedSec - prevOutbound.timestampSec;
    if (dt > 0) {
      sendBitrate =
        ((outbound.bytesSent - prevOutbound.bytesSent) * 8) / dt / 1000;
    }
  }

  if (prevInbound) {
    const dt = elapsedSec - prevInbound.timestampSec;
    if (dt > 0) {
      recvBitrate =
        ((inbound.bytesReceived - prevInbound.bytesReceived) * 8) / dt / 1000;
    }
  }

  /**************************************************
   * CHROME-SPECIFIC / EXTENDED METRICS (SAFE)
   **************************************************/
  const framesDecoded = inbound.framesDecoded ?? null;
  const framesDropped = inbound.framesDropped ?? null;
  const decodeMs = inbound.googDecodeMs ?? null;
  const jitterBufferMs = inbound.googJitterBufferMs ?? null;
  const frameRateDecoded = inbound.googFrameRateDecoded ?? null;
  const frameRateOutput = inbound.googFrameRateOutput ?? null;
  const currentDelayMs = inbound.googCurrentDelayMs ?? null;
  const nackCount = inbound.nackCount ?? null;
  const pliCount = inbound.pliCount ?? null;
  const firCount = inbound.firCount ?? null;
  const qpSumInbound = inbound.qpSum ?? null;

  let framesEncoded = null;
  let encodeUsagePercent = null;
  let frameRateInput = null;
  let frameRateSent = null;
  let qpSumOutbound = null;

  if (outbound) {
    framesEncoded = outbound.framesEncoded ?? null;
    encodeUsagePercent = outbound.googEncodeUsagePercent ?? null;
    frameRateInput = outbound.googFrameRateInput ?? null;
    frameRateSent = outbound.googFrameRateSent ?? null;
    qpSumOutbound = outbound.qpSum ?? null;
  }

  const availableSendBandwidth =
    candidate.availableOutgoingBitrate ??
    candidate.googAvailableSendBandwidth ??
    null;
  const availableRecvBandwidth =
    candidate.availableIncomingBitrate ??
    candidate.googAvailableReceiveBandwidth ??
    null;

  const targetEncBitrate = candidate.googTargetEncBitrate ?? null;
  const actualEncBitrate = candidate.googActualEncBitrate ?? null;
  const retransmitBitrate = candidate.googRetransmitBitrate ?? null;
  const bucketDelay = candidate.googBucketDelay ?? null;

  const snapshot = {
    timestampSec: elapsedSec,
    metrics: {
      // existing metrics
      sendBitrateKbps: sendBitrate,
      recvBitrateKbps: recvBitrate,
      fpsSend: outbound ? outbound.framesPerSecond || null : null,
      fpsRecv: inbound.framesPerSecond || null,
      jitterMs: inbound.jitter ? inbound.jitter * 1000 : null,
      rttMs: candidate.currentRoundTripTime
        ? candidate.currentRoundTripTime * 1000
        : null,
      packetsLost: inbound.packetsLost || 0,
      packetsReceived: inbound.packetsReceived || 0,
      width: inbound.frameWidth || null,
      height: inbound.frameHeight || null,

      // extended Chrome / internal metrics
      framesDecoded,
      framesDropped,
      decodeMs,
      jitterBufferMs,
      frameRateDecoded,
      frameRateOutput,
      currentDelayMs,
      nackCount,
      pliCount,
      firCount,
      qpSumInbound,

      framesEncoded,
      encodeUsagePercent,
      frameRateInput,
      frameRateSent,
      qpSumOutbound,

      availableSendBandwidth,
      availableRecvBandwidth,
      targetEncBitrate,
      actualEncBitrate,
      retransmitBitrate,
      bucketDelay
    }
  };

  sessionStats.push(snapshot);
  log("Snapshot added. Total snapshots: " + sessionStats.length);

  if (outbound) {
    prevOutbound = { ...outbound, timestampSec: elapsedSec };
  }
  prevInbound = { ...inbound, timestampSec: elapsedSec };
}

/**************************************************
 * STOP CALL + DOWNLOAD JSON
 **************************************************/
async function stopCall() {
  log("Stopping call...");

  document.getElementById("stopBtn").disabled = true;
  document.getElementById("startBtn").disabled = false;

  if (statsInterval) {
    clearInterval(statsInterval);
    statsInterval = null;
  }

  if (pc1) pc1.close();
  if (pc2) pc2.close();

  if (localStream) {
    localStream.getTracks().forEach(t => t.stop());
  }

  if (!sessionStats.length) {
    log("No stats collected!");
    alert("No stats collected!");
    return;
  }

  log("Downloading stats...");

  const blob = new Blob([JSON.stringify(sessionStats, null, 2)], {
    type: "application/json"
  });

  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const ts = new Date().toISOString().replace(/[:.]/g, "-");

  a.href = url;
  a.download = `webrtc_session_${ts}.json`;
  a.click();

  URL.revokeObjectURL(url);

  log("Stats downloaded.");
}

/**************************************************
 * BUTTON LISTENERS
 **************************************************/
document.getElementById("startBtn").onclick = startCall;
document.getElementById("stopBtn").onclick = stopCall;