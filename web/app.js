const statusEl = document.getElementById("status");
const activeSpeakerEl = document.getElementById("activeSpeaker");
const latencyEl = document.getElementById("latency");
const rtfEl = document.getElementById("rtf");
const transcriptEl = document.getElementById("transcript");
const lineCountEl = document.getElementById("lineCount");
const languageInput = document.getElementById("language");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const clearBtn = document.getElementById("clearBtn");

let ws = null;
let mediaStream = null;
let audioContext = null;
let sourceNode = null;
let processorNode = null;
let muteGain = null;
let totalLines = 0;

// ============================================================
// UI helpers
// ============================================================
function setStatus(text, isRecording = false) {
  statusEl.textContent = text;
  statusEl.className = isRecording ? "status-value status-recording" : "status-value";
}

function formatMs(ms) {
  const total = Math.floor(ms / 1000);
  const m = String(Math.floor(total / 60)).padStart(2, "0");
  const s = String(total % 60).padStart(2, "0");
  return `${m}:${s}`;
}

function getSpeakerClass(speaker) {
  const match = speaker.match(/Speaker\s+(\d+)/i);
  const idx = match ? match[1] : "0";
  return `speaker-${idx}`;
}

function addTranscriptLine(item) {
  const line = document.createElement("div");
  line.className = "line";

  const content = document.createElement("div");
  content.className = "line-content";

  const meta = document.createElement("div");
  meta.className = "line-meta";

  const speakerBadge = document.createElement("span");
  speakerBadge.className = `speaker-badge ${getSpeakerClass(item.speaker)}`;
  speakerBadge.textContent = item.speaker;

  const timestamp = document.createElement("span");
  timestamp.className = "timestamp";
  timestamp.textContent = `[${formatMs(item.start_ms)} - ${formatMs(item.end_ms)}]`;

  meta.appendChild(speakerBadge);
  meta.appendChild(timestamp);

  const text = document.createElement("div");
  text.className = "line-text";
  text.textContent = item.text;

  content.appendChild(meta);
  content.appendChild(text);
  line.appendChild(content);

  transcriptEl.appendChild(line);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;

  totalLines += 1;
  lineCountEl.textContent = `${totalLines} dòng`;
}

function updateLatency(ms) {
  latencyEl.textContent = `${ms}ms`;
  // Color coding: green < 1s, yellow < 3s, red > 3s
  if (ms < 1000) {
    latencyEl.style.color = "var(--ok)";
  } else if (ms < 3000) {
    latencyEl.style.color = "var(--warning)";
  } else {
    latencyEl.style.color = "var(--danger)";
  }
}

function updateRTF(rtf) {
  rtfEl.textContent = `${rtf}x`;
  if (rtf >= 1) {
    rtfEl.style.color = "var(--ok)";
  } else {
    rtfEl.style.color = "var(--warning)";
  }
}

function clearTranscript() {
  transcriptEl.innerHTML = "";
  totalLines = 0;
  lineCountEl.textContent = "0 dòng";
}

// ============================================================
// Audio processing
// ============================================================
function downsampleBuffer(buffer, inputSampleRate, outputSampleRate) {
  if (outputSampleRate === inputSampleRate) {
    return buffer;
  }

  const sampleRateRatio = inputSampleRate / outputSampleRate;
  const newLength = Math.round(buffer.length / sampleRateRatio);
  const result = new Float32Array(newLength);

  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
    let accum = 0;
    let count = 0;

    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i += 1) {
      accum += buffer[i];
      count += 1;
    }

    result[offsetResult] = count > 0 ? accum / count : 0;
    offsetResult += 1;
    offsetBuffer = nextOffsetBuffer;
  }

  return result;
}

function floatTo16BitPCM(float32Array) {
  const output = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i += 1) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return output;
}

// ============================================================
// WebSocket & streaming
// ============================================================
async function startStreaming() {
  clearTranscript();
  setStatus("Connecting...");

  const protocol = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${protocol}://${location.host}/ws/realtime`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    setStatus("Connected", true);
    ws.send(
      JSON.stringify({
        type: "config",
        language: languageInput.value.trim() || "vi",
      })
    );
  };

  ws.onmessage = (event) => {
    const payload = JSON.parse(event.data);

    if (payload.type === "ready") {
      setStatus(`Streaming (${payload.sample_rate} Hz, chunk ${payload.chunk_seconds}s)`, true);
      return;
    }

    if (payload.type === "ack") {
      setStatus(payload.message, true);
      return;
    }

    if (payload.type === "transcript") {
      if (payload.active_speaker) {
        activeSpeakerEl.textContent = payload.active_speaker;
        activeSpeakerEl.className = `status-value speaker-value ${getSpeakerClass(payload.active_speaker)}`;
      }

      if (payload.processing_ms) {
        updateLatency(payload.processing_ms);
      }
      if (payload.realtime_factor) {
        updateRTF(payload.realtime_factor);
      }

      (payload.items || []).forEach(addTranscriptLine);
      return;
    }

    if (payload.type === "no_speech") {
      // Silently skip
      return;
    }

    if (payload.type === "error") {
      setStatus(`Error: ${payload.message}`);
    }
  };

  ws.onerror = () => {
    setStatus("WebSocket error");
  };

  ws.onclose = () => {
    setStatus("Socket closed");
  };

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        sampleRate: { ideal: 48000 },
      },
    });
  } catch (err) {
    setStatus(`Microphone error: ${err.message}`);
    throw err;
  }

  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  sourceNode = audioContext.createMediaStreamSource(mediaStream);

  // Use larger buffer for more stable processing
  const bufferSize = 4096;
  processorNode = audioContext.createScriptProcessor(bufferSize, 1, 1);
  muteGain = audioContext.createGain();
  muteGain.gain.value = 0;

  processorNode.onaudioprocess = (event) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return;
    }
    const input = event.inputBuffer.getChannelData(0);
    const downsampled = downsampleBuffer(input, audioContext.sampleRate, 16000);
    const pcm16 = floatTo16BitPCM(downsampled);
    ws.send(pcm16.buffer);
  };

  sourceNode.connect(processorNode);
  processorNode.connect(muteGain);
  muteGain.connect(audioContext.destination);
}

async function stopStreaming() {
  // Disconnect audio nodes
  if (processorNode) {
    processorNode.disconnect();
    processorNode.onaudioprocess = null;
    processorNode = null;
  }

  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }

  if (muteGain) {
    muteGain.disconnect();
    muteGain = null;
  }

  // Stop microphone
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  // Close audio context
  if (audioContext) {
    await audioContext.close();
    audioContext = null;
  }

  // Close WebSocket
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "stop" }));
    ws.close();
  }
  ws = null;

  setStatus("Stopped");
  activeSpeakerEl.textContent = "-";
  activeSpeakerEl.className = "status-value speaker-value";
}

// ============================================================
// Event listeners
// ============================================================
startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  stopBtn.disabled = false;
  try {
    await startStreaming();
  } catch (error) {
    setStatus(`Cannot start: ${error.message}`);
    startBtn.disabled = false;
    stopBtn.disabled = true;
  }
});

stopBtn.addEventListener("click", async () => {
  stopBtn.disabled = true;
  await stopStreaming();
  startBtn.disabled = false;
});

clearBtn.addEventListener("click", () => {
  clearTranscript();
});

languageInput.addEventListener("change", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(
      JSON.stringify({
        type: "config",
        language: languageInput.value.trim() || "vi",
      })
    );
  }
});
