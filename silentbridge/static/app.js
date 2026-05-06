const state = {
  mode: "demo",
  latest: null,
  poller: null,
  cameraStream: null,
  selectedSign: "Help",
};

const $ = (id) => document.getElementById(id);

const controls = {
  gestureSelect: $("gestureSelect"),
  confidence: $("gestureConfidence"),
  visibility: $("handVisibility"),
  distress: $("facialDistress"),
  repetition: $("repetition"),
  speed: $("gestureSpeed"),
};

function percent(value) {
  return `${Math.round(Number(value) * 100)}%`;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

function updateRangeLabels() {
  $("confidenceOut").textContent = percent(controls.confidence.value);
  $("visibilityOut").textContent = percent(controls.visibility.value);
  $("distressOut").textContent = percent(controls.distress.value);
  $("repetitionOut").textContent = percent(controls.repetition.value);
  $("speedOut").textContent = percent(controls.speed.value);
}

async function evaluateDemo() {
  updateRangeLabels();
  const result = await api("/api/evaluate", {
    method: "POST",
    body: JSON.stringify({
      label: controls.gestureSelect.value,
      gesture_confidence: Number(controls.confidence.value),
      hand_visibility: Number(controls.visibility.value),
      facial_distress: Number(controls.distress.value),
      repetition: Number(controls.repetition.value),
      gesture_speed: Number(controls.speed.value),
    }),
  });
  renderResult(result);
}

function renderResult(result) {
  state.latest = result;
  $("englishMessage").textContent = result.message.english;
  $("hindiMessage").textContent = result.message.hindi;
  $("detectedGesture").textContent = result.label;
  $("communicationScore").textContent = percent(result.communication_confidence);
  $("urgencyBadge").textContent = `Urgency ${Number(result.urgency).toFixed(2)}`;
  $("engineName").textContent = result.engine.replace("scikit-fuzzy ", "");
  $("suggestedAction").textContent = result.action_level;
  $("resultPanel").classList.toggle("emergency", Boolean(result.emergency));
  document.querySelectorAll(".sign-grid button").forEach((button) => {
    button.classList.toggle("active", button.dataset.sign === result.label);
  });
}

function setMode(mode) {
  state.mode = mode;
}

async function startCamera() {
  setMode("live");
  try {
    state.cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
  } catch (error) {
    $("cameraStatus").textContent = "Browser camera permission was denied or unavailable.";
    $("videoFeed").style.display = "none";
    $("videoPlaceholder").style.display = "grid";
    $("videoPlaceholder").textContent = "Allow camera permission in the browser, or use the sign buttons above.";
    return;
  }

  $("videoFeed").srcObject = state.cameraStream;
  $("videoPlaceholder").style.display = "none";
  $("videoFeed").style.display = "block";
  $("cameraStatus").textContent = "Camera running in browser";
  if (state.poller) clearInterval(state.poller);
  state.poller = setInterval(analyzeCameraFrame, 1200);
}

async function stopCamera() {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach((track) => track.stop());
  }
  state.cameraStream = null;
  $("videoFeed").srcObject = null;
  $("videoFeed").style.display = "none";
  $("videoPlaceholder").style.display = "grid";
  $("videoPlaceholder").textContent = "Camera preview appears here";
  $("cameraStatus").textContent = "Camera idle";
  if (state.poller) clearInterval(state.poller);
  state.poller = null;
}

async function analyzeCameraFrame() {
  const video = $("videoFeed");
  if (!state.cameraStream || video.readyState < 2) return;

  const canvas = $("cameraCanvas");
  const width = video.videoWidth || 640;
  const height = video.videoHeight || 480;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, width, height);

  try {
    const result = await api("/api/analyze-frame", {
      method: "POST",
      body: JSON.stringify({ image: canvas.toDataURL("image/jpeg", 0.72) }),
    });
    if (result.ok) {
      renderResult(result);
      $("cameraStatus").textContent = result.hand_detected ? "Hand detected" : "Camera running. Show one hand clearly.";
    } else {
      $("cameraStatus").textContent = result.error || "Camera running";
    }
  } catch (error) {
    $("cameraStatus").textContent = "Camera preview works. Backend analysis is unavailable.";
  }
}

async function speakText(text) {
  await api("/api/speak", {
    method: "POST",
    body: JSON.stringify({ text }),
  });
}

async function logCurrent() {
  if (!state.latest) return;
  await api("/api/log", {
    method: "POST",
    body: JSON.stringify({
      label: state.latest.label,
      message: state.latest.message.english,
      communication_confidence: state.latest.communication_confidence,
      urgency: state.latest.urgency,
      action_level: state.latest.action_level,
    }),
  });
  await loadHistory();
}

async function loadHistory() {
  const history = await api("/api/history");
  const list = $("historyList");
  list.innerHTML = "";
  if (!history.length) {
    list.innerHTML = `<p class="subtitle">No communication history yet.</p>`;
    return;
  }
  for (const item of history) {
    const row = document.createElement("div");
    row.className = "history-item";
    row.innerHTML = `<strong>${item.gesture || "Gesture"}</strong><span>${item.message || ""}</span><span>Urgency ${item.urgency || ""}</span>`;
    list.appendChild(row);
  }
}

async function loadMemberships() {
  const data = await api("/api/memberships");
  drawMemberships(data);
}

function drawMemberships(data) {
  const canvas = $("membershipCanvas");
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const padding = 34;
  ctx.strokeStyle = "#d8e1df";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = padding + ((height - padding * 2) * i) / 4;
    ctx.beginPath();
    ctx.moveTo(padding, y);
    ctx.lineTo(width - padding, y);
    ctx.stroke();
  }

  const colors = ["#147d75", "#d73535", "#2f6f9f", "#b66c1e", "#44515a", "#6d8f3b"];
  let colorIndex = 0;
  let legendY = 18;
  for (const [group, curves] of Object.entries(data.curves)) {
    for (const [name, points] of Object.entries(curves)) {
      const color = colors[colorIndex % colors.length];
      colorIndex += 1;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      points.forEach((point, index) => {
        const x = padding + ((width - padding * 2) * index) / (points.length - 1);
        const y = height - padding - point * (height - padding * 2);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.fillRect(width - 176, legendY - 8, 10, 10);
      ctx.fillStyle = "#17202a";
      ctx.font = "12px system-ui";
      ctx.fillText(`${group.replace("_", " ")}: ${name}`, width - 160, legendY);
      legendY += 17;
    }
  }
}

async function init() {
  const status = await api("/api/status");
  $("classifierStatus").textContent = status.classifier_trained ? "Custom classifier ready" : "Demo classifier active";
  $("cameraStatus").textContent = status.camera.running ? "Camera running" : "Camera idle";

  controls.gestureSelect.innerHTML = status.signs.map((sign) => `<option value="${sign}">${sign}</option>`).join("");
  controls.gestureSelect.value = "Help";

  $("signButtons").innerHTML = "";
  status.signs.forEach((sign) => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.sign = sign;
    button.textContent = sign;
    button.addEventListener("click", () => {
      state.selectedSign = sign;
      controls.gestureSelect.value = sign;
      evaluateDemo();
    });
    $("signButtons").appendChild(button);
  });

  $("quickReplies").innerHTML = "";
  status.quick_replies.forEach((reply) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = reply;
    button.addEventListener("click", () => speakText(reply));
    $("quickReplies").appendChild(button);
  });

  Object.values(controls).forEach((control) => control.addEventListener("input", evaluateDemo));
  $("startCamera").addEventListener("click", startCamera);
  $("stopCamera").addEventListener("click", stopCamera);
  $("speakOutput").addEventListener("click", () => state.latest && speakText(state.latest.message.voice));
  $("logOutput").addEventListener("click", logCurrent);

  await evaluateDemo();
  await loadHistory();
  await loadMemberships();
}

init().catch((error) => {
  console.error(error);
  $("cameraStatus").textContent = "App error";
});
