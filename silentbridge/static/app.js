const state = {
  mode: "demo",
  latest: null,
  poller: null,
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
  $("urgencyScore").textContent = Number(result.urgency).toFixed(2);
  $("engineName").textContent = result.engine.replace("scikit-fuzzy ", "");
  $("suggestedAction").textContent = result.action_level;
  $("resultPanel").classList.toggle("emergency", Boolean(result.emergency));
}

function setMode(mode) {
  state.mode = mode;
  $("demoPanel").classList.toggle("hidden", mode !== "demo");
  $("livePanel").classList.toggle("hidden", mode !== "live");
  $("demoMode").classList.toggle("active", mode === "demo");
  $("liveMode").classList.toggle("active", mode === "live");
}

async function startCamera() {
  setMode("live");
  const status = await api("/api/live/start", { method: "POST", body: "{}" });
  if (!status.ok) {
    $("cameraStatus").textContent = status.error || "Camera unavailable";
    return;
  }
  $("videoPlaceholder").style.display = "none";
  $("videoFeed").src = `/video_feed?t=${Date.now()}`;
  $("cameraStatus").textContent = "Camera running";
  if (state.poller) clearInterval(state.poller);
  state.poller = setInterval(refreshLiveResult, 700);
}

async function stopCamera() {
  await api("/api/live/stop", { method: "POST", body: "{}" });
  $("videoFeed").src = "";
  $("videoPlaceholder").style.display = "grid";
  $("cameraStatus").textContent = "Camera idle";
  if (state.poller) clearInterval(state.poller);
  state.poller = null;
}

async function refreshLiveResult() {
  const status = await api("/api/live/latest");
  if (status.latest) renderResult(status.latest);
  if (status.error) $("cameraStatus").textContent = status.error;
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

  $("quickReplies").innerHTML = "";
  status.quick_replies.forEach((reply) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = reply;
    button.addEventListener("click", () => speakText(reply));
    $("quickReplies").appendChild(button);
  });

  Object.values(controls).forEach((control) => control.addEventListener("input", evaluateDemo));
  $("demoMode").addEventListener("click", () => setMode("demo"));
  $("liveMode").addEventListener("click", () => setMode("live"));
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
