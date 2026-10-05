const state = {
  schema: null,
  label: null,
  language: null,
  noiseKind: null,
  promptId: null,
  blob: null,
  blobUrl: null,
  consent: false,
  recording: false,
  receipt: null,
};

const workspace = document.querySelector("#workspace");
const loading = document.querySelector("#loading");
const detail = document.querySelector("#detail");
const capture = document.querySelector("#capture");
const card = document.querySelector("#card");
const formError = document.querySelector("#form-error");
const submitButton = document.querySelector("#submit");
const resetButton = document.querySelector("#reset");
const receipt = document.querySelector("#receipt");
const preview = document.querySelector("#preview");
const meter = document.querySelector("#meter");
const captureStatus = document.querySelector("#capture-status");
const recordButton = document.querySelector("#record");

let recorder = null;
let recordTimer = null;
let audioContext = null;
let meterFrame = 0;

document.querySelectorAll(".task").forEach((button) => {
  button.addEventListener("click", () => selectLabel(button.dataset.label));
});
document.querySelector("#consent").addEventListener("change", (event) => {
  state.consent = event.target.checked;
  refresh();
});
recordButton.addEventListener("click", () => {
  if (state.recording) stopRecording();
  else startRecording();
});
document.querySelector("#file").addEventListener("change", (event) => {
  const file = event.target.files && event.target.files[0];
  if (file) setBlob(file, "Uploaded clip ready to send.");
});
submitButton.addEventListener("click", submitClip);
resetButton.addEventListener("click", discardTake);

boot();

async function boot() {
  try {
    const [schema, stats] = await Promise.all([
      fetch("/api/schema").then(readJson),
      fetch("/api/stats").then(readJson),
    ]);
    state.schema = schema;
    loading.hidden = true;
    workspace.hidden = false;
    document.querySelector("#studio").setAttribute("aria-busy", "false");
    renderStats(stats);
    refresh();
  } catch (_error) {
    loading.textContent = "The label sheet did not load. Check that the collector is still running, then refresh.";
  }
}

function selectLabel(label) {
  state.label = label;
  state.language = label === "english" ? "en" : null;
  state.noiseKind = null;
  state.promptId = null;
  state.receipt = null;
  document.querySelectorAll(".task").forEach((button) => {
    button.setAttribute("aria-pressed", button.dataset.label === label ? "true" : "false");
  });
  renderDetail();
  discardTake(false);
  refresh();
}

function renderDetail() {
  const spec = state.schema.labels.find((item) => item.id === state.label);
  detail.hidden = false;
  capture.hidden = false;
  if (state.label === "silence") {
    detail.innerHTML = `<p>${escapeHtml(spec.instructions)}</p>`;
    return;
  }
  if (state.label === "noise") {
    const kinds = state.schema.noise_kinds.map((kind) => {
      const pressed = state.noiseKind === kind.id ? "true" : "false";
      return `<button type="button" class="kind" data-kind="${kind.id}" aria-pressed="${pressed}"><strong>${escapeHtml(kind.title)}</strong><br>${escapeHtml(kind.summary)}</button>`;
    }).join("");
    detail.innerHTML = `<p>${escapeHtml(spec.instructions)}</p><div class="choices">${kinds}</div>`;
    detail.querySelectorAll(".kind").forEach((button) => {
      button.addEventListener("click", () => {
        state.noiseKind = button.dataset.kind;
        renderDetail();
        refresh();
      });
    });
    return;
  }
  if (state.label === "english") {
    const prompt = pickPrompt("en");
    state.promptId = prompt ? prompt.id : null;
    detail.innerHTML = `<p>${escapeHtml(spec.instructions)}</p><p class="prompt">${prompt ? escapeHtml(prompt.text) : ""}</p>`;
    return;
  }
  const query = state.languageQuery || "";
  const languages = state.schema.languages.filter((language) => {
    const haystack = `${language.english} ${language.endonym} ${language.code}`.toLowerCase();
    return haystack.includes(query.toLowerCase());
  });
  const rows = languages.map((language) => languageRow(language)).join("");
  const unknown = languageRow(state.schema.unknown_language);
  const prompt = state.language ? pickPrompt(state.language) : null;
  state.promptId = prompt ? prompt.id : null;
  const spoken = prompt
    ? `<p class="prompt">${escapeHtml(prompt.text)}</p>`
    : `<p>Say one short everyday sentence. Leave out your name and where you live.</p>`;
  detail.innerHTML = `
    <p>${escapeHtml(spec.instructions)}</p>
    <label>Find a language
      <input class="search" id="language-search" value="${escapeHtml(query)}" placeholder="Spanish, 中文, sw">
    </label>
    <div class="lang-list" role="listbox">${rows}${unknown}</div>
    ${state.language ? spoken : ""}
  `;
  const search = detail.querySelector("#language-search");
  search.addEventListener("input", () => {
    state.languageQuery = search.value;
    renderDetail();
    const next = detail.querySelector("#language-search");
    if (next) {
      next.focus();
      next.setSelectionRange(next.value.length, next.value.length);
    }
  });
  detail.querySelectorAll(".lang").forEach((button) => {
    button.addEventListener("click", () => {
      state.language = button.dataset.code;
      renderDetail();
      refresh();
    });
  });
}

function languageRow(language) {
  const pressed = state.language === language.code ? "true" : "false";
  return `<button type="button" class="lang" data-code="${language.code}" aria-pressed="${pressed}"><span>${escapeHtml(language.endonym)} · ${escapeHtml(language.english)}</span><span class="mono">${language.code}</span></button>`;
}

function pickPrompt(code) {
  const prompts = (state.schema.prompts && state.schema.prompts[code]) || [];
  if (!prompts.length) return null;
  const current = prompts.find((item) => item.id === state.promptId);
  if (current) return current;
  return prompts[Math.floor(Math.random() * prompts.length)];
}

async function startRecording() {
  clearError();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError("This browser cannot record. Upload a clip instead.");
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
  } catch (_error) {
    showError("Microphone permission was blocked. You can upload a clip instead.");
    return;
  }
  const mimeType = pickMimeType();
  recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
  const chunks = [];
  recorder.addEventListener("dataavailable", (event) => {
    if (event.data.size) chunks.push(event.data);
  });
  recorder.addEventListener("stop", () => {
    stream.getTracks().forEach((track) => track.stop());
    stopMeter();
    const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
    setBlob(blob, "Take ready. Listen, then submit or discard.");
    state.recording = false;
    recordButton.textContent = "Record 4 seconds";
  });
  state.recording = true;
  recordButton.textContent = "Stop";
  captureStatus.textContent = "Recording…";
  startMeter(stream);
  recorder.start();
  const seconds = state.schema.record_seconds || 4;
  let left = seconds;
  recordTimer = window.setInterval(() => {
    left -= 0.1;
    captureStatus.textContent = left > 0 ? `Recording… ${left.toFixed(1)}s` : "Finishing the take…";
    if (left <= 0) stopRecording();
  }, 100);
}

function stopRecording() {
  if (recordTimer) window.clearInterval(recordTimer);
  recordTimer = null;
  if (recorder && recorder.state !== "inactive") recorder.stop();
}

function pickMimeType() {
  const types = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
  return types.find((type) => window.MediaRecorder && MediaRecorder.isTypeSupported(type)) || "";
}

function startMeter(stream) {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) return;
  audioContext = new AudioCtx();
  const source = audioContext.createMediaStreamSource(stream);
  const analyser = audioContext.createAnalyser();
  analyser.fftSize = 256;
  source.connect(analyser);
  const data = new Uint8Array(analyser.fftSize);
  const context = meter.getContext("2d");
  const draw = () => {
    analyser.getByteTimeDomainData(data);
    context.clearRect(0, 0, meter.width, meter.height);
    context.strokeStyle = "#9c341f";
    context.lineWidth = 2;
    context.beginPath();
    const step = meter.width / data.length;
    data.forEach((value, index) => {
      const y = ((value - 128) / 128) * (meter.height / 2) + meter.height / 2;
      if (index === 0) context.moveTo(0, y);
      else context.lineTo(index * step, y);
    });
    context.stroke();
    meterFrame = requestAnimationFrame(draw);
  };
  draw();
}

function stopMeter() {
  if (meterFrame) cancelAnimationFrame(meterFrame);
  meterFrame = 0;
  if (audioContext) audioContext.close();
  audioContext = null;
}

function setBlob(blob, message) {
  if (state.blobUrl) URL.revokeObjectURL(state.blobUrl);
  state.blob = blob;
  state.blobUrl = URL.createObjectURL(blob);
  preview.hidden = false;
  preview.src = state.blobUrl;
  captureStatus.textContent = message;
  resetButton.hidden = false;
  refresh();
}

function discardTake(render) {
  if (state.recording) stopRecording();
  if (state.blobUrl) URL.revokeObjectURL(state.blobUrl);
  state.blob = null;
  state.blobUrl = null;
  state.receipt = null;
  preview.hidden = true;
  preview.removeAttribute("src");
  resetButton.hidden = true;
  receipt.hidden = true;
  captureStatus.textContent = "The browser is asked not to clean up the sound, so silence stays silent and noise stays noisy.";
  const file = document.querySelector("#file");
  if (file) file.value = "";
  if (render !== false) refresh();
}

function refresh() {
  card.textContent = JSON.stringify(labelPreview(), null, 2);
  const ready = canSubmit();
  submitButton.disabled = !ready;
  submitButton.textContent = "Submit anonymously";
  const hint = document.querySelector("#send-hint");
  if (!state.label) hint.textContent = "Choose a label to begin.";
  else hint.textContent = ready ? "The card above is the label that will be stored." : blockReason();
}

function labelPreview() {
  return {
    label: state.label,
    language: state.label === "english" ? "en" : state.label === "non_english" ? state.language : null,
    noise_kind: state.label === "noise" ? state.noiseKind : null,
    prompt_id: state.promptId,
    consent: state.consent ? "yes" : null,
  };
}

function canSubmit() {
  if (!state.schema || !state.consent || !state.blob || !state.label || state.recording) return false;
  if (state.label === "noise" && !state.noiseKind) return false;
  if (state.label === "non_english" && !state.language) return false;
  return true;
}

async function submitClip() {
  clearError();
  if (!canSubmit()) {
    showError(blockReason());
    return;
  }
  const body = new FormData();
  const extension = state.blob.type.includes("webm") ? "webm" : "wav";
  body.append("audio", state.blob, `clip.${extension}`);
  body.append("label", state.label);
  body.append("consent", "yes");
  if (state.language) body.append("language", state.language);
  if (state.noiseKind) body.append("noise_kind", state.noiseKind);
  if (state.promptId) body.append("prompt_id", state.promptId);
  submitButton.disabled = true;
  submitButton.textContent = "Sending…";
  try {
    const response = await fetch("/api/submissions", { method: "POST", body });
    const payload = await readJson(response);
    if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "The clip was not stored.");
    state.receipt = payload;
    showReceipt(payload);
    renderStats(await fetch("/api/stats").then(readJson));
  } catch (error) {
    showError(error.message || "The clip was not stored.");
    submitButton.disabled = false;
    submitButton.textContent = "Submit anonymously";
  }
}

function blockReason() {
  if (!state.label) return "Choose silence, noise, English, or another language.";
  if (state.label === "noise" && !state.noiseKind) return "Choose the kind of noise.";
  if (state.label === "non_english" && !state.language) return "Choose the language. English has its own task.";
  if (!state.blob) return "Record or upload a clip first.";
  if (!state.consent) return "Check the consent box. It is the only permission we store.";
  return "The clip is not ready to send.";
}

function showReceipt(payload) {
  receipt.hidden = false;
  receipt.innerHTML = `
    <h3>Stored</h3>
    <p>Anonymous id <span class="mono">${escapeHtml(payload.submission_id)}</span>. Label <strong>${escapeHtml(payload.label)}</strong>${payload.language ? `, language <strong>${escapeHtml(payload.language)}</strong>` : ""}${payload.noise_kind ? `, noise <strong>${escapeHtml(payload.noise_kind)}</strong>` : ""}. This page will not play it back later.</p>
    <button type="button" id="undo" class="quiet">Remove this clip</button>
  `;
  receipt.querySelector("#undo").addEventListener("click", async () => {
    const response = await fetch(`/api/submissions/${payload.submission_id}`, { method: "DELETE" });
    if (!response.ok) {
      showError("Could not remove that clip.");
      return;
    }
    receipt.innerHTML = "<p>Removed. It is no longer in the catalog.</p>";
    renderStats(await fetch("/api/stats").then(readJson));
  });
}

function renderStats(stats) {
  const labels = [
    ["Total", stats.total],
    ["Silence", stats.by_label.silence],
    ["Noise", stats.by_label.noise],
    ["English", stats.by_label.english],
    ["Other languages", stats.by_label.non_english],
  ];
  document.querySelector("#stats").innerHTML = labels.map(([name, count]) => {
    return `<div><dt>${name}</dt><dd>${count}</dd></div>`;
  }).join("");
}

function showError(message) {
  formError.hidden = false;
  formError.textContent = message;
}

function clearError() {
  formError.hidden = true;
  formError.textContent = "";
}

async function readJson(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch (_error) {
    throw new Error("The collector sent a response that was not JSON.");
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
