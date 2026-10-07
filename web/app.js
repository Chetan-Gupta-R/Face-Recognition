const video = document.querySelector("#camera-video");
const overlay = document.querySelector("#camera-overlay");
const stage = document.querySelector("#camera-stage");
const placeholder = document.querySelector("#camera-placeholder");
const connectButton = document.querySelector("#connect-button");
const recognitionButton = document.querySelector("#recognition-button");
const stopCameraButton = document.querySelector("#stop-camera-button");
const enrollButton = document.querySelector("#enroll-button");
const trainButton = document.querySelector("#train-button");
const nameInput = document.querySelector("#person-name");
const feedStatus = document.querySelector("#feed-status");
const toast = document.querySelector("#toast");
const captureCanvas = document.createElement("canvas");
const captureContext = captureCanvas.getContext("2d", { willReadFrequently: true });

let cameraStream = null;
let activeMode = "preview";
let loopTimer = null;
let frameBusy = false;
let toastTimer = null;
let lastMatches = [];
let enrollmentName = "";
let enrollmentCount = 0;
let eventCache = [];

function showToast(message, isError = false) {
  toast.textContent = message;
  toast.classList.toggle("error", isError);
  toast.classList.add("visible");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toast.classList.remove("visible"), 3200);
}

async function requestJson(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "The request could not be completed.");
  return result;
}

function setCameraControls(connected) {
  connectButton.hidden = connected;
  recognitionButton.disabled = !connected;
  stopCameraButton.disabled = !connected;
  enrollButton.disabled = !connected;
  feedStatus.classList.toggle("connected", connected);
  feedStatus.querySelector("span").textContent = connected ? "Camera connected" : "Camera off";
  stage.classList.toggle("has-video", connected);
  stage.classList.toggle("streaming", connected);
  document.querySelector("#resolution-chip").textContent = connected
    ? `${video.videoWidth} × ${video.videoHeight}`
    : "—";
}

async function connectCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    showToast("Camera access requires a secure browser context.", true);
    return;
  }
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
    });
    video.srcObject = cameraStream;
    await video.play();
    setCameraControls(true);
    showToast("Camera connected. The preview stays in this browser.");
  } catch (error) {
    showToast(error.name === "NotAllowedError" ? "Camera permission was denied in your browser." : `Could not open camera: ${error.message}`, true);
  }
}

function stopCamera() {
  stopProcessing();
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  video.srcObject = null;
  lastMatches = [];
  drawMatches([]);
  document.querySelector("#face-count").textContent = "NO FACES";
  document.querySelector("#camera-mode").textContent = "PREVIEW";
  document.querySelector("#enroll-progress").hidden = true;
  setCameraControls(false);
}

function captureFrame() {
  if (!video.videoWidth || !video.videoHeight) return null;
  const scale = Math.min(1, 640 / video.videoWidth);
  captureCanvas.width = Math.round(video.videoWidth * scale);
  captureCanvas.height = Math.round(video.videoHeight * scale);
  captureContext.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
  return {
    image: captureCanvas.toDataURL("image/jpeg", 0.72),
    width: captureCanvas.width,
    height: captureCanvas.height,
  };
}

function drawMatches(matches, frameWidth = captureCanvas.width, frameHeight = captureCanvas.height) {
  const bounds = stage.getBoundingClientRect();
  const pixelRatio = window.devicePixelRatio || 1;
  overlay.width = Math.max(1, Math.round(bounds.width * pixelRatio));
  overlay.height = Math.max(1, Math.round(bounds.height * pixelRatio));
  const context = overlay.getContext("2d");
  context.scale(pixelRatio, pixelRatio);
  context.clearRect(0, 0, bounds.width, bounds.height);
  if (!video.videoWidth || !frameWidth || !frameHeight) return;

  const coverScale = Math.max(bounds.width / video.videoWidth, bounds.height / video.videoHeight);
  const videoWidth = video.videoWidth * coverScale;
  const videoHeight = video.videoHeight * coverScale;
  const offsetX = (bounds.width - videoWidth) / 2;
  const offsetY = (bounds.height - videoHeight) / 2;
  const scaleX = videoWidth / frameWidth;
  const scaleY = videoHeight / frameHeight;

  matches.forEach((match) => {
    const [x, y, width, height] = match.box;
    const left = offsetX + x * scaleX;
    const top = offsetY + y * scaleY;
    const boxWidth = width * scaleX;
    const boxHeight = height * scaleY;
    const known = match.name !== "Unknown";
    context.strokeStyle = known ? "#c7ee65" : "#f07d67";
    context.lineWidth = 2;
    context.strokeRect(left, top, boxWidth, boxHeight);
    const label = `${match.name}${known ? ` · ${Math.round(match.distance)}` : ""}`;
    context.font = "600 11px 'Cascadia Code', Consolas, monospace";
    const labelWidth = context.measureText(label).width + 12;
    context.fillStyle = known ? "#c7ee65" : "#f07d67";
    context.fillRect(left, Math.max(0, top - 21), labelWidth, 20);
    context.fillStyle = "#172119";
    context.fillText(label, left + 6, Math.max(0, top - 7));
  });
}

function stopProcessing() {
  window.clearTimeout(loopTimer);
  loopTimer = null;
  activeMode = "preview";
  recognitionButton.innerHTML = '<span class="button-symbol" aria-hidden="true">▷</span>Start recognition';
  document.querySelector("#camera-mode").textContent = "PREVIEW";
  document.querySelector("#enroll-progress").hidden = true;
}

function startRecognition() {
  if (!cameraStream) return;
  activeMode = "recognize";
  recognitionButton.innerHTML = '<span class="button-symbol" aria-hidden="true">Ⅱ</span>Pause recognition';
  document.querySelector("#camera-mode").textContent = "RECOGNITION ACTIVE";
  showToast("Recognition is running on this device.");
  processFrame();
}

function toggleRecognition() {
  if (activeMode === "recognize") {
    stopProcessing();
    drawMatches([]);
    document.querySelector("#face-count").textContent = "PAUSED";
    return;
  }
  refreshStatus().then((status) => {
    if (!status.trained) {
      showToast("Train the model before starting recognition.", true);
      return;
    }
    startRecognition();
  }).catch((error) => showToast(error.message, true));
}

function startEnrollment() {
  const name = nameInput.value.trim();
  if (!name) {
    nameInput.focus();
    showToast("Enter a name before enrolling.", true);
    return;
  }
  if (!cameraStream) return;
  enrollmentName = name;
  enrollmentCount = 0;
  activeMode = "enroll";
  document.querySelector("#camera-mode").textContent = "ENROLLMENT";
  document.querySelector("#enroll-progress").hidden = false;
  document.querySelector("#enroll-message").textContent = `Capturing ${name}`;
  updateEnrollmentProgress(0, 30);
  recognitionButton.disabled = true;
  enrollButton.disabled = true;
  processFrame();
}

function updateEnrollmentProgress(count, target) {
  document.querySelector("#enroll-count").textContent = `${count} / ${target}`;
  document.querySelector("#enroll-bar").style.width = `${Math.min(100, (count / target) * 100)}%`;
}

async function processFrame() {
  if (!cameraStream || activeMode === "preview" || frameBusy) return;
  const modeAtStart = activeMode;
  const frame = captureFrame();
  if (!frame) {
    loopTimer = window.setTimeout(processFrame, 250);
    return;
  }
  frameBusy = true;
  try {
    if (modeAtStart === "recognize") {
      const result = await requestJson("/api/recognize", { image: frame.image });
      lastMatches = result.matches;
      drawMatches(lastMatches, frame.width, frame.height);
      const count = lastMatches.length;
      document.querySelector("#face-count").textContent = count ? `${count} FACE${count === 1 ? "" : "S"} FOUND` : "NO FACES";
      const names = lastMatches.filter((match) => match.name !== "Unknown").map((match) => match.name);
      if (names.length && !names.some((name) => eventCache[0]?.includes(`| ${name} |`))) refreshStatus();
    } else if (modeAtStart === "enroll") {
      const result = await requestJson("/api/enroll", { name: enrollmentName, image: frame.image });
      if (result.saved) {
        enrollmentCount = result.count;
        updateEnrollmentProgress(result.count, result.target);
      }
      document.querySelector("#enroll-message").textContent = result.message;
      if (result.complete) {
        stopProcessing();
        document.querySelector("#enroll-message").textContent = `${enrollmentName} is enrolled`;
        updateEnrollmentProgress(result.count, result.target);
        document.querySelector("#enroll-progress").hidden = false;
        recognitionButton.disabled = false;
        enrollButton.disabled = false;
        showToast(`${enrollmentName} enrolled. Train the model to use this face.`);
        await refreshStatus();
      }
    }
  } catch (error) {
    stopProcessing();
    showToast(error.message, true);
  } finally {
    frameBusy = false;
    if (activeMode !== "preview" && cameraStream) {
      loopTimer = window.setTimeout(processFrame, activeMode === "enroll" ? 240 : 650);
    }
  }
}

function renderPeople(people) {
  const list = document.querySelector("#people-list");
  document.querySelector("#metric-people").textContent = people.length;
  document.querySelector("#people-count").textContent = people.length;
  document.querySelector("#nav-people-count").textContent = people.length;
  document.querySelector("#metric-samples").textContent = `${people.reduce((total, person) => total + person.samples, 0)} total face samples`;
  if (!people.length) {
    list.innerHTML = '<div class="empty-state"><span aria-hidden="true">＋</span><p>No one enrolled yet</p><small>Connect your camera to add someone.</small></div>';
    return;
  }
  list.replaceChildren(...people.map((person, index) => {
    const row = document.createElement("div");
    row.className = "person-row";
    const avatar = document.createElement("span");
    avatar.className = "person-avatar";
    avatar.textContent = person.name.slice(0, 2).toUpperCase();
    const info = document.createElement("span");
    info.className = "person-info";
    const name = document.createElement("strong");
    name.textContent = person.name;
    const samples = document.createElement("small");
    samples.textContent = `${person.samples} face samples`;
    info.append(name, samples);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "remove-person";
    remove.title = `Remove ${person.name}`;
    remove.setAttribute("aria-label", `Remove ${person.name}`);
    remove.textContent = "×";
    remove.addEventListener("click", () => removePerson(person.name));
    row.append(avatar, info, remove);
    row.style.animation = `rise-in .3s ${Math.min(index * 35, 210)}ms both`;
    return row;
  }));
}

function renderActivity(lines, todayCount) {
  const list = document.querySelector("#activity-list");
  eventCache = lines;
  document.querySelector("#metric-recognitions").textContent = todayCount;
  if (!lines.length) {
    list.innerHTML = '<div class="empty-activity">No recent recognitions.<br><span>They’ll appear here as they happen.</span></div>';
    return;
  }
  list.replaceChildren(...lines.map((line) => {
    const [timestamp = "", name = "", distance = ""] = line.split(" | ");
    const row = document.createElement("div");
    row.className = "activity-row";
    const avatar = document.createElement("span");
    avatar.className = "activity-avatar";
    avatar.textContent = name.trim().slice(0, 1).toUpperCase() || "?";
    const info = document.createElement("span");
    const person = document.createElement("span");
    person.className = "activity-name";
    person.textContent = name.trim();
    const time = document.createElement("span");
    time.className = "activity-time";
    time.textContent = timestamp;
    info.append(person, time);
    const tag = document.createElement("span");
    tag.className = "activity-tag";
    tag.textContent = distance.replace("distance=", "DIST ").toUpperCase();
    row.append(avatar, info, tag);
    return row;
  }));
}

async function refreshStatus() {
  const response = await fetch("/api/status", { cache: "no-store" });
  if (!response.ok) throw new Error("Could not load workspace status.");
  const status = await response.json();
  renderPeople(status.people);
  renderActivity(status.recent, status.todayCount);
  const model = document.querySelector("#metric-model");
  const detail = document.querySelector("#model-detail");
  const dot = document.querySelector("#model-dot");
  model.textContent = status.trained ? "Ready" : "Not trained";
  detail.textContent = status.trained ? "LBPH model available" : "Train to enable recognition";
  dot.classList.toggle("status-off", !status.trained);
  recognitionButton.disabled = !cameraStream || !status.trained || activeMode === "enroll";
  if (activeMode !== "recognize") recognitionButton.innerHTML = '<span class="button-symbol" aria-hidden="true">▷</span>Start recognition';
  return status;
}

async function trainModel() {
  trainButton.disabled = true;
  trainButton.innerHTML = '<span class="button-symbol" aria-hidden="true">…</span>Training';
  try {
    const result = await requestJson("/api/train", {});
    showToast(result.message);
    await refreshStatus();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    trainButton.disabled = false;
    trainButton.innerHTML = '<span class="button-symbol" aria-hidden="true">↻</span>Train model';
  }
}

async function removePerson(name) {
  if (!window.confirm(`Remove ${name} and all of their saved face samples?`)) return;
  try {
    const result = await requestJson("/api/remove", { name });
    showToast(result.message);
    await refreshStatus();
  } catch (error) {
    showToast(error.message, true);
  }
}

connectButton.addEventListener("click", connectCamera);
recognitionButton.addEventListener("click", toggleRecognition);
stopCameraButton.addEventListener("click", stopCamera);
enrollButton.addEventListener("click", startEnrollment);
trainButton.addEventListener("click", trainModel);
nameInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") startEnrollment();
});
window.addEventListener("resize", () => drawMatches(lastMatches));
window.addEventListener("beforeunload", () => cameraStream?.getTracks().forEach((track) => track.stop()));

document.querySelector("#today-label").textContent = new Intl.DateTimeFormat(undefined, {
  weekday: "short", month: "short", day: "numeric", year: "numeric",
}).format(new Date());
document.querySelectorAll(".nav-link").forEach((link) => link.addEventListener("click", () => {
  document.querySelectorAll(".nav-link").forEach((item) => item.classList.remove("active"));
  link.classList.add("active");
}));

refreshStatus().catch((error) => showToast(error.message, true));
window.setInterval(() => refreshStatus().catch(() => {}), 8000);
