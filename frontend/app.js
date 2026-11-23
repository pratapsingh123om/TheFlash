const state = {
  forecast: null,
  boundary: null,
  selected: 0,
};

const colours = ["#26384d", "#5ed7ff", "#ffb342", "#ff4f6d"];
const canvas = document.getElementById("forecast-map");
const context = canvas.getContext("2d");
const runButton = document.getElementById("run-button");

function setStatus(kind, text) {
  const dot = document.getElementById("status-dot");
  dot.className = `status-dot ${kind || ""}`;
  document.getElementById("status-text").textContent = text;
}

function formatTime(value) {
  if (!value) return "—";
  return value.replace("T", " ").replace(/\+05:30$/, "") + " IST";
}

function projection(lon, lat, width, height) {
  const bounds = { west: 67.5, east: 98.5, south: 5.5, north: 37.8 };
  const x = ((lon - bounds.west) / (bounds.east - bounds.west)) * width;
  const y = height - ((lat - bounds.south) / (bounds.north - bounds.south)) * height;
  return [x, y];
}

function drawBoundary(width, height) {
  if (!state.boundary) return;
  context.strokeStyle = "rgba(217, 232, 247, .72)";
  context.lineWidth = 1.4;
  const features = state.boundary.features || [];
  for (const feature of features) {
    const geometry = feature.geometry || {};
    const polygons = geometry.type === "Polygon" ? [geometry.coordinates] : geometry.coordinates || [];
    for (const polygon of polygons) {
      for (const ring of polygon) {
        context.beginPath();
        ring.forEach(([lon, lat], index) => {
          const [x, y] = projection(lon, lat, width, height);
          if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
        });
        context.stroke();
      }
    }
  }
}

function drawMap() {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(320, rect.width);
  const height = Math.max(620, rect.height);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);

  const gradient = context.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, "#0b2035");
  gradient.addColorStop(1, "#07111f");
  context.fillStyle = gradient;
  context.fillRect(0, 0, width, height);
  drawBoundary(width, height);

  if (!state.forecast) return;
  const period = state.forecast.periods[state.selected];
  const pointSize = Math.max(2.1, Math.min(5.2, width / 240));
  for (const cell of period.cells) {
    const [x, y] = projection(cell.lon, cell.lat, width, height);
    const code = Number(cell.warning_code || 0);
    context.fillStyle = colours[code];
    context.globalAlpha = code === 0 ? .16 : .86;
    context.fillRect(x - pointSize / 2, y - pointSize / 2, pointSize, pointSize);
  }
  context.globalAlpha = 1;
}

function selectHorizon(index) {
  state.selected = index;
  document.querySelectorAll("#horizon-buttons button").forEach((button, itemIndex) => {
    button.classList.toggle("active", itemIndex === index);
  });
  const period = state.forecast.periods[index];
  document.getElementById("valid-time").textContent = formatTime(period.valid_time_ist);
  document.getElementById("strong-count").textContent = (
    period.warning_counts.strong + period.warning_counts.heavy
  ).toLocaleString();
  document.getElementById("heavy-count").textContent = period.warning_counts.heavy.toLocaleString();
  document.getElementById("peak-heavy").textContent = `${(period.max_heavy_probability * 100).toFixed(1)}%`;
  document.getElementById("map-title").textContent = `${period.horizon_hours}-hour outlook · ${formatTime(period.valid_time_ist)}`;
  drawMap();
}

function renderForecast(payload) {
  state.forecast = payload;
  document.getElementById("map-empty").hidden = true;
  const controls = document.getElementById("horizon-buttons");
  controls.replaceChildren();
  payload.periods.forEach((period, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `+${period.horizon_hours} h`;
    button.addEventListener("click", () => selectHorizon(index));
    controls.append(button);
  });
  selectHorizon(0);
}

async function getJson(url, options) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `${response.status} ${response.statusText}`);
  return body;
}

async function loadInitial() {
  try {
    state.boundary = await getJson("/api/boundary");
  } catch (error) {
    console.warn("Boundary unavailable", error);
  }
  try {
    const health = await getJson("/api/health");
    if (health.status === "ready") setStatus("ready", "Model ready");
    else setStatus("error", health.error || "Configuration required");
    if (health.forecast_exists) renderForecast(await getJson("/api/forecast/latest"));
  } catch (error) {
    setStatus("error", error.message);
  }
  drawMap();
}

runButton.addEventListener("click", async () => {
  runButton.disabled = true;
  runButton.textContent = "Generating forecast";
  setStatus("", "Forecast running");
  try {
    const payload = await getJson("/api/forecast/run", { method: "POST" });
    renderForecast(payload);
    setStatus("ready", "Forecast complete");
  } catch (error) {
    setStatus("error", error.message);
  } finally {
    runButton.disabled = false;
    runButton.textContent = "Generate 24-hour forecast";
  }
});

window.addEventListener("resize", drawMap);
loadInitial();
