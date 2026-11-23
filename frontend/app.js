const state = {
  forecast: null,
  boundary: null,
  selected: 0,
  selectedLayer: "warnings", // "warnings", "heavy", "severe50", "expected"
  selectedCell: null,
  hoveredCell: null,
  isPlaying: false,
  playTimer: null,
};

const warningColors = ["#1e2f42", "#5ed7ff", "#ffb342", "#ff4767"];
const warningNames = ["None", "Mild", "Strong", "Heavy"];
const warningBadges = ["Low Risk", "Mild Alert", "Strong Alert", "Heavy Alert"];

const canvas = document.getElementById("forecast-map");
const context = canvas.getContext("2d");
const runButton = document.getElementById("run-button");
const playButton = document.getElementById("play-button");
const tooltip = document.getElementById("map-tooltip");
const mapWrap = document.getElementById("map-wrap");

// Major reference cities for location context
const majorCities = [
  { name: "New Delhi", lat: 28.61, lon: 77.20 },
  { name: "Mumbai", lat: 19.07, lon: 72.87 },
  { name: "Kolkata", lat: 22.57, lon: 88.36 },
  { name: "Bengaluru", lat: 12.97, lon: 77.59 },
  { name: "Hyderabad", lat: 17.38, lon: 78.48 },
  { name: "Chennai", lat: 13.08, lon: 80.27 },
  { name: "Ahmedabad", lat: 23.02, lon: 72.57 },
  { name: "Nagpur", lat: 21.14, lon: 79.08 },
  { name: "Bhubaneswar", lat: 20.29, lon: 85.82 },
  { name: "Guwahati", lat: 26.14, lon: 91.73 },
  { name: "Patna", lat: 25.59, lon: 85.13 },
  { name: "Lucknow", lat: 26.84, lon: 80.94 },
  { name: "Jaipur", lat: 26.91, lon: 75.78 },
  { name: "Ranchi", lat: 23.34, lon: 85.30 },
  { name: "Raipur", lat: 21.25, lon: 81.62 },
  { name: "Visakhapatnam", lat: 17.68, lon: 83.21 },
  { name: "Srinagar", lat: 34.08, lon: 74.79 },
  { name: "Shimla", lat: 31.10, lon: 77.17 },
  { name: "Dehradun", lat: 30.31, lon: 78.03 },
  { name: "Bhopal", lat: 23.25, lon: 77.41 },
];

function findNearestCity(lat, lon) {
  let nearest = null;
  let minDist = 999;
  for (const city of majorCities) {
    const dLat = city.lat - lat;
    const dLon = city.lon - lon;
    const dist = Math.sqrt(dLat * dLat + dLon * dLon);
    if (dist < minDist) {
      minDist = dist;
      nearest = city;
    }
  }
  if (nearest && minDist < 1.8) {
    return `Near ${nearest.name} (${Math.round(minDist * 111)} km)`;
  }
  return "Regional Grid Cell";
}

function setStatus(kind, text) {
  const dot = document.getElementById("status-dot");
  dot.className = `status-dot ${kind || ""}`;
  document.getElementById("status-text").textContent = text;
}

function formatTime(value) {
  if (!value) return "—";
  return value.replace("T", " ").replace(/\+05:30$/, "") + " IST";
}

// Bounding box for mapping
const bounds = { west: 67.5, east: 98.5, south: 5.5, north: 37.8 };

function projection(lon, lat, width, height) {
  const x = ((lon - bounds.west) / (bounds.east - bounds.west)) * width;
  const y = height - ((lat - bounds.south) / (bounds.north - bounds.south)) * height;
  return [x, y];
}

function inverseProjection(x, y, width, height) {
  const lon = bounds.west + (x / width) * (bounds.east - bounds.west);
  const lat = bounds.south + ((height - y) / height) * (bounds.north - bounds.south);
  return [lon, lat];
}

// Color scale interpolator for continuous probability maps
function getHeatmapColor(val, maxVal = 1.0) {
  const t = Math.max(0, Math.min(1, val / maxVal));
  if (t < 0.2) {
    // 0 to 0.2: navy to cyan
    const f = t / 0.2;
    return `rgb(${Math.round(20 + 74 * f)}, ${Math.round(35 + 180 * f)}, ${Math.round(60 + 195 * f)})`;
  } else if (t < 0.5) {
    // 0.2 to 0.5: cyan to amber
    const f = (t - 0.2) / 0.3;
    return `rgb(${Math.round(94 + 161 * f)}, ${Math.round(215 - 36 * f)}, ${Math.round(255 - 189 * f)})`;
  } else {
    // 0.5 to 1.0: amber to bright red
    const f = (t - 0.5) / 0.5;
    return `rgb(${Math.round(255)}, ${Math.round(179 - 108 * f)}, ${Math.round(66 + 37 * f)})`;
  }
}

function drawGridLines(width, height) {
  context.save();
  context.strokeStyle = "rgba(40, 70, 105, 0.28)";
  context.lineWidth = 0.8;
  context.fillStyle = "rgba(120, 150, 185, 0.45)";
  context.font = "10px Inter, system-ui, sans-serif";

  // Latitude parallels (10, 15, 20, 25, 30, 35)
  for (let lat = 10; lat <= 35; lat += 5) {
    const [, y] = projection(bounds.west, lat, width, height);
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
    context.fillText(`${lat}°N`, 8, y - 4);
  }

  // Longitude meridians (70, 75, 80, 85, 90, 95)
  for (let lon = 70; lon <= 95; lon += 5) {
    const [x] = projection(lon, bounds.south, width, height);
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.stroke();
    context.fillText(`${lon}°E`, x + 4, height - 8);
  }
  context.restore();
}

function drawBoundary(width, height) {
  if (!state.boundary) return;
  context.save();
  context.strokeStyle = "rgba(200, 225, 250, 0.65)";
  context.lineWidth = 1.4;
  context.shadowColor = "rgba(94, 215, 255, 0.25)";
  context.shadowBlur = 6;

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
  context.restore();
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

  // Background ocean/continent gradient
  const gradient = context.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, "#081b2e");
  gradient.addColorStop(1, "#050f1b");
  context.fillStyle = gradient;
  context.fillRect(0, 0, width, height);

  // Coordinate grid lines
  drawGridLines(width, height);

  // India boundary
  drawBoundary(width, height);

  if (!state.forecast) return;
  const period = state.forecast.periods[state.selected];
  if (!period) return;

  const pointSize = Math.max(2.4, Math.min(5.2, width / 230));
  const layer = state.selectedLayer;

  // Draw lightning cells
  for (const cell of period.cells) {
    const [x, y] = projection(cell.lon, cell.lat, width, height);

    if (layer === "warnings") {
      const code = Number(cell.warning_code || 0);
      if (code === 0) {
        // Distinct visible grid point for None (Clear sky / Zero risk)
        context.fillStyle = "#1a324d";
        context.globalAlpha = 0.5;
      } else {
        context.fillStyle = warningColors[code];
        context.globalAlpha = 0.95;
      }
    } else if (layer === "heavy") {
      context.fillStyle = getHeatmapColor(cell.p_heavy || 0, 0.35);
      context.globalAlpha = 0.85;
    } else if (layer === "severe50") {
      context.fillStyle = getHeatmapColor(cell.p_strong_or_heavy_within50 || 0, 0.55);
      context.globalAlpha = 0.85;
    } else if (layer === "expected") {
      context.fillStyle = getHeatmapColor(cell.expected_region50_count || 0, 35.0);
      context.globalAlpha = 0.85;
    }

    context.fillRect(x - pointSize / 2, y - pointSize / 2, pointSize, pointSize);
  }
  context.globalAlpha = 1.0;

  // Highlight selected cell if pinned
  if (state.selectedCell) {
    const [sx, sy] = projection(state.selectedCell.lon, state.selectedCell.lat, width, height);
    context.save();
    context.strokeStyle = "#5ed7ff";
    context.lineWidth = 2.2;
    context.shadowColor = "#5ed7ff";
    context.shadowBlur = 10;
    context.beginPath();
    context.arc(sx, sy, Math.max(pointSize * 2.2, 8), 0, Math.PI * 2);
    context.stroke();

    // Crosshair lines
    context.beginPath();
    context.moveTo(sx - 14, sy); context.lineTo(sx - 5, sy);
    context.moveTo(sx + 5, sy); context.lineTo(sx + 14, sy);
    context.moveTo(sx, sy - 14); context.lineTo(sx, sy - 5);
    context.moveTo(sx, sy + 5); context.lineTo(sx, sy + 14);
    context.stroke();
    context.restore();
  }

  // Highlight hovered cell
  if (state.hoveredCell && state.hoveredCell !== state.selectedCell) {
    const [hx, hy] = projection(state.hoveredCell.lon, state.hoveredCell.lat, width, height);
    context.save();
    context.strokeStyle = "rgba(255, 255, 255, 0.85)";
    context.lineWidth = 1.5;
    context.beginPath();
    context.arc(hx, hy, pointSize * 1.8, 0, Math.PI * 2);
    context.stroke();
    context.restore();
  }
}

// Find closest cell to geographic coordinates
function findNearestCell(cells, clickLon, clickLat, maxDistanceDeg = 1.2) {
  let nearest = null;
  let minDistanceSq = maxDistanceDeg * maxDistanceDeg;
  for (let i = 0; i < cells.length; i++) {
    const c = cells[i];
    const dLon = c.lon - clickLon;
    const dLat = c.lat - clickLat;
    const distSq = dLon * dLon + dLat * dLat;
    if (distSq < minDistanceSq) {
      minDistanceSq = distSq;
      nearest = c;
    }
  }
  return nearest;
}

function updateInspector(cell) {
  if (!cell || !state.forecast) return;

  document.getElementById("inspector-placeholder").style.display = "none";
  document.getElementById("inspector-content").style.display = "block";
  document.getElementById("inspector-status").textContent = "Location pinned";

  const period = state.forecast.periods[state.selected];
  // Match cell in current horizon
  const currentCell = period.cells.find(c => c.grid_id === cell.grid_id) || cell;

  document.getElementById("insp-coord").textContent = `${currentCell.lat.toFixed(2)}°N, ${currentCell.lon.toFixed(2)}°E`;
  document.getElementById("insp-city").textContent = findNearestCity(currentCell.lat, currentCell.lon);

  const code = Number(currentCell.warning_code || 0);
  const badge = document.getElementById("insp-badge");
  badge.className = `alert-badge ${warningNames[code].toLowerCase()}`;
  badge.textContent = code === 0 ? "No Lightning (Clear)" : warningBadges[code];

  // Calibrated probabilities
  const pHeavy = ((currentCell.p_heavy || 0) * 100).toFixed(1);
  const pStrong = ((currentCell.p_strong || 0) * 100).toFixed(1);
  const pMild = ((currentCell.p_mild || 0) * 100).toFixed(1);
  const pNone = ((currentCell.p_none || 0) * 100).toFixed(1);

  const cityInfo = findNearestCity(currentCell.lat, currentCell.lon);
  document.getElementById("insp-city").textContent = Number(pNone) >= 95 ? `${cityInfo} · Dry, stable conditions` : cityInfo;

  document.getElementById("insp-p-heavy").textContent = `${pHeavy}%`;
  document.getElementById("bar-p-heavy").style.width = `${pHeavy}%`;

  document.getElementById("insp-p-strong").textContent = `${pStrong}%`;
  document.getElementById("bar-p-strong").style.width = `${pStrong}%`;

  document.getElementById("insp-p-mild").textContent = `${pMild}%`;
  document.getElementById("bar-p-mild").style.width = `${pMild}%`;

  document.getElementById("insp-p-none").textContent = `${pNone}%`;
  document.getElementById("bar-p-none").style.width = `${pNone}%`;

  // 50km regional metrics
  const pQ95_50 = ((currentCell.p_strong_or_heavy_within50 || 0) * 100).toFixed(1);
  document.getElementById("insp-p-q95-50").textContent = `${pQ95_50}%`;
  document.getElementById("insp-exp-count").textContent = (currentCell.expected_region50_count || 0).toFixed(1);

  // 24-hour progression at this location
  const timeline = document.getElementById("insp-timeline");
  timeline.replaceChildren();

  state.forecast.periods.forEach((p, idx) => {
    const c = p.cells.find(item => item.grid_id === cell.grid_id);
    if (!c) return;
    const cCode = Number(c.warning_code || 0);
    const item = document.createElement("div");
    item.className = `timeline-item ${idx === state.selected ? "active" : ""}`;
    item.innerHTML = `
      <span class="timeline-horizon">+${p.horizon_hours}h</span>
      <span class="timeline-risk ${warningNames[cCode].toLowerCase()}">${warningNames[cCode]} Alert</span>
      <span class="timeline-pct">P(Heavy): ${((c.p_heavy || 0) * 100).toFixed(1)}%</span>
    `;
    item.addEventListener("click", () => selectHorizon(idx));
    timeline.append(item);
  });
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
  document.getElementById("map-title").textContent = `${period.horizon_hours}-Hour Outlook · ${formatTime(period.valid_time_ist)}`;

  if (state.selectedCell) {
    updateInspector(state.selectedCell);
  }
  drawMap();
}

function selectLayer(layerName) {
  state.selectedLayer = layerName;
  document.querySelectorAll(".layer-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.layer === layerName);
  });

  const catLegend = document.getElementById("category-legend");
  const heatLegend = document.getElementById("heatmap-legend");
  const maxLabel = document.getElementById("heatmap-max-label");
  const layerLabel = document.getElementById("layer-label");

  if (layerName === "warnings") {
    catLegend.style.display = "flex";
    heatLegend.style.display = "none";
    layerLabel.textContent = "Regional Warning Map";
  } else if (layerName === "heavy") {
    catLegend.style.display = "none";
    heatLegend.style.display = "flex";
    maxLabel.textContent = "40%+";
    layerLabel.textContent = "Heavy Storm Probability (Q99)";
  } else if (layerName === "severe50") {
    catLegend.style.display = "none";
    heatLegend.style.display = "flex";
    maxLabel.textContent = "60%+";
    layerLabel.textContent = "50km Severe Storm Risk";
  } else if (layerName === "expected") {
    catLegend.style.display = "none";
    heatLegend.style.display = "flex";
    maxLabel.textContent = "40+ Flashes";
    layerLabel.textContent = "50km Expected Flash Density";
  }
  drawMap();
}

function toggleAnimation() {
  if (state.isPlaying) {
    clearInterval(state.playTimer);
    state.isPlaying = false;
    playButton.classList.remove("playing");
    document.getElementById("play-icon").textContent = "▶";
    document.getElementById("play-text").textContent = "Animate";
  } else {
    if (!state.forecast) return;
    state.isPlaying = true;
    playButton.classList.add("playing");
    document.getElementById("play-icon").textContent = "⏸";
    document.getElementById("play-text").textContent = "Pause";
    state.playTimer = setInterval(() => {
      const nextIndex = (state.selected + 1) % state.forecast.periods.length;
      selectHorizon(nextIndex);
    }, 1500);
  }
}

function renderForecast(payload) {
  state.forecast = payload;
  const emptyEl = document.getElementById("map-empty");
  if (emptyEl) {
    emptyEl.classList.add("hidden");
    emptyEl.style.display = "none";
  }
  const controls = document.getElementById("horizon-buttons");
  controls.replaceChildren();
  payload.periods.forEach((period, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `+${period.horizon_hours} h`;
    button.addEventListener("click", () => {
      if (state.isPlaying) toggleAnimation();
      selectHorizon(index);
    });
    controls.append(button);
  });
  selectHorizon(0);
}

// Canvas Mouse Interactions
canvas.addEventListener("mousemove", (e) => {
  if (!state.forecast) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  const [lon, lat] = inverseProjection(x, y, rect.width, rect.height);
  const period = state.forecast.periods[state.selected];
  const nearest = findNearestCell(period.cells, lon, lat, 1.2);

  if (nearest) {
    state.hoveredCell = nearest;
    tooltip.style.display = "block";
    tooltip.style.left = `${x}px`;
    tooltip.style.top = `${y}px`;

    const code = Number(nearest.warning_code || 0);
    document.getElementById("tt-coord").textContent = `${nearest.lat.toFixed(2)}°N, ${nearest.lon.toFixed(2)}°E`;
    const ttBadge = document.getElementById("tt-badge");
    ttBadge.className = `tooltip-badge ${warningNames[code].toLowerCase()}`;
    ttBadge.textContent = warningBadges[code];
    ttBadge.style.color = warningColors[code];

    let valText = `P(Heavy): ${((nearest.p_heavy || 0) * 100).toFixed(1)}%`;
    if (state.selectedLayer === "severe50") {
      valText = `50km Risk: ${((nearest.p_strong_or_heavy_within50 || 0) * 100).toFixed(1)}%`;
    } else if (state.selectedLayer === "expected") {
      valText = `Expected: ${(nearest.expected_region50_count || 0).toFixed(1)} strikes`;
    }
    document.getElementById("tt-value").textContent = valText;
  } else {
    state.hoveredCell = null;
    tooltip.style.display = "none";
  }
  drawMap();
});

canvas.addEventListener("mouseleave", () => {
  state.hoveredCell = null;
  tooltip.style.display = "none";
  drawMap();
});

canvas.addEventListener("click", (e) => {
  if (!state.forecast) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  const [lon, lat] = inverseProjection(x, y, rect.width, rect.height);
  const period = state.forecast.periods[state.selected];
  const nearest = findNearestCell(period.cells, lon, lat, 1.5);

  if (nearest) {
    state.selectedCell = nearest;
    updateInspector(nearest);
    drawMap();
  }
});

// Setup City Quick Jump Buttons
document.getElementById("city-chips").querySelectorAll(".chip").forEach(chip => {
  chip.addEventListener("click", () => {
    if (!state.forecast) return;
    const lat = parseFloat(chip.dataset.lat);
    const lon = parseFloat(chip.dataset.lon);
    const period = state.forecast.periods[state.selected];
    const nearest = findNearestCell(period.cells, lon, lat, 2.0);
    if (nearest) {
      state.selectedCell = nearest;
      updateInspector(nearest);
      drawMap();
      document.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
    }
  });
});

// Layer Buttons
document.querySelectorAll(".layer-btn").forEach(btn => {
  btn.addEventListener("click", () => selectLayer(btn.dataset.layer));
});

// Play/Pause button
playButton.addEventListener("click", toggleAnimation);

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
  runButton.textContent = "Generating forecast...";
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

let archiveDates = [];
let isArchivePlaying = false;
let archivePlayTimer = null;

function formatArchiveDayAsForecast(archiveData) {
  const cells = archiveData.cells.map(c => ({
    grid_id: c.id,
    lat: c.lat,
    lon: c.lon,
    warning_code: c.w,
    p_none: c.w === 0 ? 0.98 : 0.05,
    p_mild: c.w === 1 ? 0.75 : 0.10,
    p_strong: c.w === 2 ? 0.65 : 0.15,
    p_heavy: c.w === 3 ? 0.55 : 0.05,
    p_strong_or_heavy_within50: c.w >= 2 ? 0.85 : 0.20,
    p_heavy_within50: c.w === 3 ? 0.75 : 0.10,
    expected_region50_count: c.s,
  }));
  const heavyCount = archiveData.cells.filter(c => c.w === 3).length;
  const strongCount = archiveData.cells.filter(c => c.w === 2).length;
  const mildCount = archiveData.cells.filter(c => c.w === 1).length;

  return {
    status: "complete",
    archive_mode: true,
    date: archiveData.date,
    periods: [
      {
        horizon_hours: 24,
        valid_time_ist: `${archiveData.date} · 24h All-India Archive (${archiveData.total.toLocaleString()} strokes)`,
        warning_counts: {
          none: Math.max(0, 5070 - (heavyCount + strongCount + mildCount)),
          mild: mildCount,
          strong: strongCount,
          heavy: heavyCount,
        },
        max_heavy_probability: heavyCount > 0 ? 0.75 : 0.10,
        cells: cells,
      }
    ]
  };
}

async function loadArchiveDay(dateStr) {
  try {
    setStatus("", `Loading ${dateStr}...`);
    const resp = await fetch(`/api/forecast/archive/day?date=${encodeURIComponent(dateStr)}`);
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    const formatted = formatArchiveDayAsForecast(data);
    renderForecast(formatted);

    document.getElementById("slider-date-display").textContent = dateStr;
    const picker = document.getElementById("archive-date-picker");
    if (picker) picker.value = dateStr;

    const slider = document.getElementById("archive-slider");
    const idx = archiveDates.indexOf(dateStr);
    if (slider && idx >= 0) slider.value = idx;

    document.querySelectorAll(".preset-pill").forEach(p => {
      p.classList.toggle("active", p.dataset.date === dateStr);
    });

    setStatus("ready", `Loaded 2023 archive: ${dateStr} (${data.total.toLocaleString()} strokes)`);
  } catch (err) {
    setStatus("error", `Archive error: ${err.message}`);
  }
}

function toggleArchivePlay() {
  const playBtn = document.getElementById("archive-play-btn");
  if (isArchivePlaying) {
    clearInterval(archivePlayTimer);
    isArchivePlaying = false;
    if (playBtn) playBtn.textContent = "▶ Play Year";
  } else {
    if (!archiveDates.length) return;
    isArchivePlaying = true;
    if (playBtn) playBtn.textContent = "⏸ Pause";
    archivePlayTimer = setInterval(() => {
      const slider = document.getElementById("archive-slider");
      if (!slider) return;
      let nextIdx = (parseInt(slider.value, 10) + 1) % archiveDates.length;
      slider.value = nextIdx;
      loadArchiveDay(archiveDates[nextIdx]);
    }, 900);
  }
}

async function initArchiveControls() {
  try {
    const manifest = await getJson("/api/forecast/archive/dates");
    if (manifest && manifest.dates) {
      archiveDates = manifest.dates;
      const slider = document.getElementById("archive-slider");
      if (slider) {
        slider.min = "0";
        slider.max = String(archiveDates.length - 1);
        const defaultDate = "2023-07-21";
        const defIdx = archiveDates.indexOf(defaultDate);
        if (defIdx >= 0) {
          slider.value = String(defIdx);
          document.getElementById("slider-date-display").textContent = defaultDate;
        }
      }
    }
  } catch (err) {
    console.warn("Archive manifest not available:", err);
  }

  const picker = document.getElementById("archive-date-picker");
  if (picker) {
    picker.addEventListener("change", () => {
      const val = picker.value;
      if (val) loadArchiveDay(val);
    });
  }

  const slider = document.getElementById("archive-slider");
  if (slider) {
    slider.addEventListener("input", () => {
      const idx = parseInt(slider.value, 10);
      if (archiveDates[idx]) {
        document.getElementById("slider-date-display").textContent = archiveDates[idx];
      }
    });
    slider.addEventListener("change", () => {
      const idx = parseInt(slider.value, 10);
      if (archiveDates[idx]) {
        loadArchiveDay(archiveDates[idx]);
      }
    });
  }

  const loadBtn = document.getElementById("archive-load-btn");
  if (loadBtn && picker) {
    loadBtn.addEventListener("click", () => {
      if (picker.value) loadArchiveDay(picker.value);
    });
  }

  const playBtn = document.getElementById("archive-play-btn");
  if (playBtn) {
    playBtn.addEventListener("click", toggleArchivePlay);
  }

  document.querySelectorAll(".preset-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      if (pill.dataset.date) {
        loadArchiveDay(pill.dataset.date);
      }
    });
  });
}

window.addEventListener("resize", drawMap);
loadInitial();
initArchiveControls();
