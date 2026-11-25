"use strict";

const SUMMARY_URL = "/data/stats/statistical-summary.json";
const integer = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });
const compact = new Intl.NumberFormat("en-IN", { notation: "compact", maximumFractionDigits: 1 });
const dateFormat = new Intl.DateTimeFormat("en-IN", {
  day: "2-digit",
  month: "short",
  year: "numeric",
  timeZone: "Asia/Kolkata",
});

function byId(id) {
  return document.getElementById(id);
}

function text(id, value) {
  const node = byId(id);
  if (node) node.textContent = value;
}

function renderSummary(data) {
  text("summary-strokes", integer.format(data.summary.retained_strokes));
  text("summary-scored", integer.format(data.summary.scored_windows));
  text("summary-masked", integer.format(data.summary.masked_windows));
  text("summary-years", data.summary.years_covered);
}

function renderCoverage(data) {
  const scored = data.summary.scored_windows;
  const masked = data.summary.masked_windows;
  const total = scored + masked;
  const scoredPercent = (100 * scored) / total;
  const maskedPercent = 100 - scoredPercent;
  byId("coverage-scored-bar").style.width = `${scoredPercent}%`;
  byId("coverage-masked-bar").style.width = `${maskedPercent}%`;
  text("coverage-scored-label", `${integer.format(scored)} (${scoredPercent.toFixed(1)}%)`);
  text("coverage-masked-label", `${integer.format(masked)} (${maskedPercent.toFixed(1)}%)`);
  byId("coverage-meter").setAttribute(
    "aria-label",
    `${integer.format(scored)} scored windows and ${integer.format(masked)} masked windows`,
  );
  text(
    "coverage-method",
    `${integer.format(data.manifest.source_partition_count)} monthly Parquet partitions were reconciled. ${integer.format(data.manifest.source_days_missing)} missing and ${integer.format(data.manifest.source_days_unreadable)} unreadable source days remained masked; all build checks passed.`,
  );
}

function renderAnnual(items) {
  const root = byId("annual-chart");
  const maximum = Math.max(...items.map((item) => item.strokes));
  root.replaceChildren(
    ...items.map((item) => {
      const row = document.createElement("div");
      row.className = `bar-row${item.strokes === maximum ? " is-peak" : ""}`;
      row.setAttribute("title", `${item.year}: ${integer.format(item.strokes)} retained strokes`);
      row.innerHTML = `
        <strong>${item.year}</strong>
        <span class="bar-track"><span class="bar-fill" style="width:${(100 * item.strokes / maximum).toFixed(2)}%"></span></span>
        <span class="bar-value">${compact.format(item.strokes)}</span>`;
      return row;
    }),
  );
}

function renderMonthly(items) {
  const root = byId("monthly-chart");
  const maximum = Math.max(...items.map((item) => item.strokes));
  root.replaceChildren(
    ...items.map((item) => {
      const column = document.createElement("div");
      const isPeak = item.strokes === maximum;
      column.className = `month-column${isPeak ? " is-peak" : ""}`;
      column.setAttribute(
        "title",
        `${item.label}: ${integer.format(item.strokes)} strokes across five years (${item.share_percent}% of all retained strokes)`,
      );
      column.innerHTML = `
        <span class="month-value">${compact.format(item.strokes)}</span>
        <span class="month-bar" style="height:${Math.max(2, 100 * item.strokes / maximum).toFixed(2)}%"></span>
        <span class="month-label">${item.label}</span>`;
      return column;
    }),
  );
}

function renderSeasonTime(items) {
  const root = byId("season-time-matrix");
  const seasons = [...new Set(items.map((item) => item.season))];
  const periods = ["Night", "Morning", "Afternoon", "Evening"];
  const maximum = Math.max(...items.map((item) => item.strokes));
  const lookup = new Map(items.map((item) => [`${item.season}|${item.period}`, item]));
  const nodes = [];

  const corner = document.createElement("div");
  corner.className = "matrix-cell matrix-head";
  corner.textContent = "Season / IST period";
  nodes.push(corner);
  periods.forEach((period) => {
    const head = document.createElement("div");
    head.className = "matrix-cell matrix-head";
    head.textContent = period;
    nodes.push(head);
  });

  seasons.forEach((season) => {
    const rowHead = document.createElement("div");
    rowHead.className = "matrix-cell matrix-row-head";
    rowHead.textContent = season;
    nodes.push(rowHead);
    periods.forEach((period) => {
      const item = lookup.get(`${season}|${period}`);
      const cell = document.createElement("div");
      const heat = 0.08 + 0.64 * Math.sqrt(item.strokes / maximum);
      cell.className = "matrix-cell matrix-data";
      cell.style.setProperty("--heat", heat.toFixed(3));
      cell.setAttribute("title", `${season}, ${period}: ${integer.format(item.strokes)} strokes`);
      cell.innerHTML = `<strong>${compact.format(item.strokes)}</strong><small>${item.share_percent}%</small>`;
      nodes.push(cell);
    });
  });
  root.replaceChildren(...nodes);
}

function renderSpatial(data) {
  const rows = data.hotspots.slice(0, 6).map((item) => {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${item.rank}</td><td>${item.latitude.toFixed(2)}°N, ${item.longitude.toFixed(2)}°E</td><td>${integer.format(item.strokes)}</td>`;
    return row;
  });
  byId("hotspot-table").replaceChildren(...rows);

  const days = data.top_days.slice(0, 6).map((item) => {
    const row = document.createElement("li");
    const parsed = new Date(`${item.date}T12:00:00+05:30`);
    row.innerHTML = `<span>${dateFormat.format(parsed)}</span><strong>${integer.format(item.strokes)}</strong>`;
    return row;
  });
  byId("top-days").replaceChildren(...days);
}

function renderTakeaways(takeaways) {
  text("takeaway-coverage", takeaways.coverage);
  text("takeaway-annual", takeaways.annual);
  text("takeaway-monthly", takeaways.monthly);
  text("takeaway-cycle", takeaways.daily_cycle);
  text("takeaway-spatial", takeaways.spatial);
}

async function start() {
  const status = byId("data-status");
  try {
    const response = await fetch(SUMMARY_URL, { cache: "no-cache" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (data.manifest.reconciliation_status !== "passed") {
      throw new Error("data reconciliation did not pass");
    }
    renderSummary(data);
    renderCoverage(data);
    renderAnnual(data.annual);
    renderMonthly(data.monthly);
    renderSeasonTime(data.season_time);
    renderSpatial(data);
    renderTakeaways(data.takeaways);
    status.textContent = `Audit passed · ${data.manifest.source_partition_count} partitions reconciled`;
  } catch (error) {
    status.classList.add("error");
    status.textContent = "Analysis data unavailable";
    console.error("Unable to render statistical analysis", error);
  }
}

document.addEventListener("DOMContentLoaded", start);
