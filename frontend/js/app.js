/**
 * CRC-Scan Clinical AI — Frontend Application
 * Features: Heatmap, Grad-CAM, TME Radar, PDF Export, Magnifier Lens
 */

// ── Constants ─────────────────────────────────────────────────────────────────
const CLASS_COLORS = {
  ADI: "#5DBCD2", LYM: "#9470C4", MUC: "#F5A623", MUS: "#B8E986",
  NCS: "#9B9B9B", NOR: "#50E3C2", BLD: "#D0021B", FCT: "#F8E71C",
  TUM: "#FF0055", UNCERTAIN: "#FFFF00",
};
const CLASS_ORDER = ["ADI","LYM","MUC","MUS","NCS","NOR","BLD","FCT","TUM","UNCERTAIN"];
const FULL_NAMES  = {
  ADI:"Adipose", LYM:"Lymphocyte", MUC:"Mucin", MUS:"Muscle", NCS:"Necrotic Debris",
  NOR:"Normal", BLD:"Blood", FCT:"Connective Tissue", TUM:"Tumor", UNCERTAIN:"Uncertain",
};
const GLOSSARY = {
  ADI: "Adipose tissue (fat cells).",
  LYM: "Lymphocytes (immune infiltration).",
  MUC: "Mucin (secretory tissue).",
  MUS: "Smooth muscle layers.",
  NCS: "Necrotic debris (cell death).",
  NOR: "Normal mucosal architecture.",
  BLD: "Red blood cells (vascularity).",
  FCT: "Loose connective tissue (stroma).",
  TUM: "Malignant Adenocarcinoma cells.",
  UNCERTAIN: "Below confidence threshold — human review required.",
};

// ── State ─────────────────────────────────────────────────────────────────────
let currentFile    = null;
let analysisResult = null;
let originalImageUrl = null;   // Object URL for magnifier lens
let donutChart = null;
let barChart   = null;
let tmeChart   = null;

// ── DOM helpers ───────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  buildLegend();
  checkHealth();
  setupUploadZone();
  setupSlider();
  setupButtons();
});

// ── Health Check ──────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch("/api/health");
    const d = await r.json();
    const badge = $("modelStatusBadge");
    badge.className = "status-badge";
    if (d.weights_status.startsWith("✅")) {
      badge.classList.add("status-ok");
      badge.innerHTML = `<span class="badge-dot"></span>${d.weights_status}`;
    } else if (d.weights_status.startsWith("❌")) {
      badge.classList.add("status-error");
      badge.innerHTML = `<span class="badge-dot"></span>Load Error`;
    } else {
      badge.classList.add("status-loading");
      badge.innerHTML = `<span class="badge-dot"></span>No Weights`;
    }
    $("deviceBadge").textContent = d.device || "—";
  } catch {
    $("modelStatusBadge").className = "status-badge status-error";
    $("modelStatusBadge").innerHTML = `<span class="badge-dot"></span>Server Offline`;
  }
}

// ── Legend ────────────────────────────────────────────────────────────────────
function buildLegend() {
  const el = $("tissueLegend");
  el.innerHTML = Object.entries(CLASS_COLORS)
    .filter(([k]) => k !== "UNCERTAIN")
    .map(([code, color]) => `
      <div class="legend-item">
        <div class="legend-dot" style="background:${color}"></div>
        <span class="legend-code" style="color:${color}">${code}</span>
        <span class="legend-desc">${GLOSSARY[code] || ""}</span>
      </div>`).join("");
}

// ── Upload Zone ───────────────────────────────────────────────────────────────
function setupUploadZone() {
  const zone = $("uploadZone");
  const input = $("fileInput");

  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", (e) => { if (e.key==="Enter"||e.key===" ") input.click(); });
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    if (e.dataTransfer.files[0]) handleFileSelect(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => { if (input.files[0]) handleFileSelect(input.files[0]); });
  $("clearBtn").addEventListener("click", (e) => { e.stopPropagation(); clearWorkspace(); });
}

function handleFileSelect(file) {
  if (!file.type.match(/image\/(jpeg|jpg|png)/)) { alert("PNG or JPEG only."); return; }
  currentFile = file;
  showState("welcome");

  // Feature 3: create Object URL for magnifier
  if (originalImageUrl) URL.revokeObjectURL(originalImageUrl);
  originalImageUrl = URL.createObjectURL(file);

  const reader = new FileReader();
  reader.onload = (e) => {
    $("previewImg").src = e.target.result;
    $("previewFilename").textContent = file.name;
    $("validationChip").textContent = "";
    $("validationChip").className = "validation-chip";
    $("previewContainer").classList.remove("hidden");
    $("runBtn").disabled = false;
  };
  reader.readAsDataURL(file);
}

// ── Slider ────────────────────────────────────────────────────────────────────
function setupSlider() {
  const s = $("thresholdSlider");
  s.addEventListener("input", () => {
    $("thresholdDisplay").textContent = `${s.value}%`;
    updateSliderBg(s.value);
  });
  updateSliderBg(30);
}
function updateSliderBg(val) {
  $("thresholdSlider").style.background =
    `linear-gradient(to right,#FF0055 0%,#FF0055 ${val}%,#1E1E38 ${val}%)`;
}

// ── Buttons ───────────────────────────────────────────────────────────────────
function setupButtons() {
  $("runBtn").addEventListener("click", runAnalysis);
  $("resetBtn").addEventListener("click", clearWorkspace);
  $("gradcamClose").addEventListener("click", closeGradcam);
  $("exportPdfBtn").addEventListener("click", exportPDF);
}

// ══════════════════════════════════════════════════════════════════════════════
//  RUN ANALYSIS
// ══════════════════════════════════════════════════════════════════════════════
async function runAnalysis() {
  if (!currentFile) return;
  $("runBtn").disabled = true;
  $("resetBtn").classList.add("hidden");
  showState("loading");
  animateLoadingSteps();

  const fd = new FormData();
  fd.append("file", currentFile);
  fd.append("threshold", $("thresholdSlider").value);

  try {
    const r    = await fetch("/api/analyze", { method: "POST", body: fd });
    const data = await r.json();

    if (data.status === "invalid") {
      $("validationChip").textContent = `❌ ${data.reason}`;
      $("validationChip").className = "validation-chip error";
      $("invalidReason").textContent = data.reason;
      showState("invalid");
      $("runBtn").disabled = false;
      return;
    }
    if (data.status === "error") {
      alert(`Analysis error: ${data.reason}`);
      showState("welcome");
      $("runBtn").disabled = false;
      return;
    }

    $("validationChip").textContent = data.validation_msg || "✅ Valid";
    $("validationChip").className = "validation-chip ok";
    analysisResult = data;
    renderAnalysis(data);
    showState("analysis");
    $("resetBtn").classList.remove("hidden");

  } catch (err) {
    alert(`Network error: ${err.message}`);
    showState("welcome");
    $("runBtn").disabled = false;
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  RENDER ANALYSIS
// ══════════════════════════════════════════════════════════════════════════════
function renderAnalysis(data) {
  // Heatmap
  const heatmapImg = $("heatmapImg");
  heatmapImg.src = `data:image/png;base64,${data.heatmap}`;
  $("heatmapInfo").textContent = `${data.original_size.width} × ${data.original_size.height} px`;
  $("heatmapInfo").classList.remove("hidden");
  heatmapImg.onclick = (e) => handleHeatmapClick(e, heatmapImg, data.original_size);

  // Charts
  renderDonutChart(data.distribution, data.dominant_tissue);
  renderBarChart(data.distribution);

  // Feature 1: TME Radar
  renderTMERadar(data);

  // Dominant label
  const dt = data.dominant_tissue;
  const dtColor = CLASS_COLORS[dt] || "#fff";
  $("dominantLabel").innerHTML = `Dominant: <strong style="color:${dtColor}">${dt}</strong> — ${GLOSSARY[dt]||""}`;
  $("dcLabel").textContent = "TOP";
  $("dcVal").textContent = dt;

  // Clinical metrics
  const c = data.clinical;
  $("mTumorVal").textContent = `${c.tum_pct.toFixed(1)}%`;
  $("mTumorTag").textContent = c.tum_pct > 20 ? "⚠️ Malignant" : "Low Risk";
  $("mTumorTag").style.color = c.tum_pct > 20 ? "#FF6699" : "#50E3C2";
  $("mTILVal").textContent   = `${c.til_score.toFixed(1)}%`;
  $("mNecVal").textContent   = `${c.ncs_pct.toFixed(1)}%`;
  $("mUncertVal").textContent = `${c.uncert_pct.toFixed(1)}%`;
  $("mUncertVal").innerHTML += ' <span style="font-size:10px; color:#50E3C2; font-weight:normal;">[Calibrated]</span>';
  $("diagNote").innerHTML = `<strong>Tissue Dominance:</strong> ${dt} (${GLOSSARY[dt]||""})<br/><strong>Diagnostic Note:</strong> ${data.diagnostic_note}`;

  // Feature 3: Magnifier setup
  setupMagnifier();
}

// ── Donut Chart ───────────────────────────────────────────────────────────────
function renderDonutChart(dist) {
  if (donutChart) { donutChart.destroy(); donutChart = null; }
  const labels=[], vals=[], colors=[];
  CLASS_ORDER.forEach((cls) => {
    const e = dist[cls];
    if (e && e.percentage > 0) { labels.push(cls); vals.push(e.percentage); colors.push(CLASS_COLORS[cls]||"#888"); }
  });
  const ctx = $("donutChart").getContext("2d");
  donutChart = new Chart(ctx, {
    type: "doughnut",
    data: { labels, datasets: [{ data: vals, backgroundColor: colors, borderWidth: 0, hoverOffset: 10 }] },
    options: {
      responsive: true, maintainAspectRatio: true, cutout: "68%",
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#161628", borderColor: "rgba(255,255,255,0.1)", borderWidth: 1,
          callbacks: { label: (ctx) => ` ${ctx.raw.toFixed(1)}%  —  ${GLOSSARY[ctx.label]||ctx.label}` }
        }
      }
    }
  });
}

// ── Bar Chart ─────────────────────────────────────────────────────────────────
function renderBarChart(dist) {
  if (barChart) { barChart.destroy(); barChart = null; }
  const entries = CLASS_ORDER
    .filter((c) => dist[c] && dist[c].percentage > 0)
    .map((c) => ({ cls: c, pct: dist[c].percentage, color: CLASS_COLORS[c] }))
    .sort((a, b) => b.pct - a.pct);

  const gc = "rgba(255,255,255,0.05)";
  const ctx = $("barChart").getContext("2d");
  barChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: entries.map((e) => e.cls),
      datasets: [{ data: entries.map((e) => e.pct), backgroundColor: entries.map((e) => e.color+"CC"),
        hoverBackgroundColor: entries.map((e) => e.color), borderRadius: { topLeft:4, topRight:4 }, borderSkipped:false }],
    },
    options: {
      indexAxis: "y", responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: {
        backgroundColor:"#161628", borderColor:"rgba(255,255,255,0.1)", borderWidth:1,
        callbacks: { label: (ctx) => ` ${ctx.raw.toFixed(2)}%` }
      }},
      scales: {
        x: { grid:{ color:gc }, ticks:{ color:"#555577", font:{size:11}, callback:(v)=>`${v}%` }, border:{ color:gc }},
        y: { grid:{ color:"transparent" }, ticks:{
          color: (ctx) => CLASS_COLORS[entries[ctx.index]?.cls]||"#888",
          font:{ family:"'JetBrains Mono',monospace", size:11, weight:"bold" }
        }, border:{ color:gc }}
      }
    }
  });
  const canvas = $("barChart");
  canvas.parentElement.style.height = `${Math.max(220, entries.length*34)}px`;
  barChart.resize();
}

// ══════════════════════════════════════════════════════════════════════════════
//  FEATURE 1 — TME RADAR CHART
// ══════════════════════════════════════════════════════════════════════════════
function renderTMERadar(data) {
  if (tmeChart) { tmeChart.destroy(); tmeChart = null; }

  const dist = data.distribution;
  const c    = data.clinical;

  const tumPct  = dist["TUM"]?.percentage || 0;
  const lymPct  = dist["LYM"]?.percentage || 0;
  const fctPct  = dist["FCT"]?.percentage || 0;
  const musPct  = dist["MUS"]?.percentage || 0;
  const bldPct  = dist["BLD"]?.percentage || 0;
  const ncsPct  = dist["NCS"]?.percentage || 0;
  const stromal = fctPct + musPct;

  // Clinical reference maxima for normalisation
  const MAX = { tumor:60, immune:50, stromal:70, vascular:25, necrosis:40 };
  const norm = (v, m) => Math.min(100, (v/m)*100);

  const patient = [
    norm(tumPct,  MAX.tumor),
    norm(lymPct,  MAX.immune),
    norm(stromal, MAX.stromal),
    norm(bldPct,  MAX.vascular),
    norm(ncsPct,  MAX.necrosis),
  ];
  const healthy = [
    norm(5,  MAX.tumor),    // ~8
    norm(15, MAX.immune),   // 30
    norm(35, MAX.stromal),  // 50
    norm(5,  MAX.vascular), // 20
    norm(1,  MAX.necrosis), // 2.5
  ];

  const labels = ["Tumor Density","Immune Infiltration","Stromal Content","Vascular Density","Necrotic Burden"];

  const ctx = $("tmeChart").getContext("2d");
  tmeChart = new Chart(ctx, {
    type: "radar",
    data: {
      labels,
      datasets: [
        {
          label: "Patient Slide",
          data: patient,
          backgroundColor: "rgba(255,0,85,0.15)",
          borderColor: "#FF0055",
          borderWidth: 2,
          pointBackgroundColor: "#FF0055",
          pointBorderColor: "#0C0C1A",
          pointRadius: 5, pointHoverRadius: 7,
        },
        {
          label: "Healthy Reference",
          data: healthy,
          backgroundColor: "rgba(80,227,194,0.08)",
          borderColor: "rgba(80,227,194,0.6)",
          borderDash: [6,4],
          borderWidth: 2,
          pointBackgroundColor: "#50E3C2",
          pointBorderColor: "#0C0C1A",
          pointRadius: 4,
        }
      ]
    },
    options: {
      responsive:true, maintainAspectRatio:true,
      scales: {
        r: {
          min:0, max:100,
          ticks: { display:false, stepSize:25 },
          grid: { color:"rgba(255,255,255,0.06)" },
          angleLines: { color:"rgba(255,255,255,0.05)" },
          pointLabels: { color:"#8888BB", font:{ family:"'Inter',sans-serif", size:11, weight:"600" } }
        }
      },
      plugins: {
        legend: {
          display:true, position:"bottom",
          labels: { color:"#8888BB", font:{ family:"'Inter',sans-serif", size:11 }, padding:16, usePointStyle:true }
        },
        tooltip: {
          backgroundColor:"#161628", borderColor:"rgba(255,255,255,0.1)", borderWidth:1,
          callbacks: {
            label: (ctx) => {
              const raw  = { "Patient Slide": patient, "Healthy Reference": healthy };
              const actuals = { "Patient Slide":
                [tumPct,lymPct,stromal,bldPct,ncsPct].map((v)=>`${v.toFixed(1)}%`),
                "Healthy Reference": ["~5%","~15%","~35%","~5%","~1%"]
              };
              const idx = ctx.dataIndex;
              return ` ${ctx.dataset.label}: ${actuals[ctx.dataset.label]?.[idx] || ctx.raw.toFixed(1)}`;
            }
          }
        }
      }
    }
  });

  // Fill TME insights panel
  updateTMEInsights({ tumPct, lymPct, ncsPct, stromal, til: c.til_score });

  // Badge
  const riskLevel = tumPct > 30 ? "High Risk" : tumPct > 10 ? "Moderate Risk" : "Low Risk";
  $("tmeBadge").textContent = riskLevel;
  $("tmeBadge").style.background    = tumPct > 30 ? "rgba(255,0,85,0.15)" : tumPct > 10 ? "rgba(245,166,35,0.15)" : "rgba(80,227,194,0.15)";
  $("tmeBadge").style.borderColor   = tumPct > 30 ? "rgba(255,0,85,0.4)"  : tumPct > 10 ? "rgba(245,166,35,0.4)"  : "rgba(80,227,194,0.4)";
  $("tmeBadge").style.color         = tumPct > 30 ? "#FF6699"              : tumPct > 10 ? "#F5A623"               : "#50E3C2";
}

function updateTMEInsights({ tumPct, lymPct, ncsPct, stromal, til }) {
  const el = $("tmeCommentary");
  if (!el) return;

  const insights = [
    {
      riskClass: tumPct > 30 ? "risk-high" : tumPct > 10 ? "risk-medium" : "risk-low",
      icon: tumPct > 30 ? "🔴" : tumPct > 10 ? "🟠" : "🟢",
      text: tumPct > 30
        ? `<strong>High Tumor Density (${tumPct.toFixed(1)}%)</strong> — Significant malignant cell content exceeds 30%. Immediate staging workup recommended.`
        : tumPct > 10
        ? `<strong>Moderate Tumor Density (${tumPct.toFixed(1)}%)</strong> — Localized malignant clusters. Monitor for progression.`
        : `<strong>Low Tumor Density (${tumPct.toFixed(1)}%)</strong> — Minimal malignant content. Likely early-stage or benign.`,
    },
    {
      riskClass: til > 50 ? "risk-low" : til > 20 ? "risk-medium" : "risk-high",
      icon: til > 50 ? "🟢" : til > 20 ? "🟡" : "🔴",
      text: til > 50
        ? `<strong>High TIL Ratio (${til.toFixed(1)}%)</strong> — Robust immune response. High TIL is associated with improved prognosis and immunotherapy response.`
        : til > 20
        ? `<strong>Moderate TIL Ratio (${til.toFixed(1)}%)</strong> — Some immune infiltration observed relative to tumor mass.`
        : `<strong>Low TIL Ratio (${til.toFixed(1)}%)</strong> — Limited immune infiltration. May indicate immune evasion by tumor.`,
    },
    {
      riskClass: ncsPct > 5 ? "risk-high" : "risk-low",
      icon: ncsPct > 5 ? "🔴" : "🟢",
      text: ncsPct > 5
        ? `<strong>Elevated Necrotic Load (${ncsPct.toFixed(1)}%)</strong> — Significant cell death detected. May indicate hypoxia, rapid tumor growth, or poor perfusion.`
        : `<strong>Necrotic Load (${ncsPct.toFixed(1)}%)</strong> — Within expected range. No significant hypoxic necrosis detected.`,
    },
    {
      riskClass: stromal > 40 ? "risk-medium" : "risk-low",
      icon: stromal > 40 ? "🟡" : "🟢",
      text: stromal > 40
        ? `<strong>Dense Stroma (${stromal.toFixed(1)}%)</strong> — High desmoplastic stromal content (FCT+MUS). Dense stroma may create a tumor-promoting microenvironment.`
        : `<strong>Stromal Content (${stromal.toFixed(1)}%)</strong> — Normal stromal architecture. No excessive desmoplasia detected.`,
    },
  ];

  el.innerHTML = insights.map((i) =>
    `<div class="tme-insight ${i.riskClass}">${i.icon} ${i.text}</div>`
  ).join("");
}

// ══════════════════════════════════════════════════════════════════════════════
//  FEATURE 2 — PDF EXPORT
// ══════════════════════════════════════════════════════════════════════════════
async function exportPDF() {
  if (!analysisResult || !currentFile) return;
  const btn = $("exportPdfBtn");
  btn.disabled = true;
  btn.textContent = "⏳ Generating Report…";

  const fd = new FormData();
  fd.append("file", currentFile);

  // Send result without heatmap field (backend uses uploaded file as slide image)
  const { heatmap: _h, ...resultClean } = analysisResult;
  fd.append("result_json",    JSON.stringify(resultClean));
  fd.append("heatmap_b64",    analysisResult.heatmap);           
  
  if (tmeChart) {
    fd.append("radar_b64", tmeChart.toBase64Image());
  }
  
  fd.append("slide_filename", currentFile.name);
  fd.append("threshold",      $("thresholdSlider").value);

  try {
    const r = await fetch("/api/report", { method: "POST", body: fd });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.reason || `HTTP ${r.status}`);
    }
    const blob = await r.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    const slideBase = currentFile.name.replace(/\.[^/.]+$/, "");
    a.download = `PathologyReport_${slideBase}.pdf`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    alert(`PDF generation failed: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "📄 Export Clinical Report (PDF)";
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  FEATURE 3 — MAGNIFICATION LENS
// ══════════════════════════════════════════════════════════════════════════════
function setupMagnifier() {
  const wrap  = $("heatmapWrap");
  const lens  = $("magnifierLens");
  const hmImg = $("heatmapImg");
  if (!wrap || !lens || !hmImg) return;

  const ZOOM      = 2.5;
  const LENS_SIZE = 190;

  wrap.addEventListener("mousemove", onLensMove);
  wrap.addEventListener("mouseleave", onLensLeave);
  // Hide lens on tile click so Grad-CAM panel doesn't fight for attention
  hmImg.addEventListener("click", onLensLeave);

  function onLensMove(e) {
    if (!originalImageUrl) return;

    const imgRect  = hmImg.getBoundingClientRect();
    const wrapRect = wrap.getBoundingClientRect();

    const x = e.clientX - imgRect.left;
    const y = e.clientY - imgRect.top;

    // Only when cursor is actually over the image
    if (x < 0 || y < 0 || x > imgRect.width || y > imgRect.height) {
      lens.style.display = "none";
      return;
    }

    // Position lens circle centred on cursor (within wrap coordinate space)
    lens.style.left    = `${(e.clientX - wrapRect.left) - LENS_SIZE / 2}px`;
    lens.style.top     = `${(e.clientY - wrapRect.top)  - LENS_SIZE / 2}px`;
    lens.style.display = "block";

    // Zoom the *original* slide image inside the lens
    const bgW = imgRect.width  * ZOOM;
    const bgH = imgRect.height * ZOOM;
    const bgX = (LENS_SIZE / 2) - x * ZOOM;
    const bgY = (LENS_SIZE / 2) - y * ZOOM;

    lens.style.backgroundImage    = `url(${originalImageUrl})`;
    lens.style.backgroundSize     = `${bgW}px ${bgH}px`;
    lens.style.backgroundPosition = `${bgX}px ${bgY}px`;
  }

  function onLensLeave() {
    lens.style.display = "none";
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  GRAD-CAM (existing)
// ══════════════════════════════════════════════════════════════════════════════
function handleHeatmapClick(e, imgEl, origSize) {
  const rect   = imgEl.getBoundingClientRect();
  const origX  = (e.clientX - rect.left)  * (origSize.width  / rect.width);
  const origY  = (e.clientY - rect.top)   * (origSize.height / rect.height);
  const TILE   = 256;
  fetchGradcam(Math.floor(origX/TILE)*TILE, Math.floor(origY/TILE)*TILE);
}

async function fetchGradcam(tileX, tileY) {
  if (!currentFile) return;
  const card    = $("gradcamCard");
  const loading = $("gradcamLoading");
  const result  = $("gradcamResult");

  card.classList.remove("hidden");
  loading.classList.remove("hidden");
  result.classList.add("hidden");
  card.scrollIntoView({ behavior:"smooth", block:"nearest" });

  const fd = new FormData();
  fd.append("file",   currentFile);
  fd.append("tile_x", tileX);
  fd.append("tile_y", tileY);

  try {
    const r    = await fetch("/api/gradcam", { method:"POST", body:fd });
    const data = await r.json();
    if (data.status !== "ok") { loading.innerHTML = `<span style="color:#FF6699">Error: ${data.reason}</span>`; return; }

    $("gcTile").src = `data:image/png;base64,${data.tile}`;
    $("gcMap").src  = `data:image/png;base64,${data.gradcam}`;
    const col = CLASS_COLORS[data.label] || "#fff";
    const lbl = $("gcPredLabel");
    lbl.textContent   = data.label;
    lbl.style.background  = col + "22";
    lbl.style.color       = col;
    lbl.style.borderColor = col + "55";
    $("gcConf").textContent  = `Confidence: ${data.confidence.toFixed(1)}%`;
    $("gcGloss").textContent = data.glossary || GLOSSARY[data.label] || "";

    loading.classList.add("hidden");
    result.classList.remove("hidden");
  } catch (err) {
    loading.innerHTML = `<span style="color:#FF6699">Network error: ${err.message}</span>`;
  }
}

function closeGradcam() { $("gradcamCard").classList.add("hidden"); }

// ══════════════════════════════════════════════════════════════════════════════
//  STATE MACHINE
// ══════════════════════════════════════════════════════════════════════════════
function showState(name) {
  ["stateWelcome","stateInvalid","stateLoading","stateAnalysis"]
    .forEach((id) => $(id).classList.add("hidden"));
  if (name === "welcome")  $("stateWelcome").classList.remove("hidden");
  if (name === "invalid")  $("stateInvalid").classList.remove("hidden");
  if (name === "loading")  $("stateLoading").classList.remove("hidden");
  if (name === "analysis") $("stateAnalysis").classList.remove("hidden");
}

function animateLoadingSteps() {
  const steps = [$("ls1"),$("ls2"),$("ls3"),$("ls4")];
  steps.forEach((s) => { s.className = "lstep"; });
  steps[0].classList.add("active");
  let i = 1;
  const iv = setInterval(() => {
    if (i >= steps.length) { clearInterval(iv); return; }
    steps[i-1].classList.remove("active");
    steps[i-1].classList.add("done");
    steps[i].classList.add("active");
    i++;
  }, 1800);
}

// ══════════════════════════════════════════════════════════════════════════════
//  RESET
// ══════════════════════════════════════════════════════════════════════════════
function clearWorkspace() {
  currentFile    = null;
  analysisResult = null;
  $("fileInput").value = "";

  if (originalImageUrl) { URL.revokeObjectURL(originalImageUrl); originalImageUrl = null; }

  $("previewContainer").classList.add("hidden");
  $("previewImg").src = "";
  $("previewFilename").textContent = "—";
  $("validationChip").textContent  = "";
  $("validationChip").className    = "validation-chip";
  $("gradcamCard").classList.add("hidden");
  $("heatmapInfo").classList.add("hidden");
  $("runBtn").disabled = true;
  $("resetBtn").classList.add("hidden");

  if (donutChart) { donutChart.destroy(); donutChart = null; }
  if (barChart)   { barChart.destroy();   barChart   = null; }
  if (tmeChart)   { tmeChart.destroy();   tmeChart   = null; }

  showState("welcome");
  checkHealth();
}
