import { fetchArtifact, initInfoToggle, populateInfoPanel, populateCrossLink } from "./artifactInfo.js";

const params = new URLSearchParams(window.location.search);
const artifactId = params.get("id");

const statusEl = document.getElementById("status");
const canvas = document.getElementById("msiCanvas");
const ctx = canvas.getContext("2d");
const eqRow = document.getElementById("eqRow");
const recipeRow = document.getElementById("recipeRow");
const recipeCaption = document.getElementById("recipeCaption");
const exposureSlider = document.getElementById("exposureSlider");
const contrastSlider = document.getElementById("contrastSlider");
const exposureValue = document.getElementById("exposureValue");
const contrastValue = document.getElementById("contrastValue");
const resetButton = document.getElementById("resetButton");

function setStatus(text) {
  statusEl.textContent = text;
  statusEl.style.opacity = "1";
}

function fadeStatus() {
  setTimeout(() => {
    statusEl.style.opacity = "0";
  }, 1800);
}

// Rough, illustrative wavelength -> hue mapping (not a physical spectral
// model) so each fader is tinted roughly where it sits in the UV -> visible
// -> IR range: violet for UV, a rainbow sweep across the visible bands,
// deepening red for near-IR. Purely a visual affordance for scanning the
// fader row at a glance.
const HUE_STOPS = [
  { nm: 365, hue: 285 },
  { nm: 400, hue: 265 },
  { nm: 450, hue: 235 },
  { nm: 490, hue: 180 },
  { nm: 530, hue: 120 },
  { nm: 580, hue: 55 },
  { nm: 645, hue: 5 },
  { nm: 700, hue: 0 },
];

function wavelengthToColor(nm) {
  if (nm > 700) {
    // Near-IR: stay red-hued but darken/desaturate with increasing wavelength.
    const t = Math.min((nm - 700) / (940 - 700), 1);
    const lightness = 45 - t * 25;
    return `hsl(0, 70%, ${lightness}%)`;
  }
  let lo = HUE_STOPS[0];
  let hi = HUE_STOPS[HUE_STOPS.length - 1];
  for (let i = 0; i < HUE_STOPS.length - 1; i++) {
    if (nm >= HUE_STOPS[i].nm && nm <= HUE_STOPS[i + 1].nm) {
      lo = HUE_STOPS[i];
      hi = HUE_STOPS[i + 1];
      break;
    }
  }
  const span = hi.nm - lo.nm || 1;
  const t = Math.min(Math.max((nm - lo.nm) / span, 0), 1);
  const hue = lo.hue + (hi.hue - lo.hue) * t;
  return `hsl(${hue.toFixed(0)}, 75%, 55%)`;
}

initInfoToggle();

const state = {
  bands: [],
  msiBaseUrl: "",
  weights: {},
  faderInputs: {},
  faderValueEls: {},
  resolvedBitmaps: new Map(),
  loading: new Set(),
  canvasReady: false,
  redrawScheduled: false,
  exposure: 1,
  contrast: 1,
  lastRecipe: null,
};

// retrying is internal (true on the automatic retry pass) so a genuinely
// failed retry doesn't loop forever.
function loadBitmap(band, onDone, retrying) {
  if (state.resolvedBitmaps.has(band.wavelength) || state.loading.has(band.wavelength)) {
    if (onDone) onDone();
    return;
  }
  state.loading.add(band.wavelength);
  fetch(state.msiBaseUrl + band.file)
    .then((res) => {
      if (!res.ok) throw new Error(`band fetch failed: ${res.status}`);
      return res.blob();
    })
    .then((blob) => createImageBitmap(blob))
    .then((bitmap) => {
      state.resolvedBitmaps.set(band.wavelength, bitmap);
      state.loading.delete(band.wavelength);
      if (!state.canvasReady) {
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
        state.canvasReady = true;
      }
      scheduleRedraw();
      if (onDone) onDone();
    })
    .catch((err) => {
      state.loading.delete(band.wavelength);
      // A background prefetch competing with other fetches is the likely
      // cause of a dropped connection, not a permanently broken URL — one
      // quiet retry clears most of those without bothering the visitor.
      if (!retrying) {
        setTimeout(() => loadBitmap(band, onDone, true), 400);
        return;
      }
      console.error(`failed to load band ${band.wavelength}nm`, err);
      if (onDone) onDone();
    });
}

// Background-prefetches the rest of the bands (the ones not already active)
// a few at a time rather than firing all 16 fetches at once — a dev server
// or modest static host can choke on that many simultaneous multi-megabyte
// requests, and it's no better for the visitor either (all 16 competing for
// bandwidth delays the ones actually on screen).
function prefetchRemainingBands(bands, concurrency = 4) {
  let next = 0;
  function pump() {
    if (next >= bands.length) return;
    const band = bands[next++];
    if (state.resolvedBitmaps.has(band.wavelength) || state.loading.has(band.wavelength)) {
      pump();
      return;
    }
    loadBitmap(band, pump);
  }
  for (let i = 0; i < concurrency; i++) pump();
}

function scheduleRedraw() {
  if (state.redrawScheduled) return;
  state.redrawScheduled = true;
  requestAnimationFrame(() => {
    state.redrawScheduled = false;
    redraw();
  });
}

// Additive mixer: each band with weight > 0 gets drawn onto the canvas with
// globalCompositeOperation "lighter" (src+dst, clamped) so raising a fader
// visibly ADDS that wavelength's signal in rather than replacing the image
// — the equalizer metaphor the user asked for. globalAlpha is clamped to
// [0,1] by the canvas spec, so weights above 1 (headroom for punching up a
// faint band) are achieved by drawing the same band multiple times.
function redraw() {
  if (!state.canvasReady) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.globalCompositeOperation = "lighter";
  for (const band of state.bands) {
    const weight = state.weights[band.wavelength] || 0;
    if (weight <= 0) continue;
    const bitmap = state.resolvedBitmaps.get(band.wavelength);
    if (!bitmap) continue; // still loading — redraw() runs again once it lands
    let remaining = weight;
    while (remaining > 0.001) {
      ctx.globalAlpha = Math.min(remaining, 1);
      ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
      remaining -= 1;
    }
  }
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
}

function applyFilter() {
  canvas.style.filter = `brightness(${state.exposure}) contrast(${state.contrast})`;
}

function clearActiveRecipeHighlight() {
  recipeRow.querySelectorAll(".recipe-chip.active").forEach((chip) => chip.classList.remove("active"));
}

function setFader(band, weight) {
  state.weights[band.wavelength] = weight;
  const input = state.faderInputs[band.wavelength];
  const valueEl = state.faderValueEls[band.wavelength];
  if (input) input.value = weight;
  if (valueEl) valueEl.textContent = `${Math.round(weight * 100)}%`;
  if (weight > 0) loadBitmap(band);
}

function buildEq(bands) {
  eqRow.innerHTML = "";
  for (const band of bands) {
    const color = wavelengthToColor(band.wavelength);

    const fader = document.createElement("div");
    fader.className = "eq-fader";

    const region = document.createElement("span");
    region.className = "eq-region";
    region.textContent = band.region;
    region.style.color = color;

    const input = document.createElement("input");
    input.type = "range";
    input.min = "0";
    input.max = "1.5";
    input.step = "0.05";
    input.value = "0";
    input.style.accentColor = color;
    input.setAttribute("aria-label", `${band.wavelength} nanometer band amount`);

    const label = document.createElement("span");
    label.className = "eq-label";
    label.textContent = `${band.wavelength}nm`;
    label.style.color = color;

    const valueEl = document.createElement("span");
    valueEl.className = "eq-value";
    valueEl.textContent = "0%";

    input.addEventListener("input", () => {
      const weight = parseFloat(input.value);
      state.weights[band.wavelength] = weight;
      valueEl.textContent = `${Math.round(weight * 100)}%`;
      if (weight > 0) loadBitmap(band);
      clearActiveRecipeHighlight();
      scheduleRedraw();
    });

    fader.append(region, input, label, valueEl);
    eqRow.appendChild(fader);

    state.faderInputs[band.wavelength] = input;
    state.faderValueEls[band.wavelength] = valueEl;
    state.weights[band.wavelength] = 0;
  }
}

function pickDefaultBand(bands) {
  const visible = bands.filter((b) => b.region === "visible");
  const pool = visible.length ? visible : bands;
  return pool[Math.floor(pool.length / 2)];
}

function applyDefaultState() {
  for (const band of state.bands) setFader(band, 0);
  const defaultBand = pickDefaultBand(state.bands);
  if (defaultBand) setFader(defaultBand, 1);

  state.exposure = 1;
  state.contrast = 1;
  exposureSlider.value = "1";
  contrastSlider.value = "1";
  exposureValue.textContent = "1.00×";
  contrastValue.textContent = "1.00×";
  applyFilter();

  recipeCaption.textContent = "";
  clearActiveRecipeHighlight();
  state.lastRecipe = null;
  scheduleRedraw();
}

function applyRecipe(recipe) {
  for (const band of state.bands) {
    const weight = recipe.weights?.[String(band.wavelength)] ?? 0;
    setFader(band, weight);
  }

  state.exposure = recipe.exposure ?? 1;
  state.contrast = recipe.contrast ?? 1;
  exposureSlider.value = String(state.exposure);
  contrastSlider.value = String(state.contrast);
  exposureValue.textContent = `${state.exposure.toFixed(2)}×`;
  contrastValue.textContent = `${state.contrast.toFixed(2)}×`;
  applyFilter();

  recipeCaption.textContent = recipe.caption || "";
  clearActiveRecipeHighlight();
  const chip = recipeRow.querySelector(`[data-recipe-id="${CSS.escape(recipe.id)}"]`);
  if (chip) chip.classList.add("active");
  state.lastRecipe = recipe;
  scheduleRedraw();
}

function buildRecipeChips(recipes) {
  recipeRow.innerHTML = "";
  for (const recipe of recipes) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "recipe-chip";
    chip.textContent = recipe.label || recipe.id;
    chip.dataset.recipeId = recipe.id;
    chip.addEventListener("click", () => applyRecipe(recipe));
    recipeRow.appendChild(chip);
  }
}

exposureSlider.addEventListener("input", () => {
  state.exposure = parseFloat(exposureSlider.value);
  exposureValue.textContent = `${state.exposure.toFixed(2)}×`;
  applyFilter();
  clearActiveRecipeHighlight();
});

contrastSlider.addEventListener("input", () => {
  state.contrast = parseFloat(contrastSlider.value);
  contrastValue.textContent = `${state.contrast.toFixed(2)}×`;
  applyFilter();
  clearActiveRecipeHighlight();
});

resetButton.addEventListener("click", () => {
  if (state.lastRecipe) {
    applyRecipe(state.lastRecipe);
  } else {
    applyDefaultState();
  }
});

async function init() {
  if (!artifactId) {
    setStatus("No artifact specified.");
    return;
  }

  let artifact;
  try {
    artifact = await fetchArtifact(artifactId);
  } catch (err) {
    console.error(err);
    setStatus("Couldn't load artifact manifest.");
    return;
  }

  if (!artifact) {
    setStatus("Artifact not found.");
    return;
  }

  populateInfoPanel(artifact, "Spectral Bands");
  populateCrossLink(artifact, "model");

  if (!artifact.msi) {
    setStatus("This artifact has no multispectral data.");
    return;
  }

  let msiManifest;
  try {
    const res = await fetch(`artifacts/${artifact.msi}`, { cache: "no-store" });
    if (!res.ok) throw new Error(`msi manifest request failed: ${res.status}`);
    msiManifest = await res.json();
  } catch (err) {
    console.error(err);
    setStatus("Couldn't load multispectral data.");
    return;
  }

  state.bands = msiManifest.bands || [];
  if (state.bands.length === 0) {
    setStatus("No spectral bands found for this artifact.");
    return;
  }

  // artifact.msi looks like "<folder>/msi/msi_manifest.json" — band file
  // paths in msiManifest are relative to the msi/ folder itself.
  state.msiBaseUrl = `artifacts/${artifact.msi.replace(/msi_manifest\.json$/, "")}`;

  buildEq(state.bands);
  applyDefaultState();

  // Load whatever's already active first so the visitor sees something
  // right away, then quietly prefetch the rest in the background so
  // scrubbing the other faders feels instant once they land.
  for (const band of state.bands) {
    if (state.weights[band.wavelength] > 0) loadBitmap(band);
  }
  setTimeout(() => prefetchRemainingBands(state.bands), 400);

  try {
    const res = await fetch(`${state.msiBaseUrl}recipes.json`, { cache: "no-store" });
    if (res.ok) {
      const data = await res.json();
      const recipes = data.recipes || [];
      if (recipes.length > 0) {
        buildRecipeChips(recipes);
        applyRecipe(recipes[0]);
      }
    }
  } catch (err) {
    // No curator recipes yet — free-adjust mode only. Not an error.
  }

  setStatus("Bands loaded");
  fadeStatus();
}

init();
