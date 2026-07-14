import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { MTLLoader } from "three/examples/jsm/loaders/MTLLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TrackballControls } from "three/examples/jsm/controls/TrackballControls.js";
import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// .glb and loose .gltf (+ .bin + textures) both go through GLTFLoader.
// .obj goes through OBJLoader, picking up its companion .mtl if the
// artifact folder has one (OBJ has no embedded materials of its own).
function loadModel(modelPath, mtlPath, onLoad, onError) {
  const ext = modelPath.split(".").pop().toLowerCase();

  if (ext === "obj") {
    const loadObj = (materials) => {
      const objLoader = new OBJLoader();
      if (materials) objLoader.setMaterials(materials);
      objLoader.load(modelPath, onLoad, undefined, onError);
    };
    if (mtlPath) {
      new MTLLoader().load(
        mtlPath,
        (materials) => {
          materials.preload();
          loadObj(materials);
        },
        undefined,
        () => loadObj(null) // fall back to OBJ's default material if the .mtl fails
      );
    } else {
      loadObj(null);
    }
    return;
  }

  new GLTFLoader().load(modelPath, (gltf) => onLoad(gltf.scene), undefined, onError);
}

// Some papyrus exports carry a baked negative-determinant node transform —
// a mirror, not a rotation (our sample/demo GLBs decompose to mirror-X ∘
// rotX(180), det = -1). For freely-rotatable artifacts that's invisible
// because you can spin them around, but rotation-locked types are pinned to
// the +Z side, so the mirror shows up as horizontally-flipped text you can't
// turn away from. A rotation can't undo a reflection, so detect the mirrored
// world matrix and cancel it with a compensating horizontal (world-X) flip.
// Do this before any bounding-box/centering math so that's computed in the
// fixed space. Each side of a two-sided artifact needs this independently.
function cancelBakedMirror(model) {
  model.updateMatrixWorld(true);
  let mirrored = false;
  model.traverse((obj) => {
    if (obj.isMesh && obj.matrixWorld.determinant() < 0) mirrored = true;
  });
  if (mirrored) {
    model.scale.x *= -1;
    model.updateMatrixWorld(true);
  }
}

// Some scans/exports are thin shells or single-sided planes whose winding
// doesn't reliably face the camera. Force double-sided rendering so
// artifacts never disappear depending on view angle.
function forceDoubleSided(model) {
  model.traverse((obj) => {
    if (!obj.isMesh || !obj.material) return;
    const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
    materials.forEach((m) => { m.side = THREE.DoubleSide; });
  });
}

// The meshes the hand-tracking raycaster tests against (see updateHandPoint).
// Re-collected per visible side so a flip never leaves it aiming at the face
// that's now turned away.
function collectHitMeshes(root) {
  const meshes = [];
  root.traverse((obj) => {
    if (obj.isMesh) meshes.push(obj);
  });
  return meshes;
}

// Wrap a loaded model in a holder Group, centered on its own geometry so the
// holder can be rotated about the model's true center. An Object3D's
// transform is T * R * S, so rotating the model directly — it carries the
// centering offset in .position — would swing it around the wrong origin.
function makeSideHolder(model, artifactType) {
  if (isRotationLocked(artifactType)) cancelBakedMirror(model);
  forceDoubleSided(model);

  const box = new THREE.Box3().setFromObject(model);
  model.position.sub(box.getCenter(new THREE.Vector3()));

  const holder = new THREE.Group();
  holder.add(model);
  return { holder, size: box.getSize(new THREE.Vector3()) };
}

const params = new URLSearchParams(window.location.search);
const artifactId = params.get("id");

const statusEl = document.getElementById("status");
const infoToggle = document.getElementById("infoToggle");
const infoPanel = document.getElementById("infoPanel");
const infoName = document.getElementById("infoName");
const infoType = document.getElementById("infoType");
const infoDescription = document.getElementById("infoDescription");

const lightingToggle = document.getElementById("lightingToggle");
const lightingPanel = document.getElementById("lightingPanel");
const ambientSlider = document.getElementById("ambientSlider");
const directionalSlider = document.getElementById("directionalSlider");
const azimuthSlider = document.getElementById("azimuthSlider");
const elevationSlider = document.getElementById("elevationSlider");
const ambientValue = document.getElementById("ambientValue");
const directionalValue = document.getElementById("directionalValue");
const azimuthValue = document.getElementById("azimuthValue");
const elevationValue = document.getElementById("elevationValue");
const rotationLockNote = document.getElementById("rotationLockNote");

const backgroundToggle = document.getElementById("backgroundToggle");
const backgroundPanel = document.getElementById("backgroundPanel");
const backgroundEnable = document.getElementById("backgroundEnable");
const backgroundSelect = document.getElementById("backgroundSelect");

const handTrackingToggle = document.getElementById("handTrackingToggle");
const handTrackingPanel = document.getElementById("handTrackingPanel");
const handTrackingStart = document.getElementById("handTrackingStart");
const handTrackingStop = document.getElementById("handTrackingStop");
const gestureLabelEl = document.getElementById("gestureLabel");
const cameraPreviewEl = document.getElementById("cameraPreview");

const flipWrapper = document.getElementById("flipWrapper");
const flipButton = document.getElementById("flipButton");

infoToggle.addEventListener("click", () => {
  const open = infoPanel.classList.toggle("open");
  infoToggle.innerHTML = open ? "Artifact Info &#x25B2;" : "Artifact Info &#x25BC;";
});

// Lighting/Background/Hand Tracking all stack in the same bottom-right
// corner, so an open panel can grow tall enough to run behind the
// buttons of the others. Opening one hides the other two buttons
// entirely until it's closed again, instead of letting them overlap.
const stackedPanels = [
  { wrapper: document.getElementById("lightingPanelWrapper"), toggle: lightingToggle, panel: lightingPanel, label: "Lighting" },
  { wrapper: document.getElementById("backgroundPanelWrapper"), toggle: backgroundToggle, panel: backgroundPanel, label: "Background" },
  { wrapper: document.getElementById("handTrackingPanelWrapper"), toggle: handTrackingToggle, panel: handTrackingPanel, label: "Hand Tracking" }
];

// The flip button shares that corner but has no panel of its own, so it
// only ever gets hidden, never opened.
const stackedWrappers = stackedPanels.map((p) => p.wrapper).concat(flipWrapper);

stackedPanels.forEach(({ wrapper, toggle, panel, label }) => {
  toggle.addEventListener("click", () => {
    const open = panel.classList.toggle("open");
    toggle.innerHTML = open ? `${label} &#x25B2;` : `${label} &#x25BC;`;
    stackedWrappers.forEach((other) => {
      if (other !== wrapper) other.classList.toggle("panel-hidden", open);
    });
  });
});

// Artifact types whose model orbit should be locked to a fixed angle —
// papyrus is thin and was authored to be read from one side, not spun.
const ROTATION_LOCKED_TYPES = ["papyrus"];
function isRotationLocked(type) {
  return ROTATION_LOCKED_TYPES.includes((type || "").trim().toLowerCase());
}

// Two-sided artifacts (a folder with a second .glb — see build_manifest.py)
// hang both sides off one pivot that spins 180° to turn the sheet over.
// Only the side facing the camera is ever visible: the two scans are of the
// same physical object, so their geometry is coincident and rendering both
// at once would z-fight.
const FLIP_DURATION_MS = 600;
let flipPivot = null;
let frontHolder = null;
let backHolder = null;
let showingBack = false;
let flipAnim = null;

function easeInOutCubic(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

flipButton.addEventListener("click", () => {
  if (flipAnim || !backHolder) return;
  const from = flipPivot.rotation.y;
  flipAnim = { from, to: from + Math.PI, start: performance.now(), swapped: false };
  flipButton.disabled = true;
});

function updateFlip() {
  if (!flipAnim) return;

  const t = Math.min((performance.now() - flipAnim.start) / FLIP_DURATION_MS, 1);
  flipPivot.rotation.y = flipAnim.from + (flipAnim.to - flipAnim.from) * easeInOutCubic(t);

  // Swap sides at the halfway point, where the sheet is edge-on and
  // effectively invisible, so the cut never reads as a pop.
  if (!flipAnim.swapped && t >= 0.5) {
    flipAnim.swapped = true;
    showingBack = !showingBack;
    frontHolder.visible = !showingBack;
    backHolder.visible = showingBack;
    modelHitMeshes = collectHitMeshes(showingBack ? backHolder : frontHolder);
    flipButton.textContent = showingBack ? "Flip back" : "Flip to other side";
  }

  if (t >= 1) {
    // Snap to the exact resting angle rather than letting += PI accumulate
    // float error across repeated flips.
    flipPivot.rotation.y = showingBack ? Math.PI : 0;
    flipAnim = null;
    flipButton.disabled = false;
  }
}

// Loads the second side in the background once the front is up, so the
// first paint isn't held behind a second multi-megabyte GLB. The button is
// revealed right away but stays disabled until this lands.
function loadBackSide(backPath, artifactType) {
  flipWrapper.classList.remove("unavailable");
  loadModel(
    backPath,
    null,
    (model) => {
      const { holder } = makeSideHolder(model, artifactType);
      // Pre-turned to face away, so the pivot's 180° brings it to camera
      // reading the right way round.
      holder.rotation.y = Math.PI;
      holder.visible = false;
      flipPivot.add(holder);
      backHolder = holder;
      flipButton.disabled = false;
    },
    (err) => {
      // The front side is already usable — don't degrade it over a
      // second-side failure, just don't offer the flip.
      console.error("second side failed to load:", err);
      flipWrapper.classList.add("unavailable");
    }
  );
}

const LIGHT_RADIUS = 5;
const DEFAULTS = { ambient: 1.6, directional: 1.4, azimuth: 45, elevation: 35 };

// azimuth: 0 = straight at the camera, ±90 = grazing the side — kept to
// this half-range (see azimuthSlider's min/max in viewer.html) so the key
// light can never swing past the object's visible face and go dark/behind it.
function lightPositionFromAngles(azimuthDeg, elevationDeg) {
  const azimuth = (azimuthDeg * Math.PI) / 180;
  const elevation = (elevationDeg * Math.PI) / 180;
  return new THREE.Vector3(
    LIGHT_RADIUS * Math.cos(elevation) * Math.sin(azimuth),
    LIGHT_RADIUS * Math.sin(elevation),
    LIGHT_RADIUS * Math.cos(elevation) * Math.cos(azimuth)
  );
}

// Filled in once the scene exists, so the slider listeners (registered
// immediately) can reach the lights that loadScene() creates later.
let ambientLight = null;
let keyLight = null;

ambientSlider.addEventListener("input", () => {
  ambientValue.textContent = ambientSlider.value;
  if (ambientLight) ambientLight.intensity = parseFloat(ambientSlider.value);
});

directionalSlider.addEventListener("input", () => {
  directionalValue.textContent = directionalSlider.value;
  if (keyLight) keyLight.intensity = parseFloat(directionalSlider.value);
});

function updateKeyLightPosition() {
  azimuthValue.textContent = `${azimuthSlider.value}°`;
  elevationValue.textContent = `${elevationSlider.value}°`;
  if (!keyLight) return;
  keyLight.position.copy(
    lightPositionFromAngles(parseFloat(azimuthSlider.value), parseFloat(elevationSlider.value))
  );
}
azimuthSlider.addEventListener("input", updateKeyLightPosition);
elevationSlider.addEventListener("input", updateKeyLightPosition);

const DEFAULT_BACKGROUND_COLOR = 0x14110d;
const CUBE_FACE_ORDER = ["px", "nx", "py", "ny", "pz", "nz"];

// Filled in once the scene exists (see loadScene), same pattern as the
// lights above, so background controls work regardless of load order.
let activeScene = null;
let backgrounds = [];
const cubeTextureCache = new Map();

function populateBackgroundSelect() {
  backgroundSelect.innerHTML = "";

  if (backgrounds.length === 0) {
    const option = document.createElement("option");
    option.textContent = "No backgrounds available";
    backgroundSelect.appendChild(option);
    backgroundSelect.disabled = true;
    backgroundEnable.disabled = true;
    return;
  }

  backgroundSelect.disabled = false;
  backgroundEnable.disabled = false;
  for (const bg of backgrounds) {
    const option = document.createElement("option");
    option.value = bg.id;
    option.textContent = bg.name;
    backgroundSelect.appendChild(option);
  }
}

async function loadBackgroundsList() {
  try {
    const res = await fetch("backgrounds/manifest.json", { cache: "no-store" });
    if (!res.ok) throw new Error(`backgrounds manifest request failed: ${res.status}`);
    const data = await res.json();
    backgrounds = data.backgrounds || [];
  } catch (err) {
    console.warn("No backgrounds available:", err);
    backgrounds = [];
  }
  populateBackgroundSelect();
}

function getCubeTexture(bg) {
  if (cubeTextureCache.has(bg.id)) return cubeTextureCache.get(bg.id);
  const urls = CUBE_FACE_ORDER.map((face) => `backgrounds/${bg.faces[face]}`);
  const texture = new THREE.CubeTextureLoader().load(urls, undefined, undefined, () => {
    setStatus("Background failed to load.");
    fadeStatus();
  });
  cubeTextureCache.set(bg.id, texture);
  return texture;
}

function applyBackgroundState() {
  if (!activeScene) return;
  if (!backgroundEnable.checked || backgrounds.length === 0) {
    activeScene.background = new THREE.Color(DEFAULT_BACKGROUND_COLOR);
    return;
  }
  const selected = backgrounds.find((bg) => bg.id === backgroundSelect.value) || backgrounds[0];
  activeScene.background = getCubeTexture(selected);
}

backgroundEnable.addEventListener("change", applyBackgroundState);
backgroundSelect.addEventListener("change", applyBackgroundState);
loadBackgroundsList();

// ── Hand-tracked magnifying lens ──
// Ported from a standalone hand-tracking prototype built for one papyrus
// model; generalized here to work with any loaded artifact by raycasting
// against whatever meshes loadScene() collects, using the same orbit
// camera as the rest of the viewer instead of a dedicated ortho camera.
// Filled in by loadScene() once the scene/camera/model exist.
let activeCamera = null;
let modelHitMeshes = [];
let revealLight = null;
let sceneTarget = null;

const raycaster = new THREE.Raycaster();
const handNdc = new THREE.Vector2(0, 0);

let handLandmarker = null;
let mediaStream = null;
let trackingActive = false;
let lastVideoTime = -1;

let lensVisible = false;
let lensRadiusTarget = 0.17;
const LENS_ZOOM = 2.8;
let freezeLens = false;
let wasPinch = false;
let lastFreezeToggle = 0;
let gestureLabel = "none";
let revealLightTargetIntensity = 0;

const tempNormal = new THREE.Vector3();
const tempLightPos = new THREE.Vector3();
const tempTargetPos = new THREE.Vector3();
const lensCenter = new THREE.Vector2(0.5, 0.5);
const lensTarget = new THREE.Vector2(0.5, 0.5);

const postScene = new THREE.Scene();
const postCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const postMaterial = new THREE.ShaderMaterial({
  uniforms: {
    tDiffuse: { value: null },
    uCenter: { value: lensCenter.clone() },
    uRadius: { value: 0.17 },
    uZoom: { value: LENS_ZOOM },
    uSoftness: { value: 0.028 },
    uEnabled: { value: 0.0 }
  },
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = vec4(position.xy, 0.0, 1.0);
    }
  `,
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform vec2 uCenter;
    uniform float uRadius;
    uniform float uZoom;
    uniform float uSoftness;
    uniform float uEnabled;
    varying vec2 vUv;

    void main() {
      vec2 delta = vUv - uCenter;
      float dist = length(delta);
      float lensMask = (1.0 - smoothstep(uRadius - uSoftness, uRadius, dist)) * uEnabled;

      vec2 zoomUv = uCenter + delta / uZoom;
      zoomUv = clamp(zoomUv, vec2(0.0), vec2(1.0));

      vec4 zoomColor = texture2D(tDiffuse, zoomUv);
      vec3 color = pow(max(zoomColor.rgb, vec3(0.0)), vec3(1.0 / 2.2));

      float ringOuter = smoothstep(uRadius + 0.006, uRadius, dist);
      float ringInner = smoothstep(uRadius, uRadius - 0.006, dist);
      float ring = max(ringOuter - ringInner, 0.0) * uEnabled;
      color = mix(color, vec3(0.98, 0.9, 0.72), ring * 0.45);

      float alpha = max(lensMask, ring);
      gl_FragColor = vec4(color, alpha);
    }
  `,
  transparent: true,
  blending: THREE.NormalBlending,
  depthTest: false,
  depthWrite: false
});
postScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), postMaterial));

function landmarkDistance(lm, a, b) {
  const dx = lm[a].x - lm[b].x;
  const dy = lm[a].y - lm[b].y;
  return Math.hypot(dx, dy);
}

function isFingerExtended(lm, tipIndex, pipIndex) {
  return lm[tipIndex].y < lm[pipIndex].y - 0.015;
}

function detectGestures(lm) {
  const pinch = landmarkDistance(lm, 4, 8) < 0.06;
  const indexExtended = isFingerExtended(lm, 8, 6);
  const middleExtended = isFingerExtended(lm, 12, 10);
  const ringExtended = isFingerExtended(lm, 16, 14);
  const pinkyExtended = isFingerExtended(lm, 20, 18);
  return {
    pinch,
    openPalm: indexExtended && middleExtended && ringExtended && pinkyExtended,
    fist: !indexExtended && !middleExtended && !ringExtended && !pinkyExtended
  };
}

async function setupHandTracking() {
  if (handLandmarker) return handLandmarker;
  setStatus("Loading hand tracker…");
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    },
    runningMode: "VIDEO",
    numHands: 1
  });
  return handLandmarker;
}

async function startTracking() {
  if (trackingActive) return;
  try {
    await setupHandTracking();
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
    cameraPreviewEl.srcObject = mediaStream;
    cameraPreviewEl.style.display = "block";
    await cameraPreviewEl.play();

    trackingActive = true;
    revealLightTargetIntensity = 0;
    gestureLabelEl.classList.add("visible");
    handTrackingStart.disabled = true;
    handTrackingStop.disabled = false;

    setStatus("Hand tracking active");
    fadeStatus();
  } catch (err) {
    console.error(err);
    setStatus("Camera/tracking failed — check permissions.");
  }
}

function stopTracking() {
  trackingActive = false;
  lensVisible = false;
  postMaterial.uniforms.uEnabled.value = 0;
  freezeLens = false;
  gestureLabel = "none";
  wasPinch = false;
  revealLightTargetIntensity = 0;
  if (revealLight) {
    revealLight.intensity = 0;
    revealLight.visible = false;
  }
  gestureLabelEl.classList.remove("visible");

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  cameraPreviewEl.srcObject = null;
  cameraPreviewEl.style.display = "none";

  handTrackingStart.disabled = false;
  handTrackingStop.disabled = true;

  setStatus("Tracking stopped");
  fadeStatus();
}

handTrackingStart.addEventListener("click", startTracking);
handTrackingStop.addEventListener("click", stopTracking);
handTrackingStop.disabled = true;
window.addEventListener("beforeunload", stopTracking);

function updateHandPoint() {
  if (!trackingActive || !handLandmarker || !activeCamera || cameraPreviewEl.readyState < 2) return;

  const now = performance.now();
  if (cameraPreviewEl.currentTime === lastVideoTime) return;
  lastVideoTime = cameraPreviewEl.currentTime;

  const result = handLandmarker.detectForVideo(cameraPreviewEl, now);
  const landmarks = result.landmarks?.[0];
  if (!landmarks) {
    lensVisible = false;
    revealLightTargetIntensity = 0;
    gestureLabel = "no hand";
    return;
  }

  const gestures = detectGestures(landmarks);

  if (gestures.pinch && !wasPinch && now - lastFreezeToggle > 450) {
    freezeLens = !freezeLens;
    lastFreezeToggle = now;
  }
  wasPinch = gestures.pinch;

  if (gestures.openPalm) {
    lensRadiusTarget = 0.22;
    gestureLabel = "open palm — wide lens";
  } else if (gestures.fist) {
    lensRadiusTarget = 0.12;
    gestureLabel = "fist — small lens";
  } else {
    lensRadiusTarget = 0.17;
    gestureLabel = "point";
  }

  if (freezeLens) {
    gestureLabel = gestures.pinch ? "freeze toggled" : "frozen (pinch to unfreeze)";
    lensVisible = true;
    revealLightTargetIntensity = 0;
    return;
  }

  const tip = landmarks[8];
  const mirroredX = 1 - tip.x;
  handNdc.set(mirroredX * 2 - 1, -(tip.y * 2 - 1));
  raycaster.setFromCamera(handNdc, activeCamera);

  const intersects = raycaster.intersectObjects(modelHitMeshes, true);
  if (!intersects.length) {
    lensVisible = false;
    revealLightTargetIntensity = 0;
    gestureLabel = "hand off model";
    return;
  }

  const hit = intersects[0];
  lensVisible = true;
  revealLightTargetIntensity = 0.9;
  if (gestures.pinch) gestureLabel = "freeze toggled";

  const projected = hit.point.clone().project(activeCamera);
  lensTarget.set(
    THREE.MathUtils.clamp((projected.x + 1) * 0.5, 0, 1),
    THREE.MathUtils.clamp((projected.y + 1) * 0.5, 0, 1)
  );

  if (revealLight) {
    tempNormal.copy(hit.face?.normal || new THREE.Vector3(0, 0, 1));
    tempNormal.transformDirection(hit.object.matrixWorld);
    tempLightPos.copy(hit.point).addScaledVector(tempNormal, 0.5);
    tempTargetPos.copy(hit.point).addScaledVector(tempNormal, 0.01);
    revealLight.position.lerp(tempLightPos, 0.28);
    revealLight.target.position.lerp(tempTargetPos, 0.28);
  }
}

function updateRevealLight() {
  if (!revealLight) return;
  revealLight.intensity = THREE.MathUtils.lerp(revealLight.intensity, revealLightTargetIntensity, 0.2);
  revealLight.visible = revealLight.intensity > 0.01;
}

function updateLensUniforms() {
  if (!lensVisible) {
    postMaterial.uniforms.uEnabled.value *= 0.85;
    if (postMaterial.uniforms.uEnabled.value < 0.01) postMaterial.uniforms.uEnabled.value = 0;
  } else {
    lensCenter.lerp(lensTarget, 0.24);
    postMaterial.uniforms.uCenter.value.copy(lensCenter);
    postMaterial.uniforms.uRadius.value = THREE.MathUtils.lerp(
      postMaterial.uniforms.uRadius.value,
      lensRadiusTarget,
      0.2
    );
    postMaterial.uniforms.uEnabled.value = THREE.MathUtils.lerp(postMaterial.uniforms.uEnabled.value, 1.0, 0.3);
  }
  if (trackingActive) gestureLabelEl.textContent = gestureLabel;
}

function setStatus(text) {
  statusEl.textContent = text;
  statusEl.style.opacity = "1";
}

function fadeStatus() {
  setTimeout(() => {
    statusEl.style.opacity = "0";
  }, 1800);
}

function titleCase(str) {
  return str.replace(/\w\S*/g, (w) => w[0].toUpperCase() + w.slice(1).toLowerCase());
}

async function init() {
  if (!artifactId) {
    setStatus("No artifact specified.");
    return;
  }

  let artifact;
  try {
    const res = await fetch("artifacts/manifest.json", { cache: "no-store" });
    if (!res.ok) throw new Error(`manifest request failed: ${res.status}`);
    const data = await res.json();
    artifact = (data.artifacts || []).find((a) => a.id === artifactId);
  } catch (err) {
    console.error(err);
    setStatus("Couldn't load artifact manifest.");
    return;
  }

  if (!artifact) {
    setStatus("Artifact not found.");
    return;
  }

  document.title = `${artifact.name} — Artifact Viewer`;
  infoName.textContent = artifact.name;
  infoType.textContent = titleCase(artifact.type || "other");
  infoDescription.textContent = artifact.description || "";

  setStatus("Loading model…");
  loadScene(
    `artifacts/${artifact.model}`,
    artifact.mtl ? `artifacts/${artifact.mtl}` : null,
    artifact.type,
    artifact.back ? `artifacts/${artifact.back}` : null
  );
}

function loadScene(modelPath, mtlPath, artifactType, backPath) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(DEFAULT_BACKGROUND_COLOR);
  activeScene = scene;
  applyBackgroundState();

  const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 100);
  camera.position.set(0, 0, 3);
  activeCamera = camera;

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.prepend(renderer.domElement);

  // Render target the hand-tracking lens samples from (see animate()).
  const drawingBufferSize = new THREE.Vector2();
  renderer.getDrawingBufferSize(drawingBufferSize);
  sceneTarget = new THREE.WebGLRenderTarget(drawingBufferSize.x, drawingBufferSize.y);
  postMaterial.uniforms.tDiffuse.value = sceneTarget.texture;

  // Reveal spotlight that follows the tracked fingertip across the model.
  revealLight = new THREE.SpotLight(0xffefc9, 0.0, 6, Math.PI / 8, 0.42, 1.0);
  revealLight.position.set(0, 0, 4);
  revealLight.target.position.set(0, 0, 0);
  revealLight.visible = false;
  scene.add(revealLight, revealLight.target);

  let controls;
  if (isRotationLocked(artifactType)) {
    // Papyrus and other locked types: OrbitControls with rotation disabled,
    // leaving pan/zoom intact.
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.enableRotate = false;
    rotationLockNote.textContent = "Rotation locked for this artifact type";
    rotationLockNote.classList.add("visible");
  } else {
    // Tablets (and anything not rotation-locked) can be spun freely in any
    // direction. OrbitControls clamps the vertical/polar angle to (0, PI)
    // via spherical.makeSafe(), so it can never roll over the poles no
    // matter how min/maxPolarAngle are set. TrackballControls has no such
    // clamp — it gives true unbounded rotation on every axis.
    controls = new TrackballControls(camera, renderer.domElement);
    controls.rotateSpeed = 3.0;
    controls.zoomSpeed = 1.2;
    controls.panSpeed = 0.8;
    controls.staticMoving = false;
    controls.dynamicDampingFactor = 0.08;
    controls.handleResize();
  }

  ambientLight = new THREE.AmbientLight(0xffffff, DEFAULTS.ambient);
  scene.add(ambientLight);

  keyLight = new THREE.DirectionalLight(0xffffff, DEFAULTS.directional);
  keyLight.position.copy(lightPositionFromAngles(DEFAULTS.azimuth, DEFAULTS.elevation));
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0xffffff, 0.6);
  fillLight.position.set(-3, -2, -4);
  scene.add(fillLight);

  loadModel(
    modelPath,
    mtlPath,
    (model) => {
      const { holder, size } = makeSideHolder(model, artifactType);
      frontHolder = holder;
      modelHitMeshes = collectHitMeshes(frontHolder);

      flipPivot = new THREE.Group();
      flipPivot.add(frontHolder);

      const maxDim = Math.max(size.x, size.y, size.z) || 1;
      const fitDistance =
        (maxDim / (2 * Math.tan((camera.fov * Math.PI) / 360))) * 1.6;
      camera.position.set(0, 0, fitDistance);
      camera.near = fitDistance / 100;
      camera.far = fitDistance * 100;
      camera.updateProjectionMatrix();
      controls.target.set(0, 0, 0);
      controls.update();

      scene.add(flipPivot);
      setStatus("Model loaded");
      fadeStatus();

      if (backPath) loadBackSide(backPath, artifactType);
    },
    (err) => {
      console.error(err);
      setStatus("Model failed to load.");
    }
  );

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.getDrawingBufferSize(drawingBufferSize);
    sceneTarget.setSize(drawingBufferSize.x, drawingBufferSize.y);
    // TrackballControls caches the viewport size for its rotate/pan math.
    if (controls.handleResize) controls.handleResize();
  });

  function animate() {
    requestAnimationFrame(animate);
    controls.update();

    updateFlip();
    updateHandPoint();
    updateLensUniforms();
    updateRevealLight();

    // Render the base scene directly so brightness never shifts...
    renderer.setRenderTarget(null);
    renderer.render(scene, camera);

    // ...then, only when the lens is visible, capture a copy of the
    // scene into sceneTarget and overlay the magnifier on top of it.
    const lensActive = postMaterial.uniforms.uEnabled.value > 0.005;
    if (lensActive) {
      renderer.setRenderTarget(sceneTarget);
      renderer.render(scene, camera);

      renderer.setRenderTarget(null);
      renderer.autoClear = false;
      renderer.render(postScene, postCamera);
      renderer.autoClear = true;
    }
  }
  animate();
}

init();
