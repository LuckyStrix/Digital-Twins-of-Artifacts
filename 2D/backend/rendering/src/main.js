import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';
import UTIF from 'utif';

const app = document.getElementById('app');
if (app) app.remove();

// Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87CEEB);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 5;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.domElement.style.cssText = 'position: fixed; top: 0; left: 0; z-index: 0;';
document.body.appendChild(renderer.domElement);

// UI overlay — appended AFTER canvas
const ui = document.createElement('div');
ui.style.cssText = `
  position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
  background: rgba(20,20,20,0.75); backdrop-filter: blur(8px);
  border-radius: 12px; padding: 12px 20px;
  display: flex; align-items: center; gap: 14px;
  color: #fff; font-family: sans-serif; font-size: 14px;
  z-index: 100; user-select: none; min-width: 280px;
`;
ui.innerHTML = `
  <div style="display:flex; flex-direction:column; gap:8px;">
    <div style="display:flex; align-items:center; gap:14px;">
      <span>Light</span>
      <input id="intensitySlider" type="range" min="0" max="10" step="0.1" value="3" style="flex:1;">
      <span id="intensityVal">3.0</span>
    </div>
    <div style="display:flex; align-items:center; gap:14px;">
      <span>Rot X</span>
      <input id="rotXSlider" type="range" min="-180" max="180" step="1" value="-90" style="flex:1;">
      <span id="rotXVal">-90</span>
    </div>
    <div style="display:flex; align-items:center; gap:14px;">
      <span>Rot Y</span>
      <input id="rotYSlider" type="range" min="-180" max="180" step="1" value="0" style="flex:1;">
      <span id="rotYVal">0</span>
    </div>
    <div style="display:flex; align-items:center; gap:14px;">
      <span>Rot Z</span>
      <input id="rotZSlider" type="range" min="-180" max="180" step="1" value="0" style="flex:1;">
      <span id="rotZVal">0</span>
    </div>
    <button id="exportBtn" disabled style="
      margin-top:4px; padding:8px 16px; border:none; border-radius:8px;
      background:#4f8ef7; color:#fff; font-size:14px; cursor:pointer;
      opacity:0.5;
    ">Loading textures…</button>
  </div>
`;
document.body.appendChild(ui);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

let rotX = -90, rotY = 0, rotZ = 0;

function applyRotations() {
  if (plane) {
    plane.rotation.x = rotX * Math.PI / 180;
    plane.rotation.y = rotY * Math.PI / 180;
    plane.rotation.z = rotZ * Math.PI / 180;
  }
}

// Lights
const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
scene.add(ambientLight);

const frontLight = new THREE.PointLight(0xffffff, 3, 50);
scene.add(frontLight);

// Slider
const slider = document.getElementById('intensitySlider');
const valLabel = document.getElementById('intensityVal');
slider.addEventListener('input', () => {
  const v = parseFloat(slider.value);
  frontLight.intensity = v;
  valLabel.textContent = v.toFixed(1);
});

// Rotation sliders
const rotXSlider = document.getElementById('rotXSlider');
const rotXVal = document.getElementById('rotXVal');
rotXSlider.addEventListener('input', () => {
  rotX = parseFloat(rotXSlider.value);
  rotXVal.textContent = rotX;
  applyRotations();
});

const rotYSlider = document.getElementById('rotYSlider');
const rotYVal = document.getElementById('rotYVal');
rotYSlider.addEventListener('input', () => {
  rotY = parseFloat(rotYSlider.value);
  rotYVal.textContent = rotY;
  applyRotations();
});

const rotZSlider = document.getElementById('rotZSlider');
const rotZVal = document.getElementById('rotZVal');
rotZSlider.addEventListener('input', () => {
  rotZ = parseFloat(rotZSlider.value);
  rotZVal.textContent = rotZ;
  applyRotations();
});

// Default plane
let plane = new THREE.Mesh(
  new THREE.PlaneGeometry(5, 5),
  new THREE.MeshBasicMaterial({ color: 0x888888 })
);
plane.rotation.x = -Math.PI / 2; //
scene.add(plane);

// Texture loader
let diffuseMap, normalMap, specularMap, roughnessMap, alphaMap;

async function loadTiffTexture(url, { alphaFromRed = false } = {}) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} failed (${res.status})`);
  const buf = await res.arrayBuffer();
  const ifds = UTIF.decode(buf);
  UTIF.decodeImage(buf, ifds[0]);
  const rgba = UTIF.toRGBA8(ifds[0]);
  const { width, height } = ifds[0];

  // Grayscale TIFFs decode via UTIF as R=G=B=value, A=255. For maps that need
  // to ride in the alpha channel (e.g. specularIntensityMap, which glTF/three
  // sample from alpha and treat as linear data — unlike RGB "color" channels,
  // which the KHR_materials_specular loader path forces through an sRGB
  // decode), mirror the grayscale value into alpha so it survives untouched.
  if (alphaFromRed) {
    for (let i = 0; i < rgba.length; i += 4) rgba[i + 3] = rgba[i];
  }

  // ImageBitmap is a valid drawImage() source, which GLTFExporter requires.
  // imageOrientation:'flipY' replaces the DataTexture flipY workaround.
  const bitmap = await createImageBitmap(
    new ImageData(new Uint8ClampedArray(rgba), width, height),
    { imageOrientation: 'flipY' }
  );
  const tex = new THREE.Texture(bitmap);
  tex.needsUpdate = true;
  return tex;
}

// Builds a specular tint map from a color image: each pixel's RGB is scaled
// so its brightest channel hits 255, preserving hue/saturation while pinning
// peak magnitude to match the default white specularColor in that channel.
// Used as specularColorMap so highlights tint toward the diffuse hue instead
// of washing to grey, without lowering the previously-tuned specular/diffuse
// magnitude ratio.
function makeSpecularTintCanvas(source) {
  const w = source.width, h = source.height;
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(source, 0, 0);
  const imgData = ctx.getImageData(0, 0, w, h);
  const data = imgData.data;
  for (let i = 0; i < data.length; i += 4) {
    const max = Math.max(data[i], data[i + 1], data[i + 2]);
    if (max > 0) {
      const scale = 255 / max;
      data[i]     = Math.min(255, data[i]     * scale);
      data[i + 1] = Math.min(255, data[i + 1] * scale);
      data[i + 2] = Math.min(255, data[i + 2] * scale);
    }
  }
  ctx.putImageData(imgData, 0, 0);
  return canvas;
}

function updateMaterial() {
  if (!diffuseMap || !normalMap || !specularMap || !roughnessMap || !alphaMap) return;

  const aspectRatio = diffuseMap.image.width / diffuseMap.image.height;
  const planeWidth  = aspectRatio >= 1 ? 5 : 5 * aspectRatio;
  const planeHeight = aspectRatio >= 1 ? 5 / aspectRatio : 5;

  const specTintTex = new THREE.CanvasTexture(makeSpecularTintCanvas(diffuseMap.image));
  specTintTex.needsUpdate = true;

  const material = new THREE.MeshPhysicalMaterial({
    map: diffuseMap,
    normalMap: normalMap,
    roughnessMap: roughnessMap,
    // specularIntensityMap (alpha channel, populated via alphaFromRed above)
    // scales the dielectric F0 reflectance per-pixel. Unlike specularColorMap
    // (RGB), the KHR_materials_specular loader treats this as linear data, so
    // the calibrated value round-trips through export/import unmodified.
    specularIntensityMap: specularMap,
    // See makeSpecularTintCanvas: tints the reflectance toward the diffuse
    // hue without lowering its peak magnitude.
    specularColorMap: specTintTex,
    alphaMap: alphaMap,
    roughness: 1.0,
    metalness: 0.0,
    side: THREE.DoubleSide,
    transparent: true,
    alphaTest: 0.5,
  });

  scene.remove(plane);
  plane = new THREE.Mesh(new THREE.PlaneGeometry(planeWidth, planeHeight), material);
  scene.add(plane);
  applyRotations();

  const exportBtn = document.getElementById('exportBtn');
  exportBtn.disabled = false;
  exportBtn.style.opacity = '1';
  exportBtn.textContent = 'Export GLB';

  window.sceneReady = true;
}

function findAlphaBounds(alphaImage) {
  const w = alphaImage.width, h = alphaImage.height;
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  canvas.getContext('2d').drawImage(alphaImage, 0, 0);
  const data = canvas.getContext('2d').getImageData(0, 0, w, h).data;

  let xMin = w, xMax = 0, yMin = h, yMax = 0;
  // Sample every 4th pixel — fast enough for 4k textures, accurate to 4px
  for (let y = 0; y < h; y += 4) {
    for (let x = 0; x < w; x += 4) {
      if (data[(y * w + x) * 4] > 10) {
        if (x < xMin) xMin = x;
        if (x > xMax) xMax = x;
        if (y < yMin) yMin = y;
        if (y > yMax) yMax = y;
      }
    }
  }
  return { xMin, yMin, w: xMax - xMin, h: yMax - yMin };
}

function cropToCanvas(image, bounds) {
  const canvas = document.createElement('canvas');
  canvas.width = bounds.w; canvas.height = bounds.h;
  canvas.getContext('2d').drawImage(image, -bounds.xMin, -bounds.yMin);
  return canvas;
}

function floorPOT(n) {
  // Largest power-of-two <= n (avoids upscaling)
  let p = 1;
  while (p * 2 <= n) p <<= 1;
  return p;
}

function padToSquarePOT(canvas) {
  // Scale content to fit within a square POT canvas without stretching,
  // centered with transparent padding. The 5×5 square geometry then
  // displays the scroll at its natural proportions with transparent margins.
  const size    = floorPOT(Math.max(canvas.width, canvas.height));
  const scale   = size / Math.max(canvas.width, canvas.height);
  const scaledW = Math.round(canvas.width  * scale);
  const scaledH = Math.round(canvas.height * scale);
  const offsetX = Math.round((size - scaledW) / 2);
  const offsetY = Math.round((size - scaledH) / 2);

  const out = document.createElement('canvas');
  out.width = size; out.height = size; // transparent by default
  out.getContext('2d').drawImage(canvas, offsetX, offsetY, scaledW, scaledH);
  return out;
}

function buildExportMesh() {
  const alphaTex = plane.material.alphaMap;
  const diffTex  = plane.material.map;
  const normTex  = plane.material.normalMap;
  const roughTex = plane.material.roughnessMap;
  const specTex  = plane.material.specularIntensityMap;

  // Crop all textures to the tight alpha content bounds so the geometry
  // aspect ratio matches the visible content, not the padded texture.
  const bounds = findAlphaBounds(alphaTex.image);

  const croppedDiff  = cropToCanvas(diffTex.image,  bounds);
  const croppedAlpha = cropToCanvas(alphaTex.image, bounds);
  const croppedNorm  = cropToCanvas(normTex.image,  bounds);
  const croppedRough = cropToCanvas(roughTex.image, bounds);
  const croppedSpec  = cropToCanvas(specTex.image,  bounds);

  // Composite alpha (R channel) into diffuse alpha channel for GLTF.
  const ctx = croppedDiff.getContext('2d');
  const diffData  = ctx.getImageData(0, 0, bounds.w, bounds.h);
  const alphaData = croppedAlpha.getContext('2d').getImageData(0, 0, bounds.w, bounds.h);
  for (let i = 0; i < diffData.data.length; i += 4) {
    diffData.data[i + 3] = alphaData.data[i];
  }
  ctx.putImageData(diffData, 0, 0);

  // Pad all maps into a square POT canvas (transparent margins, no stretching).
  const potDiff  = padToSquarePOT(croppedDiff);
  const potNorm  = padToSquarePOT(croppedNorm);
  const potRough = padToSquarePOT(croppedRough);
  const potSpec  = padToSquarePOT(croppedSpec);

  const mkTex = c => { const t = new THREE.CanvasTexture(c); t.needsUpdate = true; return t; };

  // See makeSpecularTintCanvas: tints the specular reflectance toward the
  // diffuse hue without lowering its peak magnitude.
  const potSpecTint = makeSpecularTintCanvas(potDiff);

  const material = new THREE.MeshPhysicalMaterial({
    map:                  mkTex(potDiff),
    normalMap:            mkTex(potNorm),
    roughnessMap:         mkTex(potRough),
    specularIntensityMap: mkTex(potSpec),
    specularColorMap:     mkTex(potSpecTint),
    roughness: 1.0,
    metalness: 0.0,
    side: THREE.FrontSide,
    transparent: true,
    alphaTest: 0.5,
  });

  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(5, 5), material);
  mesh.rotation.set(Math.PI, 0, 0);
  mesh.scale.set(-1, 1, 1);
  return mesh;
}

function exportGLB() {
  return new Promise((resolve, reject) => {
    new GLTFExporter().parse(buildExportMesh(), resolve, reject, { binary: true });
  });
}

function triggerDownload(glb) {
  const blob = new Blob([glb], { type: 'model/gltf-binary' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'render.glb';
  a.click();
  URL.revokeObjectURL(a.href);
}

document.getElementById('exportBtn').addEventListener('click', async () => {
  const btn = document.getElementById('exportBtn');
  btn.disabled = true;
  btn.textContent = 'Exporting…';
  try {
    triggerDownload(await exportGLB());
  } catch (e) {
    console.error('Export failed:', e);
  }
  btn.disabled = false;
  btn.textContent = 'Export GLB';
});

// Exposed for headless automation (see export-scene.js)
window.exportGLB = async () => {
  const glb = await exportGLB();
  const bytes = new Uint8Array(glb);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
};

loadTiffTexture('textures/DiffuseMap_render.tiff').then(t    => { diffuseMap   = t; updateMaterial(); }).catch(console.error);
loadTiffTexture('textures/NormalMap_render.tiff').then(t    => { normalMap    = t; updateMaterial(); }).catch(console.error);
loadTiffTexture('textures/SpecularMap_render.tiff', { alphaFromRed: true }).then(t  => { specularMap  = t; updateMaterial(); }).catch(console.error);
loadTiffTexture('textures/RoughnessMap_render.tiff').then(t => { roughnessMap = t; updateMaterial(); }).catch(console.error);
loadTiffTexture('textures/AlphaMask_render.tiff').then(t    => { alphaMap     = t; updateMaterial(); }).catch(console.error);

// Cubemap
const cubeTexture = new THREE.CubeTextureLoader().load(
  ['cubemap/posx.jpg','cubemap/negx.jpg','cubemap/posy.jpg','cubemap/negy.jpg','cubemap/posz.jpg','cubemap/negz.jpg'],
  () => { scene.background = cubeTexture; },
  undefined,
  () => console.error('Cubemap failed')
);

// Animate
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  frontLight.position.copy(camera.position);
  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
