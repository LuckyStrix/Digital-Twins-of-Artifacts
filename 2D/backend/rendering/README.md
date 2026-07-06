# Rendering — GLB Baker

Node.js / Vite + three.js app that bakes the modeling-stage texture maps into a
textured 3D model (`render.glb`). Part of the `2D/` pipeline; see the
[repo setup guide](../../../SETUP.md) for first-time install. Normally invoked
by `2D/run.py`'s "Build 3D Model" step.

## Prerequisites

- **Node.js LTS** on PATH (https://nodejs.org/).

## Setup

```bash
npm install
```

This downloads three.js and a **headless Chromium via puppeteer** (used to bake
the GLB offline), so the first install may take a while.

## Usage

```bash
npm run dev       # live dev server for interactive work
npm run build     # production build
npm run preview   # preview a production build
```

The offline bake that produces `render.glb` is driven through `export-scene.js`
(puppeteer + headless Chromium).

## Structure

```
export-scene.js         Offline GLB bake (puppeteer headless Chromium)
index.html              Scene entry point
src/                    three.js scene / baking source
public/                 Static assets served by Vite
package.json            Scripts + deps (three, utif, vite, puppeteer)
render.glb              Baked output model
```
