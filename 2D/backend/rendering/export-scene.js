// Headless GLB exporter — requires Puppeteer and a running Vite dev server.
// Usage:
//   npm run dev          (in one terminal)
//   node export-scene.js (in another terminal)
//
// Output: render.glb in this directory.

import puppeteer from 'puppeteer';
import fs from 'fs';

const URL  = 'http://localhost:5173';
const OUT  = 'render.glb';
const TIMEOUT = 30_000; // ms to wait for textures before giving up

const browser = await puppeteer.launch({
  headless: true,
  args: ['--enable-unsafe-swiftshader'],
});
const page = await browser.newPage();

page.on('console', msg => console.log('[browser]', msg.text()));
page.on('pageerror', err => console.error('[browser error]', err.message));

console.log(`Opening ${URL}…`);
await page.goto(URL, { waitUntil: 'load' });

console.log('Waiting for textures to finish loading…');
await page.waitForFunction(() => window.sceneReady === true, { timeout: TIMEOUT });

console.log('Exporting GLB…');
const base64 = await page.evaluate(() => window.exportGLB());

fs.writeFileSync(OUT, Buffer.from(base64, 'base64'));
console.log(`Saved ${OUT} (${(fs.statSync(OUT).size / 1024).toFixed(1)} KB)`);

await browser.close();
