const CACHE = 'ai-sample-match-V5.0.0';
const ASSETS = [
  './','./index.html','./styles.css','./app.js','./jszip.min.js','./manifest.webmanifest','./version.js',
  './vendor/onnxruntime/ort.min.js','./vendor/onnxruntime/ort-wasm-simd-threaded.mjs','./vendor/onnxruntime/ort-wasm-simd-threaded.wasm',
  './vendor/onnxruntime/ort-wasm-simd-threaded.jsep.mjs','./vendor/onnxruntime/ort-wasm-simd-threaded.jsep.wasm',
  './models/dinov2-small/model_quantized.onnx'
];
self.addEventListener('install',e=>e.waitUntil(caches.open(CACHE).then(c=>c.addAll(ASSETS)).then(()=>self.skipWaiting())));
self.addEventListener('activate',e=>e.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(k=>k!==CACHE).map(k=>caches.delete(k)))).then(()=>self.clients.claim())));
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);
  if(url.protocol !== 'http:' && url.protocol !== 'https:') return;
});

self.addEventListener('fetch',e=>{if(e.request.method!=='GET')return;e.respondWith(fetch(e.request).then(r=>{const copy=r.clone();caches.open(CACHE).then(c=>c.put(e.request,copy));return r}).catch(()=>caches.match(e.request).then(r=>r||caches.match('./index.html'))))});
