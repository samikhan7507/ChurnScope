// ChurnScope — Progressive Web App Service Worker (Desktop + Mobile Install Support)
const CACHE_NAME = "churnscope-pwa-v1.1.0";
const OFFLINE_URL = "/app/static/offline.html";
const PRECACHE_ASSETS = [
  "/app/static/manifest.json",
  "/app/static/offline.html",
  "/app/static/icon-192.png",
  "/app/static/icon-512.png",
  "/app/static/icon-maskable-192.png",
  "/app/static/icon-maskable-512.png",
  "/app/static/apple-touch-icon.png",
  "/app/static/favicon-32.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) =>
        Promise.allSettled(PRECACHE_ASSETS.map((url) => cache.add(url)))
      )
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key !== CACHE_NAME)
            .map((key) => caches.delete(key))
        )
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  // Never intercept Streamlit real-time WebSocket/session streams or API health checks.
  if (
    url.pathname.startsWith("/_stcore/") ||
    url.pathname.startsWith("/health") ||
    url.pathname.startsWith("/metrics") ||
    url.pathname.startsWith("/model/")
  ) {
    return;
  }

  // Cache-first for static PWA icons, manifest, and screenshots.
  if (url.pathname.startsWith("/app/static/") || url.pathname === "/manifest.json") {
    event.respondWith(
      caches.match(request).then((cached) => {
        if (cached) return cached;
        return fetch(request).then((response) => {
          if (response && response.status === 200) {
            const copy = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(request, copy));
          }
          return response;
        });
      })
    );
    return;
  }

  // Network-first for HTML navigation with dark-HUD offline fallback page.
  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request).catch(() =>
        caches.match(OFFLINE_URL).then(
          (offline) =>
            offline ||
            new Response("ChurnScope is offline. Please reconnect.", {
              status: 503,
              headers: { "Content-Type": "text/plain; charset=utf-8" },
            })
        )
      )
    );
  }
});
