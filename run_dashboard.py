"""
run_dashboard.py — Production ASGI launcher for the ChurnScope Dashboard + Installable PWA.

Mounts root-level Web App Manifest (/manifest.json, /manifest.webmanifest), root-scoped
Service Worker (/sw.js with Service-Worker-Allowed: /), Apple touch icon (/apple-touch-icon.png),
and injects PWA <head> metadata directly into the initial HTML document so desktop browsers
(Chrome/Edge/Brave/Opera address-bar install icon) and mobile browsers (Android Chrome /
Samsung Internet install banner & iOS Safari Add-to-Home-Screen) work natively.

Usage:
  python run_dashboard.py
  PORT=8501 API_BASE_URL=https://churnscope-api.onrender.com python run_dashboard.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:
    pass

ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / "static"
SCRIPT_PATH = ROOT_DIR / "streamlit_app.py"


def _build_manifest_payload() -> dict[str, Any]:
    """Load static/manifest.json and apply optional PWA_* environment customizations."""
    with (STATIC_DIR / "manifest.json").open(encoding="utf-8") as fh:
        manifest: dict[str, Any] = json.load(fh)

    if app_name := os.getenv("PWA_APP_NAME", "").strip():
        manifest["name"] = app_name
    if short_name := os.getenv("PWA_SHORT_NAME", "").strip():
        manifest["short_name"] = short_name
    if theme_color := os.getenv("PWA_THEME_COLOR", "").strip():
        manifest["theme_color"] = theme_color
    if bg_color := os.getenv("PWA_BACKGROUND_COLOR", "").strip():
        manifest["background_color"] = bg_color
    return manifest


def _pwa_head_snippet() -> str:
    theme = os.getenv("PWA_THEME_COLOR", "#00d4ff").strip() or "#00d4ff"
    short_name = os.getenv("PWA_SHORT_NAME", "ChurnScope").strip() or "ChurnScope"
    return (
        f'<meta name="theme-color" content="{theme}" data-churnscope-pwa="1" />'
        '<meta name="mobile-web-app-capable" content="yes" />'
        '<meta name="apple-mobile-web-app-capable" content="yes" />'
        '<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />'
        f'<meta name="apple-mobile-web-app-title" content="{short_name}" />'
        f'<meta name="application-name" content="{short_name}" />'
        '<link rel="manifest" href="/manifest.json" data-churnscope-pwa="1" />'
        '<link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png" />'
        '<link rel="icon" type="image/png" sizes="32x32" href="/app/static/favicon-32.png" />'
        "<script>"
        "window.__churnscopeInstallPrompt=null;"
        'window.addEventListener("beforeinstallprompt",function(e){'
        "window.__churnscopeInstallPrompt=e;"
        'window.dispatchEvent(new CustomEvent("churnscope-pwa-ready"));'
        "});"
        'window.addEventListener("appinstalled",function(){'
        "window.__churnscopeInstallPrompt=null;"
        'window.dispatchEvent(new CustomEvent("churnscope-pwa-installed"));'
        "});"
        'if("serviceWorker" in navigator){'
        'window.addEventListener("load",function(){'
        'navigator.serviceWorker.register("/sw.js",{scope:"/"}).catch(function(){});'
        "});"
        "}"
        "</script>"
    )


def create_asgi_app(inner_app: Any | None = None) -> Any:
    """Create ASGI Streamlit application with native root-scope PWA routes and <head> tags."""
    import streamlit as st
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.requests import Request
    from starlette.responses import FileResponse, JSONResponse, Response
    from starlette.routing import Mount, Route
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

    snippet_bytes = _pwa_head_snippet().encode("utf-8")

    class PWAHeadMiddleware:
        """Pure-ASGI middleware injecting PWA manifest/SW tags into the root HTML document."""

        def __init__(self, app: ASGIApp) -> None:
            self.app = app

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] != "http" or scope.get("method") != "GET" or scope.get("path") != "/":
                await self.app(scope, receive, send)
                return

            # Request uncompressed HTML from inner app so </head> substitution is exact.
            filtered_headers = [
                (k, v) for k, v in scope.get("headers", []) if k.lower() != b"accept-encoding"
            ]
            inner_scope = dict(scope, headers=filtered_headers)

            start_message: Message | None = None
            is_html = False
            chunks: list[bytes] = []

            async def capture_send(message: Message) -> None:
                nonlocal start_message, is_html
                if message["type"] == "http.response.start":
                    start_message = message
                    headers_dict = {k.lower(): v for k, v in message.get("headers", [])}
                    ctype = headers_dict.get(b"content-type", b"").lower()
                    is_html = message.get("status") == 200 and b"text/html" in ctype
                    if not is_html:
                        await send(message)
                elif message["type"] == "http.response.body":
                    if not is_html:
                        await send(message)
                        return
                    chunks.append(message.get("body", b""))
                    if not message.get("more_body", False) and start_message is not None:
                        body = b"".join(chunks)
                        if b"</head>" in body and b"data-churnscope-pwa" not in body:
                            body = body.replace(b"</head>", snippet_bytes + b"</head>", 1)
                        new_headers = [
                            (k, v)
                            for k, v in start_message.get("headers", [])
                            if k.lower() not in {b"content-length", b"content-encoding"}
                        ]
                        new_headers.append((b"content-length", str(len(body)).encode("ascii")))
                        await send(
                            {
                                "type": "http.response.start",
                                "status": start_message["status"],
                                "headers": new_headers,
                            }
                        )
                        await send({"type": "http.response.body", "body": body, "more_body": False})

            await self.app(inner_scope, receive, capture_send)

    async def manifest_endpoint(_request: Request) -> Response:
        return JSONResponse(
            content=_build_manifest_payload(),
            media_type="application/manifest+json",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    async def sw_endpoint(_request: Request) -> Response:
        return FileResponse(
            STATIC_DIR / "sw.js",
            media_type="application/javascript",
            headers={
                "Service-Worker-Allowed": "/",
                "Cache-Control": "no-cache, no-store, must-revalidate",
            },
        )

    async def offline_endpoint(_request: Request) -> Response:
        return FileResponse(STATIC_DIR / "offline.html", media_type="text/html")

    async def apple_icon_endpoint(_request: Request) -> Response:
        return FileResponse(STATIC_DIR / "apple-touch-icon.png", media_type="image/png")

    routes = [
        Route("/manifest.json", manifest_endpoint, methods=["GET", "HEAD"]),
        Route("/manifest.webmanifest", manifest_endpoint, methods=["GET", "HEAD"]),
        Route("/sw.js", sw_endpoint, methods=["GET", "HEAD"]),
        Route("/offline.html", offline_endpoint, methods=["GET", "HEAD"]),
        Route("/apple-touch-icon.png", apple_icon_endpoint, methods=["GET", "HEAD"]),
    ]

    if inner_app is not None:
        return Starlette(
            routes=[*routes, Mount("/", app=inner_app)],
            middleware=[Middleware(PWAHeadMiddleware)],
        )

    return st.App(
        str(SCRIPT_PATH),
        routes=routes,
        middleware=[Middleware(PWAHeadMiddleware)],
    )


def main() -> None:
    """Start the ChurnScope PWA dashboard on HOST:PORT (Render & Docker compatible)."""
    host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    port = int(os.getenv("PORT") or os.getenv("DASHBOARD_PORT", "8501"))

    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    os.environ.setdefault("STREAMLIT_SERVER_ENABLE_STATIC_SERVING", "true")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    import streamlit as st

    if hasattr(st, "App"):
        import uvicorn

        asgi_app = create_asgi_app()
        uvicorn.run(asgi_app, host=host, port=port, log_level="info")
    else:
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(SCRIPT_PATH),
            "--server.address",
            host,
            "--server.port",
            str(port),
            "--server.headless",
            "true",
            "--server.enableStaticServing",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ]
        raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
