"""
ChurnScope — Interactive Analytics & Installable PWA Dashboard
==============================================================
Production dark-HUD Streamlit dashboard backed by live FastAPI predictions,
genuine TreeExplainer SHAP attributions, batch CSV scoring, and real held-out
evaluation metrics. Installable as a Progressive Web App (PWA) on both laptop
(browser address-bar install icon) and mobile (Android / iOS Home Screen).

Run:
  uvicorn api.main:app --host 0.0.0.0 --port 8000   (terminal 1)
  python run_dashboard.py                           (terminal 2 — full root PWA)
  # or: streamlit run streamlit_app.py
"""

from __future__ import annotations

import html
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:
    pass

ROOT_DIR = Path(__file__).resolve().parent
FAVICON_PATH = ROOT_DIR / "static" / "favicon-32.png"

st.set_page_config(
    page_title="ChurnScope — Customer Churn Intelligence",
    page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else "🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Runtime Configuration (Local, Docker Compose, and Render Compatible) ──────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").strip().rstrip("/")
PUBLIC_API_LABEL = os.getenv("PUBLIC_API_URL", API_BASE_URL).strip().rstrip("/")
API_KEY = os.getenv("API_KEY", "").strip()
API_TIMEOUT = float(os.getenv("API_TIMEOUT_SECONDS", "15"))

PREDICT_URL = f"{API_BASE_URL}/predict"
EXPLAIN_URL = f"{API_BASE_URL}/predict/explain"
BATCH_URL = f"{API_BASE_URL}/batch-predict"
HEALTH_URL = f"{API_BASE_URL}/health"
MODEL_METRICS_URL = f"{API_BASE_URL}/model/metrics"

ENGINEERED_FEATURE_SET = {
    "tenure_band",
    "is_new_customer",
    "is_loyal_customer",
    "log_tenure",
    "monthly_charge_per_tenure",
    "charge_discrepancy",
    "is_high_value",
    "log_monthly_charges",
    "total_services",
    "service_penetration",
    "has_internet",
    "has_fiber",
    "protection_services_count",
    "charge_per_service",
    "stickiness_score",
    "high_risk_flag",
    "is_month_to_month",
    "is_long_term_contract",
    "is_electronic_check",
    "is_auto_payment",
    "is_paperless",
    "highest_churn_risk_profile",
}


def _api_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


def _esc(value: Any) -> str:
    """HTML-escape user-provided strings before rendering inside custom HUD cards."""
    return html.escape(str(value), quote=True)


@st.cache_data(ttl=15, show_spinner=False)
def fetch_api_health() -> dict[str, Any]:
    """Check real FastAPI service and model status."""
    try:
        resp = requests.get(HEALTH_URL, headers=_api_headers(), timeout=3)
        if resp.status_code in (200, 503):
            data = resp.json()
            data["_reachable"] = True
            return data
    except requests.RequestException:
        pass
    return {"_reachable": False, "status": "offline", "is_model_loaded": False}


@st.cache_data(ttl=30, show_spinner=False)
def fetch_model_metrics() -> tuple[dict[str, Any] | None, str]:
    """Load real model evaluation metrics from API /model/metrics or local artifact JSON."""
    try:
        resp = requests.get(MODEL_METRICS_URL, headers=_api_headers(), timeout=5)
        if resp.status_code == 200:
            return resp.json(), "live_api"
    except requests.RequestException:
        pass

    local_eval = ROOT_DIR / "models" / "artifacts" / "evaluation_results.json"
    local_reg = ROOT_DIR / "models" / "registry" / "model_registry.json"
    if local_eval.exists():
        with local_eval.open(encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)
        if local_reg.exists():
            with local_reg.open(encoding="utf-8") as fh:
                reg = json.load(fh)
            models = reg.get("models", [])
            active = next((m for m in reversed(models) if m.get("status") == "active"), None)
            data["registry"] = {
                "active_model": active or (models[-1] if models else None),
                "total_versions": len(models),
            }
        return data, "local_artifact"

    return None, "missing"


def render_pwa_installer() -> None:
    """Inject PWA manifest/service worker fallback and render phone + laptop Install App bar."""
    components.html(
        """
<!DOCTYPE html>
<html>
<head>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: transparent;
    font-family: "Share Tech Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
    color: #7ba3c4;
    overflow: hidden;
  }
  .pwa-bar {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 8px;
    padding: 2px 0;
    flex-wrap: wrap;
  }
  .pwa-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    background: rgba(0, 212, 255, 0.08);
    color: #00d4ff;
    border: 1px solid rgba(0, 212, 255, 0.45);
    border-radius: 2px;
    font: inherit;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.2s ease;
    text-decoration: none;
  }
  .pwa-btn:hover {
    background: rgba(0, 212, 255, 0.18);
    box-shadow: 0 0 12px rgba(0, 212, 255, 0.35);
  }
  .pwa-btn.installed {
    color: #00ff88;
    border-color: rgba(0, 255, 136, 0.4);
    background: rgba(0, 255, 136, 0.08);
    cursor: default;
  }
  .pwa-hint {
    font-size: 11px;
    color: #e2f0ff;
    background: #0c1520;
    border: 1px solid #1a3a55;
    border-left: 2px solid #00d4ff;
    padding: 4px 10px;
    border-radius: 2px;
    display: none;
  }
</style>
</head>
<body>
  <div class="pwa-bar">
    <span id="pwaHint" class="pwa-hint"></span>
    <button id="installBtn" class="pwa-btn" type="button" title="Install ChurnScope app on phone or laptop">
      ⬇ INSTALL APP
    </button>
  </div>
  <script>
    (function () {
      var topWin = window;
      var topDoc = document;
      try {
        if (window.parent && window.parent.document) {
          topWin = window.parent;
          topDoc = window.parent.document;
        }
      } catch (err) {}

      // Inject manifest, iOS/Android icons & theme-color if running via plain `streamlit run`.
      try {
        var head = topDoc.head;
        if (head && !head.querySelector('link[rel="manifest"]')) {
          var addTag = function (tag, attrs) {
            var el = topDoc.createElement(tag);
            Object.keys(attrs).forEach(function (k) { el.setAttribute(k, attrs[k]); });
            head.appendChild(el);
          };
          addTag("meta", { name: "theme-color", content: "#00d4ff" });
          addTag("meta", { name: "mobile-web-app-capable", content: "yes" });
          addTag("meta", { name: "apple-mobile-web-app-capable", content: "yes" });
          addTag("meta", { name: "apple-mobile-web-app-status-bar-style", content: "black-translucent" });
          addTag("meta", { name: "apple-mobile-web-app-title", content: "ChurnScope" });
          addTag("link", { rel: "manifest", href: "/app/static/manifest.json" });
          addTag("link", { rel: "apple-touch-icon", sizes: "180x180", href: "/app/static/apple-touch-icon.png" });
        }
        if ("serviceWorker" in topWin.navigator) {
          topWin.navigator.serviceWorker.getRegistration().then(function (reg) {
            if (!reg) {
              topWin.navigator.serviceWorker
                .register("/sw.js", { scope: "/" })
                .catch(function () {
                  topWin.navigator.serviceWorker
                    .register("/app/static/sw.js", { scope: "/app/static/" })
                    .catch(function () {});
                });
            }
          });
        }
        topWin.addEventListener("beforeinstallprompt", function (e) {
          topWin.__churnscopeInstallPrompt = e;
        });
      } catch (err) {}

      var btn = document.getElementById("installBtn");
      var hint = document.getElementById("pwaHint");

      var isStandalone = false;
      try {
        isStandalone =
          (topWin.matchMedia && topWin.matchMedia("(display-mode: standalone)").matches) ||
          topWin.navigator.standalone === true;
      } catch (err) {}

      if (isStandalone) {
        btn.textContent = "✔ APP INSTALLED";
        btn.classList.add("installed");
        return;
      }

      var markInstalled = function () {
        btn.textContent = "✔ APP INSTALLED";
        btn.classList.add("installed");
        hint.style.display = "none";
      };

      try {
        topWin.addEventListener("appinstalled", markInstalled);
        topWin.addEventListener("churnscope-pwa-installed", markInstalled);
      } catch (err) {}

      btn.addEventListener("click", async function () {
        var promptEvent = null;
        try { promptEvent = topWin.__churnscopeInstallPrompt; } catch (err) {}

        if (promptEvent && typeof promptEvent.prompt === "function") {
          promptEvent.prompt();
          try {
            var outcome = await promptEvent.userChoice;
            if (outcome && outcome.outcome === "accepted") {
              topWin.__churnscopeInstallPrompt = null;
              markInstalled();
            }
          } catch (err) {}
          return;
        }

        var ua = (navigator.userAgent || "").toLowerCase();
        var isIOS = /iphone|ipad|ipod/.test(ua);
        var isAndroid = /android/.test(ua);
        if (isIOS) {
          hint.textContent = "iPhone/iPad: tap Share (⬆) → 'Add to Home Screen'";
        } else if (isAndroid) {
          hint.textContent = "Android: tap browser menu (⋮) → 'Install app'";
        } else {
          hint.textContent = "Laptop: click the Install icon (⊕) at the right end of your address bar";
        }
        hint.style.display = "inline-block";
      });
    })();
  </script>
</body>
</html>
        """,
        height=38,
    )


# ── HUD CSS (Desktop + Mobile PWA Responsive) ─────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&display=swap');
:root{
  --void:#020408;--deep:#050a10;--panel:#080f18;--card:#0c1520;
  --b-dim:#0f2033;--b-med:#1a3a55;
  --cyan:#00d4ff;--blue:#0ea5e9;--teal:#14b8a6;
  --pri:#e2f0ff;--sec:#7ba3c4;--muted:#4b7496;
  --red:#ff3b3b;--green:#00ff88;--amber:#ffaa00;--purple:#a855f7;
  --fhud:"Orbitron",monospace;--fmono:"Share Tech Mono",monospace;--fbody:"Rajdhani",sans-serif;
}
*,*::before,*::after{box-sizing:border-box}
html,body,[class*="css"],.stApp{background:var(--void)!important;color:var(--pri)!important;font-family:var(--fbody)!important}
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:var(--deep)}
::-webkit-scrollbar-thumb{background:var(--b-med);border-radius:2px}
#MainMenu,footer,[data-testid="stHeader"],[data-testid="stToolbar"],
[data-testid="stDecoration"],[data-testid="collapsedControl"]{display:none!important}
.stDeployButton{display:none!important}
section[data-testid="stSidebar"]{display:none!important}
.main .block-container{padding:0.7rem 1.8rem 2.5rem 1.8rem!important;max-width:1700px!important}
.hdr-row{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;padding:.4rem 0 0 0}
.nav-bar{display:flex;gap:4px;padding:.5rem 0 0 0;border-bottom:1px solid var(--b-dim);margin-bottom:.9rem;flex-wrap:wrap}
.nav-tab{font-family:var(--fhud);font-size:.58rem;font-weight:700;letter-spacing:3px;
  text-transform:uppercase;padding:7px 16px;border:1px solid var(--b-dim);
  border-bottom:none;border-radius:2px 2px 0 0;color:var(--muted);background:var(--deep)}
.nav-tab.active{color:var(--cyan);background:var(--card);border-color:var(--b-med);
  border-bottom:2px solid var(--cyan);text-shadow:0 0 8px rgba(0,212,255,.5)}
.hud-title{font-family:var(--fhud);font-size:1.35rem;font-weight:900;letter-spacing:5px;
  color:var(--cyan);text-shadow:0 0 10px rgba(0,212,255,.75),0 0 28px rgba(0,212,255,.35)}
.hud-ver{font-family:var(--fmono);font-size:.58rem;color:var(--sec);letter-spacing:2px;
  margin-left:10px;border:1px solid var(--b-med);padding:2px 7px;border-radius:2px;vertical-align:middle}
.hud-sub{font-family:var(--fmono);font-size:.65rem;color:var(--sec);letter-spacing:2px;opacity:.8;margin-top:3px}
.status-ind{display:inline-flex;align-items:center;gap:6px;font-family:var(--fmono);font-size:.6rem;
  letter-spacing:2px;padding:4px 10px;border-radius:2px}
.status-online{color:var(--green);border:1px solid rgba(0,255,136,.35);background:rgba(0,255,136,.06)}
.status-degraded{color:var(--amber);border:1px solid rgba(255,170,0,.35);background:rgba(255,170,0,.06)}
.status-offline{color:var(--red);border:1px solid rgba(255,59,59,.35);background:rgba(255,59,59,.06)}
.pulse-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.status-online .pulse-dot{background:var(--green)}
.status-degraded .pulse-dot{background:var(--amber)}
.status-offline .pulse-dot{background:var(--red)}
.hud-line{height:1px;background:linear-gradient(90deg,transparent,var(--cyan),var(--blue),transparent);margin:.45rem 0 .6rem 0}
.sec-hdr{display:flex;align-items:center;gap:8px;margin-bottom:.75rem;margin-top:.3rem}
.sec-txt{font-family:var(--fhud);font-size:.56rem;font-weight:700;letter-spacing:3.5px;text-transform:uppercase;color:var(--blue);white-space:nowrap}
.sec-bar{flex:1;height:1px;background:linear-gradient(90deg,var(--b-med),transparent)}
.sec-dot{width:5px;height:5px;border-top:1px solid var(--cyan);border-right:1px solid var(--cyan);flex-shrink:0}
.panel-title{font-family:var(--fhud);font-size:.6rem;font-weight:700;letter-spacing:3.5px;
  color:var(--cyan);text-transform:uppercase;margin-bottom:.8rem;display:flex;align-items:center;gap:8px}
.panel-title::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--b-med),transparent)}
.hud-card{background:var(--card);border:1px solid var(--b-dim);border-radius:4px;padding:1rem 1.2rem;
  margin-bottom:.6rem;position:relative;overflow:hidden}
.hud-card::before{content:"";position:absolute;top:0;left:0;right:0;height:2px}
.card-churn::before{background:linear-gradient(90deg,var(--red),#ff6e40,transparent)}
.card-safe::before{background:linear-gradient(90deg,var(--green),#00e676,transparent)}
.card-neutral::before{background:linear-gradient(90deg,var(--blue),var(--teal),transparent)}
.card-lbl{font-family:var(--fmono);font-size:.6rem;color:var(--muted);letter-spacing:3px;text-transform:uppercase;margin-bottom:4px}
.card-val-lg{font-family:var(--fhud);font-size:1.1rem;font-weight:700;letter-spacing:2px;line-height:1.15;word-break:break-word}
.card-sub{font-family:var(--fmono);font-size:.64rem;color:var(--sec);margin-top:4px}
.t-red{color:var(--red)}.t-green{color:var(--green)}.t-cyan{color:var(--cyan)}
.t-amber{color:var(--amber)}.t-blue{color:var(--blue)}.t-purple{color:var(--purple)}.t-muted{color:var(--sec)}
.verdict-churn{font-family:var(--fhud);font-size:1.7rem;font-weight:900;letter-spacing:4px;
  color:var(--red);text-shadow:0 0 18px rgba(255,59,59,.6)}
.verdict-safe{font-family:var(--fhud);font-size:1.7rem;font-weight:900;letter-spacing:4px;
  color:var(--green);text-shadow:0 0 18px rgba(0,255,136,.6)}
.risk-badge{display:inline-flex;align-items:center;gap:6px;padding:3px 12px 3px 8px;border-radius:2px;
  font-family:var(--fhud);font-size:.6rem;font-weight:700;letter-spacing:3px;text-transform:uppercase;margin-top:7px}
.badge-critical{background:rgba(168,85,247,.12);color:var(--purple);border:1px solid rgba(168,85,247,.4)}
.badge-high{background:rgba(255,59,59,.12);color:var(--red);border:1px solid rgba(255,59,59,.4)}
.badge-medium{background:rgba(255,170,0,.1);color:var(--amber);border:1px solid rgba(255,170,0,.35)}
.badge-low{background:rgba(0,255,136,.08);color:var(--green);border:1px solid rgba(0,255,136,.3)}
.metric-strip{display:flex;gap:8px;margin-bottom:.8rem;flex-wrap:wrap}
.metric-chip{background:var(--card);border:1px solid var(--b-dim);border-radius:2px;padding:8px 12px;flex:1;min-width:105px;position:relative}
.metric-chip::before{content:"";position:absolute;top:0;left:0;width:100%;height:1px;background:var(--b-med)}
.chip-lbl{font-family:var(--fmono);font-size:.54rem;color:var(--muted);letter-spacing:2px;text-transform:uppercase}
.chip-val{font-family:var(--fhud);font-size:.9rem;font-weight:700;letter-spacing:1px;margin-top:2px}
.big-metric{background:var(--card);border:1px solid var(--b-dim);border-radius:4px;padding:1.05rem;text-align:center;position:relative;overflow:hidden;margin-bottom:.5rem}
.big-metric::before{content:"";position:absolute;top:0;left:0;right:0;height:2px}
.metric-accuracy::before{background:linear-gradient(90deg,var(--green),transparent)}
.metric-precision::before{background:linear-gradient(90deg,var(--cyan),transparent)}
.metric-recall::before{background:linear-gradient(90deg,var(--amber),transparent)}
.metric-auc::before{background:linear-gradient(90deg,var(--purple),transparent)}
.metric-f1::before{background:linear-gradient(90deg,var(--teal),transparent)}
.big-metric .m-lbl{font-family:var(--fmono);font-size:.58rem;color:var(--muted);letter-spacing:2.5px;text-transform:uppercase}
.big-metric .m-val{font-family:var(--fhud);font-size:1.65rem;font-weight:800;letter-spacing:1px;margin:5px 0 0 0}
.big-metric .m-sub{font-family:var(--fmono);font-size:.6rem;color:var(--sec);margin-top:3px}
.empty-state{background:var(--panel);border:1px dashed var(--b-med);border-radius:4px;padding:2rem 1rem;
  text-align:center;font-family:var(--fmono);font-size:.7rem;color:var(--sec);letter-spacing:2px}
.hud-divider{height:1px;background:linear-gradient(90deg,transparent,var(--b-med),transparent);margin:1rem 0}
.hud-footer{font-family:var(--fmono);font-size:.58rem;color:var(--muted);text-align:center;
  letter-spacing:2.5px;text-transform:uppercase;padding:.8rem 0;border-top:1px solid var(--b-dim);margin-top:1.5rem;opacity:.65}
.insight-box{background:var(--panel);border:1px solid var(--b-dim);border-left:3px solid var(--cyan);
  border-radius:0 4px 4px 0;padding:.8rem 1rem;margin-bottom:.6rem}
.insight-box.warn{border-left-color:var(--amber)}.insight-box.danger{border-left-color:var(--red)}.insight-box.ok{border-left-color:var(--green)}
.insight-title{font-family:var(--fhud);font-size:.58rem;letter-spacing:3px;text-transform:uppercase;margin-bottom:3px}
.insight-body{font-family:var(--fmono);font-size:.72rem;color:var(--sec);line-height:1.6}
label,.stSelectbox label,.stNumberInput label,.stTextInput label{
  font-family:var(--fmono)!important;font-size:.62rem!important;color:var(--sec)!important;
  letter-spacing:1.8px!important;text-transform:uppercase!important}
.stButton>button{background:transparent!important;color:var(--cyan)!important;border:1px solid var(--cyan)!important;
  border-radius:2px!important;font-family:var(--fhud)!important;font-size:.62rem!important;font-weight:700!important;
  letter-spacing:3px!important;text-transform:uppercase!important;padding:.55rem .8rem!important;width:100%!important}
.stButton>button:hover{background:rgba(0,212,255,.12)!important;box-shadow:0 0 14px rgba(0,212,255,.3)!important}
@media (max-width: 768px) {
  .main .block-container{padding:0.5rem 0.75rem 2rem 0.75rem!important}
  .hud-title{font-size:1.05rem;letter-spacing:3px}
  .big-metric .m-val{font-size:1.3rem}
  .nav-tab{padding:5px 10px;font-size:.52rem;letter-spacing:1.8px}
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Session State & PWA Shortcut Routing ───────────────────────────────────────
VALID_PAGES = {"prediction", "batch", "explainability", "metrics"}
query_page = st.query_params.get("page", "")
default_page = query_page if query_page in VALID_PAGES else "prediction"

for key, default_val in [
    ("history", []),
    ("last_result", None),
    ("batch_df", None),
    ("page", default_page),
]:
    if key not in st.session_state:
        st.session_state[key] = default_val

# ── Plot Styling Helpers ───────────────────────────────────────────────────────
PAPER = "#050a10"
PLOT = "#080f18"
GRID = "#0f2033"
TC = "#5c85a6"
TF = {"family": "Share Tech Mono", "size": 9, "color": TC}
TTF = {"family": "Orbitron", "size": 11, "color": "#7ba3c4"}


def base_layout(
    h: int = 280, t: int = 40, b: int = 40, left: int = 12, r: int = 12
) -> dict[str, Any]:
    return {
        "paper_bgcolor": PAPER,
        "plot_bgcolor": PLOT,
        "height": h,
        "margin": {"t": t, "b": b, "l": left, "r": r},
        "font": {"family": "Share Tech Mono", "color": TC},
    }


def section_header(label: str) -> None:
    st.markdown(
        f'<div class="sec-hdr"><div class="sec-dot"></div>'
        f'<div class="sec-txt">{_esc(label)}</div><div class="sec-bar"></div></div>',
        unsafe_allow_html=True,
    )


# ── Global Header + Live API Health + PWA Install Button ───────────────────────
health = fetch_api_health()
now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

if health.get("_reachable") and health.get("is_model_loaded"):
    algo_name = str(health.get("algorithm") or "LightGBM").upper()
    ver_name = str(health.get("model_version") or "v1")
    status_cls = "status-online"
    status_txt = f"API ONLINE · {algo_name} {ver_name} · {now_str}"
elif health.get("_reachable"):
    status_cls = "status-degraded"
    status_txt = f"API DEGRADED · MODEL UNLOADED · {now_str}"
else:
    status_cls = "status-offline"
    status_txt = f"API OFFLINE · {now_str}"

st.markdown(
    f"""
<div>
  <div class="hdr-row">
    <div>
      <span class="hud-title">🔮 CHURNSCOPE</span>
      <span class="hud-ver">v1.1 · PWA</span>
    </div>
    <span class="status-ind {status_cls}">
      <span class="pulse-dot"></span>{_esc(status_txt)}
    </span>
  </div>
  <div class="hud-sub">CUSTOMER ATTRITION INTELLIGENCE · 41 FEATURES · REAL-TIME SHAP · {_esc(PUBLIC_API_LABEL)}</div>
</div>
""",
    unsafe_allow_html=True,
)
render_pwa_installer()
st.markdown('<div class="hud-line"></div>', unsafe_allow_html=True)

# ── Navigation ─────────────────────────────────────────────────────────────────
NAV_ITEMS = [
    ("prediction", "⟢  Prediction"),
    ("batch", "▤  Batch CSV"),
    ("explainability", "◈  SHAP Explainability"),
    ("metrics", "▸  Model Metrics"),
]
tabs_html = (
    '<div class="nav-bar">'
    + "".join(
        f'<span class="nav-tab {"active" if st.session_state.page == pid else ""}">{plbl}</span>'
        for pid, plbl in NAV_ITEMS
    )
    + "</div>"
)
st.markdown(tabs_html, unsafe_allow_html=True)

nav_cols = st.columns(len(NAV_ITEMS), gap="small")
for col, (pid, plbl) in zip(nav_cols, NAV_ITEMS, strict=True):
    with col:
        if st.button(plbl.upper(), key=f"nav_{pid}"):
            st.session_state.page = pid
            st.rerun()

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SINGLE CUSTOMER PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "prediction":
    with st.expander(
        "⌨  CUSTOMER INPUT PANEL — Configure profile & execute scoring", expanded=True
    ):
        st.markdown(
            '<div class="panel-title">◈ Identity & Demographics</div>', unsafe_allow_html=True
        )
        c1, c2, c3, c4, c5, c6 = st.columns(6, gap="small")
        with c1:
            customer_id = st.text_input("Customer ID", "CUST-12345", max_chars=64, key="cid")
        with c2:
            gender = st.selectbox("Gender", ["Female", "Male"], key="gen")
        with c3:
            senior = st.selectbox("Senior Citizen", ["0", "1"], key="sen")
        with c4:
            partner = st.selectbox("Partner", ["Yes", "No"], key="par")
        with c5:
            dependents = st.selectbox("Dependents", ["No", "Yes"], key="dep")
        with c6:
            tenure = st.number_input(
                "Tenure (months)", min_value=0, max_value=120, value=3, key="ten"
            )

        st.markdown(
            '<div class="panel-title" style="margin-top:.7rem">◈ Subscribed Services</div>',
            unsafe_allow_html=True,
        )
        s1, s2, s3, s4, s5 = st.columns(5, gap="small")
        with s1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"], key="ph")
        ml_opts = ["No", "Yes"] if phone_service == "Yes" else ["No phone service"]
        with s2:
            multiple_lines = st.selectbox("Multiple Lines", ml_opts, key="ml")
        with s3:
            internet_svc = st.selectbox(
                "Internet Service", ["Fiber optic", "DSL", "No"], key="inet"
            )

        addon_opts = ["No", "Yes"] if internet_svc != "No" else ["No internet service"]
        stream_opts = ["Yes", "No"] if internet_svc != "No" else ["No internet service"]
        with s4:
            online_sec = st.selectbox("Online Security", addon_opts, key="os")
        with s5:
            online_backup = st.selectbox("Online Backup", addon_opts, key="ob")

        t1, t2, t3, t4 = st.columns(4, gap="small")
        with t1:
            device_prot = st.selectbox("Device Protection", addon_opts, key="dp")
        with t2:
            tech_support = st.selectbox("Tech Support", addon_opts, key="ts")
        with t3:
            streaming_tv = st.selectbox("Streaming TV", stream_opts, key="stv")
        with t4:
            streaming_mov = st.selectbox("Streaming Movies", stream_opts, key="smov")

        st.markdown(
            '<div class="panel-title" style="margin-top:.7rem">◈ Contract & Billing</div>',
            unsafe_allow_html=True,
        )
        b1, b2, b3, b4, b5 = st.columns(5, gap="small")
        with b1:
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"], key="con"
            )
        with b2:
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"], key="pb")
        with b3:
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                key="pay",
            )
        with b4:
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=500.0,
                value=95.5,
                step=0.5,
                key="mc",
            )
        with b5:
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=100000.0,
                value=286.5,
                step=1.0,
                key="tc",
            )

        predict_btn = st.button("⟢  EXECUTE PREDICTION & SHAP ANALYSIS", key="pred_btn")

    payload: dict[str, Any] = {
        "customerID": customer_id.strip() or None,
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_svc,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_prot,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_mov,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    if predict_btn:
        with st.spinner("Scoring customer and computing TreeExplainer SHAP values..."):
            try:
                resp = requests.post(
                    EXPLAIN_URL, json=payload, headers=_api_headers(), timeout=API_TIMEOUT
                )
                if resp.status_code == 404:
                    resp = requests.post(
                        PREDICT_URL, json=payload, headers=_api_headers(), timeout=API_TIMEOUT
                    )
                resp.raise_for_status()
                result = resp.json()
                result["_input"] = payload.copy()
                result["_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.last_result = result
                st.session_state.history.insert(0, result)
                st.session_state.history = st.session_state.history[:50]
                fetch_api_health.clear()
                st.rerun()
            except requests.exceptions.ConnectionError:
                st.error(
                    f"⚠ SIGNAL LOST — Cannot reach API at {API_BASE_URL}. "
                    "Start the server with `uvicorn api.main:app --host 0.0.0.0 --port 8000`."
                )
            except requests.exceptions.HTTPError as exc:
                detail = ""
                if exc.response is not None:
                    try:
                        detail = exc.response.json().get("detail", exc.response.text)
                    except ValueError:
                        detail = exc.response.text
                st.error(
                    f"⚠ API ERROR ({exc.response.status_code if exc.response is not None else '?'}) — {detail}"
                )
            except Exception as exc:
                st.error(f"⚠ SYSTEM FAULT — {exc}")

    hist: list[dict[str, Any]] = st.session_state.history
    total = len(hist)
    n_churn = sum(1 for h in hist if h.get("churn_label") == "Churn")
    n_safe = total - n_churn
    avg_prob = (
        round(sum(float(h.get("churn_probability", 0)) for h in hist) / total * 100, 1)
        if total
        else 0.0
    )
    n_high = sum(1 for h in hist if h.get("risk_tier") in {"High", "Critical"})
    high_color = "#ff3b3b" if n_high > 0 else "#5c85a6"

    st.markdown(
        f'<div class="metric-strip">'
        f'<div class="metric-chip"><div class="chip-lbl">Total Scans</div><div class="chip-val t-cyan">{total}</div></div>'
        f'<div class="metric-chip"><div class="chip-lbl">Churn Detected</div><div class="chip-val t-red">{n_churn}</div></div>'
        f'<div class="metric-chip"><div class="chip-lbl">Retained / Safe</div><div class="chip-val t-green">{n_safe}</div></div>'
        f'<div class="metric-chip"><div class="chip-lbl">Avg Probability</div><div class="chip-val t-amber">{avg_prob}%</div></div>'
        f'<div class="metric-chip"><div class="chip-lbl">High / Critical</div><div class="chip-val" style="color:{high_color}">{n_high}</div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1, 1.9], gap="large")

    with col_left:
        section_header("Latest Prediction Result")
        r = st.session_state.last_result
        if r is None:
            st.markdown(
                '<div class="empty-state"><div style="font-size:1.8rem;opacity:.35;margin-bottom:.6rem">⊘</div>'
                "AWAITING TARGET CUSTOMER<br>"
                '<span style="font-size:.6rem;opacity:.65">CONFIGURE PROFILE ABOVE → EXECUTE PREDICTION</span></div>',
                unsafe_allow_html=True,
            )
        else:
            prob = float(r.get("churn_probability", 0.0))
            label = str(r.get("churn_label", "?"))
            risk = str(r.get("risk_tier", "Medium"))
            conf = float(r.get("confidence", 0.0))
            thr = float(r.get("threshold_used", 0.5))
            raw_cid = r.get("customer_id") or r.get("_input", {}).get("customerID") or "ANONYMOUS"
            ts = r.get("_timestamp", "")
            is_churn = label == "Churn"
            card_cls = "card-churn" if is_churn else "card-safe"
            badge_map = {
                "Critical": "badge-critical",
                "High": "badge-high",
                "Medium": "badge-medium",
                "Low": "badge-low",
            }
            badge_cls = badge_map.get(risk, "badge-medium")
            ten_v = r.get("_input", {}).get("tenure", "—")
            mch_v = r.get("_input", {}).get("MonthlyCharges", "—")
            verdict_cls = "verdict-churn" if is_churn else "verdict-safe"
            verdict_txt = "⚠ CHURN" if is_churn else "✔ NO CHURN"

            st.markdown(
                f'<div class="hud-card card-neutral">'
                f'<div class="card-lbl">◈ Customer</div>'
                f'<div class="card-val-lg t-cyan">{_esc(raw_cid)}</div>'
                f'<div class="card-sub">{_esc(ts)} · Tenure {_esc(ten_v)}mo · ${_esc(mch_v)}/mo</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="hud-card {card_cls}">'
                f'<div class="card-lbl">◈ Verdict</div>'
                f'<div class="{verdict_cls}">{verdict_txt}</div>'
                f'<div><span class="risk-badge {badge_cls}">● {_esc(risk)} Risk</span></div></div>',
                unsafe_allow_html=True,
            )

            gauge_clr = "#ff3b3b" if is_churn else "#00ff88"
            thr_pct = round(thr * 100, 1)
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=round(prob * 100, 1),
                    number={
                        "suffix": "%",
                        "font": {"size": 32, "family": "Orbitron", "color": gauge_clr},
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickwidth": 1,
                            "tickcolor": TC,
                            "tickfont": {"family": "Share Tech Mono", "size": 9, "color": TC},
                            "tickvals": [0, 25, 50, 75, 100],
                        },
                        "bar": {"color": gauge_clr, "thickness": 0.22},
                        "bgcolor": PLOT,
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, thr_pct], "color": "rgba(0,255,136,.07)"},
                            {"range": [thr_pct, 65], "color": "rgba(255,170,0,.07)"},
                            {"range": [65, 100], "color": "rgba(255,59,59,.08)"},
                        ],
                        "threshold": {
                            "line": {"color": "#0ea5e9", "width": 2},
                            "thickness": 0.8,
                            "value": thr_pct,
                        },
                    },
                    title={
                        "text": "CALIBRATED CHURN PROBABILITY",
                        "font": {"size": 10, "family": "Orbitron", "color": TC},
                    },
                )
            )
            fig_gauge.update_layout(
                paper_bgcolor=PAPER, height=205, margin={"t": 32, "b": 6, "l": 18, "r": 18}
            )
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

            conf_cls = "t-green" if conf > 0.6 else ("t-amber" if conf > 0.3 else "t-red")
            st.markdown(
                f'<div class="metric-strip">'
                f'<div class="metric-chip"><div class="chip-lbl">Confidence</div><div class="chip-val {conf_cls}">{conf * 100:.1f}%</div></div>'
                f'<div class="metric-chip"><div class="chip-lbl">Threshold</div><div class="chip-val t-blue">{thr_pct}%</div></div>'
                f'<div class="metric-chip"><div class="chip-lbl">Probability</div><div class="chip-val t-cyan">{prob:.4f}</div></div>'
                f"</div>",
                unsafe_allow_html=True,
            )

            with st.expander("⌨ RAW API RESPONSE"):
                st.json({k: v for k, v in r.items() if not k.startswith("_")})
            if st.button("◈ INSPECT REAL SHAP DRIVERS →", key="go_expl"):
                st.session_state.page = "explainability"
                st.rerun()

    with col_right:
        section_header("Risk Intelligence · Session Telemetry")
        if total == 0:
            st.markdown(
                '<div class="empty-state">NO SCANS YET — EXECUTE A PREDICTION TO POPULATE LIVE CHARTS</div>',
                unsafe_allow_html=True,
            )
        else:
            hdf = pd.DataFrame(
                [
                    {
                        "customer_id": h.get("customer_id")
                        or h.get("_input", {}).get("customerID")
                        or f"SCAN-{idx + 1}",
                        "probability": round(float(h.get("churn_probability", 0)) * 100, 1),
                        "risk_tier": str(h.get("risk_tier", "?")),
                        "label": str(h.get("churn_label", "?")),
                        "monthly": float(h.get("_input", {}).get("MonthlyCharges", 0)),
                        "tenure": int(h.get("_input", {}).get("tenure", 0)),
                    }
                    for idx, h in enumerate(hist)
                ]
            )
            tier_clr = {
                "Critical": "#a855f7",
                "High": "#ff3b3b",
                "Medium": "#ffaa00",
                "Low": "#00ff88",
            }

            cha, chb = st.columns(2, gap="small")
            with cha:
                bar_colors = [tier_clr.get(t, "#0ea5e9") for t in hdf["risk_tier"]]
                fb = go.Figure(
                    go.Bar(
                        x=hdf["customer_id"],
                        y=hdf["probability"],
                        marker={"color": bar_colors, "opacity": 0.88},
                        text=[f"{p}%" for p in hdf["probability"]],
                        textposition="outside",
                        textfont={"family": "Share Tech Mono", "size": 9, "color": TC},
                        hovertemplate="<b>%{x}</b><br>%{y}%<extra></extra>",
                    )
                )
                if st.session_state.last_result:
                    fb.add_hline(
                        y=float(st.session_state.last_result.get("threshold_used", 0.5)) * 100,
                        line_dash="dashdot",
                        line_color="#0ea5e9",
                        line_width=1,
                        annotation_text="threshold",
                        annotation_font={
                            "family": "Share Tech Mono",
                            "size": 9,
                            "color": "#0ea5e9",
                        },
                    )
                fb.update_layout(
                    **base_layout(240),
                    title={"text": "CHURN PROBABILITY BY CUSTOMER", "font": TTF, "x": 0},
                    xaxis={"tickfont": TF, "gridcolor": GRID, "showgrid": False},
                    yaxis={"range": [0, 118], "tickfont": TF, "gridcolor": GRID, "ticksuffix": "%"},
                    showlegend=False,
                    bargap=0.25,
                )
                st.plotly_chart(fb, use_container_width=True, config={"displayModeBar": False})

            with chb:
                tc_df = hdf["risk_tier"].value_counts().reset_index()
                tc_df.columns = ["risk_tier", "count"]
                fd = go.Figure(
                    go.Pie(
                        labels=tc_df["risk_tier"],
                        values=tc_df["count"],
                        hole=0.62,
                        marker={
                            "colors": [tier_clr.get(t, "#0ea5e9") for t in tc_df["risk_tier"]],
                            "line": {"color": PAPER, "width": 3},
                        },
                        textfont={"family": "Share Tech Mono", "size": 10},
                    )
                )
                fd.update_layout(
                    **base_layout(240),
                    title={"text": "RISK TIER DISTRIBUTION", "font": TTF, "x": 0},
                    legend={"font": TF, "bgcolor": "rgba(0,0,0,0)"},
                    annotations=[
                        {
                            "text": f"<b>{total}</b><br><span style='font-size:9px'>SCANS</span>",
                            "x": 0.5,
                            "y": 0.5,
                            "font": {"family": "Orbitron", "size": 13, "color": "#7ba3c4"},
                            "showarrow": False,
                        }
                    ],
                )
                st.plotly_chart(fd, use_container_width=True, config={"displayModeBar": False})

            if total >= 2:
                chc, chd = st.columns(2, gap="small")
                with chc:
                    ordered = hdf.iloc[::-1].reset_index(drop=True)
                    fl = go.Figure(
                        go.Scatter(
                            x=list(range(1, len(ordered) + 1)),
                            y=ordered["probability"],
                            mode="lines+markers",
                            line={"color": "#0ea5e9", "width": 2},
                            marker={
                                "color": [tier_clr.get(t, "#0ea5e9") for t in ordered["risk_tier"]],
                                "size": 8,
                            },
                            text=ordered["customer_id"],
                            fill="tozeroy",
                            fillcolor="rgba(14,165,233,.06)",
                        )
                    )
                    fl.update_layout(
                        **base_layout(210),
                        title={"text": "SESSION PROBABILITY TIMELINE", "font": TTF, "x": 0},
                        xaxis={"tickfont": TF, "showgrid": False},
                        yaxis={
                            "range": [0, 110],
                            "tickfont": TF,
                            "gridcolor": GRID,
                            "ticksuffix": "%",
                        },
                        showlegend=False,
                    )
                    st.plotly_chart(fl, use_container_width=True, config={"displayModeBar": False})

                with chd:
                    fs = go.Figure(
                        go.Scatter(
                            x=hdf["monthly"],
                            y=hdf["probability"],
                            mode="markers",
                            marker={
                                "color": [tier_clr.get(t, "#0ea5e9") for t in hdf["risk_tier"]],
                                "size": 11,
                                "opacity": 0.85,
                            },
                            text=hdf["customer_id"],
                            hovertemplate="<b>%{text}</b><br>$%{x}/mo · %{y}%<extra></extra>",
                        )
                    )
                    fs.update_layout(
                        **base_layout(210),
                        title={"text": "MONTHLY CHARGES vs CHURN RISK", "font": TTF, "x": 0},
                        xaxis={
                            "tickfont": TF,
                            "gridcolor": GRID,
                            "title": {"text": "MONTHLY ($)", "font": TF},
                        },
                        yaxis={
                            "range": [0, 110],
                            "tickfont": TF,
                            "gridcolor": GRID,
                            "ticksuffix": "%",
                        },
                        showlegend=False,
                    )
                    st.plotly_chart(fs, use_container_width=True, config={"displayModeBar": False})

        section_header("Prediction Log · Session Memory")
        if total == 0:
            st.markdown('<div class="empty-state">SESSION LOG EMPTY</div>', unsafe_allow_html=True)
        else:
            log_df = pd.DataFrame(
                [
                    {
                        "Timestamp": h.get("_timestamp", "—"),
                        "Customer ID": h.get("customer_id")
                        or h.get("_input", {}).get("customerID")
                        or "—",
                        "Churn Prob": f"{float(h.get('churn_probability', 0)) * 100:.1f}%",
                        "Verdict": h.get("churn_label", "—"),
                        "Risk Tier": h.get("risk_tier", "—"),
                        "Confidence": f"{float(h.get('confidence', 0)) * 100:.1f}%",
                        "Contract": h.get("_input", {}).get("Contract", "—"),
                    }
                    for h in hist
                ]
            )
            st.dataframe(log_df, use_container_width=True, hide_index=True, height=190)
            lb1, lb2, _ = st.columns([1, 1, 2], gap="small")
            with lb1:
                st.download_button(
                    "⬇ EXPORT CSV",
                    log_df.to_csv(index=False).encode("utf-8"),
                    f"churnscope_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                )
            with lb2:
                if st.button("⌫ CLEAR LOG", key="clr"):
                    st.session_state.history = []
                    st.session_state.last_result = None
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BATCH CSV SCORING
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "batch":
    section_header("Batch Customer Scoring (Up to 1,000 Rows per API Request)")
    uploaded = st.file_uploader(
        "Upload a Telco customer CSV (matching the 19 required customer features)",
        type=["csv"],
        key="batch_upload",
    )

    if uploaded is not None:
        raw_batch = pd.read_csv(uploaded)
        st.caption(f"Loaded {len(raw_batch):,} rows × {raw_batch.shape[1]} columns.")
        if st.button("▤  SCORE UPLOADED COHORT", key="run_batch_btn"):
            work = raw_batch.head(1000).copy()
            if "SeniorCitizen" in work.columns:
                work["SeniorCitizen"] = (
                    pd.to_numeric(work["SeniorCitizen"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                    .astype(str)
                )
            if "TotalCharges" in work.columns:
                work["TotalCharges"] = pd.to_numeric(work["TotalCharges"], errors="coerce").fillna(
                    0.0
                )
            if "MonthlyCharges" in work.columns:
                work["MonthlyCharges"] = pd.to_numeric(
                    work["MonthlyCharges"], errors="coerce"
                ).fillna(0.0)
            if "tenure" in work.columns:
                work["tenure"] = (
                    pd.to_numeric(work["tenure"], errors="coerce").fillna(0).astype(int)
                )
            records = work.drop(columns=["Churn"], errors="ignore").to_dict(orient="records")

            with st.spinner(f"Scoring {len(records):,} customers via /batch-predict..."):
                try:
                    resp = requests.post(
                        BATCH_URL,
                        json={"customers": records},
                        headers=_api_headers(),
                        timeout=max(API_TIMEOUT, 30.0),
                    )
                    resp.raise_for_status()
                    bdata = resp.json()
                    preds = pd.DataFrame(bdata["predictions"])
                    merged = pd.concat([work.reset_index(drop=True), preds], axis=1)
                    merged = merged.sort_values("churn_probability", ascending=False)
                    st.session_state.batch_df = merged
                except Exception as exc:
                    st.error(f"⚠ Batch scoring failed: {exc}")

    if st.session_state.batch_df is not None:
        bdf: pd.DataFrame = st.session_state.batch_df
        b_total = len(bdf)
        b_churn = int((bdf["churn_prediction"] == 1).sum())
        b_crit = int(bdf["risk_tier"].isin(["High", "Critical"]).sum())
        st.markdown(
            f'<div class="metric-strip">'
            f'<div class="metric-chip"><div class="chip-lbl">Scored Customers</div><div class="chip-val t-cyan">{b_total:,}</div></div>'
            f'<div class="metric-chip"><div class="chip-lbl">Predicted Churners</div><div class="chip-val t-red">{b_churn:,} ({b_churn / max(b_total, 1) * 100:.1f}%)</div></div>'
            f'<div class="metric-chip"><div class="chip-lbl">High / Critical</div><div class="chip-val t-amber">{b_crit:,}</div></div>'
            f'<div class="metric-chip"><div class="chip-lbl">Mean Probability</div><div class="chip-val t-purple">{bdf["churn_probability"].mean() * 100:.1f}%</div></div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(bdf, use_container_width=True, hide_index=True, height=360)
        st.download_button(
            "⬇ DOWNLOAD SCORED COHORT CSV",
            bdf.to_csv(index=False).encode("utf-8"),
            f"churnscope_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — REAL TREEEXPLAINER SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "explainability":
    r = st.session_state.last_result
    if r is None or not r.get("shap_values"):
        st.markdown(
            '<div class="empty-state" style="margin-top:1.5rem;padding:2.8rem">'
            '<div style="font-size:2rem;opacity:.35;margin-bottom:.8rem">◈</div>'
            "NO SHAP ATTRIBUTION LOADED<br>"
            '<span style="font-size:.62rem;opacity:.65">RUN A PREDICTION ON THE PREDICTION TAB TO COMPUTE LIVE TREEEXPLAINER SHAP VALUES</span></div>',
            unsafe_allow_html=True,
        )
    else:
        prob = float(r.get("churn_probability", 0.0))
        label = str(r.get("churn_label", "?"))
        base_val = float(r.get("base_value", 0.0))
        raw_cid = r.get("customer_id") or r.get("_input", {}).get("customerID") or "ANONYMOUS"
        shap_items: list[dict[str, Any]] = list(r.get("shap_values", []))
        is_churn = label == "Churn"

        ea, eb = st.columns([1.3, 1.7], gap="large")
        with ea:
            section_header(f"TreeExplainer SHAP Attribution · {raw_cid}")
            card_cls = "card-churn" if is_churn else "card-safe"
            txt_cls = "t-red" if is_churn else "t-green"
            st.markdown(
                f'<div class="hud-card {card_cls}">'
                f'<div class="card-lbl">◈ Local Model Explanation (Log-Odds SHAP)</div>'
                f'<div class="card-val-lg {txt_cls}">{_esc(raw_cid)}</div>'
                f'<div class="card-sub">Probability: {prob * 100:.1f}% · Verdict: {_esc(label)} · Base Log-Odds: {base_val:+.3f}</div></div>',
                unsafe_allow_html=True,
            )

            feats = [item.get("display_name", item["feature"]) for item in shap_items]
            vals = [float(item["shap_value"]) for item in shap_items]
            colors = ["#ff3b3b" if v > 0 else "#00ff88" for v in vals]
            txt_vals = [f"{v:+.4f}" for v in vals]

            fsh = go.Figure(
                go.Bar(
                    x=vals,
                    y=feats,
                    orientation="h",
                    marker={"color": colors, "opacity": 0.88},
                    text=txt_vals,
                    textposition="outside",
                    textfont={"family": "Share Tech Mono", "size": 9, "color": TC},
                    hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>",
                )
            )
            fsh.add_vline(x=0, line_color=TC, line_width=1)
            fsh.update_layout(
                **base_layout(h=400, t=30, b=30, left=10, r=90),
                title={"text": "LOCAL SHAP VALUES  (+) pushes toward churn", "font": TTF, "x": 0},
                xaxis={
                    "tickfont": TF,
                    "gridcolor": GRID,
                    "title": {"text": "SHAP CONTRIBUTION (LOG-ODDS)", "font": TF},
                },
                yaxis={
                    "tickfont": {"family": "Share Tech Mono", "size": 10, "color": "#7ba3c4"},
                    "autorange": "reversed",
                },
                showlegend=False,
            )
            st.plotly_chart(fsh, use_container_width=True, config={"displayModeBar": False})

        with eb:
            section_header("Top 8 Feature Impacts")
            top8 = shap_items[:8]
            fwf = go.Figure(
                go.Bar(
                    x=[it.get("display_name", it["feature"]) for it in top8],
                    y=[abs(float(it["shap_value"])) for it in top8],
                    marker={
                        "color": [
                            "#ff3b3b" if float(it["shap_value"]) > 0 else "#00ff88" for it in top8
                        ],
                        "opacity": 0.88,
                    },
                    text=[
                        f"{'↑' if float(it['shap_value']) > 0 else '↓'}{abs(float(it['shap_value'])):.3f}"
                        for it in top8
                    ],
                    textposition="outside",
                    textfont={"family": "Share Tech Mono", "size": 9, "color": TC},
                )
            )
            fwf.update_layout(
                **base_layout(h=260, t=34, b=70, left=10, r=10),
                title={"text": "ABSOLUTE SHAP MAGNITUDE (TOP 8)", "font": TTF, "x": 0},
                xaxis={
                    "tickfont": {"family": "Share Tech Mono", "size": 8, "color": TC},
                    "tickangle": -28,
                },
                yaxis={"tickfont": TF, "gridcolor": GRID},
                showlegend=False,
            )
            st.plotly_chart(fwf, use_container_width=True, config={"displayModeBar": False})

            section_header("Actionable Retention Intelligence")
            pos_drivers = [it for it in shap_items if float(it["shap_value"]) > 0][:3]
            neg_drivers = [it for it in shap_items if float(it["shap_value"]) < 0][:3]

            if pos_drivers:
                dh = " · ".join(
                    f'<span class="t-red">{_esc(it.get("display_name", it["feature"]))}</span> '
                    f"(+{float(it['shap_value']):.3f})"
                    for it in pos_drivers
                )
                st.markdown(
                    f'<div class="insight-box danger">'
                    f'<div class="insight-title t-red">⚠ Primary Churn Drivers</div>'
                    f'<div class="insight-body">{dh}</div></div>',
                    unsafe_allow_html=True,
                )
            if neg_drivers:
                rh = " · ".join(
                    f'<span class="t-green">{_esc(it.get("display_name", it["feature"]))}</span> '
                    f"({float(it['shap_value']):.3f})"
                    for it in neg_drivers
                )
                st.markdown(
                    f'<div class="insight-box ok">'
                    f'<div class="insight-title t-green">✔ Protective Retention Factors</div>'
                    f'<div class="insight-body">{rh}</div></div>',
                    unsafe_allow_html=True,
                )

            recs = r.get("recommendations", [])
            if recs:
                rec_html = "<br>".join(f"▸ {_esc(item)}" for item in recs)
                st.markdown(
                    f'<div class="insight-box warn">'
                    f'<div class="insight-title t-amber">◈ Recommended Retention Playbook</div>'
                    f'<div class="insight-body">{rec_html}</div></div>',
                    unsafe_allow_html=True,
                )

            inp = r.get("_input", {})
            section_header("Customer Profile Snapshot")
            profile_rows = [
                ("Contract", inp.get("Contract", "—")),
                ("Tenure", f"{inp.get('tenure', '—')} months"),
                ("Internet Service", inp.get("InternetService", "—")),
                ("Monthly Charges", f"${inp.get('MonthlyCharges', '—')}"),
                ("Total Charges", f"${inp.get('TotalCharges', '—')}"),
                ("Tech Support", inp.get("TechSupport", "—")),
                ("Online Security", inp.get("OnlineSecurity", "—")),
                ("Payment Method", inp.get("PaymentMethod", "—")),
            ]
            st.dataframe(
                pd.DataFrame(profile_rows, columns=["Attribute", "Value"]),
                use_container_width=True,
                hide_index=True,
                height=220,
            )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — REAL HELD-OUT MODEL METRICS (ZERO MOCK DATA)
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "metrics":
    metrics, source = fetch_model_metrics()
    if metrics is None:
        st.markdown(
            '<div class="empty-state" style="margin-top:1.5rem;padding:2.8rem">'
            "NO EVALUATION ARTIFACTS FOUND<br>"
            '<span style="font-size:.62rem;opacity:.65">RUN `python train_pipeline.py` TO GENERATE HELD-OUT METRICS</span></div>',
            unsafe_allow_html=True,
        )
    else:
        src_badge = (
            "LIVE API (/model/metrics)"
            if source == "live_api"
            else "LOCAL ARTIFACT (evaluation_results.json)"
        )
        section_header(f"Held-Out Test Set Performance · Source: {src_badge}")

        cm = metrics.get("confusion_matrix", {})
        tp = int(cm.get("true_positive", 0))
        fp = int(cm.get("false_positive", 0))
        fn = int(cm.get("false_negative", 0))
        tn = int(cm.get("true_negative", 0))
        tot_cm = max(tp + fp + fn + tn, 1)
        accuracy = float(metrics.get("accuracy", (tp + tn) / tot_cm))
        precision = float(metrics.get("precision", 0.0))
        recall = float(metrics.get("recall", 0.0))
        f1_val = float(metrics.get("f1", 0.0))
        roc_auc = float(metrics.get("roc_auc", 0.0))
        pr_auc = float(metrics.get("pr_auc", 0.0))

        kpi_cols = st.columns(6, gap="small")
        kpi_specs = [
            (
                kpi_cols[0],
                "ROC-AUC",
                f"{roc_auc:.3f}",
                "metric-auc",
                "ranking discrimination",
                "#a855f7",
            ),
            (
                kpi_cols[1],
                "PR-AUC",
                f"{pr_auc:.3f}",
                "metric-precision",
                "vs 0.265 base rate",
                "#00d4ff",
            ),
            (
                kpi_cols[2],
                "Recall",
                f"{recall * 100:.1f}%",
                "metric-recall",
                "true churners caught",
                "#ffaa00",
            ),
            (
                kpi_cols[3],
                "Precision",
                f"{precision * 100:.1f}%",
                "metric-precision",
                "flagged who churn",
                "#0ea5e9",
            ),
            (
                kpi_cols[4],
                "F1 Score",
                f"{f1_val:.3f}",
                "metric-f1",
                "val-tuned threshold",
                "#14b8a6",
            ),
            (
                kpi_cols[5],
                "Accuracy",
                f"{accuracy * 100:.1f}%",
                "metric-accuracy",
                f"n={tot_cm:,} test set",
                "#00ff88",
            ),
        ]
        for col, lbl, val, css, sub, clr in kpi_specs:
            with col:
                st.markdown(
                    f'<div class="big-metric {css}">'
                    f'<div class="m-lbl">{lbl}</div>'
                    f'<div class="m-val" style="color:{clr}">{val}</div>'
                    f'<div class="m-sub">{sub}</div></div>',
                    unsafe_allow_html=True,
                )

        biz = metrics.get("business_metrics", {})
        rank = metrics.get("ranking_metrics", {})
        if biz:
            net_usd = float(biz.get("net_business_benefit_usd", 0.0))
            roi_pct = float(biz.get("campaign_roi_pct", 0.0))
            lift20 = float(rank.get("lift_at_20pct", 0.0))
            rec20 = float(rank.get("recall_at_20pct", 0.0))
            brier = float(metrics.get("brier_score", 0.0))
            st.markdown(
                f'<div class="metric-strip">'
                f'<div class="metric-chip"><div class="chip-lbl">Est. Net LTV Benefit</div><div class="chip-val t-green">${net_usd:,.0f}</div></div>'
                f'<div class="metric-chip"><div class="chip-lbl">Campaign ROI</div><div class="chip-val t-cyan">{roi_pct:.1f}%</div></div>'
                f'<div class="metric-chip"><div class="chip-lbl">Top-20% Lift</div><div class="chip-val t-purple">{lift20:.2f}x</div></div>'
                f'<div class="metric-chip"><div class="chip-lbl">Recall @ Top 20%</div><div class="chip-val t-amber">{rec20 * 100:.1f}%</div></div>'
                f'<div class="metric-chip"><div class="chip-lbl">Brier Calibration</div><div class="chip-val t-blue">{brier:.4f}</div></div>'
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown('<div class="hud-divider"></div>', unsafe_allow_html=True)

        r1c, r2c = st.columns(2, gap="small")
        with r1c:
            section_header("ROC Curve")
            fr = go.Figure()
            fr.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    line={"color": TC, "dash": "dash", "width": 1},
                    showlegend=False,
                )
            )
            fr.add_trace(
                go.Scatter(
                    x=metrics["roc_curve"]["fpr"],
                    y=metrics["roc_curve"]["tpr"],
                    mode="lines",
                    line={"color": "#a855f7", "width": 2.5},
                    fill="tozeroy",
                    fillcolor="rgba(168,85,247,.08)",
                    name=f"AUC = {roc_auc:.4f}",
                )
            )
            fr.update_layout(
                **base_layout(290),
                title={"text": f"ROC CURVE · AUC = {roc_auc:.4f}", "font": TTF, "x": 0},
                xaxis={
                    "tickfont": TF,
                    "gridcolor": GRID,
                    "range": [0, 1],
                    "title": {"text": "FALSE POSITIVE RATE", "font": TF},
                },
                yaxis={
                    "tickfont": TF,
                    "gridcolor": GRID,
                    "range": [0, 1],
                    "title": {"text": "TRUE POSITIVE RATE", "font": TF},
                },
                legend={"font": TF, "bgcolor": "rgba(0,0,0,0)"},
            )
            st.plotly_chart(fr, use_container_width=True, config={"displayModeBar": False})

        with r2c:
            section_header("Precision-Recall Curve")
            fp_fig = go.Figure(
                go.Scatter(
                    x=metrics["pr_curve"]["recall"],
                    y=metrics["pr_curve"]["precision"],
                    mode="lines",
                    line={"color": "#00d4ff", "width": 2.5},
                    fill="tozeroy",
                    fillcolor="rgba(0,212,255,.08)",
                )
            )
            fp_fig.add_hline(
                y=float(metrics.get("positive_rate", 0.265)),
                line_dash="dash",
                line_color=TC,
                line_width=1,
            )
            fp_fig.update_layout(
                **base_layout(290),
                title={
                    "text": f"PRECISION-RECALL CURVE · PR-AUC = {pr_auc:.4f}",
                    "font": TTF,
                    "x": 0,
                },
                xaxis={
                    "tickfont": TF,
                    "gridcolor": GRID,
                    "range": [0, 1],
                    "title": {"text": "RECALL", "font": TF},
                },
                yaxis={
                    "tickfont": TF,
                    "gridcolor": GRID,
                    "range": [0, 1],
                    "title": {"text": "PRECISION", "font": TF},
                },
                showlegend=False,
            )
            st.plotly_chart(fp_fig, use_container_width=True, config={"displayModeBar": False})

        r3c, r4c = st.columns(2, gap="small")
        with r3c:
            section_header("Confusion Matrix (Validation-Tuned Threshold)")
            mat = [[tn, fp], [fn, tp]]
            pct_txt = [[f"{v / tot_cm * 100:.1f}%<br>{v}" for v in row] for row in mat]
            fcm = go.Figure(
                go.Heatmap(
                    z=mat,
                    x=["Predicted: No Churn", "Predicted: Churn"],
                    y=["Actual: No Churn", "Actual: Churn"],
                    text=pct_txt,
                    texttemplate="%{text}",
                    colorscale=[[0, "#050a10"], [0.3, "#0f2033"], [0.7, "#1a3a55"], [1, "#0ea5e9"]],
                    showscale=False,
                )
            )
            fcm.update_layout(
                **base_layout(280, t=36, b=36, left=10, r=10),
                title={
                    "text": f"CONFUSION MATRIX (THRESH = {float(metrics.get('threshold', 0.5)):.4f})",
                    "font": TTF,
                    "x": 0,
                },
                xaxis={"tickfont": {"family": "Share Tech Mono", "size": 9, "color": "#7ba3c4"}},
                yaxis={"tickfont": {"family": "Share Tech Mono", "size": 9, "color": "#7ba3c4"}},
            )
            st.plotly_chart(fcm, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f'<div class="metric-strip">'
                f'<div class="metric-chip"><div class="chip-lbl">True Pos</div><div class="chip-val t-green">{tp}</div></div>'
                f'<div class="metric-chip"><div class="chip-lbl">False Pos</div><div class="chip-val t-amber">{fp}</div></div>'
                f'<div class="metric-chip"><div class="chip-lbl">False Neg</div><div class="chip-val t-red">{fn}</div></div>'
                f'<div class="metric-chip"><div class="chip-lbl">True Neg</div><div class="chip-val t-blue">{tn}</div></div>'
                f"</div>",
                unsafe_allow_html=True,
            )

        with r4c:
            section_header("Global SHAP Feature Importance (Top 15 · Cyan = Engineered)")
            fi_raw = metrics.get("feature_importance", [])
            top_fi = fi_raw[:15]
            if top_fi:
                fi_names = [item["feature"] for item in top_fi]
                fi_vals = [round(float(item["mean_abs_shap"]), 4) for item in top_fi]
                fi_colors = [
                    "#00d4ff" if name in ENGINEERED_FEATURE_SET else "#a855f7" for name in fi_names
                ]
                ffi = go.Figure(
                    go.Bar(
                        x=fi_vals,
                        y=fi_names,
                        orientation="h",
                        marker={"color": fi_colors, "opacity": 0.9},
                        text=[f"{v:.3f}" for v in fi_vals],
                        textposition="outside",
                        textfont={"family": "Share Tech Mono", "size": 9, "color": TC},
                    )
                )
                ffi.update_layout(
                    **base_layout(345, t=28, b=28, left=10, r=65),
                    title={
                        "text": "MEAN |SHAP| IMPORTANCE (41-FEATURE PIPELINE)",
                        "font": TTF,
                        "x": 0,
                    },
                    xaxis={"tickfont": TF, "gridcolor": GRID},
                    yaxis={
                        "tickfont": {"family": "Share Tech Mono", "size": 10, "color": "#7ba3c4"},
                        "autorange": "reversed",
                    },
                    showlegend=False,
                )
                st.plotly_chart(ffi, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="hud-divider"></div>', unsafe_allow_html=True)
        section_header("Model Registry & Training Provenance")
        summary = metrics.get("model_summary", {})
        active_reg = (metrics.get("registry") or {}).get("active_model") or {}
        algo_lbl = str(
            summary.get("algorithm") or active_reg.get("algorithm") or "lightgbm"
        ).upper()
        cv_auc = summary.get("cv_roc_auc_mean") or active_reg.get("cv_roc_auc")
        n_train = summary.get("n_train") or active_reg.get("train_samples") or 0
        n_val = summary.get("n_val") or active_reg.get("val_samples") or 0
        n_feat = summary.get("n_features") or active_reg.get("n_features") or 41
        log_loss_v = float(metrics.get("log_loss", 0.0))

        st.markdown(
            f'<div class="metric-strip">'
            f'<div class="metric-chip"><div class="chip-lbl">Algorithm</div><div class="chip-val t-cyan">{_esc(algo_lbl)}</div></div>'
            f'<div class="metric-chip"><div class="chip-lbl">Features</div><div class="chip-val t-purple">{int(n_feat)}</div></div>'
            f'<div class="metric-chip"><div class="chip-lbl">5-Fold CV AUC</div><div class="chip-val t-green">{f"{float(cv_auc):.4f}" if cv_auc else "—"}</div></div>'
            f'<div class="metric-chip"><div class="chip-lbl">Train / Val / Test</div><div class="chip-val t-blue">{int(n_train):,} / {int(n_val):,} / {tot_cm:,}</div></div>'
            f'<div class="metric-chip"><div class="chip-lbl">Val Threshold</div><div class="chip-val t-amber">{float(metrics.get("threshold", 0.5)):.4f}</div></div>'
            f'<div class="metric-chip"><div class="chip-lbl">Log Loss</div><div class="chip-val t-muted">{log_loss_v:.4f}</div></div>'
            f"</div>",
            unsafe_allow_html=True,
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
today = datetime.now().strftime("%Y-%m-%d")
st.markdown(
    f'<div class="hud-footer">'
    f"CHURNSCOPE v1.1 · LIGHTGBM + PLATT CALIBRATION · TREEEXPLAINER SHAP · INSTALLABLE PWA · {today}"
    f"</div>",
    unsafe_allow_html=True,
)
