"""Generate deterministic PWA icons and install-dialog screenshots for ChurnScope."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG_VOID = (2, 4, 8, 255)
BG_PANEL = (8, 15, 24, 255)
BG_CARD = (12, 21, 32, 255)
BORDER_DIM = (26, 58, 85, 255)
CYAN = (0, 212, 255, 255)
BLUE = (14, 165, 233, 255)
GREEN = (0, 255, 136, 255)
RED = (255, 59, 59, 255)
AMBER = (255, 170, 0, 255)
PURPLE = (168, 85, 247, 255)
TEXT_PRI = (226, 240, 255, 255)
TEXT_SEC = (123, 163, 196, 255)


def draw_icon(size: int, maskable: bool = False) -> Image.Image:
    """Render the ChurnScope HUD radar/scope emblem at the requested pixel resolution."""
    scale = 2
    s = size * scale
    img = Image.new("RGBA", (s, s), BG_VOID)
    draw = ImageDraw.Draw(img)

    cx = cy = s // 2
    safe = 0.34 if maskable else 0.42
    outer_r = int(s * safe)

    if not maskable:
        pad = int(s * 0.04)
        corner = int(s * 0.20)
        draw.rounded_rectangle(
            [pad, pad, s - pad, s - pad],
            radius=corner,
            fill=BG_PANEL,
            outline=BORDER_DIM,
            width=max(2, s // 96),
        )
    else:
        draw.rectangle([0, 0, s, s], fill=BG_PANEL)

    # Subtle radial glow rings
    for idx, frac in enumerate((1.12, 0.96, 0.72, 0.48)):
        r = int(outer_r * frac)
        color = BORDER_DIM if idx in (0, 2) else CYAN
        width = max(2, s // (90 if idx == 1 else 140))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=width)

    # Highlighted arc segment (cyan/green radar sweep)
    arc_r = int(outer_r * 0.96)
    arc_w = max(4, s // 34)
    draw.arc(
        [cx - arc_r, cy - arc_r, cx + arc_r, cy + arc_r],
        start=210,
        end=345,
        fill=CYAN,
        width=arc_w,
    )
    draw.arc(
        [cx - arc_r, cy - arc_r, cx + arc_r, cy + arc_r],
        start=25,
        end=115,
        fill=GREEN,
        width=max(3, arc_w - 2),
    )

    # Crosshair reticle ticks
    tick_in = int(outer_r * 0.80)
    tick_out = int(outer_r * 1.10)
    tick_w = max(2, s // 90)
    for angle_deg in (0, 90, 180, 270):
        rad = math.radians(angle_deg)
        x1, y1 = cx + int(tick_in * math.cos(rad)), cy + int(tick_in * math.sin(rad))
        x2, y2 = cx + int(tick_out * math.cos(rad)), cy + int(tick_out * math.sin(rad))
        draw.line([x1, y1, x2, y2], fill=CYAN, width=tick_w)

    # Telemetry pulse line across center
    pts = [
        (cx - int(outer_r * 0.62), cy + int(outer_r * 0.10)),
        (cx - int(outer_r * 0.30), cy + int(outer_r * 0.10)),
        (cx - int(outer_r * 0.14), cy - int(outer_r * 0.36)),
        (cx + int(outer_r * 0.06), cy + int(outer_r * 0.32)),
        (cx + int(outer_r * 0.24), cy - int(outer_r * 0.18)),
        (cx + int(outer_r * 0.42), cy - int(outer_r * 0.04)),
        (cx + int(outer_r * 0.62), cy - int(outer_r * 0.04)),
    ]
    draw.line(pts, fill=CYAN, width=max(4, s // 38), joint="curve")

    # Risk beacon dot
    bx, by = cx + int(outer_r * 0.44), cy - int(outer_r * 0.42)
    br = max(4, s // 30)
    draw.ellipse([bx - br, by - br, bx + br, by + br], fill=GREEN)

    return img.resize((size, size), Image.Resampling.LANCZOS).convert("RGB")


def draw_screenshot(width: int, height: int, mobile: bool = False) -> Image.Image:
    """Create a clean preview screenshot for Chrome/Edge/Android richer PWA install prompts."""
    img = Image.new("RGBA", (width, height), BG_VOID)
    draw = ImageDraw.Draw(img)
    font_sm = ImageFont.load_default()

    pad = 28 if mobile else 40
    # Top HUD header bar
    draw.rectangle([0, 0, width, 74], fill=BG_PANEL)
    draw.line([0, 74, width, 74], fill=CYAN, width=2)
    draw.text(
        (pad, 22), "CHURNSCOPE v1.1  |  CUSTOMER ATTRITION INTELLIGENCE", fill=CYAN, font=font_sm
    )
    draw.text(
        (pad, 44),
        "LightGBM (41 Features)  *  Real-Time SHAP  *  FastAPI",
        fill=TEXT_SEC,
        font=font_sm,
    )

    # KPI strip cards
    kpi_items = [
        ("ROC-AUC", "0.846", PURPLE),
        ("PR-AUC", "0.686", CYAN),
        ("RECALL", "79.1%", AMBER),
        ("TOP-20% LIFT", "2.61x", GREEN),
    ]
    cols = 2 if mobile else 4
    gap = 16
    card_w = (width - 2 * pad - (cols - 1) * gap) // cols
    card_h = 92
    y0 = 102

    for i, (label, val, color) in enumerate(kpi_items):
        r, c = divmod(i, cols)
        x1 = pad + c * (card_w + gap)
        y1 = y0 + r * (card_h + gap)
        x2, y2 = x1 + card_w, y1 + card_h
        draw.rounded_rectangle(
            [x1, y1, x2, y2], radius=6, fill=BG_CARD, outline=BORDER_DIM, width=2
        )
        draw.line([x1, y1, x2, y1], fill=color, width=3)
        draw.text((x1 + 16, y1 + 18), label, fill=TEXT_SEC, font=font_sm)
        draw.text((x1 + 16, y1 + 48), val, fill=color, font=font_sm)

    # Main content panels (Gauge + SHAP bars)
    panel_top = y0 + ((len(kpi_items) + cols - 1) // cols) * (card_h + gap) + 18
    if mobile:
        g_box = [pad, panel_top, width - pad, panel_top + 380]
        s_box = [pad, panel_top + 400, width - pad, height - pad - 40]
    else:
        split_x = int(width * 0.40)
        g_box = [pad, panel_top, split_x, height - pad]
        s_box = [split_x + gap, panel_top, width - pad, height - pad]

    # Left/top panel: Churn Probability Gauge
    draw.rounded_rectangle(g_box, radius=8, fill=BG_CARD, outline=BORDER_DIM, width=2)
    draw.line([g_box[0], g_box[1], g_box[2], g_box[1]], fill=RED, width=3)
    draw.text(
        (g_box[0] + 20, g_box[1] + 18), "LATEST PREDICTION  *  CUST-12345", fill=CYAN, font=font_sm
    )
    draw.text(
        (g_box[0] + 20, g_box[1] + 44), "VERDICT: HIGH CHURN RISK (68.4%)", fill=RED, font=font_sm
    )
    gcx = (g_box[0] + g_box[2]) // 2
    gcy = g_box[1] + 245
    gr = min((g_box[2] - g_box[0]) // 3, 130)
    draw.arc(
        [gcx - gr, gcy - gr, gcx + gr, gcy + gr], start=180, end=360, fill=BORDER_DIM, width=22
    )
    draw.arc([gcx - gr, gcy - gr, gcx + gr, gcy + gr], start=180, end=303, fill=RED, width=22)
    draw.text((gcx - 24, gcy - 28), "68.4%", fill=TEXT_PRI, font=font_sm)
    draw.text((gcx - 52, gcy - 6), "THRESHOLD: 27.6%", fill=BLUE, font=font_sm)

    # Right/bottom panel: Real SHAP Attribution Bars
    draw.rounded_rectangle(s_box, radius=8, fill=BG_CARD, outline=BORDER_DIM, width=2)
    draw.line([s_box[0], s_box[1], s_box[2], s_box[1]], fill=CYAN, width=3)
    draw.text(
        (s_box[0] + 20, s_box[1] + 18),
        "REAL-TIME SHAP FEATURE ATTRIBUTION",
        fill=CYAN,
        font=font_sm,
    )

    bars = [
        ("Month-to-Month Contract", 0.88, RED),
        ("Charge / Tenure Ratio", 0.64, RED),
        ("Fiber Optic Internet", 0.49, RED),
        ("Online Security (No)", 0.36, RED),
        ("Electronic Check", 0.28, RED),
        ("Service Stickiness", -0.34, GREEN),
        ("Has Partner (Yes)", -0.22, GREEN),
    ]
    zero_x = s_box[0] + int((s_box[2] - s_box[0]) * 0.50)
    max_span = int((s_box[2] - s_box[0]) * 0.40)
    row_start = s_box[1] + 64
    row_step = max(42, (s_box[3] - row_start - 24) // len(bars))
    draw.line([zero_x, row_start - 8, zero_x, s_box[3] - 20], fill=BORDER_DIM, width=2)

    for idx, (fname, weight, clr) in enumerate(bars):
        ry = row_start + idx * row_step
        draw.text((s_box[0] + 18, ry + 4), fname[:24], fill=TEXT_SEC, font=font_sm)
        bw = int(abs(weight) * max_span)
        if weight >= 0:
            draw.rectangle([zero_x + 2, ry, zero_x + 2 + bw, ry + 20], fill=clr)
        else:
            draw.rectangle([zero_x - 2 - bw, ry, zero_x - 2, ry + 20], fill=clr)

    return img.convert("RGB")


def main() -> None:
    static_dir = Path(__file__).resolve().parent.parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    assets: list[tuple[str, Image.Image]] = [
        ("icon-192.png", draw_icon(192, maskable=False)),
        ("icon-512.png", draw_icon(512, maskable=False)),
        ("icon-maskable-192.png", draw_icon(192, maskable=True)),
        ("icon-maskable-512.png", draw_icon(512, maskable=True)),
        ("apple-touch-icon.png", draw_icon(180, maskable=True)),
        ("favicon-32.png", draw_icon(32, maskable=False)),
        ("screenshot-wide.png", draw_screenshot(1280, 720, mobile=False)),
        ("screenshot-mobile.png", draw_screenshot(720, 1280, mobile=True)),
    ]

    for filename, image in assets:
        out = static_dir / filename
        image.save(out, format="PNG", optimize=True)
        print(f"Generated {out.relative_to(static_dir.parent)} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
