"""
Phase 1 – Dataset Preparation & Preprocessing
نسخة مطوّرة جمالياً: صناديق مدورة، ظلال ناعمة، لوحة ألوان أكاديمية هادئة
(الأرقام مطابقة تماماً للجدول 4.2 في الأطروحة)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.lines as mlines
import numpy as np

# ─────────────────────────────────────────────────────────────────
# GLOBAL STYLE — refined academic palette
# ─────────────────────────────────────────────────────────────────
NAVY        = "#1B2A4A"   # deep navy - titles, primary text
NAVY_SOFT   = "#2E4468"   # secondary navy - subtitles/lines
TEAL        = "#1B8A8A"   # accent - acquisition
TEAL_LIGHT  = "#E6F4F4"
AMBER       = "#C97B2E"   # accent - normalisation
AMBER_LIGHT = "#FBF0E4"
PLUM        = "#7A4B8C"   # accent - augmentation
PLUM_LIGHT  = "#F3ECF6"
SLATE       = "#5B6B79"   # neutral accent - validation/test boxes
BG          = "#FDFDFC"   # page background (warm off-white)
CARD_WHITE  = "#FFFFFF"
GRID_LINE   = "#D8DCE1"
SHADOW      = "#C9CDD3"

DPI          = 220
FONT_FAMILY  = "DejaVu Sans"

plt.rcParams.update({
    "font.family":       FONT_FAMILY,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.spines.bottom":False,
})

# ─────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────

def arrow(ax, x1, y1, x2, y2, lw=2.2, head_w=0.28, head_l=0.2,
          color=NAVY_SOFT, style="->", connectionstyle=None):
    kw = dict(
        arrowstyle=f"{style}, head_width={head_w}, head_length={head_l}",
        color=color, lw=lw, shrinkA=0, shrinkB=0
    )
    if connectionstyle:
        kw["connectionstyle"] = connectionstyle
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=kw, zorder=5)


def box(ax, cx, cy, w, h,
        title, subtitle=None,
        fill=CARD_WHITE, edge=NAVY, edge_lw=1.8,
        title_size=15, title_color=NAVY, title_weight="bold",
        sub_size=12.5, sub_color=NAVY_SOFT,
        rounding=0.06, shadow=True, accent_bar=None):
    """Rounded card with optional drop-shadow and a left accent bar."""

    if shadow:
        sh = FancyBboxPatch(
            (cx - w/2 + 0.05, cy - h/2 - 0.06), w, h,
            boxstyle=f"round,pad=0.0,rounding_size={rounding}",
            facecolor=SHADOW, edgecolor="none",
            zorder=1.5, alpha=0.55
        )
        ax.add_patch(sh)

    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={rounding}",
        facecolor=fill, edgecolor=edge,
        linewidth=edge_lw, zorder=3
    )
    ax.add_patch(rect)

    if accent_bar:
        bar = FancyBboxPatch(
            (cx - w/2, cy - h/2), 0.12, h,
            boxstyle=f"round,pad=0.0,rounding_size=0.05",
            facecolor=accent_bar, edgecolor="none", zorder=3.2
        )
        ax.add_patch(bar)

    if subtitle:
        ax.text(cx, cy + h * 0.15, title,
                ha="center", va="center",
                fontsize=title_size, fontweight=title_weight,
                color=title_color, zorder=4)
        ax.text(cx, cy - h * 0.24, subtitle,
                ha="center", va="center",
                fontsize=sub_size,
                color=sub_color, zorder=4)
    else:
        ax.text(cx, cy, title,
                ha="center", va="center",
                fontsize=title_size, fontweight=title_weight,
                color=title_color, zorder=4)


def section_title(ax, x, y, text, fs=20, color=NAVY):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=color,
            fontfamily=FONT_FAMILY)


def kicker(ax, x, y, text, color=TEAL, fs=12.5):
    """Small uppercase eyebrow label above a section title."""
    ax.text(x, y, text.upper(), ha="center", va="center",
            fontsize=fs, fontweight="bold", color=color,
            fontfamily=FONT_FAMILY)


def divider(ax, x0, x1, y, color=GRID_LINE, lw=1.4, dashed=False):
    ls = (0, (7, 5)) if dashed else "solid"
    ax.plot([x0, x1], [y, y], color=color, lw=lw, linestyle=ls, zorder=1)


def badge(ax, cx, cy, r, number, color):
    """Small circular numbered badge (stage marker)."""
    circ = plt.Circle((cx, cy), r, facecolor=color, edgecolor="none", zorder=6)
    ax.add_patch(circ)
    ax.text(cx, cy, str(number), ha="center", va="center",
            fontsize=13, fontweight="bold", color="white", zorder=7)


# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — Sequential Pipeline
# ══════════════════════════════════════════════════════════════════

def fig_pipeline():
    W, H = 16, 5.6
    fig, ax = plt.subplots(figsize=(W, H))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis("off")

    kicker(ax, W/2, 5.05, "Phase 1", color=TEAL)
    section_title(ax, W/2, 4.6, "Dataset Preparation Pipeline", fs=19)
    divider(ax, 3.0, W-3.0, 4.2)

    stages = [
        (2.8,  2.4, 4.0, 1.9, "Data Acquisition", "12,064 MRI images", TEAL, TEAL_LIGHT, 1),
        (8.0,  2.4, 4.0, 1.9, "Normalisation",     "128 × 128 px",      AMBER, AMBER_LIGHT, 2),
        (13.2, 2.4, 4.0, 1.9, "Augmentation",      "Training set only", PLUM, PLUM_LIGHT, 3),
    ]

    for (cx, cy, w, h, ttl, sub, accent, fill, n) in stages:
        box(ax, cx, cy, w, h, ttl, sub,
            fill=fill, edge=accent, edge_lw=1.6,
            title_size=15.5, sub_size=13, title_color=NAVY,
            accent_bar=accent)
        badge(ax, cx - w/2 + 0.02, cy + h/2 + 0.02, 0.26, n, accent)

    arrow(ax, 4.9, 2.4, 5.9, 2.4, lw=2.6, head_w=0.3, head_l=0.22, color=NAVY_SOFT)
    arrow(ax, 10.1, 2.4, 11.1, 2.4, lw=2.6, head_w=0.3, head_l=0.22, color=NAVY_SOFT)

    plt.tight_layout(pad=0.6)
    plt.savefig("fig_1_pipeline_simple.png",
                dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("✓  Figure 1: Pipeline saved.")


# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — Data Acquisition
#   Class distribution matches Table 4.2:
#   Glioma 3,773 (31.28%) | Meningioma 2,729 (22.62%)
#   Pituitary 3,130 (25.95%) | No Tumor 2,432 (20.16%)
# ══════════════════════════════════════════════════════════════════

def fig_integrity():
    W, H = 16, 9.5
    fig, ax = plt.subplots(figsize=(W, H))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis("off")

    kicker(ax, W/2, 8.95, "Phase 1 · Stage 1", color=TEAL)
    section_title(ax, W/2, 8.5, "Data Acquisition", fs=19)
    divider(ax, 3.0, W-3.0, 8.15)

    # ── SOURCE box ─────────────────────────────────────────────
    box(ax, 8.0, 7.2, 8.4, 1.3,
        "Brain Tumor MRI Dataset",
        "12,064 T1-weighted images",
        fill=NAVY, edge=NAVY, edge_lw=0,
        title_size=16.5, sub_size=13.5,
        title_color="white", sub_color="#CBD5E1")

    class_xs = [2.8, 5.7, 10.3, 13.2]
    for cx in class_xs:
        arrow(ax, 8.0, 6.55, cx, 5.85, lw=1.8,
              head_w=0.2, head_l=0.15, color=GRID_LINE,
              connectionstyle="arc3,rad=0.0")

    # Distinct accent colour per class for visual clarity
    classes = [
        (2.8,  5.1, 3.05, 1.3, "Glioma",      "3,773 · 31.28%", "#2E86AB", "#EAF3F8"),
        (5.7,  5.1, 3.05, 1.3, "Meningioma",  "2,729 · 22.62%", "#1B8A8A", "#E6F4F4"),
        (10.3, 5.1, 3.05, 1.3, "Pituitary",   "3,130 · 25.95%", "#C97B2E", "#FBF0E4"),
        (13.2, 5.1, 3.05, 1.3, "No Tumor",    "2,432 · 20.16%", "#5B6B79", "#EEF1F3"),
    ]
    for (cx, cy, w, h, ttl, sub, accent, fill) in classes:
        box(ax, cx, cy, w, h, ttl, sub,
            fill=fill, edge=accent, edge_lw=1.5,
            title_size=14.5, sub_size=12, accent_bar=accent)

    ax.text(8.0, 4.15,
            "Total  ·  12,064 images", ha="center", va="center",
            fontsize=13.5, fontweight="bold", color=NAVY_SOFT)

    divider(ax, 2.5, W-2.5, 3.75, dashed=True)

    # ── Integrity panel ────────────────────────────────────────
    box(ax, 8.0, 2.9, 12.4, 1.15,
        "Integrity Verification",
        "MD5 cryptographic hash checking",
        fill=AMBER_LIGHT, edge=AMBER, edge_lw=1.6,
        title_size=15, sub_size=13, accent_bar=AMBER)

    arrow(ax, 4.5, 2.32, 4.5, 1.85, lw=1.9, head_w=0.18, head_l=0.14, color=GRID_LINE)
    arrow(ax, 8.0, 2.32, 8.0, 1.85, lw=1.9, head_w=0.18, head_l=0.14, color=GRID_LINE)
    arrow(ax, 11.5, 2.32, 11.5, 1.85, lw=1.9, head_w=0.18, head_l=0.14, color=GRID_LINE)

    results = [
        (4.5,  1.15, 3.5, 1.15, "0% Duplication", "No duplicates found"),
        (8.0,  1.15, 3.5, 1.15, "100% Valid",     "All pixels verified"),
        (11.5, 1.15, 3.5, 1.15, "9,650 Images",   "Train + validation pool"),
    ]
    for (cx, cy, w, h, ttl, sub) in results:
        box(ax, cx, cy, w, h, ttl, sub,
            fill=TEAL_LIGHT, edge=TEAL, edge_lw=1.5,
            title_size=14, sub_size=12, title_color=NAVY, accent_bar=TEAL)

    plt.tight_layout(pad=0.6)
    plt.savefig("fig_2_acquisition_simple.png",
                dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("✓  Figure 2: Data Acquisition saved.")


# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — Normalisation
# ══════════════════════════════════════════════════════════════════

def fig_normalisation():
    W, H = 16, 10.4
    fig, ax = plt.subplots(figsize=(W, H))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis("off")

    kicker(ax, W/2, 9.85, "Phase 1 · Stage 2", color=AMBER)
    section_title(ax, W/2, 9.4, "Spatial and Intensity Normalisation", fs=19)
    divider(ax, 3.0, W-3.0, 9.05)

    # ══ PART A — Spatial ═══════════════════════════════════════
    badge(ax, 6.65, 8.3, 0.22, "A", TEAL)
    ax.text(7.0, 8.3, "Spatial Normalisation", ha="left", va="center",
            fontsize=15.5, fontweight="bold", color=NAVY)

    box(ax, 4.5, 7.15, 2.7, 1.15, "Variable Size", None,
        fill=CARD_WHITE, edge=GRID_LINE, edge_lw=1.4,
        title_size=13.5, title_color=NAVY_SOFT, title_weight="normal")

    arrow(ax, 5.85, 7.15, 6.825, 7.15, lw=2.2, head_w=0.24, head_l=0.18, color=TEAL)

    box(ax, 8.0, 7.15, 2.35, 1.15, "Resize", None,
        fill=TEAL_LIGHT, edge=TEAL, edge_lw=1.5,
        title_size=14)

    arrow(ax, 9.175, 7.15, 10.15, 7.15, lw=2.2, head_w=0.24, head_l=0.18, color=TEAL)

    box(ax, 11.5, 7.15, 2.7, 1.15, "128 × 128 px", None,
        fill=NAVY, edge=NAVY, edge_lw=0,
        title_size=15, title_color="white")

    divider(ax, 1.5, W-1.5, 6.15, dashed=True)

    # ══ PART B — Intensity ═════════════════════════════════════
    badge(ax, 6.85, 5.55, 0.22, "B", AMBER)
    ax.text(7.2, 5.55, "Intensity Normalisation", ha="left", va="center",
            fontsize=15.5, fontweight="bold", color=NAVY)

    box(ax, 8.0, 4.65, 5.2, 1.05,
        r"$x_{\mathrm{norm}} = \dfrac{x - \mu}{\sigma}$", None,
        fill=AMBER_LIGHT, edge=AMBER, edge_lw=1.6,
        title_size=19, accent_bar=AMBER)

    arrow(ax, 8.0, 4.1, 8.0, 3.6, lw=2.2, head_w=0.24, head_l=0.18, color=AMBER)

    channels = [
        (3.8,  2.65, 3.9, 1.35, "Red Channel",   "μ = 0.485\nσ = 0.229", "#C0392B", "#FBEAE9"),
        (8.0,  2.65, 3.9, 1.35, "Green Channel", "μ = 0.456\nσ = 0.224", "#27824A", "#E9F5EC"),
        (12.2, 2.65, 3.9, 1.35, "Blue Channel",  "μ = 0.406\nσ = 0.225", "#2E6DA4", "#E9F0F8"),
    ]
    for (cx, cy, w, h, ttl, sub, accent, fill) in channels:
        box(ax, cx, cy, w, h, ttl, sub,
            fill=fill, edge=accent, edge_lw=1.5,
            title_size=14, sub_size=12.5, accent_bar=accent)
        arrow(ax, 8.0, 3.5, cx, cy + h/2 + 0.05, lw=1.6,
              head_w=0.16, head_l=0.13, color=GRID_LINE)

    box(ax, 8.0, 1.05, 12.4, 0.85,
        "Zero-mean, unit-variance normalisation across all channels", None,
        fill=NAVY, edge=NAVY, edge_lw=0,
        title_size=13.5, title_color="white")

    plt.tight_layout(pad=0.6)
    plt.savefig("fig_3_normalisation_simple.png",
                dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("✓  Figure 3: Normalisation saved.")


# ══════════════════════════════════════════════════════════════════
# FIGURE 4 — Data Augmentation
# ══════════════════════════════════════════════════════════════════

def fig_augmentation():
    W, H = 16, 10.4
    fig, ax = plt.subplots(figsize=(W, H))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis("off")

    kicker(ax, W/2, 9.85, "Phase 1 · Stage 3", color=PLUM)
    section_title(ax, W/2, 9.4, "Data Augmentation Strategy", fs=19)
    divider(ax, 3.0, W-3.0, 9.05)

    ax.text(8.0, 8.55, "Dataset Partitions",
            ha="center", fontsize=15, fontweight="bold", color=NAVY)

    partitions = [
        (3.8,  7.55, 4.2, 1.25, "Training Set", "9,650 images · Augmented",
         PLUM, PLUM_LIGHT),
        (8.4,  7.55, 3.4, 1.25, "Validation", "No change", SLATE, "#EEF1F3"),
        (12.5, 7.55, 3.4, 1.25, "Test Set", "No change", SLATE, "#EEF1F3"),
    ]
    for (cx, cy, w, h, ttl, sub, accent, fill) in partitions:
        box(ax, cx, cy, w, h, ttl, sub,
            fill=fill, edge=accent, edge_lw=1.5,
            title_size=14.5, sub_size=12.5, accent_bar=accent)

    arrow(ax, 3.8, 6.9, 3.8, 6.35, lw=2.2, head_w=0.24, head_l=0.18, color=PLUM)

    divider(ax, 1.5, W-1.5, 6.05, dashed=True)

    ax.text(8.0, 5.6, "Augmentation Operations",
            ha="center", fontsize=15, fontweight="bold", color=NAVY)

    ops = [
        (3.2,  4.4, 4.3, 1.35, "Horizontal Flip",   "p = 0.40", "#2E86AB", "#EAF3F8"),
        (8.0,  4.4, 4.3, 1.35, "Vertical Flip",     "p = 0.30", "#1B8A8A", "#E6F4F4"),
        (12.8, 4.4, 4.3, 1.35, "Rotation",          "± 20°",    "#C97B2E", "#FBF0E4"),
        (3.2,  2.7, 4.3, 1.35, "Color Jitter",      "Brightness / Contrast", "#7A4B8C", "#F3ECF6"),
        (8.0,  2.7, 4.3, 1.35, "Gaussian Blur",     "3 × 3 kernel", "#C0392B", "#FBEAE9"),
        (12.8, 2.7, 4.3, 1.35, "Elastic Transform", "p = 0.20", "#27824A", "#E9F5EC"),
    ]
    for (cx, cy, w, h, ttl, sub, accent, fill) in ops:
        box(ax, cx, cy, w, h, ttl, sub,
            fill=fill, edge=accent, edge_lw=1.4,
            title_size=13.5, sub_size=12, accent_bar=accent)

    box(ax, 8.0, 1.15, 9.0, 0.8,
        "Applied to the training set only", None,
        fill=NAVY, edge=NAVY, edge_lw=0,
        title_size=13, title_color="white")

    plt.tight_layout(pad=0.6)
    plt.savefig("fig_4_augmentation_simple.png",
                dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("✓  Figure 4: Augmentation saved.")


# ══════════════════════════════════════════════════════════════════
# FIGURE 5 — Full Phase 1 Overview
# ══════════════════════════════════════════════════════════════════

def fig_phase1_overview():
    W, H = 18, 15.6
    fig, ax = plt.subplots(figsize=(W, H))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis("off")

    kicker(ax, W/2, 15.05, "Phase 1 · Complete Pipeline", color=NAVY_SOFT)
    section_title(ax, W/2, 14.6, "Dataset Preparation Overview", fs=20)
    divider(ax, 2.0, W-2.0, 14.2, lw=1.6)

    # ════════════════════════════════════
    # STAGE 1
    # ════════════════════════════════════
    badge(ax, 1.3, 13.55, 0.24, 1, TEAL)
    ax.text(1.65, 13.55, "Data Acquisition", ha="left", va="center",
            fontsize=15.5, fontweight="bold", color=NAVY)

    box(ax, 9.0, 12.65, 8.2, 1.1,
        "Brain Tumor MRI Dataset", "12,064 images",
        fill=NAVY, edge=NAVY, edge_lw=0,
        title_size=15.5, sub_size=13, title_color="white", sub_color="#CBD5E1")

    cls_x = [5.4, 7.6, 10.4, 12.6]
    cls = [("Glioma", "3,773", "#2E86AB", "#EAF3F8"),
           ("Meningioma", "2,729", "#1B8A8A", "#E6F4F4"),
           ("Pituitary", "3,130", "#C97B2E", "#FBF0E4"),
           ("No Tumor", "2,432", "#5B6B79", "#EEF1F3")]

    for cx, (lbl, n, accent, fill) in zip(cls_x, cls):
        arrow(ax, cx, 12.08, cx, 11.62, lw=1.6, head_w=0.16, head_l=0.13, color=GRID_LINE)
        box(ax, cx, 11.05, 2.05, 0.95, lbl, n,
            fill=fill, edge=accent, edge_lw=1.3,
            title_size=12.5, sub_size=12, accent_bar=accent)

    arrow(ax, 9.0, 10.55, 9.0, 10.1, lw=2.0, head_w=0.2, head_l=0.15, color=TEAL)

    box(ax, 9.0, 9.65, 6.8, 0.85,
        "MD5 Check  ·  0% duplicates  ·  100% valid", None,
        fill=TEAL_LIGHT, edge=TEAL, edge_lw=1.4,
        title_size=13, accent_bar=TEAL)

    divider(ax, 1.0, W-1.0, 9.05, dashed=True)

    # ════════════════════════════════════
    # STAGE 2
    # ════════════════════════════════════
    badge(ax, 1.3, 8.5, 0.24, 2, AMBER)
    ax.text(1.65, 8.5, "Normalisation", ha="left", va="center",
            fontsize=15.5, fontweight="bold", color=NAVY)

    box(ax, 3.6, 7.65, 3.6, 1.05,
        "Spatial", "128 × 128 px",
        fill=AMBER_LIGHT, edge=AMBER, edge_lw=1.5,
        title_size=14, sub_size=12.5, accent_bar=AMBER)

    intensity_channels = [
        (8.6,  7.65, 2.7, 1.05, "Red",   "μ=0.485 σ=0.229", "#C0392B", "#FBEAE9"),
        (11.4, 7.65, 2.7, 1.05, "Green", "μ=0.456 σ=0.224", "#27824A", "#E9F5EC"),
        (14.2, 7.65, 2.7, 1.05, "Blue",  "μ=0.406 σ=0.225", "#2E6DA4", "#E9F0F8"),
    ]
    for (cx, cy, w, h, ttl, sub, accent, fill) in intensity_channels:
        box(ax, cx, cy, w, h, ttl, sub,
            fill=fill, edge=accent, edge_lw=1.3,
            title_size=13, sub_size=11.5, accent_bar=accent)

    divider(ax, 1.0, W-1.0, 6.85, dashed=True)

    # ════════════════════════════════════
    # STAGE 3
    # ════════════════════════════════════
    badge(ax, 1.3, 6.35, 0.24, 3, PLUM)
    ax.text(1.65, 6.35, "Augmentation", ha="left", va="center",
            fontsize=15.5, fontweight="bold", color=NAVY)

    box(ax, 9.0, 5.55, 8.2, 0.9,
        "Training Set (9,650 images) — Augmented", None,
        fill=PLUM_LIGHT, edge=PLUM, edge_lw=1.5,
        title_size=14, accent_bar=PLUM)

    for cx, lbl in [(3.0, "Validation"), (15.0, "Test")]:
        box(ax, cx, 5.55, 3.0, 0.9,
            lbl, "No change",
            fill=CARD_WHITE, edge=GRID_LINE, edge_lw=1.3,
            title_size=13, sub_size=11.5, title_color=NAVY_SOFT)

    ops6 = [
        (3.5,  4.15, 4.05, 1.15, "Horizontal Flip",   "p = 0.40", "#2E86AB", "#EAF3F8"),
        (9.0,  4.15, 4.05, 1.15, "Vertical Flip",     "p = 0.30", "#1B8A8A", "#E6F4F4"),
        (14.5, 4.15, 4.05, 1.15, "Rotation",          "± 20°",    "#C97B2E", "#FBF0E4"),
        (3.5,  2.75, 4.05, 1.15, "Color Jitter",      "Brightness/Contrast", "#7A4B8C", "#F3ECF6"),
        (9.0,  2.75, 4.05, 1.15, "Gaussian Blur",     "3×3 kernel", "#C0392B", "#FBEAE9"),
        (14.5, 2.75, 4.05, 1.15, "Elastic Transform", "p = 0.20", "#27824A", "#E9F5EC"),
    ]
    for (cx, cy, w, h, ttl, sub, accent, fill) in ops6:
        arrow(ax, 9.0, 5.1, cx, cy + h/2 + 0.05, lw=1.3,
              head_w=0.13, head_l=0.11, color=GRID_LINE)
        box(ax, cx, cy, w, h, ttl, sub,
            fill=fill, edge=accent, edge_lw=1.3,
            title_size=12.5, sub_size=11.5, accent_bar=accent)

    divider(ax, 1.0, W-1.0, 1.95, dashed=True)

    # ── Pipeline flow footer ─────────────────────────────────────
    pipe_stages = [
        (4.0,  1.15, 4.2, 0.95, "Stage 1\nAcquisition", TEAL, TEAL_LIGHT),
        (9.0,  1.15, 4.2, 0.95, "Stage 2\nNormalisation", AMBER, AMBER_LIGHT),
        (14.0, 1.15, 4.2, 0.95, "Stage 3\nAugmentation", PLUM, PLUM_LIGHT),
    ]
    for (cx, cy, w, h, ttl, accent, fill) in pipe_stages:
        box(ax, cx, cy, w, h, ttl, None,
            fill=fill, edge=accent, edge_lw=1.6,
            title_size=13, accent_bar=accent)

    arrow(ax, 6.1, 1.15, 6.9, 1.15, lw=2.2, head_w=0.24, head_l=0.18, color=NAVY_SOFT)
    arrow(ax, 11.1, 1.15, 11.9, 1.15, lw=2.2, head_w=0.24, head_l=0.18, color=NAVY_SOFT)

    plt.tight_layout(pad=0.6)
    plt.savefig("fig_5_overview_simple.png",
                dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("✓  Figure 5: Overview saved.")


# ══════════════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    fig_pipeline()
    fig_integrity()
    fig_normalisation()
    fig_augmentation()
    fig_phase1_overview()
    print("\n✅  All 5 redesigned figures generated successfully.")