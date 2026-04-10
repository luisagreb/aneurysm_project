"""
Pipeline Figure: 3D Computational Phenotyping of TAA Cells
Output: thesis_figures/00_pipeline/pipeline_overview.{pdf,png}

Usage:
    python src/thesis_figures/plot_pipeline.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE / "thesis_figures" / "00_pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   9,
    "figure.dpi":  150,
})

C_BLUE   = "#AED6F1"   # data / preprocessing
C_ORANGE = "#FAD7A0"   # nnU-Net segmentation
C_PURPLE = "#D7BDE2"   # feature extraction
C_GREEN  = "#A9DFBF"   # integration / labels
C_PINK   = "#F1948A"   # classification / evaluation
C_GRAY   = "#D5D8DC"   # metadata
C_DARK   = "#2C3E50"

# ── Helpers ───────────────────────────────────────────────────────────────────

def box(ax, cx, cy, w, h, title, color, lines=(), fontsize=9):
    """Rounded rectangle with bold title and optional body lines."""
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.07",
        facecolor=color, edgecolor="#555", linewidth=0.9, zorder=3,
    ))
    if lines:
        title_y = cy + h / 2 - 0.28
        ax.text(cx, title_y, title, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=C_DARK, zorder=4)
        ax.plot([cx - w/2 + 0.12, cx + w/2 - 0.12],
                [cy + h/2 - 0.50, cy + h/2 - 0.50],
                color="#999", lw=0.7, zorder=4)
        ax.text(cx, cy - 0.12, "\n".join(lines),
                ha="center", va="center",
                fontsize=fontsize - 1.5, color="#333",
                linespacing=1.55, zorder=4)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=C_DARK, zorder=4)


def arr(ax, x1, y1, x2, y2, color="#555"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.1), zorder=2)


def section_label(ax, x, y, txt):
    ax.text(x, y, txt, fontsize=8, fontweight="bold",
            color="#5D6D7E", va="center")


# ── Figure ────────────────────────────────────────────────────────────────────
W, H = 22, 14
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("white")

ax.text(W / 2, H - 0.35,
        "3D Computational Phenotyping Pipeline — TAA Cell Morphology",
        ha="center", va="center", fontsize=14, fontweight="bold", color=C_DARK)

# ═════════════════════════════════════════════════════════════════════════════
# ROW 1 — Data acquisition & segmentation  (y = 11.0)
# ═════════════════════════════════════════════════════════════════════════════
R1, BH1 = 11.0, 1.35

section_label(ax, 0.45, 12.1, "① DATA ACQUISITION & SEGMENTATION")

box(ax,  2.2, R1, 2.9, BH1, "Raw Input (.oir)", C_BLUE,
    lines=["Olympus confocal z-stacks",
           "Ch00 Nucleus · Ch01 Actin",
           "Ch02 Mitochondria"])

box(ax,  5.5, R1, 2.8, BH1, "OIR → NRRD", C_BLUE,
    lines=["Per-channel NRRD files",
           "Voxel spacing preserved",
           "(AICSImage library)"])

box(ax,  8.8, R1, 3.0, BH1, "NRRD → NIfTI", C_BLUE,
    lines=["Spacing / orientation fix",
           "imagesTr / labelsTr / imagesTs",
           "nnU-Net folder structure"])

box(ax, 12.6, R1, 3.4, BH1, "nnU-Net v2\n(3 independent models)", C_ORANGE,
    lines=["Dataset001 — Actin",
           "Dataset002 — Mitochondria",
           "Dataset003 — Nucleus"])

box(ax, 16.9, R1, 3.2, BH1, "Binary Segmentation\nMasks (.nii.gz)", C_ORANGE,
    lines=["200 cells total",
           "Inference_Raw +",
           "Training_Labels"])

# row-1 arrows
for x1, x2 in [(3.65, 4.1), (6.9, 7.3), (10.3, 10.9), (14.3, 15.3)]:
    arr(ax, x1, R1, x2, R1)

# ═════════════════════════════════════════════════════════════════════════════
# ROW 2 — Feature extraction  (y = 7.6)
# ═════════════════════════════════════════════════════════════════════════════
R2, FH = 7.6, 2.7

section_label(ax, 0.45, 9.6, "② FEATURE EXTRACTION  (voxel-size corrected, µm)")

box(ax, 3.9, R2, 4.4, FH, "Nucleus Features", C_PURPLE,
    lines=["Volume · Surface Area",
           "Sphericity · Circularity",
           "Elongation · Flatness",
           "Principal axes",
           "(inertia tensor eigenvectors)"])

box(ax, 9.6, R2, 4.4, FH, "Actin Features", C_PURPLE,
    lines=["Volume · Skeleton Length",
           "Convex Hull Volume",
           "Solidity · Extent",
           "Fractional Anisotropy",
           "Major / Intermediate / Minor Axes"])

box(ax, 16.0, R2, 5.2, FH, "Mitochondria Features", C_PURPLE,
    lines=["Volume · Surface Area · Sphericity",
           "Fragment Count · Mean Fragment Volume",
           "Fragment Sphericity (mean/std/min/max)",
           "Junction Count · Branch Count",
           "Mean Branch Length · Tortuosity",
           "Cyclomatic Number   (skan skeleton)"])

# Arrows: seg masks → 3 feature boxes (branch from masks bottom)
seg_cx, seg_bot = 16.9, R1 - BH1 / 2
for feat_cx, feat_top in [(3.9, R2 + FH/2), (9.6, R2 + FH/2), (16.0, R2 + FH/2)]:
    arr(ax, seg_cx, seg_bot, feat_cx, feat_top, color="#888")

# ── Wide CSV  (y = 4.9) ───────────────────────────────────────────────────────
CSV_Y = 4.9
box(ax, 9.8, CSV_Y, 6.4, 1.05,
    "Advanced_Features_Raw_200.csv", C_GREEN,
    lines=["200 cells × 30+ features   |   Actin_*  ·  Mito_*  ·  Nucleus_*   (wide format — one row per cell)"])

arr(ax,  3.9, R2 - FH/2,  7.2, CSV_Y + 0.35)
arr(ax,  9.6, R2 - FH/2,  9.8, CSV_Y + 0.52)
arr(ax, 16.0, R2 - FH/2, 12.4, CSV_Y + 0.35)

# ═════════════════════════════════════════════════════════════════════════════
# ROW 3 — Labels · Classification · Analysis  (y = 2.4)
# ═════════════════════════════════════════════════════════════════════════════
R3, BH3 = 2.4, 1.9

section_label(ax, 0.45, 4.2, "③ LABEL ASSIGNMENT · CLASSIFICATION · ANALYSIS")

box(ax,  1.8, R3, 2.6, BH3, "Metadata", C_GRAY,
    lines=["Book1.xlsx",
           "Patient IDs",
           "Group labels"])

box(ax,  5.1, R3, 3.2, BH3, "Label Assignment", C_GREEN,
    lines=["Healthy (Non-TAA)",
           "TAA — No Collagen",
           "TAA + Collagen"])

box(ax,  9.2, R3, 3.6, BH3, "3 Binary Tasks", C_GREEN,
    lines=["① Healthy  vs  TAA",
           "② TAA  vs  TAA+Collagen",
           "③ Rescue classification"])

box(ax, 13.7, R3, 3.4, BH3, "4 ML Models", C_PINK,
    lines=["Random Forest",
           "XGBoost",
           "MLP Neural Network",
           "Logistic Regression"])

box(ax, 18.2, R3, 3.6, BH3, "Evaluation &\nStatistical Analysis", C_PINK,
    lines=["5-fold Stratified CV",
           "Accuracy · AUC · F1",
           "LMM · Pairwise Tests",
           "PCA · Feature Importance"])

# row-3 arrows
arr(ax,  3.1, R3,  3.5, R3)   # metadata → label assignment
arr(ax,  6.7, R3,  7.4, R3)   # label assignment → tasks
arr(ax, 11.0, R3, 11.9, R3)   # tasks → models
arr(ax, 15.4, R3, 16.4, R3)   # models → evaluation

# CSV → label assignment (diagonal)
arr(ax,  7.8, CSV_Y - 0.52,  5.1, R3 + BH3/2)

# ── Legend ────────────────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(fc=C_BLUE,   ec="#555", label="Data Preprocessing"),
    mpatches.Patch(fc=C_ORANGE, ec="#555", label="nnU-Net Segmentation"),
    mpatches.Patch(fc=C_PURPLE, ec="#555", label="Feature Extraction"),
    mpatches.Patch(fc=C_GREEN,  ec="#555", label="Integration & Labels"),
    mpatches.Patch(fc=C_PINK,   ec="#555", label="Classification & Evaluation"),
    mpatches.Patch(fc=C_GRAY,   ec="#555", label="Metadata"),
]
ax.legend(handles=patches, loc="lower left", bbox_to_anchor=(0.01, 0.01),
          fontsize=8, ncol=3, frameon=True, framealpha=0.92,
          edgecolor="#ccc")

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    out = OUT_DIR / f"pipeline_overview.{ext}"
    plt.savefig(out, bbox_inches="tight", dpi=200 if ext == "png" else None)
    print(f"Saved → {out}")

plt.show()
