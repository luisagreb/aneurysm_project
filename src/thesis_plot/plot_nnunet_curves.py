"""
Generate publication-quality learning curve plots from nnU-Net training logs.
Outputs saved to aneurysm_project/thesis_figures/01_nnunet_training/
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent.parent
LOGS = {
    "Actin": BASE / "data/nnUNet/nnUNet_results/Dataset001_Actin/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/training_log_2026_1_1_16_38_52.txt",
    "Mitochondria": BASE / "data/nnUNet/nnUNet_results/Dataset002_Mito/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/training_log_2026_1_3_13_16_24.txt",
    "Nucleus": BASE / "data/nnUNet/nnUNet_results/Dataset003_Nucleus/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/training_log_2026_1_5_09_46_25.txt",
}
OUT_DIR = BASE / "thesis_figures/01_nnunet_training"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "legend.frameon": False,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})

COLORS = {
    "train": "#2196F3",   # blue
    "val":   "#F44336",   # red
    "dice":  "#4CAF50",   # green
    "lr":    "#9C27B0",   # purple
}


# ── Parser ────────────────────────────────────────────────────────────────────
def parse_log(path: Path) -> dict:
    train_loss, val_loss, dice, lr = [], [], [], []
    with open(path) as f:
        for line in f:
            if m := re.search(r"train_loss\s+([-\d.]+)", line):
                train_loss.append(float(m.group(1)))
            elif m := re.search(r"val_loss\s+([-\d.]+)", line):
                val_loss.append(float(m.group(1)))
            elif m := re.search(r"Pseudo dice \[np\.float32\(([\d.]+)\)", line):
                dice.append(float(m.group(1)))
            elif m := re.search(r"Current learning rate:\s+([\d.e+-]+)", line):
                lr.append(float(m.group(1)))
    n = min(len(train_loss), len(val_loss), len(dice))
    return {
        "train_loss": np.array(train_loss[:n]),
        "val_loss":   np.array(val_loss[:n]),
        "dice":       np.array(dice[:n]),
        "lr":         np.array(lr[:n]),
        "epochs":     np.arange(n),
    }


def smooth(x, w=20):
    """Simple moving average."""
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")


# ── Plot per dataset ──────────────────────────────────────────────────────────
for name, log_path in LOGS.items():
    data = parse_log(log_path)
    epochs = data["epochs"]
    n = len(epochs)
    w = 20  # smoothing window

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                             gridspec_kw={"hspace": 0.08})

    # ── Top: Loss ──────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, data["train_loss"], color=COLORS["train"], alpha=0.18, linewidth=0.7)
    ax.plot(epochs, data["val_loss"],   color=COLORS["val"],   alpha=0.18, linewidth=0.7)
    ax.plot(epochs[w-1:], smooth(data["train_loss"], w),
            color=COLORS["train"], linewidth=1.8, label="Train loss")
    ax.plot(epochs[w-1:], smooth(data["val_loss"], w),
            color=COLORS["val"],   linewidth=1.8, label="Val loss")
    ax.set_ylabel("Loss (negative Dice)")
    ax.legend(loc="lower right")
    ax.set_title(f"{name}  —  nnU-Net 3D fullres training (fold 0)")

    # ── Bottom: Pseudo Dice ────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, data["dice"], color=COLORS["dice"], alpha=0.18, linewidth=0.7)
    ax.plot(epochs[w-1:], smooth(data["dice"], w),
            color=COLORS["dice"], linewidth=1.8, label="Pseudo Dice (val)")
    ax.set_ylabel("Pseudo Dice")
    ax.set_xlabel("Epoch")
    ax.set_xlim(0, n - 1)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="lower right")

    out = OUT_DIR / f"{name.lower()}_learning_curve.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {out.name}")


# ── Combined overview (all 3 on one figure) ───────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=False,
                         gridspec_kw={"hspace": 0.35, "wspace": 0.3})

for col, (name, log_path) in enumerate(LOGS.items()):
    data = parse_log(log_path)
    epochs = data["epochs"]
    n = len(epochs)
    w = 20

    # Loss
    ax = axes[0, col]
    ax.plot(epochs, data["train_loss"], color=COLORS["train"], alpha=0.15, linewidth=0.6)
    ax.plot(epochs, data["val_loss"],   color=COLORS["val"],   alpha=0.15, linewidth=0.6)
    ax.plot(epochs[w-1:], smooth(data["train_loss"], w), color=COLORS["train"],
            linewidth=1.8, label="Train")
    ax.plot(epochs[w-1:], smooth(data["val_loss"], w),   color=COLORS["val"],
            linewidth=1.8, label="Val")
    ax.set_title(name, fontweight="bold")
    ax.set_xlim(0, n - 1)
    if col == 0:
        ax.set_ylabel("Loss")
    ax.legend(fontsize=9)

    # Dice
    ax = axes[1, col]
    ax.plot(epochs, data["dice"], color=COLORS["dice"], alpha=0.15, linewidth=0.6)
    ax.plot(epochs[w-1:], smooth(data["dice"], w), color=COLORS["dice"], linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_xlim(0, n - 1)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    if col == 0:
        ax.set_ylabel("Pseudo Dice")

fig.suptitle("nnU-Net 3D fullres — Training curves", fontsize=14, fontweight="bold", y=1.01)
out = OUT_DIR / "all_learning_curves.png"
fig.savefig(out, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"Saved {out.name}")
