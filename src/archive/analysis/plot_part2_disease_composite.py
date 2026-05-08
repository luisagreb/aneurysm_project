"""
Composite disease comparison figures — Part 2 (TAV-ATAA vs BAV-ATAA).
Stitches existing per_feature boxplot PNGs into two grids:
  1. part2_disease_significant.png   — L1-significant features only
  2. part2_disease_supplementary.png — remaining (non-significant) features
Both grouped by organelle (Actin / Mitochondria / Nucleus).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
SRC  = PROJ / 'outputs/final_thesis_part2/comparison/boxplots/per_feature'
OUT  = PROJ / 'outputs/figures'
OUT.mkdir(parents=True, exist_ok=True)

ORGANELLE_COLORS = {
    'Actin':        '#2ECC71',
    'Mitochondria': '#E74C3C',
    'Nucleus':      '#1558A0',
}

# ── L1 patient-level significant features (Part 2) ───────────────────────────
SIG_FEATURES = {
    'Actin_Extent_ratio', 'Actin_Skeleton_Length_µm', 'Actin_Solidity_ratio',
    'Mito_Branch_Count_n', 'Mito_Cyclomatic_Number_n', 'Mito_Junction_Count_n',
    'Mito_Max_Fragment_Sphericity_ratio', 'Mito_Mean_Branch_Length_µm',
    'Mito_Mean_Tortuosity_ratio', 'Mito_Std_Fragment_Sphericity_ratio',
    'Mito_Total_Network_Length_µm',
    'Nucleus_Elongation_ratio', 'Nucleus_Sphericity_ratio', 'Nucleus_Volume_µm³',
}

GROUPS = {
    'Actin': [
        'Actin_Convex_Hull_Volume_µm³',
        'Actin_Extent_ratio',
        'Actin_Fractional_Anisotropy_ratio',
        'Actin_Intermediate_Axis_µm',
        'Actin_Major_Axis_µm',
        'Actin_Minor_Axis_µm',
        'Actin_Skeleton_Length_µm',
        'Actin_Solidity_ratio',
        'Actin_Volume_µm³',
    ],
    'Mitochondria': [
        'Mito_Branch_Count_n',
        'Mito_Cyclomatic_Number_n',
        'Mito_Fragment_Count_n',
        'Mito_Junction_Count_n',
        'Mito_Max_Fragment_Sphericity_ratio',
        'Mito_Mean_Branch_Length_µm',
        'Mito_Mean_Fragment_Sphericity_ratio',
        'Mito_Mean_Fragment_Volume_µm³',
        'Mito_Mean_Tortuosity_ratio',
        'Mito_Min_Fragment_Sphericity_ratio',
        'Mito_Sphericity_ratio',
        'Mito_Std_Fragment_Sphericity_ratio',
        'Mito_Surface_Area_µm²',
        'Mito_Total_Network_Length_µm',
        'Mito_Volume_µm³',
    ],
    'Nucleus': [
        'Nucleus_Circularity_ratio',
        'Nucleus_Elongation_ratio',
        'Nucleus_Flatness_ratio',
        'Nucleus_Solidity_ratio',
        'Nucleus_Sphericity_ratio',
        'Nucleus_Volume_µm³',
    ],
}

NCOLS = 3

plt.rcParams.update({
    'font.family':     'Arial',
    'font.sans-serif': ['Arial'],
    'text.usetex':     False,
})

def build_entries(sig_only: bool):
    entries = []
    for org, feats in GROUPS.items():
        for feat in feats:
            if (feat in SIG_FEATURES) != sig_only:
                continue
            p = SRC / f'L1_{feat}.png'
            if p.exists():
                entries.append((org, p))
    return entries

def make_composite(entries, out_path):
    if not entries:
        print(f'No entries for {out_path.name}, skipping.')
        return

    n_rows  = (len(entries) + NCOLS - 1) // NCOLS
    n_cells = n_rows * NCOLS
    imgs    = [np.array(Image.open(p).convert('RGB')) for _, p in entries]

    cell_w, cell_h = 3.2, 2.6
    fig, axes = plt.subplots(n_rows, NCOLS,
                             figsize=(NCOLS * cell_w, n_rows * cell_h + 0.7),
                             constrained_layout=False)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01,
                        hspace=0.08, wspace=0.04)

    for idx in range(n_cells):
        row, col = divmod(idx, NCOLS)
        ax = axes[row, col]
        if idx < len(imgs):
            org, _ = entries[idx]
            ax.imshow(imgs[idx], aspect='auto')
            for spine in ax.spines.values():
                spine.set_edgecolor(ORGANELLE_COLORS[org])
                spine.set_linewidth(2.5)
        else:
            ax.set_visible(False)
            continue
        ax.set_xticks([])
        ax.set_yticks([])

    org_seen = set()
    for idx, (org, _) in enumerate(entries):
        if org not in org_seen:
            org_seen.add(org)
            row, _ = divmod(idx, NCOLS)
            axes[row, 0].set_ylabel(org, fontsize=11, fontweight='bold',
                                    color=ORGANELLE_COLORS[org],
                                    rotation=90, labelpad=6)

    patches = [mpatches.Patch(facecolor=ORGANELLE_COLORS[o], label=o, alpha=0.85)
               for o in ['Actin', 'Mitochondria', 'Nucleus']
               if any(e[0] == o for e in entries)]
    fig.legend(handles=patches, loc='upper center', ncol=3,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.995))

    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    print(f'Saved → {out_path}')

make_composite(build_entries(sig_only=True),
               OUT / 'part2_disease_significant.png')
make_composite(build_entries(sig_only=False),
               OUT / 'part2_disease_supplementary.png')
