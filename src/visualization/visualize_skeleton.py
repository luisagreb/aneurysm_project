"""
Visualize mitochondrial skeletonization with branch-type colour coding.

Panels:
  1. XY max-projection of segmentation mask
  2. Skeleton overlaid on mask (all branches blue)
  3. Branch types colour-coded:
       red   = junction-to-junction  (loop-forming, contributes to cyclomatic number)
       green = endpoint branches      (free tips)
       blue  = other / slab pixels

Usage:
  python src/visualization/visualize_skeleton.py --seg_file path/to/Mito_Test_XXXX.nii.gz
  python src/visualization/visualize_skeleton.py --seg_file path/to/seg.nii.gz --output outputs/skel.png
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import morphology
from skan import Skeleton, summarize


BRANCH_COLORS = {
    # string keys (older skan versions)
    'junction-to-junction': (0.91, 0.30, 0.24),
    'isolated cycle':       (0.91, 0.30, 0.24),
    'endpoint-to-endpoint': (0.18, 0.80, 0.44),
    'junction-to-endpoint': (0.18, 0.80, 0.44),
    # integer keys (newer skan): 0=ep-ep, 1=junc-ep, 2=junc-junc, 3=isolated cycle
    0: (0.18, 0.80, 0.44),  # green
    1: (0.18, 0.80, 0.44),  # green
    2: (0.91, 0.30, 0.24),  # red — loop-forming
    3: (0.91, 0.30, 0.24),  # red
}
DEFAULT_COLOR = (0.20, 0.60, 1.00)  # blue — fallback / slab


def load_seg(path):
    img = nib.load(path)
    return (img.get_fdata() > 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_file', required=True, help='Mito segmentation .nii.gz (X,Y,Z)')
    parser.add_argument('--output', default='outputs/skeleton_visualization.png')
    parser.add_argument('--voxel_x', type=float, default=0.18)
    parser.add_argument('--voxel_y', type=float, default=0.18)
    parser.add_argument('--voxel_z', type=float, default=0.37)
    args = parser.parse_args()

    voxel_size = (args.voxel_x, args.voxel_y, args.voxel_z)

    print(f"Loading {args.seg_file} ...")
    mask = load_seg(args.seg_file)
    print(f"  Shape (X,Y,Z): {mask.shape}  foreground voxels: {np.sum(mask):,}")

    print("Skeletonizing ...")
    skel_img = morphology.skeletonize(mask)
    if np.sum(skel_img) < 2:
        print("Skeleton too small — aborting.")
        sys.exit(1)

    skel_obj = Skeleton(skel_img, spacing=voxel_size)
    branch_data = summarize(skel_obj, separator='-')
    print(f"  Total branches: {len(branch_data)}")
    btype_col = 'branch-type' if 'branch-type' in branch_data.columns else 'branch_type'
    print(branch_data[btype_col].value_counts().to_string())

    # ── XY max-projections (collapse axis 2 = Z) ────────────────────────────
    proj_mask = mask.max(axis=2).astype(float)        # (X, Y)
    proj_skel = skel_img.max(axis=2).astype(float)    # (X, Y)

    # Build per-pixel colour map from branch paths
    H, W = mask.shape[:2]
    branch_rgb   = np.zeros((H, W, 3), dtype=np.float32)
    branch_alpha = np.zeros((H, W),    dtype=np.float32)

    for idx, row in branch_data.iterrows():
        btype = row.get('branch-type', row.get('branch_type', ''))
        color = BRANCH_COLORS.get(btype, DEFAULT_COLOR)
        try:
            coords = skel_obj.path_coordinates(idx)  # (N, ndim) in (X,Y,Z)
        except Exception:
            continue
        for coord in coords:
            xi, yi = int(round(coord[0])), int(round(coord[1]))
            if 0 <= xi < H and 0 <= yi < W:
                branch_rgb[xi, yi]   = color
                branch_alpha[xi, yi] = 1.0

    # imshow expects (rows=Y, cols=X) with origin='lower', so transpose
    def T(arr):
        return arr.T

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#1a1a2e')
    for ax in axes:
        ax.set_facecolor('#1a1a2e')

    # Panel 1 — segmentation
    axes[0].imshow(T(proj_mask), cmap='gray', origin='lower', interpolation='nearest')
    axes[0].set_title('Segmentation (XY max-proj)', color='white', fontsize=12)
    axes[0].axis('off')

    # Panel 2 — skeleton (uniform blue) on dimmed mask
    axes[1].imshow(T(proj_mask), cmap='gray', origin='lower', interpolation='nearest', alpha=0.35)
    skel_rgba = np.zeros((*T(proj_skel).shape, 4), dtype=np.float32)
    skel_rgba[T(proj_skel).astype(bool)] = [*DEFAULT_COLOR, 1.0]
    axes[1].imshow(skel_rgba, origin='lower', interpolation='nearest')
    axes[1].set_title('Skeleton overlay', color='white', fontsize=12)
    axes[1].axis('off')

    # Panel 3 — branch types
    axes[2].imshow(T(proj_mask), cmap='gray', origin='lower', interpolation='nearest', alpha=0.35)
    branch_rgba_img = np.concatenate(
        [branch_rgb.transpose(1, 0, 2), T(branch_alpha)[:, :, None]], axis=2
    )
    axes[2].imshow(branch_rgba_img, origin='lower', interpolation='nearest')
    axes[2].set_title('Branch types', color='white', fontsize=12)
    axes[2].axis('off')

    legend_patches = [
        mpatches.Patch(color=(0.91, 0.30, 0.24), label='Junction–junction (loop)'),
        mpatches.Patch(color=(0.18, 0.80, 0.44), label='Endpoint branch'),
        mpatches.Patch(color=DEFAULT_COLOR,        label='Other'),
    ]
    axes[2].legend(handles=legend_patches, loc='lower right',
                   facecolor='#1a1a2e', edgecolor='white',
                   labelcolor='white', fontsize=9)

    cell_name = Path(args.seg_file).name.replace('.nii.gz', '')
    fig.suptitle(f'Mitochondrial Skeleton — {cell_name}', color='white', fontsize=14, y=1.01)

    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"\nSaved → {out}")


if __name__ == '__main__':
    main()
