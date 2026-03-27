"""
Cyclomatic Number Visualization
--------------------------------
Shows what the mitochondrial cyclomatic number means graphically:
  - Concept panel: Y-tree / ring / lollipop examples
  - MIP of the mask and skeleton for a real cell
  - Full network graph with loop-forming edges highlighted in red

Usage:
    python src/visualization/visualize_cyclomatic.py
    python src/visualization/visualize_cyclomatic.py --seg_file path/to/cell_segmentation.nii.gz
    python src/visualization/visualize_cyclomatic.py --seg_file ... --voxel_size 0.37 0.207 0.207

Output:
    outputs/cyclomatic_visualization.png
"""

import argparse
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from skimage import morphology
from skan import Skeleton, summarize
import networkx as nx
from scipy.ndimage import binary_closing

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_SEG = (
    '/home/luisa/aneurysm_project/experiments/V2/restored_names/'
    'Mito/Inference_Raw/01C-0096 +coll 60x vehicleZstack cell3_segmentation.nii.gz'
)
DEFAULT_VOXEL = [0.37, 0.207, 0.207]
DEFAULT_OUT   = 'outputs/cyclomatic_visualization.png'
MIN_BRANCH    = 1.0   # µm — prune noise branches below this length


def build_graph(seg_file, voxel_size):
    """Load segmentation, skeletonise, prune, and return skeleton + graph."""
    nii  = nib.load(seg_file)
    mask = (nii.get_fdata() > 0).astype(np.uint8)
    mask = binary_closing(mask, structure=np.ones((3, 3, 3))).astype(np.uint8)

    skel     = morphology.skeletonize(mask)
    skel_obj = Skeleton(skel, spacing=voxel_size)
    branches = summarize(skel_obj)
    real     = branches[branches['branch-distance'] > MIN_BRANCH].copy()

    G = nx.from_pandas_edgelist(
        real, source='node-id-src', target='node-id-dst',
        create_using=nx.MultiGraph()
    )

    E = G.number_of_edges()
    N = G.number_of_nodes()
    C = nx.number_connected_components(G)
    cyclomatic = max(0, E - N + C)

    return mask, skel, skel_obj, real, G, E, N, C, cyclomatic


def get_loop_edges(G):
    """Return set of (min,max) node pairs that form independent cycles."""
    G_simple   = nx.Graph(G)
    cycles     = nx.cycle_basis(G_simple)
    loop_edges = set()
    for cycle in cycles:
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            loop_edges.add((min(u, v), max(u, v)))
    return loop_edges, len(cycles)


def node_pixel_coords(skel_obj, G, voxel_size):
    """Map each graph node ID to (x_px, y_px) by projecting along Z."""
    coords   = skel_obj.coordinates   # (n, 3) in µm: [z, y, x]
    node_ids = list(G.nodes())
    node_px  = {}
    for nid in node_ids:
        if nid < len(coords):
            _, y_um, x_um = coords[nid]
            node_px[nid] = (x_um / voxel_size[2], y_um / voxel_size[1])
    return node_px


def draw_concept_panel(ax):
    """Panel 1: three toy graphs illustrating cyclomatic numbers 0, 1, 2."""
    ax.set_facecolor('#1a1a2e')
    ax.set_title('Concept: Cyclomatic = E − N + C',
                 color='white', fontsize=11, fontweight='bold', pad=8)

    examples = [
        # (node_positions, edges, label, cyclomatic_value)
        (
            [(0, 1), (1, 1), (2, 1), (1, 0)],
            [(0, 1), (1, 2), (1, 3)],
            'Y-tree\nE=3, N=4, C=1\ncyclomatic = 0', 0
        ),
        (
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            [(0, 1), (1, 2), (2, 3), (3, 0)],
            'Ring\nE=4, N=4, C=1\ncyclomatic = 1', 1
        ),
        (
            [(0, 0), (1, 0), (2, 0), (1, 1), (1, -1)],
            [(0, 1), (1, 2), (1, 3), (1, 4), (3, 0), (4, 0)],
            'Lollipop\nE=6, N=5, C=1\ncyclomatic = 2', 2
        ),
    ]
    offsets = [(-3.5, 1.5), (0, 1.5), (3.0, 1.5)]

    for (pos, edges, label, cyc), (ox, oy) in zip(examples, offsets):
        for u, v in edges:
            x0, y0 = pos[u][0] + ox, pos[u][1] + oy
            x1, y1 = pos[v][0] + ox, pos[v][1] + oy
            ax.plot([x0, x1], [y0, y1], color='#4FC3F7', lw=2.5)
        for x, y in pos:
            ax.plot(x + ox, y + oy, 'o', color='#FFD54F', ms=10, zorder=5)
        color = '#FF5252' if cyc > 0 else '#B2DFDB'
        ax.text(ox + 0.5, oy - 1.3, label, ha='center', color=color, fontsize=8)

    ax.set_xlim(-5, 6)
    ax.set_ylim(-0.8, 3.5)
    ax.axis('off')


def draw_graph_panel(ax, real_branches, G, node_px, loop_edges, E, N, C, cyclomatic, cell_name):
    """Panel 4: full network graph with loop edges in red."""
    ax.set_facecolor('#0a0a1a')
    ax.set_title(
        f'Mitochondrial Network Graph — Loops Highlighted    [{cell_name}]\n'
        f'E={E:,} branches  |  N={N:,} nodes  |  C={C:,} components  |  '
        f'Cyclomatic = {E:,} − {N:,} + {C:,} = {cyclomatic:,}',
        color='white', fontsize=10, fontweight='bold', pad=8
    )

    n_loop_edges = n_tree_edges = 0
    for _, row in real_branches.iterrows():
        src, dst = int(row['node-id-src']), int(row['node-id-dst'])
        if src not in node_px or dst not in node_px:
            continue
        x0, y0 = node_px[src]
        x1, y1 = node_px[dst]
        is_loop = (min(src, dst), max(src, dst)) in loop_edges
        if is_loop:
            ax.plot([x0, x1], [y0, y1], color='#FF5252', lw=1.5, alpha=0.9, zorder=3)
            n_loop_edges += 1
        else:
            ax.plot([x0, x1], [y0, y1], color='#4FC3F7', lw=0.5, alpha=0.3, zorder=2)
            n_tree_edges += 1

    junctions = [n for n, d in G.degree() if d > 2 and n in node_px]
    endpoints  = [n for n, d in G.degree() if d <= 2 and n in node_px]

    if endpoints:
        ax.scatter([node_px[n][0] for n in endpoints],
                   [node_px[n][1] for n in endpoints],
                   c='#B2DFDB', s=2, alpha=0.4, zorder=4)
    if junctions:
        ax.scatter([node_px[n][0] for n in junctions],
                   [node_px[n][1] for n in junctions],
                   c='#FFD54F', s=10, alpha=0.85, zorder=5)

    legend = [
        Line2D([0], [0], color='#4FC3F7', lw=1.5,
               label=f'Tree branches ({n_tree_edges:,})'),
        Line2D([0], [0], color='#FF5252', lw=2.0,
               label=f'Loop branches ({n_loop_edges:,})  →  cyclomatic = {cyclomatic}'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD54F', ms=7,
               label=f'Junctions ({len(junctions):,})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#B2DFDB', ms=5,
               label=f'Endpoints ({len(endpoints):,})'),
    ]
    ax.legend(handles=legend, loc='upper right',
              facecolor='#1a1a2e', edgecolor='gray', labelcolor='white', fontsize=9)
    ax.axis('off')


def main():
    parser = argparse.ArgumentParser(description='Visualize mitochondrial cyclomatic number.')
    parser.add_argument('--seg_file',   default=DEFAULT_SEG,
                        help='Path to mito segmentation .nii.gz file')
    parser.add_argument('--voxel_size', nargs=3, type=float, default=DEFAULT_VOXEL,
                        metavar=('VZ', 'VY', 'VX'),
                        help='Voxel size in Z Y X (µm). Default: 0.37 0.207 0.207')
    parser.add_argument('--output',     default=DEFAULT_OUT,
                        help='Output PNG path')
    args = parser.parse_args()

    cell_name = args.seg_file.split('/')[-1].replace('_segmentation.nii.gz', '')
    print(f"Processing: {cell_name}")

    mask, skel, skel_obj, real_branches, G, E, N, C, cyclomatic = build_graph(
        args.seg_file, args.voxel_size)

    loop_edges, n_cycles = get_loop_edges(G)
    node_px = node_pixel_coords(skel_obj, G, args.voxel_size)

    print(f"  E={E:,}  N={N:,}  C={C:,}  Cyclomatic={cyclomatic}")
    print(f"  Cycles from nx.cycle_basis: {n_cycles}  |  Loop edges: {len(loop_edges)}")

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor('#0d0d0d')

    ax_concept = fig.add_subplot(2, 3, 1)
    ax_mask    = fig.add_subplot(2, 3, 2)
    ax_skel    = fig.add_subplot(2, 3, 3)
    ax_graph   = fig.add_subplot(2, 1, 2)

    # Panel 1 — concept
    draw_concept_panel(ax_concept)

    # Panel 2 — mask MIP
    ax_mask.imshow(mask.max(axis=0), cmap='hot', origin='upper', aspect='auto')
    ax_mask.set_title(f'Mito Mask — MIP (Z-projection)\n{cell_name[:45]}',
                      color='white', fontsize=9, fontweight='bold')
    ax_mask.axis('off')
    ax_mask.set_facecolor('black')

    # Panel 3 — skeleton MIP
    ax_skel.imshow(skel.max(axis=0), cmap='cool', origin='upper', aspect='auto')
    ax_skel.set_title('Skeleton — MIP (Z-projection)',
                      color='white', fontsize=9, fontweight='bold')
    ax_skel.axis('off')
    ax_skel.set_facecolor('black')

    # Panel 4 — graph
    draw_graph_panel(ax_graph, real_branches, G, node_px, loop_edges,
                     E, N, C, cyclomatic, cell_name[:50])

    for ax in [ax_concept, ax_mask, ax_skel, ax_graph]:
        ax.title.set_color('white')

    plt.suptitle('Mito Cyclomatic Number — Concept & Real Cell Visualization',
                 color='white', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches='tight', facecolor='#0d0d0d')
    plt.close()
    print(f"Saved → {args.output}")


if __name__ == '__main__':
    main()
