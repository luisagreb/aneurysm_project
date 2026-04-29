"""
Composite figure — Part 1 L3 + L5.
  A  LMM heatmap          (Disease / Age / Sex terms)
  B  Sex heatmap          (Cohen's d Male−Female within TAV-NA and TAV-ATAA)
  C  Sex effect bar chart (Patient + Cell level)

Modifications vs originals:
  - No dashed lines on LMM colourbar
  - "(n/29 sig)" replaced by significance stars in panel C titles
  - No grey footer text
Output: outputs/figures/part1_l3_l5_composite.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

PROJ = Path(__file__).resolve().parents[2]
OUT  = PROJ / 'outputs' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

ORGANELLE_COLORS = {
    'Actin':        '#2ECC71',
    'Mitochondria': '#E74C3C',
    'Nucleus':      '#1558A0',
}
GROUP_LABELS = {'Healthy': 'TAV-NA', 'TAA': 'TAV-ATAA'}
ALPHA = 0.05

# ── helpers ───────────────────────────────────────────────────────────────────
def cohens_d(a, b):
    s = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(b) - np.mean(a)) / s if s > 0 else 0.0

def sig_stars(q):
    if q < 0.001: return '***'
    if q < 0.01:  return '**'
    if q < 0.05:  return '*'
    return ''

def bh_fdr(pvals):
    _, q, _, _ = multipletests(np.nan_to_num(pvals, nan=1.0), method='fdr_bh')
    return q

def abbrev(feat):
    org_map = {'Actin': 'Act', 'Mito': 'Mit', 'Nucleus': 'Nuc'}
    for k, v in org_map.items():
        feat = feat.replace(k + '_', v + '_')
    feat = (feat.replace('_ratio', '').replace('_µm³', '')
                .replace('_µm²', '').replace('_µm', '')
                .replace('_n', '').replace('_', ''))
    return feat

# ── load data ─────────────────────────────────────────────────────────────────
lmm_full  = pd.read_csv(PROJ / 'outputs/final_thesis_part1/lmm/L3_lmm_full_results.csv')
sex_pt    = pd.read_csv(PROJ / 'outputs/final_thesis_part1/sex/L5_sex_pt.csv')
sex_cell  = pd.read_csv(PROJ / 'outputs/final_thesis_part1/sex/L5_sex_cell.csv')
pt_means  = pd.read_csv(PROJ / 'outputs/final_thesis_part1/patient/patient_means.csv')

feat_cols = [c for c in pt_means.columns
             if any(c.startswith(o) for o in ['Actin_', 'Mito_', 'Nucleus_'])]

def get_organelle(f):
    if f.startswith('Actin'):   return 'Actin'
    if f.startswith('Mito'):    return 'Mitochondria'
    if f.startswith('Nucleus'): return 'Nucleus'
    return 'Other'

# ── paper rc ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'Arial',
    'font.sans-serif':    ['Arial'],
    'font.size':          11,
    'font.weight':        'bold',
    'axes.labelsize':     12,
    'axes.labelweight':   'bold',
    'axes.titlesize':     11,
    'axes.titleweight':   'bold',
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    10,
    'axes.linewidth':     0.8,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'xtick.major.size':   3,
    'ytick.major.size':   3,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'savefig.dpi':        300,
    'text.usetex':        False,
})

# ── figure layout ─────────────────────────────────────────────────────────────
# Top:    A — LMM heatmap (full width, horizontal: features=cols, terms=rows)
# Bottom: B — Sex heatmap | C — Sex bars (patient + cell)
n_feat = len(feat_cols)
fig = plt.figure(figsize=(20, n_feat * 0.42 + 8))
gs  = fig.add_gridspec(2, 1, height_ratios=[1, 2.5],
                        left=0.06, right=0.97, top=0.95, bottom=0.04,
                        hspace=0.55)
ax_lmm   = fig.add_subplot(gs[0])
gs_bot   = gs[1].subgridspec(1, 3, width_ratios=[1.0, 3.2, 3.2], wspace=0.55)
ax_sexh  = fig.add_subplot(gs_bot[0])
ax_pt    = fig.add_subplot(gs_bot[1])
ax_cell  = fig.add_subplot(gs_bot[2])

# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — LMM heatmap (horizontal: terms=rows, features=columns)
# ══════════════════════════════════════════════════════════════════════════════
lmm = lmm_full.copy()
term_order = [t for t in ['Disease', 'Age', 'Sex'] if t in lmm['Term'].unique()]
lmm = lmm[lmm['Term'].isin(term_order)]
lmm['score'] = -np.log10(lmm['BH_q'].clip(1e-10)) * np.sign(lmm['Coef'])

feat_order_lmm = (lmm.drop_duplicates('Feature')
                     .sort_values(['Organelle', 'Feature'])['Feature'].tolist())
pivot   = (lmm.pivot(index='Feature', columns='Term', values='score')
              .reindex(index=feat_order_lmm, columns=term_order).fillna(0))
sig_piv = (lmm.pivot(index='Feature', columns='Term', values='Significant')
              .reindex(index=feat_order_lmm, columns=term_order).fillna(False))

x_labels_lmm = [abbrev(f) for f in feat_order_lmm]
n_f, n_t = len(feat_order_lmm), len(term_order)
vmax_lmm = max(3.0, float(np.abs(pivot.values).max()))

# Transpose: rows=terms, cols=features
data_lmm = pivot.T.values
sig_lmm  = sig_piv.T.values

im_lmm = ax_lmm.imshow(data_lmm, aspect='auto',
                        cmap='RdBu_r', vmin=-vmax_lmm, vmax=vmax_lmm)
ax_lmm.set_yticks(np.arange(n_t))
ax_lmm.set_yticklabels(term_order, fontsize=11, fontweight='bold')
ax_lmm.set_xticks(np.arange(n_f))
ax_lmm.set_xticklabels(x_labels_lmm, fontsize=9, fontweight='bold',
                        rotation=45, ha='right', rotation_mode='anchor')
ax_lmm.set_xticks(np.arange(-0.5, n_f, 1), minor=True)
ax_lmm.set_yticks(np.arange(-0.5, n_t, 1), minor=True)
ax_lmm.grid(which='minor', color='white', linewidth=0.6)
ax_lmm.tick_params(which='minor', bottom=False, left=False)

sig_map = (lmm.pivot(index='Feature', columns='Term', values='BH_q')
              .reindex(index=feat_order_lmm, columns=term_order).fillna(1.0))
for i in range(n_t):
    for j in range(n_f):
        if sig_lmm[i, j]:
            ax_lmm.text(j, i, sig_stars(sig_map.iloc[j, i]),
                        ha='center', va='center', fontsize=9, color='black')

cbar_lmm = fig.colorbar(im_lmm, ax=ax_lmm, fraction=0.02, pad=0.01)
cbar_lmm.set_label(r'$-\log_{10}(q)\times\mathrm{sign(\beta)}$', fontsize=10, fontweight='bold')
cbar_lmm.ax.tick_params(labelsize=9)

ax_lmm.text(-0.04, 1.08, 'A', transform=ax_lmm.transAxes,
            fontsize=13, fontweight='bold', va='bottom')
ax_lmm.set_title('LMM — fixed effects', fontsize=10, pad=6, loc='left', fontweight='bold')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — Sex heatmap (Cohen's d Male−Female within each group)
# ══════════════════════════════════════════════════════════════════════════════
groups    = ['Healthy', 'TAA']
d_cols, q_cols, col_labels = [], [], []

for g in groups:
    sub_g = pt_means[pt_means['Disease'] == g]
    nm = int((sub_g['Gender'] == 'Male').sum())
    nf = int((sub_g['Gender'] == 'Female').sum())
    col_labels.append(f"{GROUP_LABELS[g]}\n(M={nm}, F={nf})")
    ds, pvals = [], []
    for feat in feat_cols:
        m = sub_g[sub_g['Gender'] == 'Male'][feat].dropna().values
        f = sub_g[sub_g['Gender'] == 'Female'][feat].dropna().values
        if len(m) >= 2 and len(f) >= 2:
            d = cohens_d(f, m)
            try:
                _, p = mannwhitneyu(m, f, alternative='two-sided')
            except Exception:
                p = np.nan
        else:
            d, p = 0.0, np.nan
        ds.append(d)
        pvals.append(p if not np.isnan(p) else 1.0)
    qs = bh_fdr(np.array(pvals))
    d_cols.append(ds)
    q_cols.append(qs)

d_df   = pd.DataFrame(np.column_stack(d_cols), index=feat_cols, columns=col_labels)
q_df   = pd.DataFrame(np.column_stack(q_cols), index=feat_cols, columns=col_labels)
sig_df = q_df < ALPHA

order  = d_df.abs().mean(axis=1).sort_values().index
d_df   = d_df.loc[order]
q_df   = q_df.loc[order]
sig_df = sig_df.loc[order]

y_labels_sexh = [abbrev(f) for f in d_df.index]
n_fh, n_ch = len(d_df), len(groups)
vmax_sexh = max(2.0, float(d_df.abs().values.max()))

im_sexh = ax_sexh.imshow(d_df.values, aspect='auto',
                          cmap='RdBu_r', vmin=-vmax_sexh, vmax=vmax_sexh)
ax_sexh.set_xticks(range(n_ch))
ax_sexh.set_xticklabels(col_labels, fontsize=10, fontweight='bold')
ax_sexh.set_yticks(range(n_fh))
ax_sexh.set_yticklabels(y_labels_sexh, fontsize=10, fontweight='bold')
ax_sexh.set_xticks(np.arange(-0.5, n_ch, 1), minor=True)
ax_sexh.set_yticks(np.arange(-0.5, n_fh, 1), minor=True)
ax_sexh.grid(which='minor', color='white', linewidth=0.6)
ax_sexh.tick_params(which='minor', bottom=False, left=False)

for i in range(n_fh):
    for j in range(n_ch):
        if sig_df.iloc[i, j]:
            ax_sexh.text(j, i, sig_stars(q_df.iloc[i, j]),
                         ha='center', va='center', fontsize=8, color='black')

cbar_sexh = fig.colorbar(im_sexh, ax=ax_sexh, fraction=0.08, pad=0.04)
cbar_sexh.set_label("Cohen's $d$ (Male−Female)", fontsize=10, fontweight='bold')
cbar_sexh.ax.tick_params(labelsize=9)

ax_sexh.text(-0.30, 1.02, 'B', transform=ax_sexh.transAxes,
             fontsize=13, fontweight='bold', va='bottom')
ax_sexh.set_title('Sex heatmap', fontsize=10, pad=4, loc='left', fontweight='bold')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL C — Sex effect bars (patient + cell)
# ══════════════════════════════════════════════════════════════════════════════
def draw_sex_bars(ax, res, panel_letter, sublabel):
    sub    = res.sort_values('Cohen_d', ascending=True).copy()
    colors = [ORGANELLE_COLORS.get(get_organelle(f), '#AAA') for f in sub['Feature']]
    ax.barh(range(len(sub)), sub['Cohen_d'], color=colors,
            edgecolor='white', linewidth=0, height=0.78, alpha=0.85)
    ax.axvline(0, color='#333333', lw=0.8)

    labels = []
    for f in sub['Feature']:
        parts = f.split('_', 1)
        labels.append(parts[1].replace('_', ' ') if len(parts) > 1 else f.replace('_', ' '))
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_xlabel("Cohen's $d$  (Female → Male)", fontsize=12, fontweight='bold')

    # stars annotation
    xlim = ax.get_xlim()
    pad  = (xlim[1] - xlim[0]) * 0.04
    star_xs = []
    for i, row in sub.iterrows():
        if row.get('Significant', False):
            x  = row['Cohen_d']
            sx = x + (pad if x >= 0 else -pad)
            star_xs.append(sx)
            ax.text(sx, list(sub.index).index(i),
                    sig_stars(row['BH_q']),
                    va='center', ha='left' if x >= 0 else 'right',
                    fontsize=8, color='#222222')
    if star_xs:
        extra     = (xlim[1] - xlim[0]) * 0.10
        ax.set_xlim(min(xlim[0], min(star_xs)) - extra,
                    max(xlim[1], max(star_xs)) + extra)

    # title with stars instead of (n/29 sig)
    n_sig = int(sub['Significant'].sum())
    if n_sig == 0:
        title_str = sublabel
    else:
        min_q  = sub.loc[sub['Significant'], 'BH_q'].min()
        title_str = f'{sublabel}  {sig_stars(min_q)}'

    ax.set_title(title_str, fontsize=10, pad=4, loc='left', fontweight='bold')

    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color='#dddddd', linewidth=0.5, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if panel_letter:
        ax.text(-0.20, 1.02, panel_letter, transform=ax.transAxes,
                fontsize=13, fontweight='bold', va='bottom')

draw_sex_bars(ax_pt,   sex_pt,   'C', 'Patient level')
draw_sex_bars(ax_cell, sex_cell, '',  'Cell level')

# ── shared organelle legend ───────────────────────────────────────────────────
patches = [mpatches.Patch(facecolor=ORGANELLE_COLORS[o], label=o, alpha=0.85)
           for o in ['Actin', 'Mitochondria', 'Nucleus']]
fig.legend(handles=patches, loc='upper center', ncol=3,
           frameon=False, fontsize=9,
           bbox_to_anchor=(0.60, 0.995))

fig.text(0.60, 0.005,
         '* $q$<0.05   ** $q$<0.01   *** $q$<0.001   (BH-FDR corrected)',
         ha='center', fontsize=7.5, color='#666666', style='italic')

out_path = OUT / 'part1_l3_l5_composite.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close(fig)
print(f'Saved → {out_path}')
