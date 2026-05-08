"""
Cohort demographics figure: age distribution + sex composition.
Outputs: outputs/figures/cohort_demographics.png
"""

import csv
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import kruskal, chi2_contingency

PROJ = Path(__file__).resolve().parents[2]
OUT  = PROJ / 'outputs' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# ── colours matching shared_stats.py ─────────────────────────────────────────
GROUP_COLORS = {
    'Healthy': '#4C9BE8',
    'TAA':     '#E74C3C',
    'BAV':     '#F39C12',
}
GROUP_LABELS = {
    'Healthy': 'TAV-NA',
    'TAA':     'TAV-ATAA',
    'BAV':     'BAV-ATAA',
}
SEX_MARKER = {'Male': 'o', 'Female': '^'}
SEX_FACE   = {'Male': 'white', 'Female': None}   # open = male, filled = female

# ── load data ─────────────────────────────────────────────────────────────────
def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))

p1 = load(PROJ / 'outputs' / 'final_thesis_part1' / 'patient' / 'patient_means.csv')
p2 = load(PROJ / 'outputs' / 'final_thesis_part2' / 'patient' / 'patient_means.csv')

all_pts = {r['Subject']: r for r in p1}
for r in p2:
    if r['Subject'] not in all_pts:
        all_pts[r['Subject']] = r

groups  = ['Healthy', 'TAA', 'BAV']
disease_map = {'Healthy': 'Healthy', 'TAA': 'TAA', 'BAV': 'BAV'}

pts_by_group = {g: [] for g in groups}
for r in all_pts.values():
    d = r['Disease']
    if d in pts_by_group:
        pts_by_group[d].append(r)

# ── stats ─────────────────────────────────────────────────────────────────────
def ages(rows):
    return [float(r['Age']) for r in rows if r['Age'] not in ('', 'nan', 'NaN')]

ages_h = ages(pts_by_group['Healthy'])
ages_t = ages(pts_by_group['TAA'])
ages_b = ages(pts_by_group['BAV'])

kw_stat, kw_p = kruskal(ages_h, ages_t, ages_b)

sex_counts = {}
for g in groups:
    rows = pts_by_group[g]
    sex_counts[g] = {
        'Male':   sum(1 for r in rows if r['Gender'] == 'Male'),
        'Female': sum(1 for r in rows if r['Gender'] == 'Female'),
    }
table = [[sex_counts[g]['Male'] for g in groups],
         [sex_counts[g]['Female'] for g in groups]]
_, chi2_p, _, _ = chi2_contingency(table)

# ── paper rc ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':     'Arial',
    'font.sans-serif': ['Arial'],
    'font.size':       9,
    'axes.titlesize':  9,
    'axes.labelsize':  9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth':  0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'savefig.dpi':       300,
})

# ── figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.2, 3.2))
# left panel wider (age), right panel narrower (sex)
gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.38,
                      left=0.09, right=0.97, top=0.88, bottom=0.14)
ax_age = fig.add_subplot(gs[0])
ax_sex = fig.add_subplot(gs[1])

# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — age strip plot with mean ± SD
# ══════════════════════════════════════════════════════════════════════════════
group_x = {'Healthy': 1, 'TAA': 2, 'BAV': 3}
jitter_seed = 42
rng = np.random.default_rng(jitter_seed)

for g in groups:
    rows  = pts_by_group[g]
    xc    = group_x[g]
    color = GROUP_COLORS[g]
    ag    = ages(rows)

    # mean ± SD bar
    mn  = np.mean(ag)
    sd  = np.std(ag, ddof=1)
    ax_age.plot([xc - 0.18, xc + 0.18], [mn, mn],
                color=color, lw=2.0, solid_capstyle='round', zorder=4)
    ax_age.errorbar(xc, mn, yerr=sd, fmt='none',
                    color=color, lw=1.4, capsize=4, capthick=1.4, zorder=4)

    # individual dots, jittered, coded by sex
    jitter = rng.uniform(-0.14, 0.14, size=len(rows))
    for i, r in enumerate(rows):
        if r['Age'] in ('', 'nan', 'NaN'):
            continue
        age = float(r['Age'])
        sex = r['Gender']
        marker  = SEX_MARKER.get(sex, 'o')
        fc      = 'white' if sex == 'Male' else color
        ec      = color
        ax_age.scatter(xc + jitter[i], age,
                       marker=marker, s=38, facecolor=fc,
                       edgecolor=ec, linewidth=1.2, zorder=5)

ax_age.set_xticks([1, 2, 3])
ax_age.set_xticklabels([GROUP_LABELS[g] for g in groups])
ax_age.set_ylabel('Age (years)')
ax_age.set_xlim(0.5, 3.5)
ax_age.set_ylim(30, 90)
ax_age.set_yticks([30, 40, 50, 60, 70, 80, 90])

# KW annotation
kw_str = f'Kruskal–Wallis\n$p$ = {kw_p:.3f}'
ax_age.text(0.97, 0.97, kw_str,
            transform=ax_age.transAxes, fontsize=7.5,
            ha='right', va='top', color='#444444',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', lw=0.7))

ax_age.set_title('A   Age distribution', loc='left', fontweight='bold', fontsize=9)

# sex legend
leg_handles = [
    mpatches.Patch(facecolor='#888888', edgecolor='#888888', label='Male (open)'),
]
male_handle = plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='white', markeredgecolor='#555555',
                          markersize=6, label='Male', linewidth=0)
fem_handle  = plt.Line2D([0], [0], marker='^', color='w',
                          markerfacecolor='#555555', markeredgecolor='#555555',
                          markersize=6, label='Female', linewidth=0)
ax_age.legend(handles=[male_handle, fem_handle],
              loc='lower right', frameon=False, fontsize=7.5,
              handletextpad=0.4, borderpad=0.2)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — sex composition stacked bar
# ══════════════════════════════════════════════════════════════════════════════
x_pos   = np.arange(len(groups))
males   = [sex_counts[g]['Male']   for g in groups]
females = [sex_counts[g]['Female'] for g in groups]
totals  = [males[i] + females[i] for i in range(len(groups))]

bar_w = 0.52
colors_m = [GROUP_COLORS[g] for g in groups]
colors_f = [GROUP_COLORS[g] for g in groups]

bars_m = ax_sex.bar(x_pos, males,   bar_w, color=colors_m,
                    alpha=0.40, label='Male',   edgecolor='none')
bars_f = ax_sex.bar(x_pos, females, bar_w, color=colors_f,
                    alpha=0.85, bottom=males, label='Female', edgecolor='none')

# count labels inside bars
for i, (m, f, tot) in enumerate(zip(males, females, totals)):
    if m > 0:
        ax_sex.text(i, m / 2, f'{m}M', ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white')
    if f > 0:
        ax_sex.text(i, m + f / 2, f'{f}F', ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white')
    ax_sex.text(i, tot + 0.15, f'n={tot}', ha='center', va='bottom',
                fontsize=7.5, color='#444444')

ax_sex.set_xticks(x_pos)
ax_sex.set_xticklabels([GROUP_LABELS[g] for g in groups])
ax_sex.set_ylabel('Number of patients')
ax_sex.set_ylim(0, max(totals) + 1.8)
ax_sex.set_yticks(range(0, max(totals) + 2))

chi2_str = f'$\chi^2$ test\n$p$ = {chi2_p:.3f}'
ax_sex.text(0.97, 0.97, chi2_str,
            transform=ax_sex.transAxes, fontsize=7.5,
            ha='right', va='top', color='#444444',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', lw=0.7))

# legend
male_patch  = mpatches.Patch(color='#888888', alpha=0.40, label='Male')
fem_patch   = mpatches.Patch(color='#888888', alpha=0.85, label='Female')
ax_sex.legend(handles=[male_patch, fem_patch],
              loc='upper left', frameon=False, fontsize=7.5,
              handletextpad=0.4, borderpad=0.2)

ax_sex.set_title('B   Sex composition', loc='left', fontweight='bold', fontsize=9)

# ── save ──────────────────────────────────────────────────────────────────────
out_path = OUT / 'cohort_demographics.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close(fig)
print(f'Saved → {out_path}')
