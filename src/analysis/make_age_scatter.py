"""
Age residualization illustration — scatter plot.

Uses the exact same dataset and group assignments as the 5-layer SMC analysis
(outputs/Advanced_Features_Raw_200.csv, Non-Aneurysmal vs Aneurysmal).

X: Age, Y: Mito Branch Count
Dots: teal = Non-Aneurysmal (healthy), salmon = Aneurysmal (TAA)
Single OLS regression line through all dots
Annotation: β₁ = slope (change per year)

Usage:
  cd /home/luisa/aneurysm_project
  python src/analysis/make_age_scatter.py
"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pathlib import Path

# ── Same group lists and helpers as complete_stat_analysis.py ─────────────────
NON_ANEURYSMAL = ['01asc-180','01asc-222','01asc-230',
                  '01c-96','01c-97','01c-113','01c-117','01c-202']
ANEURYSMAL     = ['03asc-24','03asc-43','03rt-45','03asc-46','03asc-47',
                  '03asc-51','03asc-54','03asc-55','03asc-56','03asc-57']

def normalize(sid):
    return re.sub(r'-0*(\d+)$', r'-\1', sid.lower()) if sid else None

def extract_id(cell_name):
    m = re.search(r'(0[123][A-Za-z]+-\d+)', str(cell_name))
    if not m:
        m = re.search(r'(0[123][A-Za-z]+)(\d+)', str(cell_name))
        if not m: return None
        return (m.group(1) + '-' + m.group(2)).lower()
    return m.group(1).lower()

na_norm = [normalize(x) for x in NON_ANEURYSMAL]
an_norm = [normalize(x) for x in ANEURYSMAL]

# ── Data ──────────────────────────────────────────────────────────────────────
df_all = pd.read_csv('outputs/Advanced_Features_Raw_200.csv')
df_all['SpecimenID'] = df_all['CellName'].apply(extract_id).apply(
    lambda x: normalize(x) if x else None)

df_all = df_all[df_all['SpecimenID'].isin(na_norm + an_norm)].copy()
df_all['DiseaseGroup'] = df_all['SpecimenID'].apply(
    lambda x: 'Non-Aneurysmal' if x in na_norm else 'Aneurysmal (TAA)')

# No-collagen only (same as age correction pass in complete_stat_analysis.py
# which uses the full df — but scatter is cleaner with one condition)
df = df_all[df_all['Collagen_Status'] == 'NoCollagen'].copy()

FEAT = 'Actin_Skeleton_Length'  # highest R² vs Age (R²=0.196) among all 21 features
sub  = df[['Age', FEAT, 'DiseaseGroup']].dropna()

COLORS = {'Non-Aneurysmal': '#00897B', 'Aneurysmal (TAA)': '#EF9A9A'}

# ── Regression on ALL cells ────────────────────────────────────────────────────
X = sub['Age'].values.reshape(-1, 1)
y = sub[FEAT].values
model = LinearRegression().fit(X, y)
slope     = model.coef_[0]
intercept = model.intercept_

age_line = np.array([sub['Age'].min() - 1, sub['Age'].max() + 1])
y_line   = model.predict(age_line.reshape(-1, 1))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4.5))

for grp, color in COLORS.items():
    mask = sub['DiseaseGroup'] == grp
    label = 'Non-Aneurysmal\n(healthy)' if grp == 'Non-Aneurysmal' else 'Aneurysmal\n(TAA)'
    ax.scatter(sub.loc[mask, 'Age'], sub.loc[mask, FEAT],
               color=color, alpha=0.55, s=28, linewidths=0,
               label=label, zorder=2)

# Regression line
ax.plot(age_line, y_line, color='#333333', lw=2.0, zorder=3)

# β₁ annotation — place near the middle of the line
x_mid = sub['Age'].mean()
y_mid = model.predict([[x_mid]])[0]
sign  = '+' if slope >= 0 else ''
ax.annotate(
    f'$\\beta_1$ = {sign}{slope:.1f} per year',
    xy=(x_mid, y_mid),
    xytext=(x_mid + 3, y_mid + 120),
    fontsize=10,
    color='#333333',
    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
              edgecolor='#cccccc', alpha=0.9),
)

ax.set_xlabel('Age (years)', fontsize=11)
ax.set_ylabel('Actin Skeleton Length (µm)', fontsize=11)
ax.set_title('Age residualization — single regression\nfit across all cells  (best-fit feature: R² = 0.20)',
             fontsize=11, fontweight='bold')

ax.legend(framealpha=0.9, fontsize=9, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', color='#eeeeee', lw=0.8, zorder=0)

plt.tight_layout()

out = Path('classification_results/smc_analysis/5layer/layer2/layer2_age_scatter.png')
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=180, bbox_inches='tight')
plt.close()
print(f"Saved {out}")
print(f"β₁ = {slope:.2f} branches/year  (intercept = {intercept:.1f})")
