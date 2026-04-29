"""
Part 2 — TAV-ATAA vs BAV-ATAA
==============================
Layers:
  L1  Disease effect         (adaptive test + BH-FDR, patient + cell)
  L2  Age-corrected          (residualize age, repeat L1)
  L3  LMM confirmatory       (Disease + Age + Sex, Subject random intercept)
  L4  Collagen rescue        (paired Wilcoxon patient / Mann-Whitney cell)
  L5  Sex effect             (Male vs Female)
  L6  Hypertension effect    (Yes vs No)
  L7  Aorta diameter         (Spearman correlation)

Outputs: outputs/final_thesis_part2/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared_stats as ss

# ── output directories ────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[3] / 'outputs' / 'final_thesis_part2'
PT_DIR    = BASE / 'patient'
CELL_DIR  = BASE / 'cell'
LMM_DIR   = BASE / 'lmm'
COMP_DIR  = BASE / 'comparison'
COL_DIR   = BASE / 'collagen'
SEX_DIR   = BASE / 'sex'
HT_DIR    = BASE / 'hypertension'
DIAM_DIR  = BASE / 'diameter'
for d in [PT_DIR, CELL_DIR, LMM_DIR, COMP_DIR, COL_DIR, SEX_DIR, HT_DIR, DIAM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

G1, G2 = 'TAA', 'BAV'
G1_LABEL, G2_LABEL = ss.GROUP_LABELS[G1], ss.GROUP_LABELS[G2]

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Data ─────────────────────────────────────────────────────────────────
    df, df_nc, df_pt, feat_cols = ss.load_data(G1, G2, out_dir=PT_DIR)

    # ── L1 Disease effect ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("L1 — DISEASE EFFECT")
    print("=" * 70)
    res_pt   = ss.run_pairwise(df_pt, G1, G2, feat_cols, label='Patient-level')
    res_cell = ss.run_pairwise(df_nc, G1, G2, feat_cols, label='Cell-level')

    res_pt.to_csv(PT_DIR   / 'L1_pairwise.csv', index=False)
    res_cell.to_csv(CELL_DIR / 'L1_pairwise.csv', index=False)

    ss.plot_cohens_d_bars(res_pt,   PT_DIR   / 'L1_cohens_d.png',
                          G1_LABEL, G2_LABEL, level='patient')
    ss.plot_cohens_d_bars(res_cell, CELL_DIR / 'L1_cohens_d.png',
                          G1_LABEL, G2_LABEL, level='cell')
    ss.plot_level_comparison(res_pt, res_cell, COMP_DIR / 'L1_patient_vs_cell.png')
    ss.plot_boxplots_per_feature(df_pt, df_nc, res_pt, res_cell,
                                  G1, G2, COMP_DIR / 'boxplots', prefix='L1')

    # ── L2 Age-corrected ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("L2 — AGE-CORRECTED DISEASE EFFECT")
    print("=" * 70)
    df_pt_r   = ss.run_age_correction(df_pt, feat_cols, 'Patient-level')
    df_cell_r = ss.run_age_correction(df_nc, feat_cols, 'Cell-level')

    res_pt2   = ss.run_pairwise(df_pt_r,   G1, G2, feat_cols, label='Patient-level (age-corrected)')
    res_cell2 = ss.run_pairwise(df_cell_r, G1, G2, feat_cols, label='Cell-level (age-corrected)')

    res_pt2.to_csv(PT_DIR   / 'L2_age_corrected.csv', index=False)
    res_cell2.to_csv(CELL_DIR / 'L2_age_corrected.csv', index=False)

    ss.plot_cohens_d_bars(res_pt2,   PT_DIR   / 'L2_age_corrected_cohens_d.png',
                          G1_LABEL, G2_LABEL, level='patient')
    ss.plot_cohens_d_bars(res_cell2, CELL_DIR / 'L2_age_corrected_cohens_d.png',
                          G1_LABEL, G2_LABEL, level='cell')
    ss.plot_level_comparison(res_pt2, res_cell2, COMP_DIR / 'L2_patient_vs_cell.png')

    for level_label, r1, r2 in [('Patient', res_pt, res_pt2),
                                   ('Cell',    res_cell, res_cell2)]:
        s1 = set(r1.loc[r1['Significant'], 'Feature']) if not r1.empty else set()
        s2 = set(r2.loc[r2['Significant'], 'Feature']) if not r2.empty else set()
        print(f"  {level_label}: {len(s1)} → {len(s2)} sig  "
              f"(kept={len(s1 & s2)}, lost={len(s1 - s2)}, gained={len(s2 - s1)})")

    # ── L3 LMM ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("L3 — LMM CONFIRMATORY")
    print("=" * 70)
    lmm = ss.run_lmm(df, G1, G2, feat_cols, sig_feats=None, label='LMM')
    if lmm is not None:
        lmm.to_csv(LMM_DIR / 'L3_lmm_results.csv', index=False)
        ss.plot_lmm_forest(lmm, LMM_DIR / 'L3_lmm_forest.png')
        dis_rows = lmm[lmm['Term'] == f'{G2_LABEL} vs {G1_LABEL}']
        best_feat = (dis_rows.sort_values('p').iloc[0]['Feature']
                     if not dis_rows.empty else feat_cols[0])
        ss.plot_lmm_caterpillar(df, G1, G2, best_feat,
                                LMM_DIR / 'L3_lmm_caterpillar.png',
                                title=f'LMM Random Effects — {best_feat.replace("_", " ")}')

    lmm_full = ss.run_lmm_full(df, G1, G2, feat_cols, label='LMM-full')
    if lmm_full is not None:
        lmm_full.to_csv(LMM_DIR / 'L3_lmm_full_results.csv', index=False)
        ss.plot_lmm_heatmap(lmm_full, LMM_DIR / 'L3_lmm_heatmap.png')

    # ── L4 Collagen rescue ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("L4 — COLLAGEN RESCUE")
    print("=" * 70)
    pt_coll, cell_coll = ss.collagen_rescue(
        df, G1, G2, feat_cols,
        label=f'{G1_LABEL} vs {G2_LABEL}',
        out_dir=COL_DIR
    )
    ss.plot_collagen_bars(pt_coll, cell_coll, G1, G2,
                          COL_DIR / 'L4_collagen_rescue.png')
    ss.plot_boxplots_collagen_per_feature(df, feat_cols, G1, G2,
                                          COL_DIR, prefix='L4', res_pt=res_pt, res_cell=res_cell)
    ss.plot_boxplots_rescue_per_feature(df_pt, df_nc, df, feat_cols,
                                        res_pt, res_cell,
                                        G1, G2, COL_DIR, prefix='L4_rescue')

    # ── L5 Sex effect ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("L5 — SEX EFFECT")
    print("=" * 70)
    sex_pt   = ss.sex_effect(df_pt, feat_cols, label='Patient-level')
    sex_cell = ss.sex_effect(df_nc, feat_cols, label='Cell-level')

    if not sex_pt.empty:
        sex_pt.to_csv(SEX_DIR / 'L5_sex_pt.csv', index=False)
    if not sex_cell.empty:
        sex_cell.to_csv(SEX_DIR / 'L5_sex_cell.csv', index=False)

    ss.plot_sex_bars(sex_pt, sex_cell, SEX_DIR / 'L5_sex_effect.png')
    ss.plot_sex_heatmap(df_pt, feat_cols, G1, G2, SEX_DIR / 'L5_sex_heatmap.png')

    # ── L6 Hypertension ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("L6 — HYPERTENSION EFFECT")
    print("=" * 70)
    ht_pt   = ss.hypertension_effect(df_pt, feat_cols, label='Patient-level')
    ht_cell = ss.hypertension_effect(df_nc, feat_cols, label='Cell-level')

    if not ht_pt.empty:
        ht_pt.to_csv(HT_DIR / 'L6_hypertension_pt.csv', index=False)
    if not ht_cell.empty:
        ht_cell.to_csv(HT_DIR / 'L6_hypertension_cell.csv', index=False)

    if not ht_pt.empty:
        ss.plot_cohens_d_bars(ht_pt, HT_DIR / 'L6_hypertension_pt.png',
                              'No-HT', 'HT', level='patient')
    if not ht_cell.empty:
        ss.plot_cohens_d_bars(ht_cell, HT_DIR / 'L6_hypertension_cell.png',
                              'No-HT', 'HT', level='cell')

    # ── L7 Aorta diameter ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("L7 — AORTA DIAMETER CORRELATIONS")
    print("=" * 70)
    diam_pt   = ss.diameter_correlation(df_pt, feat_cols, label='Patient-level')
    diam_cell = ss.diameter_correlation(df_nc, feat_cols, label='Cell-level')

    if not diam_pt.empty:
        diam_pt.to_csv(DIAM_DIR / 'L7_diameter_pt.csv', index=False)
        n_subj = df_pt['AortaDiam_mm'].notna().sum()
        ss.plot_diameter_spearman(diam_pt, DIAM_DIR / 'L7_diameter_pt.png')
    if not diam_cell.empty:
        diam_cell.to_csv(DIAM_DIR / 'L7_diameter_cell.csv', index=False)
        ss.plot_diameter_spearman(diam_cell, DIAM_DIR / 'L7_diameter_cell.png')

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    ss.plot_summary(df_pt, feat_cols, res_pt, G1, G2, BASE)

    print("\n" + "=" * 70)
    print("PART 2 COMPLETE")
    print(f"  Outputs → {BASE}")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL HELPER: Spearman bar chart
# ══════════════════════════════════════════════════════════════════════════════
def _plot_spearman_bars(res, path, title):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    if res.empty:
        return
    sub    = res.sort_values('Spearman_r', ascending=True).copy()
    colors = [ss.ORGANELLE_COLORS.get(o, '#AAA') for o in sub['Organelle']]

    fig, ax = plt.subplots(figsize=(10, max(6, len(sub) * 0.45)))
    ax.barh(range(len(sub)), sub['Spearman_r'], color=colors,
            edgecolor='white', height=0.7)
    ax.axvline(0, color='black', lw=0.9)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([f.replace('_', ' ') for f in sub['Feature']], fontsize=10)
    ax.set_xlabel("Spearman ρ (Aorta Diameter)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    xlim = ax.get_xlim()
    pad  = (xlim[1] - xlim[0]) * 0.02
    for i, (_, row) in enumerate(sub.iterrows()):
        if row.get('Significant', False):
            x = row['Spearman_r']
            ax.text(x + (pad if x >= 0 else -pad), i, ss.sig_stars(row['BH_q']),
                    va='center', ha='left' if x >= 0 else 'right', fontsize=11)

    patches = [mpatches.Patch(color=c, label=o) for o, c in ss.ORGANELLE_COLORS.items()]
    ax.legend(handles=patches, fontsize=10, loc='lower right')
    plt.tight_layout()
    ss.save(fig, path)


if __name__ == '__main__':
    import os
    os.chdir(ss.PROJ_ROOT)
    main()
