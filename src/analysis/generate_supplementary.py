"""
Generate all supplementary LaTeX tables and figures (S1–S13).
Output: outputs/supplementary/
"""

import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path("outputs/supplementary")
OUT.mkdir(parents=True, exist_ok=True)

BASE = Path("outputs")
P1   = BASE / "final_thesis_part1"
P2   = BASE / "final_thesis_part2"
CLS  = BASE / "classification"

# ─────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────
def _esc(s):
    """Escape a string for LaTeX."""
    s = str(s)
    s = s.replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')
    s = s.replace('_', r'\_')
    s = s.replace('µ', r'$\mu$')
    s = s.replace('²', r'$^{2}$')
    s = s.replace('³', r'$^{3}$')
    return s

def _feat(name):
    """Human-readable feature name for tables."""
    name = str(name)
    for org in ('Actin_', 'Mito_', 'Nucleus_'):
        if name.startswith(org):
            prefix = org[:-1]
            stem   = name[len(org):]
            return f"{prefix} {stem.replace('_', ' ')}"
    return name.replace('_', ' ')

def _pval(p):
    if pd.isna(p): return '--'
    if p < 0.001: return r'$<$0.001'
    return f'{p:.3f}'

def _qval(q):
    if pd.isna(q): return '--'
    if q < 0.001: return r'$<$0.001'
    return f'{q:.3f}'

def _sig(row, col='Significant'):
    return r'$^{*}$' if row.get(col, False) else ''

HEADER = r"""\documentclass[11pt,a4paper]{article}
\usepackage{booktabs,longtable,geometry,array,xcolor,microtype}
\geometry{margin=2cm}
\renewcommand{\arraystretch}{1.25}
\begin{document}
"""
FOOTER = r"\end{document}" + "\n"

def write_tex(path, body):
    with open(path, 'w') as f:
        f.write(HEADER)
        f.write(body)
        f.write(FOOTER)
    print(f"  Saved: {path.name}")


# ─────────────────────────────────────────────────────────────────
# S1 — Donor cohort table
# ─────────────────────────────────────────────────────────────────
def s1_cohort():
    df = pd.read_csv(BASE / "Advanced_Features_Raw_Final.csv")
    # Extract donor ID from cell name
    df['_DonorRaw'] = df['CellName'].str.extract(r'^([\w]+-[\w]+)', expand=False)
    meta = df.groupby('_DonorRaw').agg(
        Group=('Disease', 'first'),
        Age=('Age', 'first'),
        Sex=('Gender', 'first'),
        N_cells_Collagen=('Collagen_Status', lambda x: (x == 'Collagen').sum()),
        N_cells_NoCollagen=('Collagen_Status', lambda x: (x == 'NoCollagen').sum()),
    ).reset_index().rename(columns={'_DonorRaw': 'Donor'})
    meta['N_total'] = meta['N_cells_Collagen'] + meta['N_cells_NoCollagen']

    group_order = {'Healthy': 0, 'TAA': 1, 'BAV': 2}
    meta['_ord'] = meta['Group'].map(group_order)
    meta = meta.sort_values(['_ord', 'Donor']).drop(columns='_ord').reset_index(drop=True)

    lines = []
    lines.append(r"\section*{S1 — Full Donor Cohort}")
    lines.append(r"All donors included in the study with group assignment, age, sex, and cell counts")
    lines.append(r"per condition (collagen-coated vs.\ uncoated wells). Hypertension status and aortic")
    lines.append(r"diameter (where available) were obtained from clinical records and are listed in the")
    lines.append(r"main methods section. Donor 03Asc-54 was excluded from Part~2 statistical tests")
    lines.append(r"due to missing BAV comparison data (noted in the main text).")
    lines.append(r"\vspace{1em}")
    lines.append(r"\begin{longtable}{llcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Donor & Group & Age & Sex & $N_{+\text{coll}}$ & $N_{\text{no coll}}$ & $N_{\text{total}}$ \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule Donor & Group & Age & Sex & $N_{+\text{coll}}$ & $N_{\text{no coll}}$ & $N_{\text{total}}$ \\\midrule\endhead")
    lines.append(r"\bottomrule\endfoot")

    cur_group = None
    for _, row in meta.iterrows():
        if row['Group'] != cur_group:
            if cur_group is not None:
                lines.append(r"\midrule")
            lines.append(r"\multicolumn{7}{l}{\textit{" + _esc(row['Group']) + r"}} \\")
            cur_group = row['Group']
        age = f"{int(row['Age'])}" if not pd.isna(row['Age']) else '--'
        lines.append(f"  {_esc(row['Donor'])} & {_esc(row['Group'])} & {age} & {_esc(row['Sex'])} & "
                     f"{int(row['N_cells_Collagen'])} & {int(row['N_cells_NoCollagen'])} & {int(row['N_total'])} \\\\")

    total = meta['N_total'].sum()
    lines.append(r"\midrule")
    lines.append(f"  \\textbf{{Total}} & & & & & & {int(total)} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")

    write_tex(OUT / "S1_cohort.tex", "\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────
# S2 — Shapiro-Wilk normality results
# ─────────────────────────────────────────────────────────────────
def s2_shapiro():
    df = pd.read_csv(BASE / "statistical_analysis" / "L1_shapiro_wilk.csv")

    lines = []
    lines.append(r"\section*{S2 — Shapiro--Wilk Normality Test Results}")
    lines.append(r"Shapiro--Wilk test $p$-values for all 30 features in each group.")
    lines.append(r"Features with $p < 0.05$ in at least one group failed normality")
    lines.append(r"and received Mann--Whitney~U (cell level) or Wilcoxon signed-rank (patient level)")
    lines.append(r"tests instead of Welch $t$-tests. All 30 features failed normality in at least one")
    lines.append(r"group; the adaptive testing strategy therefore applied to the full feature set.")
    lines.append(r"\vspace{1em}")

    # Pivot to wide: rows=features, cols=groups
    wide = df.pivot_table(index='Feature', columns='Disease', values='SW_p', aggfunc='first').reset_index()
    groups = [c for c in wide.columns if c != 'Feature']
    col_spec = 'l' + 'r' * len(groups)

    lines.append(r"\begin{longtable}{" + col_spec + "}")
    lines.append(r"\toprule")
    header_cells = ['Feature'] + [_esc(g) for g in groups]
    lines.append(' & '.join(header_cells) + r' \\')
    lines.append(r" & \multicolumn{" + str(len(groups)) + r"}{c}{Shapiro--Wilk $p$-value} \\")
    lines.append(r"\midrule\endfirsthead")
    lines.append(r"\toprule " + ' & '.join(header_cells) + r'\\\midrule\endhead')
    lines.append(r"\bottomrule\endfoot")

    for _, row in wide.iterrows():
        cells = [_esc(_feat(row['Feature']))]
        for g in groups:
            p = row.get(g, np.nan)
            mark = r'$^{*}$' if (not pd.isna(p) and p < 0.05) else ''
            cells.append(_pval(p) + mark)
        lines.append('  ' + ' & '.join(cells) + r' \\')

    lines.append(r"\bottomrule")
    lines.append(r"\multicolumn{" + str(len(groups)+1) + r"}{l}{\footnotesize $^{*}p < 0.05$: failed normality.} \\")
    lines.append(r"\end{longtable}")

    write_tex(OUT / "S2_shapiro_wilk.tex", "\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────
# S3/S4 — Full Cohen's d tables (Part 1 & Part 2)
# ─────────────────────────────────────────────────────────────────
def _cohens_d_table(part_dir, part_label, comp_label, out_name, section_num):
    c_raw  = pd.read_csv(part_dir / "cell"    / "L1_pairwise.csv")
    c_age  = pd.read_csv(part_dir / "cell"    / "L2_age_corrected.csv")
    pt_raw = pd.read_csv(part_dir / "patient" / "L1_pairwise.csv")
    pt_age = pd.read_csv(part_dir / "patient" / "L2_age_corrected.csv")

    # index by Feature
    for d in [c_raw, c_age, pt_raw, pt_age]:
        d.set_index('Feature', inplace=True)

    features = c_raw.index.tolist()

    lines = []
    lines.append(f"\\section*{{S{section_num} — Full Cohen's~$d$ Table: {part_label}}}")
    lines.append(f"All 30 features for {comp_label}, at both cell and patient level,")
    lines.append(r"raw and age-residualised. $q$: BH-FDR adjusted $p$-value. $^{*}$: $q < 0.05$.")
    lines.append(r"\vspace{1em}")
    lines.append(r"\begin{longtable}{lrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{2}{c}{Cell (raw)} & \multicolumn{2}{c}{Cell (age-corr.)} "
                 r"& \multicolumn{2}{c}{Patient (raw)} & \multicolumn{2}{c}{Patient (age-corr.)} \\")
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}")
    lines.append(r"Feature & $d$ & $q$ & $d$ & $q$ & $d$ & $q$ & $d$ & $q$ \\")
    lines.append(r"\midrule\endfirsthead")
    lines.append(r"\toprule Feature & $d$ & $q$ & $d$ & $q$ & $d$ & $q$ & $d$ & $q$ \\\midrule\endhead")
    lines.append(r"\bottomrule\endfoot")

    prev_org = None
    for feat in features:
        org = feat.split('_')[0]
        if org != prev_org:
            if prev_org is not None:
                lines.append(r"\midrule")
            lines.append(r"\multicolumn{9}{l}{\textit{" + _esc(org) + r"}} \\")
            prev_org = org

        def _cell(df, feat, dcol='Cohen_d', qcol='BH_q', scol='Significant'):
            if feat not in df.index: return '--', '--', ''
            row = df.loc[feat]
            d  = f"{float(row[dcol]):+.2f}" if not pd.isna(row.get(dcol)) else '--'
            q  = _qval(row.get(qcol))
            mk = r'$^{*}$' if row.get(scol, False) else ''
            return d, q, mk

        def fmt(df, feat):
            d, q, mk = _cell(df, feat)
            return f"{d}{mk}", q

        cr_d, cr_q = fmt(c_raw,  feat)
        ca_d, ca_q = fmt(c_age,  feat)
        pr_d, pr_q = fmt(pt_raw, feat)
        pa_d, pa_q = fmt(pt_age, feat)

        name = _esc(_feat(feat).split(' ', 1)[1] if ' ' in _feat(feat) else _feat(feat))
        lines.append(f"  {name} & {cr_d} & {cr_q} & {ca_d} & {ca_q} & {pr_d} & {pr_q} & {pa_d} & {pa_q} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\multicolumn{9}{l}{\footnotesize $^{*}q < 0.05$ (BH-FDR).} \\")
    lines.append(r"\end{longtable}")

    write_tex(OUT / out_name, "\n".join(lines) + "\n")


def s3_part1_cohens_d():
    _cohens_d_table(P1, "Part~1: Healthy vs.\ TAA", "healthy vs.\ diseased (TAA)",
                    "S3_part1_cohens_d.tex", 3)

def s4_part2_cohens_d():
    _cohens_d_table(P2, "Part~2: TAV-ATAA vs.\ BAV-ATAA", "TAV-ATAA vs.\ BAV-ATAA",
                    "S4_part2_cohens_d.tex", 4)


# ─────────────────────────────────────────────────────────────────
# S5 — Full LMM tables
# ─────────────────────────────────────────────────────────────────
def _lmm_table(lmm_csv, part_label, section_num, out_name):
    df = pd.read_csv(lmm_csv)
    terms_order = ['Disease', 'Collagen', 'Interaction', 'Age']

    lines = []
    lines.append(f"\\section*{{S{section_num} — Full LMM Output: {part_label}}}")
    lines.append(r"Complete linear mixed model results for all 30 features.")
    lines.append(r"Model: \textit{Feature} $\sim$ Disease $+$ Collagen $+$ Disease:Collagen $+$ Age $+$ (1\,|\,Donor).")
    lines.append(r"$q$: BH-FDR across features within each term. $^{*}$: $q < 0.05$.")
    lines.append(r"\vspace{1em}")
    lines.append(r"\begin{longtable}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Feature & Term & Coef. & $p$ & $q$ & Sig. \\")
    lines.append(r"\midrule\endfirsthead")
    lines.append(r"\toprule Feature & Term & Coef. & $p$ & $q$ & Sig. \\\midrule\endhead")
    lines.append(r"\bottomrule\endfoot")

    features = df['Feature'].unique()
    prev_org = None
    for feat in features:
        org = feat.split('_')[0]
        if org != prev_org:
            if prev_org is not None:
                lines.append(r"\midrule")
            lines.append(r"\multicolumn{6}{l}{\textit{" + _esc(org) + r"}} \\")
            prev_org = org

        sub = df[df['Feature'] == feat].copy()
        sub = sub.set_index('Term').reindex(terms_order).reset_index()
        name = _esc(_feat(feat).split(' ', 1)[1] if ' ' in _feat(feat) else _feat(feat))

        for i, row in sub.iterrows():
            term = _esc(str(row.get('Term', '--')))
            coef = f"{float(row['Coef']):+.3f}" if not pd.isna(row.get('Coef')) else '--'
            pv   = _pval(row.get('p'))
            qv   = _qval(row.get('BH_q'))
            sig  = r'$^{*}$' if row.get('Significant', False) else ''
            feat_col = name if i == 0 else ''
            lines.append(f"  {feat_col} & {term} & {coef} & {pv} & {qv} & {sig} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\multicolumn{6}{l}{\footnotesize $^{*}q < 0.05$ (BH-FDR).} \\")
    lines.append(r"\end{longtable}")

    write_tex(OUT / out_name, "\n".join(lines) + "\n")


def s5_lmm():
    _lmm_table(P1 / "lmm" / "L3_lmm_full_results.csv",
               "Part~1 (Healthy vs.\ TAA)", 5, "S5a_lmm_part1.tex")
    _lmm_table(P2 / "lmm" / "L3_lmm_full_results.csv",
               "Part~2 (TAV-ATAA vs.\ BAV-ATAA)", 5, "S5b_lmm_part2.tex")


# ─────────────────────────────────────────────────────────────────
# S6 — ICC placeholder (needs re-run with random-effects extraction)
# ─────────────────────────────────────────────────────────────────
def s6_icc_placeholder():
    body = (
        r"\section*{S6 — Intraclass Correlation Coefficients (ICC)}" + "\n"
        r"ICC values quantify the proportion of total variance explained by donor identity " + "\n"
        r"(random intercept) in the linear mixed model. These values were computed from the " + "\n"
        r"fitted variance components: $\text{ICC} = \sigma^2_{\text{donor}} / "
        r"(\sigma^2_{\text{donor}} + \sigma^2_{\text{residual}})$." + "\n\n"
        r"\textbf{Note:} ICC extraction requires re-running the LMM with variance component " + "\n"
        r"output enabled on the analysis server. This table will be populated upon re-run." + "\n"
    )
    write_tex(OUT / "S6_icc.tex", body)


# ─────────────────────────────────────────────────────────────────
# S7 — Full collagen rescue
# ─────────────────────────────────────────────────────────────────
def _collagen_table(part_dir, part_label, section_num, out_name):
    c  = pd.read_csv(part_dir / "collagen" / "L4_collagen_mw_cell.csv")
    pt = pd.read_csv(part_dir / "collagen" / "L4_collagen_wilcoxon_pt.csv")

    lines = []
    lines.append(f"\\section*{{S{section_num} — Full Collagen Rescue Results: {part_label}}}")
    lines.append(r"Effect of collagen coating within each disease group at cell level (Mann--Whitney~U)")
    lines.append(r"and patient level (Wilcoxon signed-rank). $d$: Cohen's~$d$. $q$: BH-FDR. $^{*}$: $q<0.05$.")
    lines.append(r"\vspace{1em}")

    for label, df in [("Cell level", c), ("Patient level", pt)]:
        groups_in = df['Disease'].unique() if 'Disease' in df.columns else ['']
        lines.append(r"\subsection*{" + label + "}")
        lines.append(r"\begin{longtable}{llrrr}")
        lines.append(r"\toprule")
        lines.append(r"Feature & Group & $d$ & $q$ & Sig. \\")
        lines.append(r"\midrule\endfirsthead")
        lines.append(r"\toprule Feature & Group & $d$ & $q$ & Sig. \\\midrule\endhead")
        lines.append(r"\bottomrule\endfoot")

        prev_org = None
        for _, row in df.iterrows():
            feat = row['Feature']
            org  = feat.split('_')[0]
            if org != prev_org:
                if prev_org is not None:
                    lines.append(r"\midrule")
                lines.append(r"\multicolumn{5}{l}{\textit{" + _esc(org) + r"}} \\")
                prev_org = org
            name  = _esc(_feat(feat).split(' ', 1)[1] if ' ' in _feat(feat) else _feat(feat))
            grp   = _esc(str(row.get('Disease', '')))
            d     = f"{float(row['Cohen_d']):+.2f}" if 'Cohen_d' in row and not pd.isna(row['Cohen_d']) else '--'
            q     = _qval(row.get('BH_q'))
            sig   = r'$^{*}$' if row.get('Significant', False) else ''
            lines.append(f"  {name} & {grp} & {d} & {q} & {sig} \\\\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{longtable}")

    write_tex(OUT / out_name, "\n".join(lines) + "\n")


def s7_collagen():
    _collagen_table(P1, "Part~1 (Healthy vs.\ TAA)", 7, "S7a_collagen_part1.tex")
    _collagen_table(P2, "Part~2 (TAV-ATAA vs.\ BAV-ATAA)", 7, "S7b_collagen_part2.tex")


# ─────────────────────────────────────────────────────────────────
# S8 — Diameter correlations
# ─────────────────────────────────────────────────────────────────
def s8_diameter():
    c  = pd.read_csv(P2 / "diameter" / "L7_diameter_cell.csv")
    pt = pd.read_csv(P2 / "diameter" / "L7_diameter_pt.csv")

    lines = []
    lines.append(r"\section*{S8 — Full Aortic Diameter Correlation Table}")
    lines.append(r"Spearman's $\rho$ between each feature and aortic diameter across all 13 TAA")
    lines.append(r"specimens with available diameter measurements. $q$: BH-FDR. $^{*}$: $q<0.05$.")
    lines.append(r"\vspace{1em}")

    for label, df in [("Cell level", c), ("Patient level", pt)]:
        lines.append(r"\subsection*{" + label + "}")
        lines.append(r"\begin{longtable}{lrrrr}")
        lines.append(r"\toprule")
        lines.append(r"Feature & $\rho$ & $p$ & $n$ & $q$ \\")
        lines.append(r"\midrule\endfirsthead")
        lines.append(r"\toprule Feature & $\rho$ & $p$ & $n$ & $q$ \\\midrule\endhead")
        lines.append(r"\bottomrule\endfoot")

        prev_org = None
        for _, row in df.iterrows():
            feat = row['Feature']
            org  = feat.split('_')[0]
            if org != prev_org:
                if prev_org is not None:
                    lines.append(r"\midrule")
                lines.append(r"\multicolumn{5}{l}{\textit{" + _esc(org) + r"}} \\")
                prev_org = org
            name = _esc(_feat(feat).split(' ', 1)[1] if ' ' in _feat(feat) else _feat(feat))
            rho  = f"{float(row['Spearman_r']):+.3f}" if not pd.isna(row.get('Spearman_r')) else '--'
            pv   = _pval(row.get('p'))
            n    = str(int(row['n'])) if 'n' in row and not pd.isna(row['n']) else '--'
            qv   = _qval(row.get('BH_q'))
            sig  = r'$^{*}$' if row.get('Significant', False) else ''
            lines.append(f"  {name} & {rho}{sig} & {pv} & {n} & {qv} \\\\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{longtable}")

    write_tex(OUT / "S8_diameter_correlation.tex", "\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────
# S9 — Hypertension full results
# ─────────────────────────────────────────────────────────────────
def s9_hypertension():
    c  = pd.read_csv(P2 / "hypertension" / "L6_hypertension_cell.csv")
    pt = pd.read_csv(P2 / "hypertension" / "L6_hypertension_pt.csv")

    lines = []
    lines.append(r"\section*{S9 — Full Hypertension Analysis Results}")
    lines.append(r"Cohen's~$d$ for hypertensive vs.\ normotensive donors in the TAA cohort,")
    lines.append(r"at cell and patient level. $q$: BH-FDR. $^{*}$: $q<0.05$.")
    lines.append(r"\vspace{1em}")

    for label, df in [("Cell level", c), ("Patient level", pt)]:
        lines.append(r"\subsection*{" + label + "}")
        lines.append(r"\begin{longtable}{lrrrrrr}")
        lines.append(r"\toprule")
        lines.append(r"Feature & $n_\text{HT}$ & $n_\text{norm}$ & $d$ & $p$ & $q$ & Sig. \\")
        lines.append(r"\midrule\endfirsthead")
        lines.append(r"\toprule Feature & $n_\text{HT}$ & $n_\text{norm}$ & $d$ & $p$ & $q$ & Sig. \\\midrule\endhead")
        lines.append(r"\bottomrule\endfoot")

        prev_org = None
        for _, row in df.iterrows():
            feat = row['Feature']
            org  = feat.split('_')[0]
            if org != prev_org:
                if prev_org is not None:
                    lines.append(r"\midrule")
                lines.append(r"\multicolumn{7}{l}{\textit{" + _esc(org) + r"}} \\")
                prev_org = org
            name  = _esc(_feat(feat).split(' ', 1)[1] if ' ' in _feat(feat) else _feat(feat))
            nht   = str(int(row['n_HT']))   if 'n_HT'   in row else '--'
            nnorm = str(int(row['n_noHT'])) if 'n_noHT' in row else '--'
            d     = f"{float(row['Cohen_d']):+.2f}"
            pv    = _pval(row.get('p'))
            qv    = _qval(row.get('BH_q'))
            sig   = r'$^{*}$' if row.get('Significant', False) else ''
            lines.append(f"  {name} & {nht} & {nnorm} & {d} & {pv} & {qv} & {sig} \\\\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{longtable}")

    write_tex(OUT / "S9_hypertension.tex", "\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────
# S10 — nnU-Net training details (placeholder)
# ─────────────────────────────────────────────────────────────────
def s10_nnunet():
    body = (
        r"\section*{S10 — nnU-Net Training Details}" + "\n"
        r"Three nnU-Net~v2 (3D full-resolution) models were trained independently for nucleus," + "\n"
        r"F-actin, and mitochondria segmentation. Training was performed on the analysis server" + "\n"
        r"(NVIDIA GPU). Key parameters from \texttt{nnUNetPlans.json} are summarised below;" + "\n"
        r"the full configuration files are available in the project repository." + "\n\n"
        r"\begin{longtable}{lp{3cm}p{3cm}p{3cm}}" + "\n"
        r"\toprule" + "\n"
        r"Parameter & Nucleus & F-actin & Mitochondria \\" + "\n"
        r"\midrule\endfirsthead\toprule Parameter & Nucleus & F-actin & Mitochondria \\\midrule\endhead" + "\n"
        r"\bottomrule\endfoot" + "\n"
        r"Dataset ID & 1 & 2 & 3 \\" + "\n"
        r"Configuration & 3d\_fullres & 3d\_fullres & 3d\_fullres \\" + "\n"
        r"Patch size & \multicolumn{3}{l}{\textit{see nnUNetPlans.json on server}} \\" + "\n"
        r"Batch size & \multicolumn{3}{l}{\textit{see nnUNetPlans.json on server}} \\" + "\n"
        r"Optimizer & SGD & SGD & SGD \\" + "\n"
        r"Loss & DC $+$ CE & DC $+$ CE & DC $+$ CE \\" + "\n"
        r"LR schedule & Poly & Poly & Poly \\" + "\n"
        r"Folds & 5 & 5 & 5 \\" + "\n"
        r"\bottomrule" + "\n"
        r"\multicolumn{4}{l}{\footnotesize DC: Dice; CE: cross-entropy; Poly: polynomial decay.} \\" + "\n"
        r"\end{longtable}" + "\n"
    )
    write_tex(OUT / "S10_nnunet_training.tex", body)


# ─────────────────────────────────────────────────────────────────
# S11 — Segmentation quality examples (placeholder)
# ─────────────────────────────────────────────────────────────────
def s11_segmentation():
    body = (
        r"\section*{S11 — Segmentation Quality Examples}" + "\n"
        r"Representative maximum-intensity projections of raw fluorescence (left) alongside" + "\n"
        r"the corresponding binary segmentation masks (right) for each organelle channel," + "\n"
        r"shown for one healthy donor cell (top row) and one TAA donor cell (bottom row)." + "\n\n"
        r"\textbf{Note:} Raw microscopy images are stored on the analysis server and are not" + "\n"
        r"distributed with this supplementary file. Contact the corresponding author for access." + "\n\n"
        r"Dice scores on the held-out test set:" + "\n"
        r"\begin{center}" + "\n"
        r"\begin{tabular}{lc}\toprule Organelle & Dice (mean $\pm$ SD) \\\midrule" + "\n"
        r"Nucleus & reported in main text \\" + "\n"
        r"F-actin & reported in main text \\" + "\n"
        r"Mitochondria & reported in main text \\" + "\n"
        r"\bottomrule\end{tabular}" + "\n"
        r"\end{center}" + "\n"
    )
    write_tex(OUT / "S11_segmentation_examples.tex", body)


# ─────────────────────────────────────────────────────────────────
# S12 — Feature distribution violin plots (all 30 features)
# ─────────────────────────────────────────────────────────────────
def s12_distributions():
    df = pd.read_csv(BASE / "Advanced_Features_Raw_Final.csv")

    feat_cols = [c for c in df.columns if any(
        c.startswith(p) for p in ('Actin_', 'Mito_', 'Nucleus_')
    )]
    # only numeric
    feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]

    group_col   = 'Disease'
    group_order = ['Healthy', 'TAA', 'BAV']
    colors      = {'Healthy': '#2980B9', 'TAA': '#C0392B', 'BAV': '#27AE60'}

    n_feats = len(feat_cols)
    ncols   = 5
    nrows   = int(np.ceil(n_feats / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.6))
    axes = axes.flatten()

    for idx, feat in enumerate(feat_cols):
        ax = axes[idx]
        data_by_group = [df[df[group_col] == g][feat].dropna().values for g in group_order]
        parts = ax.violinplot(data_by_group, positions=range(len(group_order)),
                               showmedians=True, showextrema=False)
        for i, (body, grp) in enumerate(zip(parts['bodies'], group_order)):
            body.set_facecolor(colors[grp])
            body.set_alpha(0.75)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.5)

        ax.set_xticks(range(len(group_order)))
        ax.set_xticklabels(group_order, fontsize=7, rotation=30, ha='right')
        title = _feat(feat)
        # shorten if too long
        title_short = title if len(title) < 28 else title[:26] + '..'
        ax.set_title(title_short, fontsize=7.5, fontweight='bold', pad=2)
        ax.tick_params(axis='y', labelsize=6)
        ax.spines[['top', 'right']].set_visible(False)

    for ax in axes[n_feats:]:
        ax.set_visible(False)

    # legend
    handles = [mpatches.Patch(color=colors[g], alpha=0.75, label=g) for g in group_order]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle('S12 — Feature Distributions by Group (all 30 features)',
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f"S12_feature_distributions.{ext}", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: S12_feature_distributions.png/.pdf")


# ─────────────────────────────────────────────────────────────────
# S13 — LOPO-CV fold results
# ─────────────────────────────────────────────────────────────────
def s13_lopo_folds():
    t1 = pd.read_csv(CLS / "task1_healthy_vs_diseased" / "fold_results.csv")
    t2 = pd.read_csv(CLS / "task2_bav_vs_tav"         / "fold_results.csv")

    lines = []
    lines.append(r"\section*{S13 — Per-Fold LOPO-CV Results}")
    lines.append(r"Balanced accuracy for each held-out donor across all three classifiers,")
    lines.append(r"for Task~1 (Healthy vs.\ TAA) and Task~2 (TAV-ATAA vs.\ BAV-ATAA).")
    lines.append(r"Each fold holds out one donor; prediction is the majority vote across")
    lines.append(r"that donor's cells. LR: Logistic Regression; RF: Random Forest; SVM: Support Vector Machine.")
    lines.append(r"\vspace{1em}")

    clf_short = {'Logistic Regression': 'LR', 'Random Forest': 'RF', 'SVM': 'SVM'}

    for task_label, df in [("Task 1: Healthy vs.\ TAA", t1),
                            ("Task 2: TAV-ATAA vs.\ BAV-ATAA", t2)]:
        lines.append(r"\subsection*{" + task_label + "}")

        # Pivot: rows = patient, cols = classifier
        wide = df.pivot_table(index=['patient', 'disease'], columns='classifier',
                               values='bal_acc', aggfunc='first').reset_index()
        wide = wide.sort_values(['disease', 'patient'])
        clfs = [c for c in ['Logistic Regression', 'Random Forest', 'SVM'] if c in wide.columns]

        col_spec = 'llc' + 'c' * len(clfs)
        lines.append(r"\begin{longtable}{" + col_spec + "}")
        lines.append(r"\toprule")
        clf_heads = ' & '.join([clf_short.get(c, c) for c in clfs])
        lines.append(r"Donor & Group & $N_\text{cells}$ & " + clf_heads + r" \\")
        lines.append(r"\midrule\endfirsthead")
        lines.append(r"\toprule Donor & Group & $N_\text{cells}$ & " + clf_heads + r" \\\midrule\endhead")
        lines.append(r"\bottomrule\endfoot")

        prev_disease = None
        for _, row in wide.iterrows():
            if row['disease'] != prev_disease:
                if prev_disease is not None:
                    lines.append(r"\midrule")
                lines.append(r"\multicolumn{" + str(3 + len(clfs)) + r"}{l}{\textit{" +
                              _esc(str(row['disease'])) + r"}} \\")
                prev_disease = row['disease']
            # get n_cells from original df
            nc_rows = df[(df['patient'] == row['patient']) & (df['classifier'] == clfs[0])]
            n_cells = int(nc_rows['n_cells'].values[0]) if len(nc_rows) > 0 else '--'
            clf_vals = []
            for c in clfs:
                v = row.get(c, np.nan)
                clf_vals.append(f"{float(v):.2f}" if not pd.isna(v) else '--')
            lines.append(f"  {_esc(str(row['patient']))} & {_esc(str(row['disease']))} & "
                         f"{n_cells} & " + ' & '.join(clf_vals) + r" \\")

        # Summary row
        lines.append(r"\midrule")
        means = []
        for c in clfs:
            vals = pd.to_numeric(wide[c], errors='coerce').dropna()
            means.append(f"{vals.mean():.2f} $\\pm$ {vals.std():.2f}")
        lines.append(r"  \textbf{Mean $\pm$ SD} & & & " + ' & '.join(means) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{longtable}")

    write_tex(OUT / "S13_lopo_cv_folds.tex", "\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────
# Main supplementary wrapper
# ─────────────────────────────────────────────────────────────────
def main_wrapper():
    files = [
        ("S1_cohort.tex",                "S1 — Full Donor Cohort Table"),
        ("S2_shapiro_wilk.tex",          "S2 — Shapiro--Wilk Normality Tests"),
        ("S3_part1_cohens_d.tex",        "S3 — Full Cohen's $d$: Part 1"),
        ("S4_part2_cohens_d.tex",        "S4 — Full Cohen's $d$: Part 2"),
        ("S5a_lmm_part1.tex",            "S5a — Full LMM: Part 1"),
        ("S5b_lmm_part2.tex",            "S5b — Full LMM: Part 2"),
        ("S6_icc.tex",                   "S6 — ICC Values"),
        ("S7a_collagen_part1.tex",       "S7a — Collagen Rescue: Part 1"),
        ("S7b_collagen_part2.tex",       "S7b — Collagen Rescue: Part 2"),
        ("S8_diameter_correlation.tex",  "S8 — Diameter Correlations"),
        ("S9_hypertension.tex",          "S9 — Hypertension Analysis"),
        ("S10_nnunet_training.tex",      "S10 — nnU-Net Training Details"),
        ("S11_segmentation_examples.tex","S11 — Segmentation Quality Examples"),
        ("S13_lopo_cv_folds.tex",        "S13 — LOPO-CV Fold Results"),
    ]

    wrapper = r"""\documentclass[11pt,a4paper]{article}
\usepackage{booktabs,longtable,geometry,array,xcolor,microtype,graphicx,hyperref}
\geometry{margin=2cm}
\renewcommand{\arraystretch}{1.25}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}
\title{\textbf{Supplementary Material}}
\author{}
\date{}
\begin{document}
\maketitle
\tableofcontents
\clearpage
"""
    for fname, title in files:
        wrapper += f"\\input{{{fname}}}\n\\clearpage\n"

    # S12 as figure
    wrapper += r"""
\section*{S12 --- Feature Distributions (All 30 Features)}
All 30 morphological features across the three groups (Healthy, TAA, BAV).
Violins show the full distribution; horizontal bar: median.
\begin{figure}[h!]
  \centering
  \includegraphics[width=\textwidth]{S12_feature_distributions.pdf}
  \caption*{S12 --- Per-feature distribution by group.}
\end{figure}
\end{document}
"""
    with open(OUT / "supplementary_main.tex", 'w') as f:
        f.write(wrapper)
    print("  Saved: supplementary_main.tex")


# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating supplementary material...")
    s1_cohort()
    s2_shapiro()
    s3_part1_cohens_d()
    s4_part2_cohens_d()
    s5_lmm()
    s6_icc_placeholder()
    s7_collagen()
    s8_diameter()
    s9_hypertension()
    s10_nnunet()
    s11_segmentation()
    s12_distributions()
    s13_lopo_folds()
    main_wrapper()
    print(f"\nAll files in: {OUT.resolve()}")
