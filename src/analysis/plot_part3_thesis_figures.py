"""
Part 3 — 6 publication-quality thesis figures.

Figure 1  — Performance overview bar chart (bal. acc ± SD, all classifiers × tasks)
Figure 2  — ROC curves (2 panels: Task 1 | Task 2)
Figure 3  — Normalised confusion matrices (3 clf × 2 tasks grid)
Figure 4  — Per-class Precision / Recall / F1 bar chart
Figure 5  — Per-patient fold balanced accuracy scatter
Figure 6  — Feature importance 2×2 (LR + RF, Task 1 + Task 2)

Output: outputs/final_thesis_part3/figures/
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.svm           import SVC
from sklearn.calibration   import CalibratedClassifierCV
from sklearn.metrics       import (balanced_accuracy_score, roc_auc_score,
                                   confusion_matrix,
                                   precision_recall_fscore_support,
                                   roc_curve)

warnings.filterwarnings('ignore')

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import shared_stats as ss

OUT = PROJ / 'outputs' / 'final_thesis_part3' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────────
ORG_COLORS = {
    'Nucleus':      '#1558A0',
    'Actin':        '#2ECC71',
    'Mitochondria': '#E74C3C',
}
CLF_COLORS = {
    'LR':  '#8E44AD',
    'RF':  '#27AE60',
    'SVM': '#E67E22',
}
TASK_COLORS = {
    'Task 1': '#2C3E50',
    'Task 2': '#16A085',
}
DIS_COLORS = {
    'Healthy':  '#4C9BE8',
    'TAA':      '#E74C3C',
    'BAV':      '#F39C12',
}
DIS_MARKERS = {'Healthy': 'o', 'TAA': 's', 'BAV': '^'}

RC = {
    'font.family':        'Arial',
    'font.sans-serif':    ['Arial'],
    'font.size':          10,
    'axes.labelsize':     10,
    'axes.titlesize':     11,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'axes.linewidth':     0.8,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'xtick.major.size':   3.5,
    'ytick.major.size':   3.5,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'savefig.dpi':        300,
    'text.usetex':        False,
}

CLF_KEYS  = ['Logistic Regression', 'Random Forest', 'SVM']
CLF_SHORT = {'Logistic Regression': 'LR', 'Random Forest': 'RF', 'SVM': 'SVM'}


def rc():    plt.rcParams.update(RC)
def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
def ygrid(ax):
    ax.yaxis.grid(True, color='#E5E5E5', lw=0.6, zorder=0)
    ax.set_axisbelow(True)
def xgrid(ax):
    ax.xaxis.grid(True, color='#E5E5E5', lw=0.6, zorder=0)
    ax.set_axisbelow(True)
def save(fig, name):
    png = OUT / f'{name}.png'
    pdf = OUT / f'{name}.pdf'
    fig.savefig(png, dpi=300, bbox_inches='tight', pad_inches=0.06)
    fig.savefig(pdf, dpi=300, bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)
    print(f'  Saved: {name}.png/.pdf')


def get_organelle(feat):
    fl = feat.lower()
    if 'nucleus' in fl: return 'Nucleus'
    if 'actin'   in fl: return 'Actin'
    if 'mito'    in fl: return 'Mitochondria'
    return 'Other'


def get_classifiers():
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=2000, class_weight='balanced', C=0.1,
            solver='lbfgs', random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, class_weight='balanced', max_depth=6,
            min_samples_leaf=3, random_state=42, n_jobs=-1),
        'SVM': CalibratedClassifierCV(
            SVC(kernel='rbf', class_weight='balanced', C=1.0,
                gamma='scale', random_state=42), cv=3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# LOPO-CV
# ══════════════════════════════════════════════════════════════════════════════
def run_lopo(df, feat_cols, class_names, label_col):
    patients      = sorted(df['Subject'].unique())
    fold_results  = defaultdict(list)
    importances   = defaultdict(list)

    for held_out in patients:
        tr = df['Subject'] != held_out
        te = df['Subject'] == held_out

        X_tr = df.loc[tr, feat_cols].values
        y_tr = df.loc[tr, label_col].values
        X_te = df.loc[te, feat_cols].values
        y_te = df.loc[te, label_col].values

        if len(np.unique(y_tr)) < 2:
            continue

        sc   = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        disease = df.loc[te, 'Disease'].iloc[0]

        for cname, clf in get_classifiers().items():
            try:
                clf.fit(X_tr, y_tr)
                y_pred     = clf.predict(X_te)
                y_prob_pos = None
                if hasattr(clf, 'predict_proba'):
                    try:
                        proba      = clf.predict_proba(X_te)
                        pos_idx    = list(clf.classes_).index(class_names[1])
                        y_prob_pos = proba[:, pos_idx]
                    except Exception:
                        pass

                bal = balanced_accuracy_score(y_te, y_pred)
                auc = np.nan
                if y_prob_pos is not None:
                    try:
                        auc = roc_auc_score(
                            (y_te == class_names[1]).astype(int), y_prob_pos)
                    except Exception:
                        pass

                fold_results[cname].append({
                    'patient': held_out, 'disease': disease,
                    'bal_acc': bal, 'auc': auc,
                    'y_true': y_te.tolist(), 'y_pred': y_pred.tolist(),
                    'y_prob': y_prob_pos.tolist() if y_prob_pos is not None else None,
                })

                if cname == 'Random Forest':
                    importances[cname].append(clf.feature_importances_)
                elif cname == 'Logistic Regression':
                    importances[cname].append(clf.coef_[0])
            except Exception as e:
                print(f'  {held_out} {cname}: {e}')

    summary = {}
    for cname, folds in fold_results.items():
        if not folds:
            continue
        bals = [f['bal_acc'] for f in folds]
        aucs = [f['auc']     for f in folds if not np.isnan(f['auc'])]
        yt   = np.concatenate([f['y_true'] for f in folds])
        yp   = np.concatenate([f['y_pred'] for f in folds])
        cm   = confusion_matrix(yt, yp, labels=class_names)
        p, r, f1, _ = precision_recall_fscore_support(
            yt, yp, labels=class_names, zero_division=0)

        fwp   = [f for f in folds if f['y_prob'] is not None]
        y_prob_all, y_bin_all = None, None
        if fwp:
            y_prob_all = np.concatenate([f['y_prob'] for f in fwp])
            yt2        = np.concatenate([f['y_true'] for f in fwp])
            y_bin_all  = (np.array(yt2) == class_names[1]).astype(int)

        summary[cname] = {
            'folds':      folds,
            'bal_mean':   np.mean(bals),
            'bal_std':    np.std(bals),
            'auc_mean':   np.mean(aucs) if aucs else np.nan,
            'auc_std':    np.std(aucs)  if aucs else np.nan,
            'cm':         cm,
            'precision':  dict(zip(class_names, p)),
            'recall':     dict(zip(class_names, r)),
            'f1':         dict(zip(class_names, f1)),
            'class_names': class_names,
            'y_prob_all': y_prob_all,
            'y_bin_all':  y_bin_all,
        }
    return summary, importances


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Performance overview bar chart
# ══════════════════════════════════════════════════════════════════════════════
def fig1_performance(s1, s2):
    rc()
    clfs   = CLF_KEYS
    labels = [CLF_SHORT[c] for c in clfs]
    x      = np.arange(len(clfs))
    w      = 0.30

    t1_m = [s1[c]['bal_mean'] for c in clfs]
    t1_s = [s1[c]['bal_std']  for c in clfs]
    t2_m = [s2[c]['bal_mean'] for c in clfs]
    t2_s = [s2[c]['bal_std']  for c in clfs]

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    eb      = dict(elinewidth=0.9, capthick=0.9, capsize=4, ecolor='#333333')

    bars1 = ax.bar(x - w/2, t1_m, w, yerr=t1_s, error_kw=eb,
                   color='#5B6FA6', alpha=0.88, edgecolor='white', lw=0,
                   label='Task 1: Healthy vs Diseased')
    bars2 = ax.bar(x + w/2, t2_m, w, yerr=t2_s, error_kw=eb,
                   color='#2ECC71', alpha=0.88, edgecolor='white', lw=0,
                   label='Task 2: TAV-ATAA vs BAV-ATAA')

    ax.axhline(0.5, color='#C0392B', ls='--', lw=1.0, alpha=0.85,
               label='Chance level (0.5)')
    ax.set_ylim(0, 1.25)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Balanced accuracy (mean ± SD, LOPO folds)', fontsize=10)
    ax.set_xlabel('Classifier', fontsize=10)
    ax.set_title('Classification performance — Task 1 vs Task 2',
                 fontsize=12, fontweight='bold', pad=8)

    for bars, ms, ss_ in [(bars1, t1_m, t1_s), (bars2, t2_m, t2_s)]:
        for bar, val, std in zip(bars, ms, ss_):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (std or 0) + 0.028,
                        f'{val:.2f}', ha='center', va='bottom',
                        fontsize=8.5, fontweight='bold')

    ax.legend(frameon=False, loc='upper center', fontsize=9,
              bbox_to_anchor=(0.5, -0.16), ncol=3)
    despine(ax)
    ygrid(ax)
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    save(fig, 'fig1_performance_overview')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — ROC curves (2 panels)
# ══════════════════════════════════════════════════════════════════════════════
def fig2_roc(s1, s2):
    rc()
    task_pairs = [
        ('Task 1: Healthy vs Diseased', s1),
        ('Task 2: TAV-ATAA vs BAV-ATAA', s2),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0))

    for ax, (task_label, s) in zip(axes, task_pairs):
        ax.plot([0, 1], [0, 1], color='#AAAAAA', ls='--', lw=0.9,
                label='Chance  (AUC = 0.50)')
        for cname in CLF_KEYS:
            sv = s.get(cname)
            if sv is None or sv['y_prob_all'] is None:
                continue
            try:
                fpr, tpr, _ = roc_curve(sv['y_bin_all'], sv['y_prob_all'])
                auc_val      = roc_auc_score(sv['y_bin_all'], sv['y_prob_all'])
                short        = CLF_SHORT[cname]
                ax.plot(fpr, tpr, color=CLF_COLORS[short], lw=2.0,
                        label=f'{short}  (AUC = {auc_val:.2f})')
            except Exception:
                pass

        ax.set_xlabel('False positive rate', fontsize=10)
        ax.set_ylabel('True positive rate', fontsize=10)
        ax.set_title(task_label, fontsize=11, fontweight='bold', pad=6)
        ax.legend(frameon=False, loc='lower right', fontsize=9)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.04)
        despine(ax)

    plt.tight_layout(w_pad=3.0)
    save(fig, 'fig2_roc_curves')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Confusion matrices (3 clf × 2 tasks)
# ══════════════════════════════════════════════════════════════════════════════
def fig3_confusion(s1, s2):
    rc()
    from matplotlib.colors import LinearSegmentedColormap

    # Natural slate-teal colormap: white → muted steel blue (not the default Blues)
    CM_CMAP = LinearSegmentedColormap.from_list(
        'cm_natural', ['#f7f7f7', '#c6d8e8', '#5a8db5', '#1e4d72'], N=256
    )

    task_pairs = [
        ('Task 1\nHealthy vs Diseased', s1),
        ('Task 2\nTAV-ATAA vs BAV-ATAA', s2),
    ]

    fig, axes = plt.subplots(len(CLF_KEYS), 2,
                             figsize=(6.5, len(CLF_KEYS) * 2.7))
    plt.subplots_adjust(hspace=0.45, wspace=0.35)

    for col, (task_label, s) in enumerate(task_pairs):
        for row, cname in enumerate(CLF_KEYS):
            ax = axes[row, col]
            sv = s.get(cname)
            if sv is None:
                ax.axis('off')
                continue

            cm       = sv['cm'].astype(float)
            cn       = sv['class_names']
            cm_norm  = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))

            im = ax.imshow(cm_norm, cmap=CM_CMAP, vmin=0, vmax=1, aspect='auto')

            for i in range(len(cn)):
                for j in range(len(cn)):
                    val = cm_norm[i, j]
                    raw = int(cm[i, j])
                    ax.text(j, i, f'{val:.2f}\n({raw})',
                            ha='center', va='center', fontsize=9,
                            fontfamily='Arial',
                            color='white' if val > 0.52 else '#2d2d2d',
                            fontweight='bold' if val > 0.52 else 'normal')

            ax.set_xticks(range(len(cn)))
            ax.set_yticks(range(len(cn)))
            short_cn = [c.replace('TAV-ATAA', 'TAV').replace('BAV-ATAA', 'BAV')
                        for c in cn]
            ax.set_xticklabels(short_cn, fontsize=8.5, fontfamily='Arial')
            ax.set_yticklabels(short_cn, fontsize=8.5, fontfamily='Arial')

            # Thin border, no extra spine clutter
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
                spine.set_edgecolor('#aaaaaa')

            if row == len(CLF_KEYS) - 1:
                ax.set_xlabel('Predicted', fontsize=9, fontfamily='Arial',
                              labelpad=4, color='#333333')
            if col == 0:
                ax.set_ylabel('True', fontsize=9, fontfamily='Arial',
                              labelpad=4, color='#333333')

            bal = sv['bal_mean']
            ax.set_title(f'{CLF_SHORT[cname]} — Bal. Acc = {bal:.2f}',
                         fontsize=9.5, fontweight='bold', pad=4,
                         fontfamily='Arial', color='#222222')

            if row == 0:
                ax.text(0.5, 1.30, task_label, transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=10,
                        fontweight='bold', fontfamily='Arial', color='#222222')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.018, 0.70])
    sm = plt.cm.ScalarMappable(cmap=CM_CMAP, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Recall (row-normalised)', fontsize=9, fontfamily='Arial',
                   color='#333333')
    cbar.ax.tick_params(labelsize=8, colors='#333333')
    cbar.outline.set_linewidth(0.5)

    save(fig, 'fig3_confusion_matrices')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Per-class P / R / F1
# ══════════════════════════════════════════════════════════════════════════════
def fig4_per_class(s1, s2):
    rc()
    task_pairs = [
        ('Task 1: Healthy vs Diseased', s1),
        ('Task 2: TAV-ATAA vs BAV-ATAA', s2),
    ]
    metrics = ['precision', 'recall', 'f1']
    mlabels = {'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1-score'}
    mcolors = {'precision': '#2980B9', 'recall': '#C0392B', 'f1': '#27AE60'}

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.5))
    plt.subplots_adjust(hspace=0.55, wspace=0.38)

    for row, (task_label, s) in enumerate(task_pairs):
        class_names = s[CLF_KEYS[0]]['class_names']
        for col, cls in enumerate(class_names):
            ax  = axes[row, col]
            x   = np.arange(len(CLF_KEYS))
            w   = 0.22

            for i, metric in enumerate(metrics):
                vals = [s[c][metric][cls] for c in CLF_KEYS]
                ax.bar(x + (i - 1) * w, vals, w,
                       color=mcolors[metric], alpha=0.85,
                       edgecolor='white', lw=0,
                       label=mlabels[metric] if (row == 0 and col == 0) else '_')

            ax.set_xticks(x)
            ax.set_xticklabels([CLF_SHORT[c] for c in CLF_KEYS], fontsize=9.5)
            ax.set_ylim(0, 1.18)
            ax.axhline(0.5, color='#AAAAAA', ls=':', lw=0.8)
            ax.set_title(f'{cls}', fontsize=10.5, fontweight='bold', pad=4)
            if col == 0:
                ax.set_ylabel('Score', fontsize=10)
            if row == 0 and col == 0:
                ax.legend(frameon=False, loc='lower left', fontsize=8.5)
            despine(ax)
            ygrid(ax)

        # task label on left
        axes[row, 0].annotate(task_label,
            xy=(-0.32, 0.5), xycoords='axes fraction',
            fontsize=9.5, fontweight='bold', va='center', ha='right',
            rotation=90, color='#2C3E50')

    plt.tight_layout()
    save(fig, 'fig4_per_class_metrics')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Per-patient fold balanced accuracy scatter
# ══════════════════════════════════════════════════════════════════════════════
def fig5_per_fold(s1, s2):
    rc()
    task_pairs = [
        ('Task 1: Healthy vs Diseased', s1),
        ('Task 2: TAV-ATAA vs BAV-ATAA', s2),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.5))
    rng = np.random.RandomState(42)

    for ax, (task_label, s) in zip(axes, task_pairs):
        # collect all folds across classifiers
        all_rows = []
        for cname in CLF_KEYS:
            sv = s.get(cname)
            if sv is None:
                continue
            for f in sv['folds']:
                all_rows.append({
                    'clf':     CLF_SHORT[cname],
                    'patient': f['patient'],
                    'disease': f['disease'],
                    'bal_acc': f['bal_acc'],
                })
        df_plot = pd.DataFrame(all_rows)
        if df_plot.empty:
            ax.axis('off')
            continue

        # order patients by disease then name
        dis_order = {'Healthy': 0, 'TAA': 1, 'BAV': 2}
        pt_info   = (df_plot[['patient','disease']].drop_duplicates()
                     .assign(dis_rank=lambda d: d['disease'].map(dis_order))
                     .sort_values(['dis_rank','patient']))
        pt_order  = pt_info['patient'].tolist()
        pt_x      = {p: i for i, p in enumerate(pt_order)}
        pt_dis    = dict(zip(pt_info['patient'], pt_info['disease']))

        clf_offsets = {c: (i - 1) * 0.12
                       for i, c in enumerate(CLF_KEYS)}

        for cname in CLF_KEYS:
            short = CLF_SHORT[cname]
            sub   = df_plot[df_plot['clf'] == short]
            for _, row in sub.iterrows():
                pt   = row['patient']
                dis  = pt_dis.get(pt, row['disease'])
                xpos = pt_x[pt] + clf_offsets.get(cname, 0)
                jit  = rng.uniform(-0.04, 0.04)
                ax.scatter(xpos + jit, row['bal_acc'],
                           c=DIS_COLORS.get(dis, '#888888'),
                           marker=DIS_MARKERS.get(dis, 'o'),
                           s=38, alpha=0.80,
                           edgecolors='white', linewidths=0.4, zorder=3)

        # per-patient mean (across classifiers)
        for pt in pt_order:
            sub_pt   = df_plot[df_plot['patient'] == pt]
            pt_mean  = sub_pt['bal_acc'].mean()
            xp       = pt_x[pt]
            ax.plot([xp - 0.22, xp + 0.22], [pt_mean, pt_mean],
                    color='#222222', lw=1.5, zorder=4)

        overall_mean = df_plot['bal_acc'].mean()
        ax.axhline(0.5, color='#C0392B', ls='--', lw=1.0, alpha=0.85,
                   label='Chance (0.5)')
        ax.axhline(overall_mean, color='#222222', ls='-', lw=1.2, alpha=0.5,
                   label=f'Overall mean ({overall_mean:.2f})')

        ax.set_xticks(range(len(pt_order)))
        short_labels = [p.split('-')[0] + '-' + p.split('-')[1]
                        if '-' in p else p for p in pt_order]
        ax.set_xticklabels(short_labels, rotation=45, ha='right',
                           fontsize=7.5)
        ax.set_ylim(-0.05, 1.18)
        ax.set_ylabel('Balanced accuracy (per LOPO fold)', fontsize=10)
        ax.set_xlabel('Patient', fontsize=10)
        ax.set_title(task_label, fontsize=11, fontweight='bold', pad=6)

        # legend: disease groups
        dis_labels = df_plot['disease'].unique().tolist()
        handles = [plt.scatter([], [], c=DIS_COLORS.get(d,'#888'),
                               marker=DIS_MARKERS.get(d,'o'), s=32,
                               label=d, edgecolors='white')
                   for d in ['Healthy','TAA','BAV'] if d in dis_labels]
        handles += [plt.Line2D([0],[0], color='#C0392B', ls='--', lw=1.0,
                               label='Chance (0.5)'),
                    plt.Line2D([0],[0], color='#222222', ls='-', lw=1.2,
                               alpha=0.5, label=f'Mean ({overall_mean:.2f})')]
        # clf marker legend
        for cname in CLF_KEYS:
            handles.append(mpatches.Patch(facecolor='#CCCCCC', alpha=0.9,
                                          label=f'{CLF_SHORT[cname]} (offset)'))
        ax.legend(handles=handles, frameon=False, loc='upper right',
                  fontsize=8.0, handletextpad=0.4)
        despine(ax)
        ygrid(ax)

    plt.tight_layout(w_pad=3.0)
    save(fig, 'fig5_per_fold_scatter')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Feature importance 2×2 (LR + RF  ×  Task 1 + Task 2)
# ══════════════════════════════════════════════════════════════════════════════
def fig6_importances(feat_cols, imp1, imp2, top_n=15):
    rc()
    tasks = [
        ('Task 1: Healthy vs Diseased',    imp1),
        ('Task 2: TAV-ATAA vs BAV-ATAA',   imp2),
    ]
    clfs_imp = [
        ('Logistic Regression', 'LR coefficients\n(+ = Diseased / BAV-ATAA direction)',
         True),
        ('Random Forest',       'RF feature importances\n(mean impurity decrease)',
         False),
    ]

    fig = plt.figure(figsize=(13.0, 11.0))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.55, wspace=0.55,
                            left=0.25, right=0.97,
                            top=0.94, bottom=0.04)

    panel_letters = [['A', 'B'], ['C', 'D']]

    for row, (clf_key, clf_label, signed) in enumerate(clfs_imp):
        for col, (task_label, imp) in enumerate(tasks):
            ax = fig.add_subplot(gs[row, col])

            if clf_key not in imp or len(imp[clf_key]) == 0:
                ax.axis('off')
                continue

            arr  = np.array(imp[clf_key])
            vals = np.mean(arr, axis=0)
            stds = np.std(arr,  axis=0)
            feat = np.array(feat_cols)

            if signed:
                idx = np.argsort(np.abs(vals))[-top_n:]
                idx = idx[np.argsort(vals[idx])]
            else:
                idx = np.argsort(vals)[-top_n:]

            f_sel  = feat[idx]
            v_sel  = vals[idx]
            s_sel  = stds[idx]
            colors = [ORG_COLORS.get(get_organelle(f), '#AAAAAA') for f in f_sel]

            labels = []
            for f in f_sel:
                parts = f.split('_', 1)
                labels.append(parts[1].replace('_', ' ')
                              if len(parts) > 1 else f.replace('_', ' '))

            eb_kw = dict(elinewidth=0.8, capthick=0.8, capsize=3,
                         ecolor='#444444')
            ax.barh(range(len(f_sel)), v_sel, xerr=s_sel, error_kw=eb_kw,
                    color=colors, edgecolor='white', lw=0,
                    height=0.65, alpha=0.88)

            if signed:
                ax.axvline(0, color='#333333', lw=0.9)

            ax.set_yticks(range(len(f_sel)))
            ax.set_yticklabels(labels, fontsize=8.0)
            ax.set_xlabel('Mean ± SD across LOPO folds', fontsize=9)
            ax.set_title(f'{task_label}\n{clf_label}',
                         fontsize=9.5, fontweight='bold', pad=5)

            letter = panel_letters[row][col]
            ax.text(-0.32, 1.04, letter, transform=ax.transAxes,
                    fontsize=13, fontweight='bold', va='bottom')

            despine(ax)
            xgrid(ax)

    patches = [mpatches.Patch(facecolor=c, label=o, alpha=0.88)
               for o, c in ORG_COLORS.items()]
    fig.legend(handles=patches, loc='upper center', ncol=3,
               frameon=False, fontsize=9.5, bbox_to_anchor=(0.6, 0.995))

    save(fig, 'fig6_feature_importances')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print('Loading data...')
    df = pd.read_csv(ss.FEATURES_FILE)
    df = df[df['Disease'].isin(['Healthy', 'TAA', 'BAV'])].copy()
    df = df[df['Collagen_Status'] == 'NoCollagen'].copy()
    df['Subject'] = df['CellName'].apply(ss.norm_id)

    feat_cols = sorted([c for c in df.columns
                        if c.startswith(('Actin_', 'Mito_', 'Nucleus_'))])
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f'  {len(df)} cells | {df["Subject"].nunique()} patients | '
          f'{len(feat_cols)} features')

    # Task 1
    print('\nRunning Task 1 (Healthy vs Diseased)...')
    df1 = df.copy()
    df1['label'] = df1['Disease'].map(
        {'Healthy': 'Healthy', 'TAA': 'Diseased', 'BAV': 'Diseased'})
    s1, imp1 = run_lopo(df1, feat_cols,
                         class_names=['Healthy', 'Diseased'],
                         label_col='label')

    # Task 2
    print('Running Task 2 (TAV-ATAA vs BAV-ATAA)...')
    df2 = df[df['Disease'].isin(['TAA', 'BAV'])].copy()
    df2['label'] = df2['Disease'].map({'TAA': 'TAV-ATAA', 'BAV': 'BAV-ATAA'})
    s2, imp2 = run_lopo(df2, feat_cols,
                         class_names=['TAV-ATAA', 'BAV-ATAA'],
                         label_col='label')

    # print summary
    for task_label, s in [('Task 1', s1), ('Task 2', s2)]:
        print(f'\n  {task_label}')
        for c in CLF_KEYS:
            sv = s.get(c, {})
            print(f'    {c:25s}  '
                  f'Bal.Acc={sv.get("bal_mean",0):.3f}±{sv.get("bal_std",0):.3f}')

    print('\nGenerating figures...')
    fig1_performance(s1, s2)
    fig2_roc(s1, s2)
    fig3_confusion(s1, s2)
    fig4_per_class(s1, s2)
    fig5_per_fold(s1, s2)
    fig6_importances(feat_cols, imp1, imp2)

    print(f'\nAll 6 figures saved to: {OUT}')


if __name__ == '__main__':
    import os
    os.chdir(ss.PROJ_ROOT)
    main()
