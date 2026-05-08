"""
Part 3 — Publication-quality classification figures.
Runs LOPO-CV and saves paper-ready plots to:
  outputs/final_thesis_part3/task1_diseased_vs_healthy/
  outputs/final_thesis_part3/task2_tav_vs_bav/
  outputs/final_thesis_part3/comparison/
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
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

OUT_BASE = PROJ / 'outputs' / 'final_thesis_part3'
T1_DIR   = OUT_BASE / 'task1_diseased_vs_healthy'
T2_DIR   = OUT_BASE / 'task2_tav_vs_bav'
CMP_DIR  = OUT_BASE / 'comparison'
for d in [T1_DIR, T2_DIR, CMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ORGANELLE_COLORS = {
    'Nucleus':      '#1558A0',
    'Actin':        '#2ECC71',
    'Mitochondria': '#E74C3C',
}
CLF_COLORS = {
    'Logistic Regression': '#8E44AD',
    'Random Forest':       '#27AE60',
    'SVM':                 '#E67E22',
}
CLF_SHORT = {
    'Logistic Regression': 'LR',
    'Random Forest':       'RF',
    'SVM':                 'SVM',
}

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
    'axes.grid':          False,
    'savefig.dpi':        300,
    'text.usetex':        False,
}


def get_organelle(feat):
    fl = feat.lower()
    if 'nucleus' in fl: return 'Nucleus'
    if 'actin'   in fl: return 'Actin'
    if 'mito'    in fl: return 'Mitochondria'
    return 'Other'


def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def ygrid(ax):
    ax.yaxis.grid(True, color='#E0E0E0', linewidth=0.6, linestyle='-', zorder=0)
    ax.set_axisbelow(True)


def xgrid(ax):
    ax.xaxis.grid(True, color='#E0E0E0', linewidth=0.6, linestyle='-', zorder=0)
    ax.set_axisbelow(True)


def save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def get_classifiers():
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=2000, class_weight='balanced',
            C=0.1, solver='lbfgs', random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, class_weight='balanced',
            max_depth=6, min_samples_leaf=3,
            random_state=42, n_jobs=-1),
        'SVM': CalibratedClassifierCV(
            SVC(kernel='rbf', class_weight='balanced',
                C=1.0, gamma='scale', random_state=42),
            cv=3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# LOPO-CV
# ══════════════════════════════════════════════════════════════════════════════
def run_lopo_cv(df, feat_cols, class_names, label_col):
    patients      = sorted(df['Subject'].unique())
    fold_results  = defaultdict(list)
    all_importances = defaultdict(list)

    for held_out in patients:
        train_mask = df['Subject'] != held_out
        test_mask  = df['Subject'] == held_out

        X_train = df.loc[train_mask, feat_cols].values
        y_train = df.loc[train_mask, label_col].values
        X_test  = df.loc[test_mask,  feat_cols].values
        y_test  = df.loc[test_mask,  label_col].values

        if len(np.unique(y_train)) < 2:
            continue

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
        disease = df.loc[test_mask, 'Disease'].iloc[0]

        for clf_name, clf in get_classifiers().items():
            try:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_prob_pos = None
                if hasattr(clf, 'predict_proba') and len(class_names) == 2:
                    try:
                        proba   = clf.predict_proba(X_test)
                        pos_idx = list(clf.classes_).index(class_names[1])
                        y_prob_pos = proba[:, pos_idx]
                    except Exception:
                        pass

                bal_acc = balanced_accuracy_score(y_test, y_pred)
                auc_val = np.nan
                if y_prob_pos is not None:
                    try:
                        auc_val = roc_auc_score(
                            (y_test == class_names[1]).astype(int), y_prob_pos)
                    except Exception:
                        pass

                fold_results[clf_name].append({
                    'patient': held_out, 'disease': disease,
                    'bal_acc': bal_acc, 'auc': auc_val,
                    'y_true': y_test.tolist(), 'y_pred': y_pred.tolist(),
                    'y_prob': y_prob_pos.tolist() if y_prob_pos is not None else None,
                })

                if clf_name == 'Random Forest':
                    all_importances[clf_name].append(clf.feature_importances_)
                elif clf_name == 'Logistic Regression':
                    all_importances[clf_name].append(clf.coef_[0])

            except Exception as e:
                print(f"  {held_out} {clf_name} failed: {e}")

    summary = {}
    for clf_name, folds in fold_results.items():
        if not folds:
            continue
        bal_accs = [f['bal_acc'] for f in folds]
        aucs     = [f['auc'] for f in folds if not np.isnan(f['auc'])]
        y_true_all = np.concatenate([f['y_true'] for f in folds])
        y_pred_all = np.concatenate([f['y_pred'] for f in folds])
        cm = confusion_matrix(y_true_all, y_pred_all, labels=class_names)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true_all, y_pred_all, labels=class_names, zero_division=0)
        y_prob_all, y_bin_all = None, None
        folds_with_prob = [f for f in folds if f['y_prob'] is not None]
        if folds_with_prob:
            y_prob_all = np.concatenate([f['y_prob'] for f in folds_with_prob])
            y_true_prob = np.concatenate([f['y_true'] for f in folds_with_prob])
            y_bin_all   = (np.array(y_true_prob) == class_names[1]).astype(int)

        summary[clf_name] = {
            'folds':        folds,
            'bal_acc_mean': np.mean(bal_accs),
            'bal_acc_std':  np.std(bal_accs),
            'auc_mean':     np.mean(aucs) if aucs else np.nan,
            'auc_std':      np.std(aucs)  if aucs else np.nan,
            'confusion':    cm,
            'precision':    dict(zip(class_names, p)),
            'recall':       dict(zip(class_names, r)),
            'f1':           dict(zip(class_names, f1)),
            'class_names':  class_names,
            'y_prob_all':   y_prob_all,
            'y_bin_all':    y_bin_all,
        }

    return summary, all_importances


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrices(summary, class_names, task_label, out_dir):
    plt.rcParams.update(RC)
    clf_names   = list(summary.keys())
    n = len(clf_names)

    fig, axes = plt.subplots(1, n, figsize=(n * 3.0, 3.2))
    if n == 1:
        axes = [axes]

    for ax, clf_name in zip(axes, clf_names):
        cm      = summary[clf_name]['confusion'].astype(float)
        cm_norm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))

        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = cm_norm[i, j]
                raw = int(cm[i, j])
                ax.text(j, i, f'{val:.2f}\n({raw})',
                        ha='center', va='center', fontsize=9,
                        color='white' if val > 0.55 else '#222222',
                        fontweight='bold' if val > 0.55 else 'normal')

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel('Predicted label', fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel('True label', fontsize=10)

        bal = summary[clf_name]['bal_acc_mean']
        auc = summary[clf_name]['auc_mean']
        astr = f', AUC={auc:.2f}' if not np.isnan(auc) else ''
        ax.set_title(f'{CLF_SHORT.get(clf_name, clf_name)}\nBal. Acc = {bal:.2f}{astr}',
                     fontsize=10, pad=5)

    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.04)
    cbar.set_label('Recall (row-normalised)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(task_label, fontsize=11, fontweight='bold', y=1.02)
    save(fig, out_dir / 'confusion_matrices.png')


def plot_performance_bar(summary, task_label, out_dir):
    plt.rcParams.update(RC)
    clf_names = list(summary.keys())
    x  = np.arange(len(clf_names))
    w  = 0.33

    bal_m = [summary[c]['bal_acc_mean'] for c in clf_names]
    bal_s = [summary[c]['bal_acc_std']  for c in clf_names]
    auc_m = [summary[c]['auc_mean']     for c in clf_names]
    auc_s = [summary[c]['auc_std']      for c in clf_names]

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    eb_kw = dict(elinewidth=0.9, capthick=0.9, capsize=4, ecolor='#444444')

    bars1 = ax.bar(x - w/2, bal_m, w, yerr=bal_s, error_kw=eb_kw,
                   color=[CLF_COLORS[c] for c in clf_names],
                   alpha=0.88, edgecolor='white', linewidth=0, label='Balanced accuracy')
    ax.bar(x + w/2, auc_m, w, yerr=auc_s, error_kw=eb_kw,
           color=[CLF_COLORS[c] for c in clf_names],
           alpha=0.38, edgecolor=[CLF_COLORS[c] for c in clf_names],
           linewidth=1.0, hatch='///', label='AUC-ROC')

    ax.axhline(0.5, color='#C0392B', ls='--', lw=1.0, alpha=0.85, label='Chance (0.5)')
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels([CLF_SHORT.get(c, c) for c in clf_names], fontsize=10)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_xlabel('Classifier', fontsize=10)
    ax.set_title(task_label, fontsize=11, fontweight='bold', pad=6)

    for bar, val, std in zip(bars1, bal_m, bal_s):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (std or 0) + 0.025,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8.5,
                    fontweight='bold')

    ax.legend(frameon=False, loc='upper left', fontsize=9)
    despine(ax)
    ygrid(ax)
    plt.tight_layout()
    save(fig, out_dir / 'performance_bar.png')


def plot_roc(summary, class_names, task_label, out_dir):
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.plot([0, 1], [0, 1], color='#AAAAAA', ls='--', lw=0.9,
            label='Chance (AUC = 0.50)')

    for clf_name, s in summary.items():
        if s['y_prob_all'] is None:
            continue
        try:
            fpr, tpr, _ = roc_curve(s['y_bin_all'], s['y_prob_all'])
            auc_val = roc_auc_score(s['y_bin_all'], s['y_prob_all'])
            short   = CLF_SHORT.get(clf_name, clf_name)
            ax.plot(fpr, tpr, color=CLF_COLORS[clf_name], lw=1.8,
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
    plt.tight_layout()
    save(fig, out_dir / 'roc_curves.png')


def plot_per_fold(fold_results_dict, task_label, out_dir):
    plt.rcParams.update(RC)
    clf_names = list(fold_results_dict.keys())
    rows = []
    for clf_name, folds in fold_results_dict.items():
        for f in folds:
            rows.append({'Classifier': clf_name, 'Disease': f['disease'],
                         'Bal_Acc': f['bal_acc']})
    df_plot = pd.DataFrame(rows)
    if df_plot.empty:
        return

    x_pos = {c: i for i, c in enumerate(clf_names)}
    rng   = np.random.RandomState(42)
    disease_set = df_plot['Disease'].unique().tolist()

    disease_styles = {
        'Healthy': ('o', '#4C9BE8'),
        'TAA':     ('s', '#E74C3C'),
        'BAV':     ('^', '#F39C12'),
        'Diseased':('D', '#C0392B'),
    }

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for clf_name in clf_names:
        sub    = df_plot[df_plot['Classifier'] == clf_name]
        jitter = rng.uniform(-0.18, 0.18, len(sub))
        for dis in disease_set:
            mask = sub['Disease'] == dis
            if not mask.any():
                continue
            idx = np.where(mask.values)[0]
            marker, color = disease_styles.get(dis, ('o', '#888888'))
            ax.scatter(x_pos[clf_name] + jitter[idx],
                       sub.loc[mask, 'Bal_Acc'].values,
                       c=color, marker=marker, s=35, alpha=0.78,
                       edgecolors='white', linewidths=0.4,
                       label=dis if clf_name == clf_names[0] else '_nolegend_',
                       zorder=3)

        mean_acc = sub['Bal_Acc'].mean()
        ax.plot([x_pos[clf_name] - 0.28, x_pos[clf_name] + 0.28],
                [mean_acc, mean_acc], color='#222222', lw=2.0, zorder=5)
        ax.text(x_pos[clf_name] + 0.32, mean_acc,
                f'{mean_acc:.2f}', va='center', fontsize=8.5,
                color='#222222', fontweight='bold')

    ax.axhline(0.5, color='#C0392B', ls='--', lw=1.0, alpha=0.85,
               label='Chance (0.5)')
    ax.set_xticks(range(len(clf_names)))
    ax.set_xticklabels([CLF_SHORT.get(c, c) for c in clf_names], fontsize=10)
    ax.set_ylabel('Balanced accuracy per patient fold', fontsize=10)
    ax.set_xlabel('Classifier', fontsize=10)
    ax.set_ylim(-0.05, 1.18)
    ax.set_title(task_label, fontsize=11, fontweight='bold', pad=6)

    handles = []
    for dis in disease_set:
        marker, color = disease_styles.get(dis, ('o', '#888888'))
        handles.append(plt.scatter([], [], c=color, marker=marker, s=28,
                                   label=dis, edgecolors='white'))
    handles.append(plt.Line2D([0], [0], color='#C0392B', ls='--', lw=1.0,
                               label='Chance (0.5)'))
    ax.legend(handles=handles, frameon=False, loc='lower right',
              handletextpad=0.4, fontsize=9)
    despine(ax)
    ygrid(ax)
    plt.tight_layout()
    save(fig, out_dir / 'per_fold_accuracy.png')


def plot_feature_importance(feat_cols, values, stds, title,
                             signed, out_dir, fname, top_n=15):
    plt.rcParams.update(RC)
    feat_arr = np.array(feat_cols)
    val_arr  = np.array(values)
    std_arr  = np.array(stds)

    if signed:
        idx = np.argsort(np.abs(val_arr))[-top_n:]
        idx = idx[np.argsort(val_arr[idx])]
    else:
        idx = np.argsort(val_arr)[-top_n:]

    feats  = feat_arr[idx]
    vals   = val_arr[idx]
    stds_s = std_arr[idx]
    colors = [ORGANELLE_COLORS.get(get_organelle(f), '#AAAAAA') for f in feats]

    labels = []
    for f in feats:
        parts = f.split('_', 1)
        labels.append(parts[1].replace('_', ' ') if len(parts) > 1 else f.replace('_', ' '))

    fig_h  = max(4.0, len(feats) * 0.35)
    fig, ax = plt.subplots(figsize=(5.8, fig_h))
    eb_kw = dict(elinewidth=0.8, capthick=0.8, capsize=3, ecolor='#444444')
    ax.barh(range(len(feats)), vals, xerr=stds_s, error_kw=eb_kw,
            color=colors, edgecolor='white', linewidth=0, height=0.65, alpha=0.88)

    if signed:
        ax.axvline(0, color='#333333', lw=0.9)

    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(labels, fontsize=8.5)
    xlabel = ('Mean coefficient ± SD  (LOPO folds)' if signed
              else 'Mean importance ± SD  (LOPO folds)')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)

    patches = [mpatches.Patch(facecolor=c, label=o, alpha=0.88)
               for o, c in ORGANELLE_COLORS.items()]
    ax.legend(handles=patches, frameon=False, loc='lower right', fontsize=9)
    despine(ax)
    xgrid(ax)
    plt.tight_layout()
    save(fig, out_dir / fname)


def plot_per_class_metrics(summary, class_names, task_label, out_dir):
    plt.rcParams.update(RC)
    clf_names = list(summary.keys())
    metrics   = ['precision', 'recall', 'f1']
    mlabels   = {'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1-score'}
    mcolors   = {'precision': '#2980B9', 'recall': '#C0392B', 'f1': '#27AE60'}
    ncols     = len(class_names)

    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 3.0, 3.5), sharey=True)
    if ncols == 1:
        axes = [axes]

    x = np.arange(len(clf_names))
    w = 0.22

    for ax, cls in zip(axes, class_names):
        for i, metric in enumerate(metrics):
            vals = [summary[c][metric][cls] for c in clf_names]
            ax.bar(x + (i - 1) * w, vals, w,
                   color=mcolors[metric], alpha=0.84, edgecolor='white',
                   linewidth=0, label=mlabels[metric])
        ax.set_xticks(x)
        ax.set_xticklabels([CLF_SHORT.get(c, c) for c in clf_names], fontsize=10)
        ax.set_title(cls, fontsize=11, fontweight='bold', pad=5)
        ax.set_ylim(0, 1.15)
        ax.axhline(0.5, color='#AAAAAA', ls=':', lw=0.8)
        if ax is axes[0]:
            ax.set_ylabel('Score', fontsize=10)
            ax.legend(frameon=False, loc='lower left', fontsize=9)
        despine(ax)
        ygrid(ax)

    fig.suptitle(task_label, fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    save(fig, out_dir / 'per_class_metrics.png')


def plot_model_comparison(s1, s2, out_dir):
    """Side-by-side bar chart: Task 1 vs Task 2."""
    plt.rcParams.update(RC)
    clf_names = list(s1.keys())
    x = np.arange(len(clf_names))
    w = 0.33

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
    eb_kw = dict(elinewidth=0.9, capthick=0.9, capsize=4, ecolor='#444444')

    task_info = [
        ('Task 1: Healthy vs Diseased', s1, axes[0]),
        ('Task 2: TAV-ATAA vs BAV-ATAA', s2, axes[1]),
    ]

    for label, s, ax in task_info:
        bal_m = [s[c]['bal_acc_mean'] for c in clf_names]
        bal_s = [s[c]['bal_acc_std']  for c in clf_names]
        auc_m = [s[c]['auc_mean']     for c in clf_names]
        auc_s = [s[c]['auc_std']      for c in clf_names]

        bars = ax.bar(x - w/2, bal_m, w, yerr=bal_s, error_kw=eb_kw,
                      color=[CLF_COLORS[c] for c in clf_names],
                      alpha=0.88, edgecolor='white', linewidth=0,
                      label='Balanced accuracy')
        ax.bar(x + w/2, auc_m, w, yerr=auc_s, error_kw=eb_kw,
               color=[CLF_COLORS[c] for c in clf_names],
               alpha=0.38, edgecolor=[CLF_COLORS[c] for c in clf_names],
               linewidth=1.0, hatch='///', label='AUC-ROC')

        ax.axhline(0.5, color='#C0392B', ls='--', lw=1.0, alpha=0.85)
        ax.set_ylim(0, 1.15)
        ax.set_xticks(x)
        ax.set_xticklabels([CLF_SHORT.get(c, c) for c in clf_names], fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold', pad=6)
        ax.set_xlabel('Classifier', fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel('Score', fontsize=10)

        for bar, val, std in zip(bars, bal_m, bal_s):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (std or 0) + 0.025,
                        f'{val:.3f}', ha='center', va='bottom',
                        fontsize=8.5, fontweight='bold')
        despine(ax)
        ygrid(ax)

    handles = (
        [mpatches.Patch(facecolor=CLF_COLORS[c], label=CLF_SHORT.get(c, c), alpha=0.88)
         for c in clf_names] +
        [mpatches.Patch(facecolor='#888888', alpha=0.88, label='Balanced accuracy'),
         mpatches.Patch(facecolor='#888888', alpha=0.38, hatch='///', label='AUC-ROC'),
         plt.Line2D([0], [0], color='#C0392B', ls='--', lw=1.0, label='Chance (0.5)')]
    )
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save(fig, out_dir / 'model_comparison.png')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("Loading data...")
    df = pd.read_csv(ss.FEATURES_FILE)
    df = df[df['Disease'].isin(['Healthy', 'TAA', 'BAV'])].copy()
    df = df[df['Collagen_Status'] == 'NoCollagen'].copy()
    df['Subject'] = df['CellName'].apply(ss.norm_id)

    feat_cols = sorted([c for c in df.columns
                        if c.startswith(('Actin_', 'Mito_', 'Nucleus_'))])
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"  {len(df)} cells | {df['Subject'].nunique()} patients | {len(feat_cols)} features")

    # ── Task 1: Healthy vs Diseased ───────────────────────────────────────────
    print("\nTask 1: Healthy vs Diseased")
    df_t1 = df.copy()
    df_t1['label'] = df_t1['Disease'].map(
        {'Healthy': 'Healthy', 'TAA': 'Diseased', 'BAV': 'Diseased'})
    s1, imp1 = run_lopo_cv(df_t1, feat_cols,
                            class_names=['Healthy', 'Diseased'],
                            label_col='label')

    plot_confusion_matrices(s1, ['Healthy', 'Diseased'],
                            'Task 1: Healthy vs Diseased', T1_DIR)
    plot_performance_bar(s1, 'Task 1: Healthy vs Diseased', T1_DIR)
    plot_roc(s1, ['Healthy', 'Diseased'], 'Task 1: Healthy vs Diseased', T1_DIR)
    plot_per_class_metrics(s1, ['Healthy', 'Diseased'],
                           'Task 1: Healthy vs Diseased', T1_DIR)
    plot_per_fold({c: s1[c]['folds'] for c in s1},
                  'Task 1: Healthy vs Diseased', T1_DIR)

    if imp1.get('Logistic Regression'):
        cm = np.mean(imp1['Logistic Regression'], axis=0)
        cs = np.std(imp1['Logistic Regression'], axis=0)
        plot_feature_importance(feat_cols, cm, cs,
            'Logistic Regression — Coefficients\nTask 1: Healthy vs Diseased',
            signed=True, out_dir=T1_DIR, fname='logreg_coefficients.png')

    if imp1.get('Random Forest'):
        rm = np.mean(imp1['Random Forest'], axis=0)
        rs = np.std(imp1['Random Forest'], axis=0)
        plot_feature_importance(feat_cols, rm, rs,
            'Random Forest — Feature Importances\nTask 1: Healthy vs Diseased',
            signed=False, out_dir=T1_DIR, fname='rf_importances.png')

    # ── Task 2: TAV-ATAA vs BAV-ATAA ─────────────────────────────────────────
    print("\nTask 2: TAV-ATAA vs BAV-ATAA")
    df_t2 = df[df['Disease'].isin(['TAA', 'BAV'])].copy()
    df_t2['label'] = df_t2['Disease'].map({'TAA': 'TAV-ATAA', 'BAV': 'BAV-ATAA'})
    s2, imp2 = run_lopo_cv(df_t2, feat_cols,
                            class_names=['TAV-ATAA', 'BAV-ATAA'],
                            label_col='label')

    plot_confusion_matrices(s2, ['TAV-ATAA', 'BAV-ATAA'],
                            'Task 2: TAV-ATAA vs BAV-ATAA', T2_DIR)
    plot_performance_bar(s2, 'Task 2: TAV-ATAA vs BAV-ATAA', T2_DIR)
    plot_roc(s2, ['TAV-ATAA', 'BAV-ATAA'], 'Task 2: TAV-ATAA vs BAV-ATAA', T2_DIR)
    plot_per_class_metrics(s2, ['TAV-ATAA', 'BAV-ATAA'],
                           'Task 2: TAV-ATAA vs BAV-ATAA', T2_DIR)
    plot_per_fold({c: s2[c]['folds'] for c in s2},
                  'Task 2: TAV-ATAA vs BAV-ATAA', T2_DIR)

    if imp2.get('Logistic Regression'):
        cm2 = np.mean(imp2['Logistic Regression'], axis=0)
        cs2 = np.std(imp2['Logistic Regression'], axis=0)
        plot_feature_importance(feat_cols, cm2, cs2,
            'Logistic Regression — Coefficients\nTask 2: TAV-ATAA vs BAV-ATAA',
            signed=True, out_dir=T2_DIR, fname='logreg_coefficients.png')

    if imp2.get('Random Forest'):
        rm2 = np.mean(imp2['Random Forest'], axis=0)
        rs2 = np.std(imp2['Random Forest'], axis=0)
        plot_feature_importance(feat_cols, rm2, rs2,
            'Random Forest — Feature Importances\nTask 2: TAV-ATAA vs BAV-ATAA',
            signed=False, out_dir=T2_DIR, fname='rf_importances.png')

    # ── Comparison ────────────────────────────────────────────────────────────
    print("\nComparison plot")
    plot_model_comparison(s1, s2, CMP_DIR)

    # print summary
    print("\n" + "=" * 60)
    for task_label, s in [("Task 1 — Healthy vs Diseased", s1),
                           ("Task 2 — TAV-ATAA vs BAV-ATAA", s2)]:
        print(f"\n  {task_label}")
        for clf_name, sv in s.items():
            astr = (f"  AUC={sv['auc_mean']:.3f}±{sv['auc_std']:.3f}"
                    if not np.isnan(sv['auc_mean']) else "")
            print(f"    {clf_name:25s}  Bal.Acc={sv['bal_acc_mean']:.3f}±{sv['bal_acc_std']:.3f}{astr}")

    print(f"\nDone. Outputs in: {OUT_BASE}/")


if __name__ == '__main__':
    import os
    os.chdir(ss.PROJ_ROOT)
    main()
