"""
Morphological Classification of Aortic Aneurysm Subtypes
=========================================================
Thesis: Computational 3D Phenotyping of Mitochondrial Morphology in Aortic Aneurysm

TWO CLASSIFICATION TASKS:
  Task 1 — Healthy vs Diseased  (TAV-ATAA + BAV pooled as "Diseased")
  Task 2 — BAV vs TAV-ATAA      (aneurysm subtype discrimination)

CROSS-VALIDATION:
  Leave-One-Patient-Out CV (LOPO-CV)
  All cells from one patient held out → train on remaining patients → test
  Prevents data leakage across cells from the same donor
  Repeated for every patient → mean ± SD reported

CLASSIFIERS:
  1. Logistic Regression   — linear baseline, interpretable coefficients
  2. Random Forest         — non-linear, feature importances
  3. SVM (RBF kernel)      — strong on small-n, high-dimensional data

METRICS:
  - Balanced accuracy (primary — handles class imbalance)
  - AUC-ROC
  - Precision, Recall, F1 per class
  - Confusion matrix (aggregated across folds)
  - Feature importances / coefficients (mean ± SD across folds)

DATA:
  - NoCollagen cells only
  - StandardScaler fit on training fold only (no leakage)
  - class_weight='balanced' in all classifiers

Outputs: outputs/classification/
  task1/   — Healthy vs Diseased
  task2/   — BAV vs TAV-ATAA
  comparison/ — cross-task summary
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared_stats as ss

warnings.filterwarnings('ignore')

# ── output directories ────────────────────────────────────────────────────────
BASE      = Path(__file__).resolve().parents[3] / 'outputs' / 'classification'
TASK1_DIR = BASE / 'task1_healthy_vs_diseased'
TASK2_DIR = BASE / 'task2_bav_vs_tav'
COMP_DIR  = BASE / 'comparison'

DISEASE_COLORS = {
    'Healthy':  '#4C9BE8',
    'TAA':      '#E74C3C',
    'BAV':      '#F39C12',
    'Diseased': '#C0392B',
}
ORGANELLE_COLORS = {
    'Nucleus':      '#4C9BE8',
    'Actin':        '#E74C3C',
    'Mitochondria': '#2ECC71',
}
CLF_COLORS = {
    'Logistic Regression': '#8E44AD',
    'Random Forest':       '#27AE60',
    'SVM (RBF)':           '#E67E22',
}

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_organelle(feat):
    fl = feat.lower()
    if 'nucleus' in fl: return 'Nucleus'
    if 'actin'   in fl: return 'Actin'
    if 'mito'    in fl: return 'Mitochondria'
    return 'Other'

def save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches='tight')
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
        'SVM (RBF)': CalibratedClassifierCV(
            SVC(kernel='rbf', class_weight='balanced',
                C=1.0, gamma='scale', random_state=42),
            cv=3),
    }

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    df = pd.read_csv(ss.FEATURES_FILE)
    df = df[df['Disease'].isin(['Healthy', 'TAA', 'BAV'])].copy()
    df = df[df['Collagen_Status'] == 'NoCollagen'].copy()
    df['Subject'] = df['CellName'].apply(ss.norm_id)

    feat_cols = sorted([c for c in df.columns
                        if c.startswith(('Actin_', 'Mito_', 'Nucleus_'))])
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"  Total cells (NoCollagen): {len(df)}")
    print(f"  Total patients:           {df['Subject'].nunique()}")
    for d in ['Healthy', 'TAA', 'BAV']:
        sub = df[df['Disease'] == d]
        print(f"    {d:8s}: {len(sub):3d} cells | "
              f"{sub['Subject'].nunique()} patients")
    print(f"  Features: {len(feat_cols)}")
    return df, feat_cols

# ══════════════════════════════════════════════════════════════════════════════
# LOPO-CV ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def run_lopo_cv(df, feat_cols, task_name, label_col, class_names, out_dir):
    """
    Leave-One-Patient-Out Cross-Validation.

    Each fold:
      1. Hold out all cells from one patient (test set)
      2. Train on all cells from all other patients
      3. Fit StandardScaler on training data only
      4. Transform test data with training scaler
      5. Evaluate all three classifiers
    """
    print(f"\n{'='*70}")
    print(f"TASK: {task_name}")
    print(f"  Classes:  {class_names}")
    print(f"  Patients: {df['Subject'].nunique()}")
    print(f"  Cells:    {len(df)}")
    print(f"{'='*70}")

    patients    = sorted(df['Subject'].unique())
    fold_results    = defaultdict(list)
    all_importances = defaultdict(list)

    for fold_i, held_out in enumerate(patients):
        train_mask = df['Subject'] != held_out
        test_mask  = df['Subject'] == held_out

        X_train = df.loc[train_mask, feat_cols].values
        y_train = df.loc[train_mask, label_col].values
        X_test  = df.loc[test_mask,  feat_cols].values
        y_test  = df.loc[test_mask,  label_col].values

        if len(np.unique(y_train)) < 2:
            print(f"  Fold {fold_i+1} skipped — training set has only one class")
            continue

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        disease = df.loc[test_mask, 'Disease'].iloc[0]

        for clf_name, clf_fresh in get_classifiers().items():
            try:
                clf_fresh.fit(X_train, y_train)
                y_pred = clf_fresh.predict(X_test)

                y_prob = None
                if hasattr(clf_fresh, 'predict_proba'):
                    try:
                        y_prob = clf_fresh.predict_proba(X_test)
                    except Exception:
                        pass

                bal_acc = balanced_accuracy_score(y_test, y_pred)
                auc_val = np.nan
                y_prob_pos = None
                if y_prob is not None and len(class_names) == 2:
                    try:
                        pos_idx = list(clf_fresh.classes_).index(class_names[1])
                        y_prob_pos = y_prob[:, pos_idx]
                        auc_val = roc_auc_score(
                            (y_test == class_names[1]).astype(int), y_prob_pos)
                    except Exception:
                        pass

                fold_results[clf_name].append({
                    'patient': held_out,
                    'disease': disease,
                    'n_cells': len(y_test),
                    'bal_acc': bal_acc,
                    'auc':     auc_val,
                    'y_true':  y_test.tolist(),
                    'y_pred':  y_pred.tolist(),
                    'y_prob':  y_prob_pos.tolist() if y_prob_pos is not None else None,
                })

                if clf_name == 'Random Forest':
                    all_importances[clf_name].append(clf_fresh.feature_importances_)
                elif clf_name == 'Logistic Regression':
                    all_importances[clf_name].append(clf_fresh.coef_[0])

            except Exception as e:
                print(f"  Fold {fold_i+1} {clf_name} failed: {e}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
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

        summary[clf_name] = {
            'folds':        folds,
            'bal_acc_mean': np.mean(bal_accs),
            'bal_acc_std':  np.std(bal_accs),
            'auc_mean':     np.mean(aucs)  if aucs else np.nan,
            'auc_std':      np.std(aucs)   if aucs else np.nan,
            'confusion':    cm,
            'precision':    dict(zip(class_names, p)),
            'recall':       dict(zip(class_names, r)),
            'f1':           dict(zip(class_names, f1)),
            'class_names':  class_names,
            'y_true_all':   y_true_all,
            'y_pred_all':   y_pred_all,
        }

        print(f"\n  {clf_name}:")
        print(f"    Balanced accuracy : {np.mean(bal_accs):.3f} ± {np.std(bal_accs):.3f}")
        if aucs:
            print(f"    AUC-ROC           : {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
        for cls, prec, rec, f1s in zip(class_names, p, r, f1):
            print(f"    {cls:12s}  P={prec:.2f}  R={rec:.2f}  F1={f1s:.2f}")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    rows = []
    for clf_name, folds in fold_results.items():
        for f in folds:
            rows.append({'classifier': clf_name, 'patient': f['patient'],
                         'disease': f['disease'], 'n_cells': f['n_cells'],
                         'bal_acc': f['bal_acc'], 'auc': f['auc']})
    pd.DataFrame(rows).to_csv(out_dir / 'fold_results.csv', index=False)

    class_rows = []
    for clf_name, s in summary.items():
        for cls in class_names:
            class_rows.append({'classifier': clf_name, 'class': cls,
                               'precision': s['precision'][cls],
                               'recall':    s['recall'][cls],
                               'f1':        s['f1'][cls]})
    pd.DataFrame(class_rows).to_csv(out_dir / 'class_metrics.csv', index=False)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_performance_summary(summary, task_name,
                             out_dir / '01_performance_summary.png')
    plot_confusion_matrices(summary,
                            out_dir / '02_confusion_matrices.png')
    plot_per_fold_accuracy(fold_results,
                           out_dir / '03_per_fold_accuracy.png')
    plot_roc_curves(fold_results, class_names,
                    out_dir / '04_roc_curves.png')
    plot_per_class_metrics(summary,
                           out_dir / '05_per_class_metrics.png')

    if all_importances['Random Forest']:
        imp_mean = np.mean(all_importances['Random Forest'], axis=0)
        imp_std  = np.std(all_importances['Random Forest'],  axis=0)
        plot_feature_importance(
            feat_cols, imp_mean, imp_std,
            f'Random Forest — Feature Importances\n{task_name}',
            out_dir / '06_rf_feature_importance.png', signed=False)
        pd.DataFrame({
            'feature':         feat_cols,
            'importance_mean': imp_mean,
            'importance_std':  imp_std,
            'organelle':       [get_organelle(f) for f in feat_cols],
        }).sort_values('importance_mean', ascending=False
        ).to_csv(out_dir / 'rf_importances.csv', index=False)

    if all_importances['Logistic Regression']:
        coef_mean = np.mean(all_importances['Logistic Regression'], axis=0)
        coef_std  = np.std(all_importances['Logistic Regression'],  axis=0)
        plot_feature_importance(
            feat_cols, coef_mean, coef_std,
            f'Logistic Regression — Coefficients\n{task_name}\n'
            f'(positive = {class_names[1]})',
            out_dir / '07_logreg_coefficients.png', signed=True)
        pd.DataFrame({
            'feature':   feat_cols,
            'coef_mean': coef_mean,
            'coef_std':  coef_std,
            'organelle': [get_organelle(f) for f in feat_cols],
        }).sort_values('coef_mean', key=abs, ascending=False
        ).to_csv(out_dir / 'logreg_coefficients.csv', index=False)

    return summary, fold_results, all_importances

# ══════════════════════════════════════════════════════════════════════════════
# PUBLICATION STYLE
# ══════════════════════════════════════════════════════════════════════════════
def _paper_rc():
    plt.rcParams.update({
        'font.family':       'sans-serif',
        'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size':         9,
        'axes.labelsize':    9,
        'axes.titlesize':    9,
        'xtick.labelsize':   8,
        'ytick.labelsize':   8,
        'legend.fontsize':   8,
        'axes.linewidth':    0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size':  3,
        'ytick.major.size':  3,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.grid':         False,
        'figure.dpi':        150,
        'savefig.dpi':       300,
        'savefig.bbox':      'tight',
        'savefig.pad_inches': 0.05,
    })

def _despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def _subtick(ax, axis='y'):
    if axis == 'y':
        ax.yaxis.grid(True, color='#CCCCCC', linewidth=0.5, linestyle='-', zorder=0)
        ax.set_axisbelow(True)
    else:
        ax.xaxis.grid(True, color='#CCCCCC', linewidth=0.5, linestyle='-', zorder=0)
        ax.set_axisbelow(True)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def plot_performance_summary(summary, task_name, path):
    _paper_rc()
    clf_names = list(summary.keys())
    x = np.arange(len(clf_names))
    w = 0.32

    bal_means = [summary[c]['bal_acc_mean'] for c in clf_names]
    bal_stds  = [summary[c]['bal_acc_std']  for c in clf_names]
    auc_means = [summary[c]['auc_mean']     for c in clf_names]
    auc_stds  = [summary[c]['auc_std']      for c in clf_names]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    eb_kw = dict(elinewidth=0.8, capthick=0.8, capsize=3)

    bars1 = ax.bar(x - w/2, bal_means, w, yerr=bal_stds, error_kw=eb_kw,
                   color=[CLF_COLORS[c] for c in clf_names],
                   alpha=0.85, label='Balanced accuracy', edgecolor='white', linewidth=0)
    ax.bar(x + w/2, auc_means, w, yerr=auc_stds, error_kw=eb_kw,
           color=[CLF_COLORS[c] for c in clf_names],
           alpha=0.40, label='AUC-ROC', edgecolor=[CLF_COLORS[c] for c in clf_names],
           linewidth=0.8, hatch='///')

    ax.axhline(0.5, color='#C0392B', ls='--', lw=0.9, alpha=0.8, label='Chance level')
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(['LR', 'RF', 'SVM'], fontsize=8)
    ax.set_ylabel('Score')
    ax.set_xlabel('Classifier')

    short = task_name.split('—')[-1].strip() if '—' in task_name else task_name
    ax.set_title(short, pad=6)

    for bar, val, std in zip(bars1, bal_means, bal_stds):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (std or 0) + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    ax.legend(frameon=False, loc='upper right', ncol=1)
    _despine(ax)
    _subtick(ax, 'y')
    plt.tight_layout()
    save(fig, path)


def plot_confusion_matrices(summary, path):
    _paper_rc()
    clf_names   = list(summary.keys())
    short_names = {'Logistic Regression': 'LR', 'Random Forest': 'RF', 'SVM (RBF)': 'SVM'}

    fig, axes = plt.subplots(1, len(clf_names),
                             figsize=(len(clf_names) * 2.6, 2.8))
    if len(clf_names) == 1:
        axes = [axes]

    for ax, clf_name in zip(axes, clf_names):
        cm          = summary[clf_name]['confusion'].astype(float)
        class_names = summary[clf_name]['class_names']
        cm_norm     = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))

        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = cm_norm[i, j]
                ax.text(j, i, f'{val:.2f}',
                        ha='center', va='center', fontsize=8,
                        color='white' if val > 0.6 else '#333333')

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel('Predicted', fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel('True', fontsize=8)

        bal = summary[clf_name]['bal_acc_mean']
        auc = summary[clf_name]['auc_mean']
        astr = f', AUC={auc:.2f}' if not np.isnan(auc) else ''
        ax.set_title(f'{short_names.get(clf_name, clf_name)}\nBal.Acc={bal:.2f}{astr}',
                     fontsize=8, pad=4)

    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04, label='Recall')
    plt.tight_layout()
    save(fig, path)


def plot_per_fold_accuracy(fold_results, path):
    _paper_rc()
    rows = []
    for clf_name, folds in fold_results.items():
        for f in folds:
            rows.append({'Classifier': clf_name, 'Patient': f['patient'],
                         'Disease': f['disease'], 'Bal_Acc': f['bal_acc']})
    df_plot = pd.DataFrame(rows)
    if df_plot.empty:
        return

    clf_names   = list(fold_results.keys())
    short_names = {'Logistic Regression': 'LR', 'Random Forest': 'RF', 'SVM (RBF)': 'SVM'}
    x_pos       = {c: i for i, c in enumerate(clf_names)}
    rng         = np.random.RandomState(42)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    disease_markers = [('Healthy', 'o'), ('TAA', 's'), ('BAV', '^')]
    for clf_name in clf_names:
        sub    = df_plot[df_plot['Classifier'] == clf_name]
        jitter = rng.uniform(-0.15, 0.15, len(sub))
        for dis, marker in disease_markers:
            mask = sub['Disease'] == dis
            if not mask.any():
                continue
            idx = np.where(mask.values)[0]
            ax.scatter(x_pos[clf_name] + jitter[idx],
                       sub.loc[mask, 'Bal_Acc'].values,
                       c=DISEASE_COLORS.get(dis, '#AAA'),
                       marker=marker, s=30, alpha=0.75,
                       edgecolors='white', linewidths=0.3,
                       label=dis if clf_name == clf_names[0] else '_nolegend_',
                       zorder=3)
        mean_acc = sub['Bal_Acc'].mean()
        ax.plot([x_pos[clf_name] - 0.28, x_pos[clf_name] + 0.28],
                [mean_acc, mean_acc],
                color='#333333', lw=1.8, zorder=5)
        ax.text(x_pos[clf_name] + 0.31, mean_acc,
                f'{mean_acc:.2f}', va='center', fontsize=7, color='#333333')

    ax.axhline(0.5, color='#C0392B', ls='--', lw=0.9, alpha=0.8, label='Chance level')
    ax.set_xticks(range(len(clf_names)))
    ax.set_xticklabels([short_names.get(c, c) for c in clf_names])
    ax.set_ylabel('Balanced accuracy (per patient fold)')
    ax.set_xlabel('Classifier')
    ax.set_ylim(-0.05, 1.15)

    handles = [plt.scatter([], [], c=DISEASE_COLORS.get(d, '#AAA'),
                           marker=m, s=25, label=d, edgecolors='white')
               for d, m in disease_markers if d in df_plot['Disease'].values]
    handles.append(plt.Line2D([0], [0], color='#C0392B', ls='--', lw=0.9,
                               label='Chance level'))
    ax.legend(handles=handles, frameon=False, loc='lower right', handletextpad=0.4)
    _despine(ax)
    _subtick(ax, 'y')
    plt.tight_layout()
    save(fig, path)


def plot_roc_curves(fold_results, class_names, path):
    if len(class_names) != 2:
        return
    _paper_rc()

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.plot([0, 1], [0, 1], color='#AAAAAA', ls='--', lw=0.8, label='Chance (AUC=0.50)')

    for clf_name, folds in fold_results.items():
        folds_with_prob = [f for f in folds if f.get('y_prob') is not None]
        if not folds_with_prob:
            continue
        y_true_all = np.concatenate([f['y_true'] for f in folds_with_prob])
        y_prob_all = np.concatenate([f['y_prob'] for f in folds_with_prob])
        y_bin = (np.array(y_true_all) == class_names[1]).astype(int)
        try:
            fpr, tpr, _ = roc_curve(y_bin, y_prob_all)
            auc_val = roc_auc_score(y_bin, y_prob_all)
            short = {'Logistic Regression': 'LR', 'Random Forest': 'RF',
                     'SVM (RBF)': 'SVM'}.get(clf_name, clf_name)
            ax.plot(fpr, tpr, color=CLF_COLORS[clf_name], lw=1.6,
                    label=f'{short} (AUC={auc_val:.2f})')
        except Exception as e:
            print(f"  ROC failed for {clf_name}: {e}")

    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.legend(frameon=False, loc='lower right')
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.04)
    _despine(ax)
    plt.tight_layout()
    save(fig, path)


def plot_per_class_metrics(summary, path):
    _paper_rc()
    clf_names   = list(summary.keys())
    class_names = summary[clf_names[0]]['class_names']
    short_clf   = {'Logistic Regression': 'LR', 'Random Forest': 'RF', 'SVM (RBF)': 'SVM'}
    metrics     = ['precision', 'recall', 'f1']
    metric_labels  = {'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1'}
    metric_colors  = {'precision': '#2980B9', 'recall': '#C0392B', 'f1': '#27AE60'}

    ncols = len(class_names)
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 2.8, 3.2), sharey=True)
    if ncols == 1:
        axes = [axes]

    x = np.arange(len(clf_names))
    w = 0.22

    for ax, cls in zip(axes, class_names):
        for i, metric in enumerate(metrics):
            vals = [summary[c][metric][cls] for c in clf_names]
            ax.bar(x + (i - 1) * w, vals, w,
                   color=metric_colors[metric], alpha=0.82,
                   label=metric_labels[metric], edgecolor='white', linewidth=0)
        ax.set_xticks(x)
        ax.set_xticklabels([short_clf.get(c, c) for c in clf_names])
        ax.set_title(cls, pad=4)
        ax.set_ylim(0, 1.12)
        if ax == axes[0]:
            ax.set_ylabel('Score')
            ax.legend(frameon=False, loc='lower left', fontsize=7)
        ax.axhline(0.5, color='#AAAAAA', ls=':', lw=0.8)
        _despine(ax)
        _subtick(ax, 'y')

    plt.tight_layout()
    save(fig, path)


def plot_feature_importance(feat_cols, values, stds, title, path,
                            signed=False, top_n=15):
    _paper_rc()
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
    colors = [ORGANELLE_COLORS.get(get_organelle(f), '#AAA') for f in feats]

    fig, ax = plt.subplots(figsize=(5.5, max(3.5, len(feats) * 0.30)))
    eb_kw = dict(elinewidth=0.7, capthick=0.7, capsize=2, ecolor='#555555')
    ax.barh(range(len(feats)), vals, xerr=stds_s, error_kw=eb_kw,
            color=colors, edgecolor='white', linewidth=0, height=0.65, alpha=0.85)
    if signed:
        ax.axvline(0, color='#333333', lw=0.8)

    labels = []
    for f in feats:
        parts = f.split('_', 1)
        label = parts[1].replace('_', ' ') if len(parts) > 1 else f.replace('_', ' ')
        labels.append(label)

    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(labels, fontsize=7.5)
    xlabel = ('Coefficient (mean ± SD, LOPO folds)' if signed
              else 'Importance (mean ± SD, LOPO folds)')
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=5)
    _despine(ax)
    _subtick(ax, 'x')

    patches = [mpatches.Patch(color=c, label=o, alpha=0.85)
               for o, c in ORGANELLE_COLORS.items()]
    ax.legend(handles=patches, frameon=False, loc='lower right', fontsize=7)
    plt.tight_layout()
    save(fig, path)


def plot_task_comparison(summary_t1, summary_t2, path):
    _paper_rc()
    tasks = {
        'Task 1: Healthy vs Diseased': summary_t1,
        'Task 2: BAV vs TAV-ATAA':     summary_t2,
    }
    clf_names   = list(next(iter(tasks.values())).keys())
    short_clf   = {'Logistic Regression': 'LR', 'Random Forest': 'RF', 'SVM (RBF)': 'SVM'}
    x = np.arange(len(clf_names))
    w = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5), sharey=True)
    eb_kw = dict(elinewidth=0.8, capthick=0.8, capsize=3)

    for ax, (task_label, summary) in zip(axes, tasks.items()):
        bal_means = [summary[c]['bal_acc_mean'] for c in clf_names]
        bal_stds  = [summary[c]['bal_acc_std']  for c in clf_names]
        auc_means = [summary[c]['auc_mean']     for c in clf_names]
        auc_stds  = [summary[c]['auc_std']      for c in clf_names]

        ax.bar(x - w/2, bal_means, w, yerr=bal_stds, error_kw=eb_kw,
               color=[CLF_COLORS[c] for c in clf_names],
               alpha=0.85, edgecolor='white', linewidth=0, label='Balanced accuracy')
        ax.bar(x + w/2, auc_means, w, yerr=auc_stds, error_kw=eb_kw,
               color=[CLF_COLORS[c] for c in clf_names],
               alpha=0.38, edgecolor=[CLF_COLORS[c] for c in clf_names],
               linewidth=0.8, hatch='///', label='AUC-ROC')

        ax.axhline(0.5, color='#C0392B', ls='--', lw=0.9, alpha=0.8)
        ax.set_ylim(0, 1.12)
        ax.set_xticks(x)
        ax.set_xticklabels([short_clf.get(c, c) for c in clf_names])
        ax.set_title(task_label, pad=5)
        if ax == axes[0]:
            ax.set_ylabel('Score')
        ax.set_xlabel('Classifier')

        for i, (val, std) in enumerate(zip(bal_means, bal_stds)):
            if not np.isnan(val):
                ax.text(x[i] - w/2, val + (std or 0) + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7)
        _despine(ax)
        _subtick(ax, 'y')

    handles = ([mpatches.Patch(color=CLF_COLORS[c], label=short_clf.get(c, c))
                for c in clf_names] +
               [mpatches.Patch(facecolor='#888888', alpha=0.85, label='Balanced accuracy'),
                mpatches.Patch(facecolor='#888888', alpha=0.38, hatch='///', label='AUC-ROC'),
                plt.Line2D([0], [0], color='#C0392B', ls='--', lw=0.9, label='Chance level')])
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save(fig, path)


def plot_feature_overlap(feat_cols, imp_t1, imp_t2, path):
    _paper_rc()

    def norm01(x):
        r = np.abs(x) - np.abs(x).min()
        return r / r.max() if r.max() > 0 else r

    i1 = norm01(imp_t1)
    i2 = norm01(imp_t2)
    colors = [ORGANELLE_COLORS.get(get_organelle(f), '#AAA') for f in feat_cols]

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.scatter(i1, i2, c=colors, s=22, alpha=0.75,
               edgecolors='white', linewidths=0.3, zorder=3)

    threshold = 0.55
    for f, x, y in zip(feat_cols, i1, i2):
        if x > threshold or y > threshold:
            parts = f.split('_', 1)
            label = parts[1].replace('_', ' ') if len(parts) > 1 else f.replace('_', ' ')
            ax.annotate(label, (x, y), fontsize=6.5, alpha=0.9,
                        xytext=(4, 3), textcoords='offset points')

    ax.axhline(threshold, color='#AAAAAA', ls=':', lw=0.8)
    ax.axvline(threshold, color='#AAAAAA', ls=':', lw=0.8)
    ax.set_xlabel('Normalised importance — Task 1 (Healthy vs Diseased)')
    ax.set_ylabel('Normalised importance — Task 2 (BAV vs TAV-ATAA)')

    patches = [mpatches.Patch(color=c, label=o, alpha=0.85)
               for o, c in ORGANELLE_COLORS.items()]
    ax.legend(handles=patches, frameon=False, fontsize=7.5)
    _despine(ax)
    plt.tight_layout()
    save(fig, path)


def print_thesis_numbers(summary_t1, summary_t2):
    print("\n" + "=" * 70)
    print("NUMBERS FOR THESIS TEXT")
    print("=" * 70)
    for task_label, summary in [("Task 1 — Healthy vs Diseased", summary_t1),
                                  ("Task 2 — BAV vs TAV-ATAA",     summary_t2)]:
        print(f"\n  {task_label}")
        print(f"  {'-'*50}")
        for clf_name, s in summary.items():
            auc_str = (f"  AUC={s['auc_mean']:.2f}±{s['auc_std']:.2f}"
                       if not np.isnan(s['auc_mean']) else "")
            print(f"  {clf_name:25s}  "
                  f"Bal.Acc={s['bal_acc_mean']:.2f}±{s['bal_acc_std']:.2f}"
                  f"{auc_str}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    for d in [BASE, TASK1_DIR, TASK2_DIR, COMP_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MORPHOLOGICAL CLASSIFICATION — AORTIC ANEURYSM SUBTYPES")
    print("Leave-One-Patient-Out Cross-Validation")
    print("=" * 70)

    df, feat_cols = load_data()

    # ── TASK 1: Healthy vs Diseased ───────────────────────────────────────────
    df_t1 = df.copy()
    df_t1['label'] = df_t1['Disease'].map(
        {'Healthy': 'Healthy', 'TAA': 'Diseased', 'BAV': 'Diseased'})
    summary_t1, _, imp_t1 = run_lopo_cv(
        df_t1, feat_cols,
        task_name   = 'Task 1 — Healthy vs Diseased (TAV + BAV pooled)',
        label_col   = 'label',
        class_names = ['Healthy', 'Diseased'],
        out_dir     = TASK1_DIR)

    # ── TASK 2: BAV vs TAV-ATAA ───────────────────────────────────────────────
    df_t2 = df[df['Disease'].isin(['TAA', 'BAV'])].copy()
    df_t2['label'] = df_t2['Disease']
    summary_t2, _, imp_t2 = run_lopo_cv(
        df_t2, feat_cols,
        task_name   = 'Task 2 — BAV vs TAV-ATAA (subtype discrimination)',
        label_col   = 'label',
        class_names = ['TAA', 'BAV'],
        out_dir     = TASK2_DIR)

    # ── Cross-task comparison ─────────────────────────────────────────────────
    print("\n  Generating comparison plots...")
    plot_task_comparison(summary_t1, summary_t2,
                         COMP_DIR / '01_task_comparison.png')

    if imp_t1.get('Random Forest') and imp_t2.get('Random Forest'):
        rf_imp_t1 = np.mean(imp_t1['Random Forest'], axis=0)
        rf_imp_t2 = np.mean(imp_t2['Random Forest'], axis=0)
        plot_feature_overlap(feat_cols, rf_imp_t1, rf_imp_t2,
                             COMP_DIR / '02_feature_importance_overlap.png')

    print_thesis_numbers(summary_t1, summary_t2)

    print(f"\nAll outputs in: {BASE}/")
    for sub in [TASK1_DIR, TASK2_DIR, COMP_DIR]:
        files = sorted(sub.glob('*'))
        print(f"  {sub.name}/  ({len(files)} files)")


if __name__ == '__main__':
    import os
    os.chdir(ss.PROJ_ROOT)
    main()
