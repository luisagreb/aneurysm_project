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

import re
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
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
    plot_confusion_matrices(summary, task_name,
                            out_dir / '02_confusion_matrices.png')
    plot_per_fold_accuracy(fold_results, task_name,
                           out_dir / '03_per_fold_accuracy.png')
    plot_roc_curves(fold_results, class_names, task_name,
                    out_dir / '04_roc_curves.png')
    plot_per_class_metrics(summary, task_name,
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
# PLOT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def plot_performance_summary(summary, task_name, path):
    clf_names = list(summary.keys())
    x = np.arange(len(clf_names))
    w = 0.35

    bal_means = [summary[c]['bal_acc_mean'] for c in clf_names]
    bal_stds  = [summary[c]['bal_acc_std']  for c in clf_names]
    auc_means = [summary[c]['auc_mean']     for c in clf_names]
    auc_stds  = [summary[c]['auc_std']      for c in clf_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - w/2, bal_means, w, yerr=bal_stds, capsize=5,
                   color=[CLF_COLORS[c] for c in clf_names],
                   alpha=0.92, label='Balanced Accuracy', edgecolor='white')
    bars2 = ax.bar(x + w/2, auc_means, w, yerr=auc_stds, capsize=5,
                   color=[CLF_COLORS[c] for c in clf_names],
                   alpha=0.50, label='AUC-ROC', edgecolor='white', hatch='//')

    ax.axhline(0.5, color='red', ls='--', lw=1.5, label='Random baseline (0.5)')
    ax.set_ylim(0, 1.18)
    ax.set_xticks(x)
    ax.set_xticklabels(clf_names, fontsize=12)
    ax.set_ylabel('Score', fontsize=13)
    ax.set_title(f'{task_name}\nClassification Performance — LOPO-CV\nMean ± SD across patient folds',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    for bar, val, std in zip(bars1, bal_means, bal_stds):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + std + 0.025,
                    f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')
    for bar, val, std in zip(bars2, auc_means, auc_stds):
        if not np.isnan(val):
            s = std if not np.isnan(std) else 0
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + s + 0.025,
                    f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    save(fig, path)


def plot_confusion_matrices(summary, task_name, path):
    clf_names = list(summary.keys())
    fig, axes = plt.subplots(1, len(clf_names),
                             figsize=(len(clf_names)*5, 5))
    if len(clf_names) == 1:
        axes = [axes]
    fig.suptitle(f'{task_name}\nConfusion Matrices — LOPO-CV (normalised by true class)',
                 fontsize=13, fontweight='bold')

    for ax, clf_name in zip(axes, clf_names):
        cm          = summary[clf_name]['confusion'].astype(float)
        class_names = summary[clf_name]['class_names']
        cm_norm     = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))

        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, vmin=0, vmax=1,
                    annot_kws={'size': 13, 'fontweight': 'bold'},
                    linewidths=0.5, linecolor='lightgray')
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        bal  = summary[clf_name]['bal_acc_mean']
        auc  = summary[clf_name]['auc_mean']
        astr = f'  AUC={auc:.2f}' if not np.isnan(auc) else ''
        ax.set_title(f'{clf_name}\nBal.Acc={bal:.2f}{astr}',
                     fontsize=11, fontweight='bold')
    plt.tight_layout()
    save(fig, path)


def plot_per_fold_accuracy(fold_results, task_name, path):
    rows = []
    for clf_name, folds in fold_results.items():
        for f in folds:
            rows.append({'Classifier': clf_name, 'Patient': f['patient'],
                         'Disease': f['disease'], 'Bal_Acc': f['bal_acc']})
    df_plot = pd.DataFrame(rows)
    if df_plot.empty:
        return

    clf_names = list(fold_results.keys())
    x_pos     = {c: i for i, c in enumerate(clf_names)}
    rng       = np.random.RandomState(42)

    fig, ax = plt.subplots(figsize=(11, 6))

    disease_markers = [('Healthy', 'o'), ('TAA', 's'), ('BAV', '^')]
    for clf_name in clf_names:
        sub    = df_plot[df_plot['Classifier'] == clf_name]
        jitter = rng.uniform(-0.18, 0.18, len(sub))
        for dis, marker in disease_markers:
            mask = sub['Disease'] == dis
            if not mask.any():
                continue
            idx = np.where(mask.values)[0]
            ax.scatter(x_pos[clf_name] + jitter[idx],
                       sub.loc[mask, 'Bal_Acc'].values,
                       c=DISEASE_COLORS.get(dis, '#AAA'),
                       marker=marker, s=90, alpha=0.85,
                       edgecolors='white', lw=0.5,
                       label=dis if clf_name == clf_names[0] else '_nolegend_')
        mean_acc = sub['Bal_Acc'].mean()
        ax.plot([x_pos[clf_name]-0.3, x_pos[clf_name]+0.3],
                [mean_acc, mean_acc],
                color=CLF_COLORS[clf_name], lw=3.5, zorder=5)
        ax.text(x_pos[clf_name], mean_acc + 0.04,
                f'{mean_acc:.2f}', ha='center', fontsize=10,
                fontweight='bold', color=CLF_COLORS[clf_name])

    ax.axhline(0.5, color='red', ls='--', lw=1.5, alpha=0.7,
               label='Random baseline (0.5)')
    ax.set_xticks(range(len(clf_names)))
    ax.set_xticklabels(clf_names, fontsize=12)
    ax.set_ylabel('Balanced Accuracy (per patient fold)', fontsize=12)
    ax.set_ylim(-0.05, 1.18)
    ax.set_title(f'{task_name} — Per-Patient Fold Accuracy\n'
                 'Each point = one held-out patient  |  Horizontal bar = mean',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    handles = [plt.scatter([], [], c=DISEASE_COLORS.get(d, '#AAA'),
                           marker=m, s=70, label=d, edgecolors='white')
               for d, m in disease_markers
               if d in df_plot['Disease'].values]
    handles.append(plt.Line2D([0], [0], color='red', ls='--',
                               label='Random baseline (0.5)'))
    ax.legend(handles=handles, fontsize=10, loc='lower right')
    plt.tight_layout()
    save(fig, path)


def plot_roc_curves(fold_results, class_names, task_name, path):
    if len(class_names) != 2:
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC=0.50)')

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
            ax.plot(fpr, tpr, color=CLF_COLORS[clf_name], lw=2.5,
                    label=f'{clf_name}  (AUC={auc_val:.2f})')
        except Exception as e:
            print(f"  ROC failed for {clf_name}: {e}")

    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title(f'{task_name} — ROC Curves\nAggregated across all LOPO folds',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    plt.tight_layout()
    save(fig, path)


def plot_per_class_metrics(summary, task_name, path):
    clf_names   = list(summary.keys())
    class_names = summary[clf_names[0]]['class_names']
    metrics     = ['precision', 'recall', 'f1']
    metric_labels = {'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1'}

    ncols = len(class_names)
    fig, axes = plt.subplots(1, ncols, figsize=(ncols*5.5, 5), sharey=True)
    if ncols == 1:
        axes = [axes]
    fig.suptitle(f'{task_name} — Per-Class Metrics (LOPO-CV)',
                 fontsize=13, fontweight='bold')

    x  = np.arange(len(clf_names))
    w  = 0.25
    metric_colors = {'precision': '#3498DB', 'recall': '#E74C3C', 'f1': '#2ECC71'}

    for ax, cls in zip(axes, class_names):
        for i, metric in enumerate(metrics):
            vals = [summary[c][metric][cls] for c in clf_names]
            ax.bar(x + (i-1)*w, vals, w,
                   color=metric_colors[metric],
                   alpha=0.85, label=metric_labels[metric],
                   edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(clf_names, fontsize=10, rotation=10)
        ax.set_title(f'Class: {cls}', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Score', fontsize=11)
        ax.axhline(0.5, color='grey', ls=':', lw=1, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        if ax == axes[0]:
            ax.legend(fontsize=10)
    plt.tight_layout()
    save(fig, path)


def plot_feature_importance(feat_cols, values, stds, title, path,
                            signed=False, top_n=20):
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

    fig, ax = plt.subplots(figsize=(11, max(6, len(feats)*0.52)))
    ax.barh(range(len(feats)), vals, xerr=stds_s, color=colors,
            edgecolor='white', height=0.7, capsize=3, alpha=0.88)
    if signed:
        ax.axvline(0, color='black', lw=0.9)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([f.replace('_', ' ') for f in feats], fontsize=11)
    ax.set_xlabel('Importance / Coefficient  (mean ± SD across LOPO folds)', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    patches = [mpatches.Patch(color=c, label=o) for o, c in ORGANELLE_COLORS.items()]
    ax.legend(handles=patches, fontsize=10, loc='lower right')
    plt.tight_layout()
    save(fig, path)


def plot_task_comparison(summary_t1, summary_t2, path):
    tasks = {
        'Task 1\nHealthy vs Diseased': summary_t1,
        'Task 2\nBAV vs TAV-ATAA':     summary_t2,
    }
    clf_names = list(next(iter(tasks.values())).keys())
    x = np.arange(len(clf_names))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle('Classification Performance — Both Tasks (LOPO-CV)\n'
                 'Solid = Balanced Accuracy  |  Hatched = AUC-ROC  |  '
                 'Red dashed = random baseline',
                 fontsize=13, fontweight='bold')

    for ax, (task_label, summary) in zip(axes, tasks.items()):
        bal_means = [summary[c]['bal_acc_mean'] for c in clf_names]
        bal_stds  = [summary[c]['bal_acc_std']  for c in clf_names]
        auc_means = [summary[c]['auc_mean']     for c in clf_names]
        auc_stds  = [summary[c]['auc_std']      for c in clf_names]

        ax.bar(x - w/2, bal_means, w, yerr=bal_stds, capsize=5,
               color=[CLF_COLORS[c] for c in clf_names],
               alpha=0.92, edgecolor='white')
        ax.bar(x + w/2, auc_means, w, yerr=auc_stds, capsize=5,
               color=[CLF_COLORS[c] for c in clf_names],
               alpha=0.50, edgecolor='white', hatch='//')

        ax.axhline(0.5, color='red', ls='--', lw=1.5, alpha=0.7)
        ax.set_ylim(0, 1.18)
        ax.set_xticks(x)
        ax.set_xticklabels(clf_names, fontsize=11, rotation=10)
        ax.set_title(task_label, fontsize=13, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        for i, (val, std) in enumerate(zip(bal_means, bal_stds)):
            if not np.isnan(val):
                ax.text(x[i] - w/2, val + std + 0.03,
                        f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')

    handles = ([mpatches.Patch(color=CLF_COLORS[c], label=c) for c in clf_names] +
               [mpatches.Patch(facecolor='grey', alpha=0.9, label='Balanced Accuracy'),
                mpatches.Patch(facecolor='grey', alpha=0.5, hatch='//', label='AUC-ROC'),
                plt.Line2D([0], [0], color='red', ls='--', label='Random baseline')])
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, -0.03))
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    save(fig, path)


def plot_feature_overlap(feat_cols, imp_t1, imp_t2, path):
    def norm01(x):
        r = np.abs(x) - np.abs(x).min()
        return r / r.max() if r.max() > 0 else r

    i1 = norm01(imp_t1)
    i2 = norm01(imp_t2)
    colors = [ORGANELLE_COLORS.get(get_organelle(f), '#AAA') for f in feat_cols]

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(i1, i2, c=colors, s=80, alpha=0.82, edgecolors='white', lw=0.5)

    threshold = 0.55
    for f, x, y in zip(feat_cols, i1, i2):
        if x > threshold or y > threshold:
            ax.annotate(f.replace('_', ' '), (x, y),
                        fontsize=8, alpha=0.85,
                        xytext=(5, 5), textcoords='offset points')

    ax.axhline(threshold, color='grey', ls=':', lw=1, alpha=0.5)
    ax.axvline(threshold, color='grey', ls=':', lw=1, alpha=0.5)
    ax.set_xlabel('Normalised Importance — Task 1 (Healthy vs Diseased)', fontsize=12)
    ax.set_ylabel('Normalised Importance — Task 2 (BAV vs TAV)', fontsize=12)
    ax.set_title('Feature Importance Overlap Across Tasks\n'
                 'Top-right = discriminative for both tasks',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    patches = [mpatches.Patch(color=c, label=o) for o, c in ORGANELLE_COLORS.items()]
    ax.legend(handles=patches, fontsize=10)
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
    summary_t1, folds_t1, imp_t1 = run_lopo_cv(
        df_t1, feat_cols,
        task_name   = 'Task 1 — Healthy vs Diseased (TAV + BAV pooled)',
        label_col   = 'label',
        class_names = ['Healthy', 'Diseased'],
        out_dir     = TASK1_DIR)

    # ── TASK 2: BAV vs TAV-ATAA ───────────────────────────────────────────────
    df_t2 = df[df['Disease'].isin(['TAA', 'BAV'])].copy()
    df_t2['label'] = df_t2['Disease']
    summary_t2, folds_t2, imp_t2 = run_lopo_cv(
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
