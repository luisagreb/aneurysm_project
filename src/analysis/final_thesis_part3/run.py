"""
Part 3 — Classification
=======================
Task 1: Diseased (TAV-ATAA + BAV-ATAA) vs Healthy (TAV-NA)
Task 2: TAV-ATAA vs BAV-ATAA

Pipeline (both tasks):
  - Patient-level train/test split (GroupShuffleSplit, 80/20, stratified)
  - Leave-One-Patient-Out CV for model selection
  - Models: Random Forest, Logistic Regression, SVM
  - Cell-level prediction at test time (cells from held-out patients)
  - Two-stage option: Task 1 first, then Task 2 on predicted-diseased cells

Outputs: outputs/final_thesis_part3/
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared_stats as ss

warnings.filterwarnings('ignore')

# ── output directories ────────────────────────────────────────────────────────
BASE   = Path(__file__).resolve().parents[3] / 'outputs' / 'final_thesis_part3'
T1_DIR = BASE / 'task1_diseased_vs_healthy'
T2_DIR = BASE / 'task2_tav_vs_bav'
for d in [T1_DIR, T2_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20

MODELS = {
    'RandomForest':          RandomForestClassifier(
                                 n_estimators=300, max_depth=None,
                                 class_weight='balanced', random_state=RANDOM_STATE),
    'LogisticRegression':    LogisticRegression(
                                 max_iter=2000, class_weight='balanced',
                                 solver='lbfgs', random_state=RANDOM_STATE),
    'SVM':                   SVC(
                                 kernel='rbf', class_weight='balanced',
                                 probability=True, random_state=RANDOM_STATE),
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def load_features():
    """Load all three groups, extract Subject, return df + feat_cols."""
    df = pd.read_csv(ss.FEATURES_FILE)
    df = df[df['Disease'].isin(['Healthy', 'TAA', 'BAV'])].copy()
    df['Subject'] = df['CellName'].apply(ss.norm_id)
    feat_cols = sorted([c for c in df.columns
                        if c.startswith(('Actin_', 'Mito_', 'Nucleus_'))])
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    return df, feat_cols


def patient_split(df, feat_cols, label_col, test_size=TEST_SIZE):
    """
    Split at patient level to prevent data leakage.
    Returns (X_train, X_test, y_train, y_test, subjects_test).
    Scales features with StandardScaler fitted on train only.
    """
    subjects = df['Subject'].values
    labels   = df[label_col].values

    # Stratified patient-level split
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                            random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, labels, groups=subjects))

    X_train = df.iloc[train_idx][feat_cols].fillna(0).values
    X_test  = df.iloc[test_idx][feat_cols].fillna(0).values
    y_train = labels[train_idx]
    y_test  = labels[test_idx]
    subj_test = subjects[test_idx]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, subj_test, scaler, train_idx, test_idx


def lopo_cv(df, feat_cols, label_col, clf, clf_name):
    """
    Leave-One-Patient-Out CV at cell level.
    Returns mean balanced accuracy across folds.
    """
    logo = LeaveOneGroupOut()
    X    = df[feat_cols].fillna(0).values
    y    = df[label_col].values
    g    = df['Subject'].values

    scores = []
    for train_idx, test_idx in logo.split(X, y, groups=g):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[train_idx])
        Xte = sc.transform(X[test_idx])
        clf_ = type(clf)(**clf.get_params())
        try:
            clf_.fit(Xtr, y[train_idx])
            pred = clf_.predict(Xte)
            scores.append(balanced_accuracy_score(y[test_idx], pred))
        except Exception:
            pass
    mean_score = float(np.mean(scores)) if scores else 0.0
    print(f"    LOPO CV  {clf_name:22s}: balanced_acc = {mean_score:.3f}  "
          f"(n_folds={len(scores)})")
    return mean_score


def evaluate_and_save(clf, clf_name, X_train, X_test, y_train, y_test,
                      class_names, out_dir, task_label):
    """Fit, predict, save confusion matrix + ROC + report."""
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test) if hasattr(clf, 'predict_proba') else None

    acc  = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    print(f"    {clf_name:22s}: acc={acc:.3f}  balanced_acc={bacc:.3f}")

    # Confusion matrix
    cm  = confusion_matrix(y_test, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, colorbar=False)
    ax.set_title(f'{task_label} — {clf_name}\nbal_acc={bacc:.3f}', fontsize=12)
    plt.tight_layout()
    ss.save(fig, out_dir / f'{clf_name}_confusion.png')

    # ROC (binary or one-vs-rest for multi-class)
    if y_prob is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        if len(class_names) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1], pos_label=class_names[1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f}')
        else:
            y_bin = label_binarize(y_test, classes=class_names)
            for i, cn in enumerate(class_names):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2, label=f'{cn} AUC={roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{task_label} — {clf_name} ROC', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        ss.save(fig, out_dir / f'{clf_name}_roc.png')

    # Text report
    report = classification_report(y_test, y_pred, target_names=class_names)
    (out_dir / f'{clf_name}_report.txt').write_text(
        f"{task_label} — {clf_name}\n"
        f"acc={acc:.3f}  balanced_acc={bacc:.3f}\n\n{report}"
    )
    return clf, bacc


def feature_importance_plot(clf, clf_name, feat_cols, out_dir, title,
                             X_test=None, y_test=None):
    """Bar chart of feature importances (RF) or permutation importance."""
    import matplotlib.patches as mpatches

    if hasattr(clf, 'feature_importances_'):
        imp = clf.feature_importances_
    elif X_test is not None and y_test is not None:
        r   = permutation_importance(clf, X_test, y_test,
                                     n_repeats=10, random_state=RANDOM_STATE)
        imp = r.importances_mean
    else:
        return

    idx  = np.argsort(imp)[-20:]
    feats = [feat_cols[i] for i in idx]
    vals  = imp[idx]
    colors = [ss.ORGANELLE_COLORS.get(ss.get_organelle(f), '#AAA') for f in feats]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(feats)), vals, color=colors, edgecolor='white')
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([f.replace('_', ' ') for f in feats], fontsize=10)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'{title}\nTop-20 features', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    patches = [mpatches.Patch(color=c, label=o) for o, c in ss.ORGANELLE_COLORS.items()]
    ax.legend(handles=patches, fontsize=10)
    plt.tight_layout()
    ss.save(fig, out_dir / f'{clf_name}_feature_importance.png')


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — Diseased vs Healthy
# ══════════════════════════════════════════════════════════════════════════════
def task1(df, feat_cols):
    print("\n" + "=" * 70)
    print("TASK 1 — Diseased (TAV-ATAA + BAV-ATAA) vs Healthy (TAV-NA)")
    print("=" * 70)

    df1 = df.copy()
    df1['Label'] = df1['Disease'].map({'Healthy': 'Healthy', 'TAA': 'Diseased', 'BAV': 'Diseased'})
    class_names  = ['Healthy', 'Diseased']

    print(f"  Class counts: {df1['Label'].value_counts().to_dict()}")
    print(f"  Unique patients: {df1['Subject'].nunique()}")

    # LOPO CV
    print("\n  LOPO-CV (model selection):")
    lopo_scores = {}
    for name, clf in MODELS.items():
        lopo_scores[name] = lopo_cv(df1, feat_cols, 'Label', clf, name)

    best_name = max(lopo_scores, key=lopo_scores.get)
    print(f"\n  Best model (LOPO): {best_name} (balanced_acc={lopo_scores[best_name]:.3f})")

    # Final evaluation on held-out patients
    print("\n  Hold-out evaluation:")
    X_tr, X_te, y_tr, y_te, subj_te, scaler, train_idx, test_idx = \
        patient_split(df1, feat_cols, 'Label')

    print(f"  Train: {len(y_tr)} cells from "
          f"{df1.iloc[train_idx]['Subject'].nunique()} patients")
    print(f"  Test:  {len(y_te)} cells from "
          f"{df1.iloc[test_idx]['Subject'].nunique()} patients: "
          f"{df1.iloc[test_idx]['Subject'].unique().tolist()}")

    results = {}
    for name, clf in MODELS.items():
        clf_fit, bacc = evaluate_and_save(
            clf, name, X_tr, X_te, y_tr, y_te, class_names, T1_DIR,
            'Task 1 — Diseased vs Healthy')
        results[name] = (clf_fit, bacc)
        feature_importance_plot(clf_fit, name, feat_cols, T1_DIR,
                                f'Task 1 — {name}', X_te, y_te)

    # Summary bar
    _plot_model_comparison(lopo_scores, results, T1_DIR / 'model_comparison.png',
                           'Task 1 — Diseased vs Healthy')

    # Return best fitted clf + scaler for two-stage
    best_clf = results[best_name][0]
    return best_clf, scaler, train_idx, test_idx, df1


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — TAV-ATAA vs BAV-ATAA  (standalone + two-stage)
# ══════════════════════════════════════════════════════════════════════════════
def task2(df, feat_cols, t1_clf=None, t1_scaler=None):
    print("\n" + "=" * 70)
    print("TASK 2 — TAV-ATAA vs BAV-ATAA")
    print("=" * 70)

    df2 = df[df['Disease'].isin(['TAA', 'BAV'])].copy()
    df2['Label'] = df2['Disease'].map({'TAA': 'TAV-ATAA', 'BAV': 'BAV-ATAA'})
    class_names  = ['TAV-ATAA', 'BAV-ATAA']

    print(f"  Class counts: {df2['Label'].value_counts().to_dict()}")
    print(f"  Unique patients: {df2['Subject'].nunique()}")

    # LOPO CV
    print("\n  LOPO-CV (model selection):")
    lopo_scores = {}
    for name, clf in MODELS.items():
        lopo_scores[name] = lopo_cv(df2, feat_cols, 'Label', clf, name)

    best_name = max(lopo_scores, key=lopo_scores.get)
    print(f"\n  Best model (LOPO): {best_name} (balanced_acc={lopo_scores[best_name]:.3f})")

    # Final evaluation
    print("\n  Hold-out evaluation:")
    X_tr, X_te, y_tr, y_te, subj_te, scaler, train_idx, test_idx = \
        patient_split(df2, feat_cols, 'Label')

    print(f"  Train: {len(y_tr)} cells / Test: {len(y_te)} cells")

    results = {}
    for name, clf in MODELS.items():
        clf_fit, bacc = evaluate_and_save(
            clf, name, X_tr, X_te, y_tr, y_te, class_names, T2_DIR,
            'Task 2 — TAV-ATAA vs BAV-ATAA')
        results[name] = (clf_fit, bacc)
        feature_importance_plot(clf_fit, name, feat_cols, T2_DIR,
                                f'Task 2 — {name}', X_te, y_te)

    _plot_model_comparison(lopo_scores, results, T2_DIR / 'model_comparison.png',
                           'Task 2 — TAV-ATAA vs BAV-ATAA')

    # ── Two-stage: classify diseased first, then TAV vs BAV ───────────────────
    if t1_clf is not None and t1_scaler is not None:
        print("\n  Two-stage pipeline: apply Task 1 filter first...")
        X_all   = df[feat_cols].fillna(0).values
        X_all_s = t1_scaler.transform(X_all)
        pred_stage1 = t1_clf.predict(X_all_s)
        diseased_mask = pred_stage1 == 'Diseased'
        df_diseased   = df[diseased_mask].copy()
        true_diseased = df_diseased['Disease'].isin(['TAA', 'BAV'])

        print(f"  Stage 1: {diseased_mask.sum()} cells predicted diseased "
              f"({true_diseased.sum()} truly diseased)")

        df_two = df_diseased[df_diseased['Disease'].isin(['TAA', 'BAV'])].copy()
        df_two['Label'] = df_two['Disease'].map({'TAA': 'TAV-ATAA', 'BAV': 'BAV-ATAA'})
        if len(df_two) < 10:
            print("  Two-stage: not enough diseased cells for stage 2 evaluation.")
            return

        X_tw = df_two[feat_cols].fillna(0).values
        y_tw = df_two['Label'].values
        sc2  = StandardScaler()
        X_tw_s = sc2.fit_transform(X_tw)

        print("\n  Two-stage LOPO-CV:")
        for name, clf in MODELS.items():
            lopo_cv(df_two, feat_cols, 'Label', clf, name)

        out_two = T2_DIR / 'two_stage'
        out_two.mkdir(exist_ok=True)
        for name, clf in MODELS.items():
            clf2 = type(clf)(**clf.get_params())
            clf2.fit(X_tw_s, y_tw)
            y_pred2 = clf2.predict(X_tw_s)
            bacc2   = balanced_accuracy_score(y_tw, y_pred2)
            print(f"    Two-stage {name:22s}: balanced_acc={bacc2:.3f} (all diseased cells)")
            evaluate_and_save(clf2, f'TwoStage_{name}',
                              X_tw_s, X_tw_s, y_tw, y_tw,
                              class_names, out_two,
                              'Task 2 Two-Stage — TAV-ATAA vs BAV-ATAA')


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL HELPER: model comparison bar
# ══════════════════════════════════════════════════════════════════════════════
def _plot_model_comparison(lopo_scores, results, path, title):
    names     = list(MODELS.keys())
    lopo_vals = [lopo_scores.get(n, 0) for n in names]
    hold_vals = [results[n][1] if n in results else 0 for n in names]

    x    = np.arange(len(names))
    bw   = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - bw/2, lopo_vals, width=bw, label='LOPO-CV',   color='#8E44AD', alpha=0.85)
    ax.bar(x + bw/2, hold_vals, width=bw, label='Hold-out',  color='#E67E22', alpha=0.85)
    ax.axhline(0.5, color='red', ls='--', lw=1.2, label='Chance (0.5)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('Balanced Accuracy', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    ss.save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    df, feat_cols = load_features()
    print(f"  Total cells: {len(df)}  |  Features: {len(feat_cols)}")
    for dis in ['Healthy', 'TAA', 'BAV']:
        sub = df[df['Disease'] == dis]
        print(f"    {ss.GROUP_LABELS.get(dis, dis):12s}: "
              f"{sub['Subject'].nunique()} patients, {len(sub)} cells")

    t1_clf, t1_scaler, _, _, _ = task1(df, feat_cols)
    task2(df, feat_cols, t1_clf=t1_clf, t1_scaler=t1_scaler)

    print("\n" + "=" * 70)
    print("PART 3 COMPLETE")
    print(f"  Outputs → {BASE}")
    print("=" * 70)


if __name__ == '__main__':
    import os
    os.chdir(ss.PROJ_ROOT)
    main()
