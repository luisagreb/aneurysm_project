"""
Rigorous 3-class Cell Classifier: Healthy / TAA / BAV
======================================================

Design:
  1. StratifiedGroupKFold (outer, 5-fold) — class balance + no patient leakage
  2. GridSearchCV with StratifiedGroupKFold (inner, 3-fold) — hyperparam tuning
     on training data only (true nested CV → unbiased performance estimate)
  3. Patient-level prediction aggregation — average cell probabilities per patient
     before computing final metrics (the unit of interest is the patient, not the cell)
  4. Tasks:
       A. 3-class: Healthy vs TAA vs BAV
       B. Pairwise binary: H vs TAA, H vs BAV, TAA vs BAV
  5. Metrics: macro-F1, per-class AUC (OVR), confusion matrix
  6. Best model retrained on full data → saved for app

Usage:
    cd /home/luisa/aneurysm_project
    python src/classification/rigorous_classifier.py
"""

import re
import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from sklearn.pipeline          import Pipeline
from sklearn.preprocessing     import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.model_selection   import (StratifiedGroupKFold, GridSearchCV,
                                       cross_val_predict, LeaveOneGroupOut)
from sklearn.metrics           import (f1_score, accuracy_score, roc_auc_score,
                                       confusion_matrix, ConfusionMatrixDisplay,
                                       classification_report)

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
FEATURES_FILE = 'outputs/Advanced_Features_Raw_Final.csv'
MODEL_OUT     = 'app/models/classifier.joblib'
OUT_DIR       = Path('classification_results/rigorous')
OUT_DIR.mkdir(parents=True, exist_ok=True)
Path('app/models').mkdir(parents=True, exist_ok=True)

DISEASE_ORDER  = ['Healthy', 'TAA', 'BAV']
LABEL_MAP      = {'Healthy': 0, 'TAA': 1, 'BAV': 2}
COLOR_MAP      = {'Healthy': '#27AE60', 'TAA': '#E74C3C', 'BAV': '#F39C12'}
RANDOM_STATE   = 42

# ── Patient ID extraction ──────────────────────────────────────────────────────
def norm_patient_id(text):
    """
    Extract a unique patient identifier that includes the disease prefix.
    Handles:  01Asc-0017  (Healthy)
              02Asc-0017  (BAV)      ← different patient despite same number
              03Rt-17     (TAA)
    Returns e.g. '01-17', '02-17', '03-17'
    """
    if pd.isna(text):
        return None
    s = str(text).strip()
    # capture prefix (01 / 02 / 03) and trailing number separately
    m = re.search(r'(0[123])[A-Za-z]+-?0*(\d+)', s, re.IGNORECASE)
    if m:
        return f"{m.group(1)}-{int(m.group(2))}"
    # fallback: any trailing number
    nums = re.findall(r'\d+', s)
    return nums[-1] if nums else None

# ── Load & prepare data ────────────────────────────────────────────────────────
print("=" * 65)
print("Rigorous 3-class Classifier  (Healthy / TAA / BAV)")
print("Nested CV  |  StratifiedGroupKFold  |  Patient-level metrics")
print("=" * 65)

df = pd.read_csv(FEATURES_FILE)
print(f"\nLoaded: {df.shape[0]} cells x {df.shape[1]} columns")

# Labels
df['Label'] = df['Disease'].map(LABEL_MAP)
df = df.dropna(subset=['Label'])
df['Label'] = df['Label'].astype(int)

# Patient groups — must be unique per patient (not just numeric ID)
df['Patient_ID'] = df['CellName'].apply(norm_patient_id)
df = df.dropna(subset=['Patient_ID'])

# Encode Patient_ID as integer group index for sklearn
patient_ids    = df['Patient_ID'].values
unique_patients = sorted(df['Patient_ID'].unique())
patient_enc    = {p: i for i, p in enumerate(unique_patients)}
groups         = np.array([patient_enc[p] for p in patient_ids])

# Features
drop_cols = {'CellName', 'Label', 'Patient_ID', 'Disease',
             'Voxel_Source', 'DataSource', 'Age', 'Gender', 'Collagen_Status',
             'Voxel_Z_µm', 'Voxel_Y_µm', 'Voxel_X_µm'}
feat_cols = [c for c in df.columns if c not in drop_cols
             and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
X      = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
y      = df['Label'].values

n_patients = len(unique_patients)
print(f"\nPatients : {n_patients}  (Healthy={sum(df.groupby('Patient_ID')['Label'].first()==0)}, "
      f"TAA={sum(df.groupby('Patient_ID')['Label'].first()==1)}, "
      f"BAV={sum(df.groupby('Patient_ID')['Label'].first()==2)})")
print(f"Cells    : {len(y)}  (Healthy={sum(y==0)}, TAA={sum(y==1)}, BAV={sum(y==2)})")
print(f"Features : {len(feat_cols)}")

# ── Pipelines & hyper-parameter grids ─────────────────────────────────────────
# SelectKBest is inside the pipeline → fitted only on training fold (no leakage)
BASE_PIPES = {
    'Random Forest': Pipeline([
        ('scaler',   StandardScaler()),
        ('selector', SelectKBest(f_classif, k=15)),
        ('clf',      RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE)),
    ]),
    'Gradient Boosting': Pipeline([
        ('scaler',   StandardScaler()),
        ('selector', SelectKBest(f_classif, k=15)),
        ('clf',      GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ]),
    'Logistic Regression': Pipeline([
        ('scaler',   StandardScaler()),
        ('selector', SelectKBest(f_classif, k=15)),
        ('clf',      LogisticRegression(multi_class='multinomial', solver='lbfgs',
                                        class_weight='balanced', max_iter=2000,
                                        random_state=RANDOM_STATE)),
    ]),
}

PARAM_GRIDS = {
    'Random Forest': {
        'selector__k':          [10, 15, 20],
        'clf__n_estimators':    [100, 300],
        'clf__max_depth':       [3, 5, None],
        'clf__min_samples_leaf':[2, 4],
    },
    'Gradient Boosting': {
        'selector__k':       [10, 15, 20],
        'clf__n_estimators': [100, 200],
        'clf__max_depth':    [2, 3],
        'clf__learning_rate':[0.05, 0.1],
    },
    'Logistic Regression': {
        'selector__k': [10, 15, 20],
        'clf__C':      [0.01, 0.1, 1.0],
    },
}

# ── Nested CV ─────────────────────────────────────────────────────────────────
def run_nested_cv(X, y, groups, label_names, task_name, n_outer=5, n_inner=3,
                  use_lopo=False):
    """
    Nested cross-validation with patient-level grouping.

    Outer loop : StratifiedGroupKFold (or LeaveOneGroupOut) for evaluation
    Inner loop : GridSearchCV with StratifiedGroupKFold for hyperparameter tuning

    Returns
    -------
    results : dict  per-model metrics (cell-level and patient-level)
    best    : str   name of the best model by macro-F1 (patient-level)
    all_probas : dict  name → (n_cells, n_classes) array of OOF probabilities
    """
    print(f"\n{'─'*65}")
    print(f"Task: {task_name}  |  classes: {label_names}")
    print(f"{'─'*65}")

    n_classes  = len(label_names)
    outer_cv   = (LeaveOneGroupOut() if use_lopo
                  else StratifiedGroupKFold(n_splits=n_outer,
                                            shuffle=True,
                                            random_state=RANDOM_STATE))
    inner_cv   = StratifiedGroupKFold(n_splits=n_inner,
                                       shuffle=True,
                                       random_state=0)

    results    = {}
    all_probas = {}

    for model_name, base_pipe in BASE_PIPES.items():
        # Skip GB for multiclass > 2 (slow; keep for binary tasks)
        if n_classes > 2 and model_name == 'Gradient Boosting':
            continue

        print(f"\n  [{model_name}]")
        cell_probas  = np.zeros((len(y), n_classes))
        best_params_folds = []

        for fold_i, (tr, te) in enumerate(outer_cv.split(X, y, groups)):
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]
            g_tr       = groups[tr]

            # Inner: tune hyperparameters on training fold only
            search = GridSearchCV(
                base_pipe,
                PARAM_GRIDS[model_name],
                cv=inner_cv,
                scoring='f1_macro',
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_tr, y_tr, groups=g_tr)
            best_params_folds.append(search.best_params_)

            # Predict on held-out test fold
            cell_probas[te] = search.best_estimator_.predict_proba(X_te)

        # ── Cell-level metrics ─────────────────────────────────────────────────
        cell_preds = cell_probas.argmax(axis=1)
        cell_f1    = f1_score(y, cell_preds, average='macro')
        cell_acc   = accuracy_score(y, cell_preds)
        if n_classes == 2:
            cell_auc = roc_auc_score(y, cell_probas[:, 1])
        else:
            cell_auc = roc_auc_score(y, cell_probas, multi_class='ovr',
                                     average='macro')

        # ── Patient-level aggregation ──────────────────────────────────────────
        # Average cell probabilities per patient, then classify
        pat_df = pd.DataFrame(cell_probas, columns=[f'p_{c}' for c in label_names])
        pat_df['Patient_ID'] = patient_ids[:len(y)]   # aligned
        pat_df['True_Label'] = y
        pat_agg = pat_df.groupby('Patient_ID').agg(
            {**{f'p_{c}': 'mean' for c in label_names},
             'True_Label': 'first'}
        )
        pat_preds = pat_agg[[f'p_{c}' for c in label_names]].values.argmax(axis=1)
        pat_true  = pat_agg['True_Label'].values
        pat_f1    = f1_score(pat_true, pat_preds, average='macro')
        pat_acc   = accuracy_score(pat_true, pat_preds)
        if n_classes == 2:
            pat_auc = roc_auc_score(pat_true,
                                    pat_agg[[f'p_{c}' for c in label_names]].values[:, 1])
        else:
            pat_auc = roc_auc_score(pat_true,
                                    pat_agg[[f'p_{c}' for c in label_names]].values,
                                    multi_class='ovr', average='macro')

        results[model_name] = {
            'cell_f1': cell_f1, 'cell_acc': cell_acc, 'cell_auc': cell_auc,
            'pat_f1':  pat_f1,  'pat_acc':  pat_acc,  'pat_auc':  pat_auc,
            'pat_true': pat_true, 'pat_preds': pat_preds,
            'best_params': best_params_folds,
        }
        all_probas[model_name] = cell_probas

        print(f"    Cell-level : Acc={cell_acc:.1%}  macro-F1={cell_f1:.3f}  AUC={cell_auc:.3f}")
        print(f"    Patient-level: Acc={pat_acc:.1%}  macro-F1={pat_f1:.3f}  AUC={pat_auc:.3f}")

    best = max(results, key=lambda k: results[k]['pat_f1'])
    print(f"\n  Best: {best}  (patient macro-F1={results[best]['pat_f1']:.3f})")
    return results, best, all_probas


# ── Task A — 3-class: Healthy / TAA / BAV ─────────────────────────────────────
res3, best3, probas3 = run_nested_cv(
    X, y, groups,
    label_names=DISEASE_ORDER,
    task_name='3-class (Healthy / TAA / BAV)',
)

# ── Tasks B-D — pairwise binary ───────────────────────────────────────────────
pairwise_tasks = [
    ('Healthy vs TAA', ['Healthy', 'TAA'], [0, 1]),
    ('Healthy vs BAV', ['Healthy', 'BAV'], [0, 2]),
    ('TAA vs BAV',     ['TAA',     'BAV'], [1, 2]),
]
pw_results = {}
for task_name, label_names, keep_labels in pairwise_tasks:
    mask  = np.isin(y, keep_labels)
    y_bin = np.where(y[mask] == keep_labels[0], 0, 1)
    # remap group indices to keep dense contiguous for sklearn
    g_bin = groups[mask]
    res_bin, best_bin, _ = run_nested_cv(
        X[mask], y_bin, g_bin,
        label_names=label_names,
        task_name=task_name,
    )
    pw_results[task_name] = (res_bin, best_bin, label_names)

# ── Summary table ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY — Patient-level metrics (nested CV, no patient leakage)")
print("=" * 65)
rows = []
for model in BASE_PIPES:
    if model in res3:
        r = res3[model]
        rows.append({'Task': '3-class (H/TAA/BAV)', 'Model': model,
                     'Pat_Acc': r['pat_acc'], 'Pat_F1': r['pat_f1'],
                     'Pat_AUC': r['pat_auc']})
for task_name, (res_bin, _, _) in pw_results.items():
    for model, r in res_bin.items():
        rows.append({'Task': task_name, 'Model': model,
                     'Pat_Acc': r['pat_acc'], 'Pat_F1': r['pat_f1'],
                     'Pat_AUC': r['pat_auc']})
summary_df = pd.DataFrame(rows)
print(summary_df.to_string(index=False))
summary_df.to_csv(OUT_DIR / 'nested_cv_summary.csv', index=False)

# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_confusion(true, preds, labels, title, fname):
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(true, preds, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title, fontweight='bold', fontsize=11)
    plt.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname.name}")

def plot_patient_f1_bar(results, task_name, fname):
    models = list(results.keys())
    f1s    = [results[m]['pat_f1']  for m in models]
    aucs   = [results[m]['pat_auc'] for m in models]
    x      = np.arange(len(models))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f'{task_name} — Patient-level (nested CV)', fontweight='bold')
    for ax, vals, metric in zip(axes, [f1s, aucs], ['macro-F1', 'AUC']):
        bars = ax.bar(x, vals, color='#5b21b6', edgecolor='white', width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha='right', fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(metric)
        ax.set_title(metric)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.03,
                    f'{v:.2f}', ha='center', fontsize=10, fontweight='bold', color='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname.name}")

print("\nGenerating plots...")

# 3-class confusion matrix (best model, patient-level)
plot_confusion(
    res3[best3]['pat_true'], res3[best3]['pat_preds'], DISEASE_ORDER,
    f'3-class — {best3}  (patient-level)',
    OUT_DIR / '01_confusion_3class_patient.png'
)
plot_patient_f1_bar(res3, '3-class (Healthy / TAA / BAV)',
                    OUT_DIR / '02_f1_bar_3class.png')

# Pairwise confusion matrices
for i, (task_name, (res_bin, best_bin, labels)) in enumerate(pw_results.items(), 3):
    plot_confusion(
        res_bin[best_bin]['pat_true'], res_bin[best_bin]['pat_preds'], labels,
        f'{task_name} — {best_bin}  (patient-level)',
        OUT_DIR / f'{i:02d}_confusion_{task_name.replace(" ","_")}.png'
    )

# Per-class report for 3-class best model
print(f"\nDetailed report — {best3} (patient-level, 3-class):")
print(classification_report(
    res3[best3]['pat_true'], res3[best3]['pat_preds'],
    target_names=DISEASE_ORDER
))

# ── Retrain best model on full data ───────────────────────────────────────────
print("=" * 65)
print(f"Retraining {best3} on all {len(y)} cells for app deployment...")

# Fit the best pipeline with the most common best hyperparams from folds
best_params_list = res3[best3]['best_params']
# Use the first fold's best params (representative; for production use majority vote)
from collections import Counter
def most_common_params(param_list):
    """Pick the most common value for each hyperparameter across folds."""
    keys = param_list[0].keys()
    return {k: Counter(str(p[k]) for p in param_list).most_common(1)[0][0]
            for k in keys}

common_params = most_common_params(best_params_list)
# Convert back: strings that look like ints/floats/None
def parse_val(s):
    if s == 'None': return None
    try: return int(s)
    except ValueError: pass
    try: return float(s)
    except ValueError: pass
    return s

final_pipe = BASE_PIPES[best3]
final_pipe.set_params(**{k: parse_val(v) for k, v in common_params.items()})
final_pipe.fit(X, y)

train_preds = final_pipe.predict(X)
print(f"Train accuracy (sanity): {accuracy_score(y, train_preds):.1%}")
print(classification_report(y, train_preds, target_names=DISEASE_ORDER))

bundle = {
    'pipeline':    final_pipe,
    'features':    feat_cols,
    'label_map':   LABEL_MAP,
    'label_names': DISEASE_ORDER,
    'best_params': common_params,
}
joblib.dump(bundle, MODEL_OUT)
print(f"\nSaved → {MODEL_OUT}")
print(f"Done. All outputs in {OUT_DIR}/")
