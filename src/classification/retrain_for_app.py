"""
Retrain classifier for the CellVision web app.

Key improvements over the original run_ml_comparison_cv.py:
  - All models wrapped in a StandardScaler Pipeline
  - Patient-level GroupKFold CV (cells from the same patient never split across
    train/test, preventing data leakage between patients)
  - Best model selected by CV AUC
  - Saved as a sklearn Pipeline (scaler + model) to app/models/classifier.joblib

Usage:
    cd /home/luisa/aneurysm_project
    python src/classification/retrain_for_app.py
"""

import re
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.neural_network  import MLPClassifier
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics         import accuracy_score, classification_report

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_FILE = 'outputs/Advanced_Features_Raw_200.csv'
MODEL_OUT     = 'app/models/classifier.joblib'
RESULTS_DIR   = 'classification_results/app_model'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs('app/models', exist_ok=True)

# ── Column rename map (CSV has no units; app uses unit-suffixed names) ────────
RENAME = {
    'Voxel_Z':                       'Voxel_Z_µm',
    'Voxel_Y':                       'Voxel_Y_µm',
    'Voxel_X':                       'Voxel_X_µm',
    'Actin_Volume':                  'Actin_Volume_µm³',
    'Actin_Skeleton_Length':         'Actin_Skeleton_Length_µm',
    'Actin_Convex_Hull_Volume':      'Actin_Convex_Hull_Volume_µm³',
    'Actin_Solidity':                'Actin_Solidity_ratio',
    'Actin_Extent':                  'Actin_Extent_ratio',
    'Actin_Fractional_Anisotropy':   'Actin_Fractional_Anisotropy_ratio',
    'Actin_Major_Axis':              'Actin_Major_Axis_µm',
    'Actin_Intermediate_Axis':       'Actin_Intermediate_Axis_µm',
    'Actin_Minor_Axis':              'Actin_Minor_Axis_µm',
    'Mito_Volume':                   'Mito_Volume_µm³',
    'Mito_Surface_Area':             'Mito_Surface_Area_µm²',
    'Mito_Sphericity':               'Mito_Sphericity_ratio',
    'Mito_Fragment_Count':           'Mito_Fragment_Count_n',
    'Mito_Mean_Fragment_Sphericity': 'Mito_Mean_Fragment_Sphericity_ratio',
    'Mito_Std_Fragment_Sphericity':  'Mito_Std_Fragment_Sphericity_ratio',
    'Mito_Min_Fragment_Sphericity':  'Mito_Min_Fragment_Sphericity_ratio',
    'Mito_Max_Fragment_Sphericity':  'Mito_Max_Fragment_Sphericity_ratio',
    'Mito_Mean_Fragment_Volume':     'Mito_Mean_Fragment_Volume_µm³',
    'Mito_Junction_Count':           'Mito_Junction_Count_n',
    'Mito_Branch_Count':             'Mito_Branch_Count_n',
    'Mito_Mean_Branch_Length':       'Mito_Mean_Branch_Length_µm',
    'Mito_Total_Network_Length':     'Mito_Total_Network_Length_µm',
    'Mito_Mean_Tortuosity':          'Mito_Mean_Tortuosity_ratio',
    'Mito_Cyclomatic_Number':        'Mito_Cyclomatic_Number_n',
    'Nucleus_Volume':                'Nucleus_Volume_µm³',
    'Nucleus_Sphericity':            'Nucleus_Sphericity_ratio',
    'Nucleus_Circularity':           'Nucleus_Circularity_ratio',
    'Nucleus_Elongation':            'Nucleus_Elongation_ratio',
    'Nucleus_Flatness':              'Nucleus_Flatness_ratio',
    'Nucleus_Solidity':              'Nucleus_Solidity_ratio',
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_id(text):
    if pd.isna(text): return None
    t = str(text).strip().split(' ')[0].strip('-')
    parts = t.split('-')
    if len(parts) > 1:
        lp = parts[-1]
        if lp.isdigit(): return int(lp)
        d = re.findall(r'\d+', lp)
        if d: return int(d[0])
    d = re.findall(r'\d+', t)
    return int(d[-1]) if d else None

# ── Load data ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("CellVision App — Classifier Retraining (patient-level CV)")
print("=" * 65)

df = pd.read_csv(FEATURES_FILE)
print(f"Features loaded: {df.shape}")

# Rename columns to match app feature names (with units)
df = df.rename(columns=RENAME)

# Labels come directly from the Disease column
df['Label'] = df['Disease'].map({'Healthy': 0, 'TAA': 1})
df = df.dropna(subset=['Label'])
df['Label'] = df['Label'].astype(int)

# Patient ID for GroupKFold
df['Numeric_ID'] = df['CellName'].apply(extract_id)
df = df.dropna(subset=['Numeric_ID'])

n_healthy = (df['Label'] == 0).sum()
n_taa     = (df['Label'] == 1).sum()
n_patients = df['Numeric_ID'].nunique()
print(f"Metadata: labels read from Disease column")

# Drop non-feature columns
drop_cols = ['CellName', 'Label', 'Numeric_ID', 'Disease',
             'Voxel_Source', 'DataSource', 'Age', 'Gender', 'Collagen_Status']
feat_cols = [c for c in df.columns if c not in drop_cols]

X      = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
y      = df['Label']
groups = df['Numeric_ID'].astype(int).values

n_patients = len(np.unique(groups))
print(f"\nDataset  : {X.shape[0]} cells x {X.shape[1]} features")
print(f"Patients : {n_patients}")
print(f"Class    : Healthy={n_healthy}, TAA={n_taa}")

# ── Settings you can change ───────────────────────────────────────────────────
# Set to a model name below to force that model, or None to auto-pick by AUC
FORCE_MODEL = 'Random Forest'

# Decision threshold: lower → more TAA flagged (higher recall, more false alarms)
#   0.20 → 91% TAA recall,  8 cells missed
#   0.25 → 90% TAA recall,  9 cells missed
#   0.30 → 90% TAA recall,  9 cells missed  ← current
#   0.50 → 85% TAA recall, 15 cells missed  (default sklearn)
THRESHOLD = 0.25

# ── Candidate pipelines ───────────────────────────────────────────────────────
# n_splits capped at 3: with few patients per class, 5 folds create test sets
# with only one class, making AUC undefined. 3 folds keeps both classes in test.
n_splits = min(3, n_patients // 2)
print(f"\nUsing {n_splits}-fold patient-level GroupKFold CV")

candidates = {
    # ── Random Forest ─────────────────────────────────────────────────────────
    # n_estimators : number of trees — more = more stable, slower to train
    # max_depth    : how deep each tree grows — deeper = catches more patterns
    #                but can overfit; None = fully grown
    # min_samples_leaf : min cells per leaf — higher = smoother, less overfit
    # class_weight : {0: 1, 1: 3} means TAA errors penalised 3× more than
    #                Healthy errors → pushes model to catch more TAA cells
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=2,
            class_weight={0: 1, 1: 3},   # ← penalise missing TAA cells 3×
            random_state=42,
        )),
    ]),
    # ── Gradient Boosting ─────────────────────────────────────────────────────
    # n_estimators  : boosting rounds — more = potentially better, slower
    # max_depth     : tree depth per round — keep low (2–4) for GBM
    # learning_rate : step size — lower = more rounds needed but more robust
    # subsample     : fraction of data per round — adds randomness, less overfit
    'Gradient Boosting': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )),
    ]),
    'Logistic Regression L1': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            penalty='l1', solver='liblinear', C=0.1,
            max_iter=2000, class_weight='balanced', random_state=42,
        )),
    ]),
    'Logistic Regression L2': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            penalty='l2', C=0.1, max_iter=2000,
            class_weight='balanced', random_state=42,
        )),
    ]),
}

# ── Patient-level GroupKFold CV ───────────────────────────────────────────────
cv = GroupKFold(n_splits=n_splits)

print("\n" + "=" * 65)
print("Results (patient-level CV -- no patient leakage)")
print("=" * 65)

cv_results = {}
for name, pipe in candidates.items():
    scores = cross_validate(
        pipe, X, y,
        cv=cv, groups=groups,
        scoring=['accuracy', 'roc_auc'],
        return_train_score=False,
        n_jobs=-1,
    )
    acc = scores['test_accuracy'].mean()
    auc = scores['test_roc_auc'].mean()
    cv_results[name] = {
        'acc':     acc,
        'auc':     auc,
        'acc_std': scores['test_accuracy'].std(),
        'auc_std': scores['test_roc_auc'].std(),
    }
    print(f"  {name:<26}  Acc={acc:.1%} +/-{scores['test_accuracy'].std():.1%}"
          f"  AUC={auc:.3f} +/-{scores['test_roc_auc'].std():.3f}")

# ── Pick model ────────────────────────────────────────────────────────────────
if FORCE_MODEL and FORCE_MODEL in candidates:
    best_name = FORCE_MODEL
    print(f"\nForced model : {best_name}")
else:
    best_name = max(cv_results, key=lambda k: cv_results[k]['auc'])
    print(f"\nBest model (auto) : {best_name}")
best_pipe = candidates[best_name]
print(f"  AUC={cv_results[best_name]['auc']:.3f}  "
      f"Acc={cv_results[best_name]['acc']:.1%}")

# ── Threshold tuning — maximise TAA recall ───────────────────────────────────
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

probas_cv = cross_val_predict(
    best_pipe, X, y, cv=cv, groups=groups, method='predict_proba'
)[:, 1]
auc_cv = roc_auc_score(y, probas_cv)
print(f"\nThreshold scan (CV, {n_splits}-fold patient-level)  AUC={auc_cv:.3f}")
print(f"{'Threshold':>10}  {'Acc':>7}  {'TAA recall':>11}  {'Healthy recall':>15}  {'TAA missed':>10}")
for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    preds  = (probas_cv >= t).astype(int)
    acc    = (preds == y.values).mean()
    taa_r  = preds[y == 1].mean()
    hlt_r  = (1 - preds[y == 0]).mean()
    missed = int((y == 1).sum() * (1 - taa_r))
    print(f"{t:>10.2f}  {acc:>6.1%}  {taa_r:>10.1%}  {hlt_r:>14.1%}  {missed:>10} cells")

# Threshold=0.30 gives 90 % TAA recall while keeping overall accuracy at ~75 %
THRESHOLD = 0.30

# ── Retrain on full dataset ───────────────────────────────────────────────────
print(f"\nRetraining on all {len(X)} cells...")
best_pipe.fit(X, y)
y_tr = (best_pipe.predict_proba(X)[:, 1] >= THRESHOLD).astype(int)
print(f"Train accuracy (sanity, threshold={THRESHOLD}): {accuracy_score(y, y_tr):.1%}")
print(classification_report(y, y_tr, target_names=['Healthy', 'TAA']))

# ── Save as bundle (pipeline + feature list + threshold) ─────────────────────
bundle = {
    'pipeline':  best_pipe,
    'features':  list(X.columns),   # all 33 features
    'threshold': THRESHOLD,          # 0.30 → 90 % TAA recall
}
joblib.dump(bundle, MODEL_OUT)
print(f"Saved -> {MODEL_OUT}  (threshold={THRESHOLD}, {len(X.columns)} features)")

# ── CV summary plot ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
names  = list(cv_results.keys())
colors = ['#5b21b6' if n == best_name else '#2d2060' for n in names]

for ax, metric, title in [
    (axes[0], 'acc', f'Accuracy  {n_splits}-fold patient CV'),
    (axes[1], 'auc', f'AUC-ROC   {n_splits}-fold patient CV'),
]:
    vals = [cv_results[n][metric]           for n in names]
    stds = [cv_results[n][f'{metric}_std']  for n in names]
    bars = ax.bar(names, vals, color=colors, yerr=stds, capsize=4,
                  edgecolor='white', linewidth=0.5)
    ax.set_title(title, fontweight='bold')
    ax.set_ylim(0, 1.12)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03,
                f'{v:.2f}', ha='center', fontsize=9, fontweight='bold', color='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle(f'Best: {best_name}  (AUC={cv_results[best_name]["auc"]:.3f})',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plot_out = f'{RESULTS_DIR}/cv_comparison_patient_level.png'
plt.savefig(plot_out, dpi=150, bbox_inches='tight')
print(f"Plot  -> {plot_out}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
res_df = pd.DataFrame(cv_results).T.sort_values('auc', ascending=False)
print(res_df[['acc', 'acc_std', 'auc', 'auc_std']].round(3).to_string())
res_df.to_csv(f'{RESULTS_DIR}/cv_results_patient_level.csv')
print(f"\nDone. Restart the web app to use the new model.")
