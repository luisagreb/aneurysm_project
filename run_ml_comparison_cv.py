import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

# Configuration
FEATURES_FILE = 'outputs/Advanced_Features_Raw.csv'
METADATA_FILE = 'data/Book1.xlsx'

def extract_numeric_id(text):
    """Extract Subject ID from filename."""
    import re
    if pd.isna(text):
        return None
    text = str(text).strip()
    first_token = text.split(' ')[0]
    token = first_token.strip('-')
    parts = token.split('-')
    if len(parts) > 1:
        last_part = parts[-1]
        if last_part.isdigit():
            return int(last_part)
        sub_digits = re.findall(r'\d+', last_part)
        if sub_digits:
            return int(sub_digits[0])
    digits = re.findall(r'\d+', token)
    if digits:
        return int(digits[-1])
    return None

def load_metadata(filepath):
    print(f"Loading metadata from {filepath}...")
    try:
        df = pd.read_excel(filepath, header=None, skiprows=18)
        healthy_col = df[0]
        taa_col = df[1]
        
        healthy_ids = set()
        taa_ids = set()
        
        for x in healthy_col.dropna():
            nid = extract_numeric_id(x)
            if nid is not None:
                healthy_ids.add(nid)
        for x in taa_col.dropna():
            nid = extract_numeric_id(x)
            if nid is not None:
                taa_ids.add(nid)
        
        return healthy_ids, taa_ids
    except Exception as e:
        print(f"Error parse metadata: {e}")
        return set(), set()

print("="*80)
print("ML CLASSIFIER COMPARISON: TAA vs Healthy")
print("5-Fold Stratified Cross-Validation with Voxel-Aware Features")
print("="*80)

# Load Metadata
healthy_ids, taa_ids = load_metadata(METADATA_FILE)
print(f"Found {len(healthy_ids)} Healthy IDs")
print(f"Found {len(taa_ids)} TAA IDs")

# Load Features
df_features = pd.read_csv(FEATURES_FILE)
print(f"\nFeatures loaded: {df_features.shape}")

# Extract ID and Label
df_features['Numeric_ID'] = df_features['CellName'].apply(extract_numeric_id)
df_features = df_features.dropna(subset=['Numeric_ID'])

def get_label(nid):
    if nid in healthy_ids:
        return 0
    if nid in taa_ids:
        return 1
    return None

df_features['Label'] = df_features['Numeric_ID'].apply(get_label)
df_features = df_features.dropna(subset=['Label'])
print(f"Assigned labels to {len(df_features)}/{len(df_features)} rows.")

# Aggregate by Cell
def normalize_filename(f):
    s = str(f).replace('.nii.gz', '').replace('.tif', '')
    s = s.replace('_segmentation', '').replace('_visible', '')
    return s.strip()

df_features['Cell_ID'] = df_features['CellName'].apply(normalize_filename)

feature_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
if 'Label' in feature_cols:
    feature_cols.remove('Label')
if 'Numeric_ID' in feature_cols:
    feature_cols.remove('Numeric_ID')

agg_dict = {col: 'max' for col in feature_cols}
agg_dict['Label'] = 'first'
agg_dict['Numeric_ID'] = 'first'

df_agg = df_features.groupby('Cell_ID').agg(agg_dict)
print(f"Aggregated Dataset shape: {df_agg.shape} (Cells)")

# Prepare ML Data
X = df_agg.drop(columns=['Label', 'Numeric_ID'])
y = df_agg['Label']

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

print(f"\nFinal X: {X.shape}, y: {y.shape}")
print(f"Class Balance:")
print(y.value_counts())

# Define Classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    ),
    'MLP Classifier': MLPClassifier(
        hidden_layer_sizes=(50, 25),
        max_iter=1000,
        random_state=42,
        solver='adam',
        learning_rate_init=0.001,
        alpha=0.1,
        early_stopping=False
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    ),
    'K-Nearest Neighbors': KNeighborsClassifier(
        n_neighbors=5,
        weights='distance'
    )
}

# 5-Fold Stratified Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

print("\n" + "="*80)
print("RUNNING 5-FOLD CROSS-VALIDATION")
print("="*80)

for name, clf in classifiers.items():
    print(f"\n{name}...")
    
    # For classifiers that need scaling (MLP, KNN, LogReg)
    if name in ['MLP Classifier', 'K-Nearest Neighbors', 'Logistic Regression']:
        # Custom scorer that handles scaling
        accuracies = []
        aucs = []
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train
            clf.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = clf.predict(X_test_scaled)
            y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
            
            # Metrics
            acc = (y_pred == y_test).mean()
            auc = roc_auc_score(y_test, y_pred_proba)
            
            accuracies.append(acc)
            aucs.append(auc)
        
        mean_acc = np.mean(accuracies)
        mean_auc = np.mean(aucs)
        
    else:
        # Random Forest doesn't need scaling
        accuracy_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        
        # AUC scores
        auc_scores = []
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            clf.fit(X_train, y_train)
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            auc_scores.append(auc)
        
        mean_acc = accuracy_scores.mean()
        mean_auc = np.mean(auc_scores)
    
    results.append({
        'Classifier': name,
        'Accuracy': mean_acc,
        'AUC': mean_auc
    })
    
    print(f"  Accuracy: {mean_acc:.1%}")
    print(f"  AUC: {mean_auc:.1%}")

# Sort by accuracy
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)
results_df['Rank'] = range(1, len(results_df) + 1)

print("\n" + "="*80)
print("FINAL RESULTS (5-Fold Stratified Cross-Validation)")
print("="*80)
print()
print(results_df[['Rank', 'Classifier', 'Accuracy', 'AUC']].to_string(index=False))

# Save to CSV
results_df.to_csv('classification_results/ml_comparison_cv.csv', index=False)
print(f"\nResults saved to: classification_results/ml_comparison_cv.csv")

# ---------------------------------------------------------
# SAVE MODEL FOR WEB APP
# ---------------------------------------------------------
print("\nSaving best model (Random Forest) to app/models/classifier.joblib...")
import joblib
import os

model_dir = 'app/models'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'classifier.joblib')

# Retrain on full dataset for deployment
clf = classifiers['Random Forest']
clf.fit(X, y)
joblib.dump(clf, model_path)
print(f"Model saved to {model_path}")

# Create visualization
fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(results_df))
width = 0.35

bars1 = ax.bar(x - width/2, results_df['Accuracy']*100, width, 
               label='Accuracy', color='#3498DB', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, results_df['AUC']*100, width,
               label='AUC', color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax.set_title('ML Classifier Comparison: TAA vs Healthy\n5-Fold Stratified Cross-Validation', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(results_df['Classifier'], rotation=15, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 100])

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# Add metadata text
info_text = f"Data: {len(df_agg)} cells\nTraining: 80% (4 folds)\nTest: 20% (1 fold, stratified)"
ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
output_path = 'classification_results/ml_comparison_cv.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Visualization saved to: {output_path}")

print("\nDone!")
