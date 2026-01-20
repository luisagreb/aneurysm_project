"""
Phase 2: Binary Classifier Validation - Healthy (No Col) vs TAA (No Col)

Goal: Validate that statistical differences translate into a "digital fingerprint"
      that can accurately classify cells.

Expected Result: >80% accuracy (based on Phase 1 effect sizes)

Outputs:
- Trained Random Forest model (saved for Phase 3)
- Confusion matrix
- Feature importance plot
- Classification report
- Model predictions

Author: Antigravity AI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import joblib
import os
import re

# Configuration
FEATURES_FILE = 'outputs/Advanced_Features_Raw.csv'
METADATA_FILE = 'data/Book1.xlsx'
OUTPUT_DIR = 'src/analysis/outputs/phase2'
MODEL_PATH = f'{OUTPUT_DIR}/healthy_vs_taa_model.pkl'
SCALER_PATH = f'{OUTPUT_DIR}/scaler.pkl'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_numeric_id(text):
    """Extract subject ID from filename."""
    if pd.isna(text):
        return None
    text = str(text).strip()
    parts = text.split('-')
    if len(parts) > 1:
        last_part = parts[-1].split()[0]
        if last_part.isdigit():
            return int(last_part)
        digits = re.findall(r'\d+', last_part)
        if digits:
            return int(digits[0])
    digits = re.findall(r'\d+', text)
    if digits:
        return int(digits[-1])
    return None

def load_metadata(filepath):
    """Load metadata to classify Healthy vs TAA."""
    print(f"Loading metadata from {filepath}...")
    df = pd.read_excel(filepath, header=None, skiprows=18)
    
    healthy_ids = set()
    taa_ids = set()
    
    for x in df[0].dropna():
        nid = extract_numeric_id(x)
        if nid is not None:
            healthy_ids.add(nid)
    
    for x in df[1].dropna():
        nid = extract_numeric_id(x)
        if nid is not None:
            taa_ids.add(nid)
    
    return healthy_ids, taa_ids

def extract_collagen_status(filename):
    """Extract collagen status."""
    if pd.isna(filename):
        return None
    filename_clean = str(filename).replace(' ', '').lower()
    if '+coll' in filename_clean or '+col' in filename_clean:
        return 'Collagen'
    elif '-coll' in filename_clean or '-col' in filename_clean or 'nocoll' in filename_clean:
        return 'NoCollagen'
    return None

def main():
    print("=" * 80)
    print("PHASE 2: BINARY CLASSIFIER VALIDATION - Healthy vs TAA (No Collagen)")
    print("=" * 80)
    
    # 1. Load data
    healthy_ids, taa_ids = load_metadata(METADATA_FILE)
    df = pd.read_csv(FEATURES_FILE)
    
    # 2. Add labels
    df['Numeric_ID'] = df['CellName'].apply(extract_numeric_id)
    df['Collagen_Status'] = df['CellName'].apply(extract_collagen_status)
    
    def get_disease_label(nid):
        if nid in healthy_ids:
            return 0  # Healthy
        if nid in taa_ids:
            return 1  # TAA
        return None
    
    df['Disease_Label'] = df['Numeric_ID'].apply(get_disease_label)
    
    # 3. Filter to No Collagen only
    df_nocol = df[df['Collagen_Status'] == 'NoCollagen'].copy()
    df_nocol = df_nocol.dropna(subset=['Disease_Label'])
    
    print(f"\nDataset: {len(df_nocol)} cells")
    print(f"  Healthy (0): {sum(df_nocol['Disease_Label'] == 0)}")
    print(f"  TAA (1): {sum(df_nocol['Disease_Label'] == 1)}")
    
    # 4. Prepare features
    feature_cols = [col for col in df_nocol.columns if 
                    col.endswith('_µm') or col.endswith('_ratio') or col.endswith('_µm³') or 
                    col.endswith('_µm²') or col.endswith('_n')]
    
    X = df_nocol[feature_cols].copy()
    y = df_nocol['Disease_Label'].copy()
    
    # Handle inf/nan
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    print(f"\nFeatures: {X.shape[1]}")
    
    # 5. Train/Test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} | Test set: {len(X_test)}")
    
    # 6. Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 7. Train Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    
    # 8. Predictions
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)
    y_proba_test = rf.predict_proba(X_test)[:, 1]
    
    # 9. Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    auc = roc_auc_score(y_test, y_proba_test)
    
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Training Accuracy: {train_acc:.1%}")
    print(f"Test Accuracy: {test_acc:.1%}")
    print(f"AUC: {auc:.3f}")
    
    # 10. Cross-validation
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
    print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
    
    # 11. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=['Healthy', 'TAA']))
    
    # 12. Feature Importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    top_n = 15
    print(f"\nTop {top_n} Features:")
    for i in range(min(top_n, len(indices))):
        print(f"{i+1}. {X.columns[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # 13. Save model
    joblib.dump(rf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nModel saved: {MODEL_PATH}")
    
    # 14. Visualizations
    
    # Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'TAA'], 
                yticklabels=['Healthy', 'TAA'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Phase 2: Binary Classifier\nAccuracy: {test_acc:.1%} | AUC: {auc:.3f}', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/confusion_matrix.png', dpi=300)
    print(f"Saved: {OUTPUT_DIR}/confusion_matrix.png")
    
    # Feature Importance Plot
    plt.figure(figsize=(10, 8))
    colors = []
    for feat in X.columns[indices[:top_n]]:
        if 'Actin' in feat:
            colors.append('#E74C3C')
        elif 'Mito' in feat:
            colors.append('#27AE60')
        else:
            colors.append('#3498DB')
    
    plt.barh(range(top_n), importances[indices[:top_n]], color=colors)
    plt.yticks(range(top_n), X.columns[indices[:top_n]])
    plt.xlabel('Feature Importance', fontweight='bold')
    plt.title('Phase 2: Feature Importance (Random Forest)\nHealthy vs TAA Classifier', 
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/feature_importance.png', dpi=300)
    print(f"Saved: {OUTPUT_DIR}/feature_importance.png")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba_test)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'RF (AUC = {auc:.3f})', color='#E74C3C')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Guess')
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('Phase 2: ROC Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/roc_curve.png', dpi=300)
    print(f"Saved: {OUTPUT_DIR}/roc_curve.png")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'CellName': df_nocol.iloc[X_test.index]['CellName'],
        'True_Label': y_test.map({0: 'Healthy', 1: 'TAA'}),
        'Predicted_Label': pd.Series(y_pred_test).map({0: 'Healthy', 1: 'TAA'}),
        'Probability_TAA': y_proba_test
    })
    predictions_df.to_csv(f'{OUTPUT_DIR}/predictions.csv', index=False)
    print(f"Saved: {OUTPUT_DIR}/predictions.csv")
    
    print(f"\n{'='*80}")
    print("THESIS SECTION 3.2: Machine Learning Classification Complete")
    print(f"{'='*80}")
    
    # Thesis-ready summary
    print(f"\nTHESIS SUMMARY:")
    print(f"  Binary Random Forest classifier achieved {test_acc:.1%} accuracy")
    print(f"  (AUC = {auc:.3f}) in distinguishing Healthy from TAA cells.")
    print(f"  Top predictive feature: {X.columns[indices[0]]}")

if __name__ == '__main__':
    main()
