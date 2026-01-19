"""
4-Class Classifier - Combines Disease Status and Collagen
Classes:
  0: Healthy + No Collagen
  1: Healthy + Collagen
  2: TAA + No Collagen
  3: TAA + Collagen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configuration
FEATURES_FILE = 'outputs/Advanced_Features_Raw.csv'
METADATA_FILE = 'data/Book1.xlsx'
OUTPUT_DIR = 'classification_results/4class'

CLASS_NAMES = ['Healthy-NoColl', 'Healthy+Coll', 'TAA-NoColl', 'TAA+Coll']

def extract_numeric_id(text):
    """Extract Subject ID from filename."""
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

def extract_collagen_status(filename):
    """Extract collagen status from filename."""
    if pd.isna(filename):
        return None
    filename = str(filename).lower()
    
    # Positive Collagen
    if '+coll' in filename or '+col' in filename or 'plus coll' in filename:
        return 1
        
    # Negative Collagen (including DMSO control)
    if '-coll' in filename or '-col' in filename or 'nocoll' in filename or 'no coll' in filename or 'dmso' in filename:
        return 0
        
    return None

def load_metadata(filepath):
    """Load healthy/TAA labels from metadata file."""
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
                
        print(f"Found {len(healthy_ids)} Healthy IDs, {len(taa_ids)} TAA IDs")
        return healthy_ids, taa_ids
        
    except Exception as e:
        print(f"Error parsing metadata: {e}")
        return set(), set()

def normalize_filename(f):
    """Normalize filename to get unique cell ID."""
    s = str(f).replace('.nii.gz', '').replace('.tif', '')
    s = s.replace('_segmentation', '').replace('_visible', '')
    return s.strip()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("4-CLASS CLASSIFIER: Disease Status × Collagen")
    print("=" * 60)
    
    # 1. Load Metadata
    healthy_ids, taa_ids = load_metadata(METADATA_FILE)
    if not healthy_ids and not taa_ids:
        print("Failed to load metadata labels.")
        return
    
    # 2. Load Features
    if not os.path.exists(FEATURES_FILE):
        print(f"Error: {FEATURES_FILE} not found.")
        return
        
    df_features = pd.read_csv(FEATURES_FILE)
    print(f"Features loaded: {df_features.shape}")
    
    # 3. Extract IDs and Labels
    df_features['Numeric_ID'] = df_features['CellName'].apply(extract_numeric_id)
    df_features['Collagen'] = df_features['CellName'].apply(extract_collagen_status)
    
    # Get disease status
    def get_disease_label(nid):
        if nid in healthy_ids:
            return 0  # Healthy
        if nid in taa_ids:
            return 1  # TAA
        return None
    
    df_features['Disease'] = df_features['Numeric_ID'].apply(get_disease_label)
    
    # Create 4-class label
    def get_4class_label(row):
        if pd.isna(row['Disease']) or pd.isna(row['Collagen']):
            return None
        disease = int(row['Disease'])
        collagen = int(row['Collagen'])
        # 0: Healthy-NoColl, 1: Healthy+Coll, 2: TAA-NoColl, 3: TAA+Coll
        return disease * 2 + collagen
    
    df_features['Label'] = df_features.apply(get_4class_label, axis=1)
    
    # Drop unknown labels
    initial_count = len(df_features)
    df_features = df_features.dropna(subset=['Label'])
    print(f"Assigned 4-class labels to {len(df_features)}/{initial_count} rows.")
    
    if df_features.empty:
        print("No samples with determinable labels.")
        return
    
    # 4. Aggregate by Cell
    df_features['Cell_ID'] = df_features['CellName'].apply(normalize_filename)
    
    feature_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    for col in ['Label', 'Numeric_ID', 'Disease', 'Collagen']:
        if col in feature_cols:
            feature_cols.remove(col)
    
    agg_dict = {col: 'max' for col in feature_cols}
    agg_dict['Label'] = 'first'
    agg_dict['Disease'] = 'first'
    agg_dict['Collagen'] = 'first'
    
    df_agg = df_features.groupby('Cell_ID').agg(agg_dict)
    print(f"Aggregated to {len(df_agg)} unique cells.")
    
    # 5. Prepare ML Data
    X = df_agg.drop(columns=['Label', 'Disease', 'Collagen'])
    y = df_agg['Label'].astype(int)
    
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    print(f"\nClass Balance:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {sum(y == i)}")
    
    # 6. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 7. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ============================================================
    # RANDOM FOREST CLASSIFIER
    # ============================================================
    print("\n" + "=" * 50)
    print("RANDOM FOREST CLASSIFIER")
    print("=" * 50)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    
    print(f"\nAccuracy Score: {acc_rf:.4f}")
    print("\nConfusion Matrix:")
    print(cm_rf)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=CLASS_NAMES))
    
    # Feature Importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    top_n = 10
    print(f"\nTop {top_n} Features:")
    for f in range(min(top_n, X.shape[1])):
        print(f"{f + 1}. {X.columns[indices[f]]} ({importances[indices[f]]:.4f})")
    
    # ============================================================
    # NEURAL NETWORK CLASSIFIER
    # ============================================================
    print("\n" + "=" * 50)
    print("NEURAL NETWORK CLASSIFIER")
    print("=" * 50)
    
    nn = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=1000,
        random_state=42,
        solver='adam',
        learning_rate_init=0.001,
        alpha=0.1,
        early_stopping=False
    )
    nn.fit(X_train_scaled, y_train)
    
    y_pred_nn = nn.predict(X_test_scaled)
    acc_nn = accuracy_score(y_test, y_pred_nn)
    cm_nn = confusion_matrix(y_test, y_pred_nn)
    
    print(f"\nAccuracy Score: {acc_nn:.4f}")
    print("\nConfusion Matrix:")
    print(cm_nn)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_nn, target_names=CLASS_NAMES))
    
    # ============================================================
    # VISUALIZATION
    # ============================================================
    
    # Confusion Matrix Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    axes[0].set_title(f'Random Forest (Acc: {acc_rf:.1%})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    axes[1].set_title(f'Neural Network (Acc: {acc_nn:.1%})', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.suptitle('4-Class Classification: Disease Status × Collagen', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/4class_confusion_matrices.png', dpi=150)
    print(f"\nSaved confusion matrices to {OUTPUT_DIR}/4class_confusion_matrices.png")
    
    # Feature Importance Plot
    plt.figure(figsize=(10, 6))
    colors = []
    for feat in X.columns[indices[:top_n]]:
        if 'Actin' in feat:
            colors.append('#E74C3C')
        elif 'Mito' in feat:
            colors.append('#27AE60')
        else:
            colors.append('#3498DB')
    
    plt.bar(range(min(top_n, X.shape[1])), importances[indices[:top_n]], color=colors)
    plt.xticks(range(min(top_n, X.shape[1])), X.columns[indices[:top_n]], rotation=45, ha='right')
    plt.title('4-Class Classification - Feature Importances', fontsize=14, fontweight='bold')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/4class_feature_importance.png', dpi=150)
    print(f"Saved feature importance to {OUTPUT_DIR}/4class_feature_importance.png")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY: 4-Class Classification")
    print("=" * 50)
    print(f"\n{'Classifier':<20} {'Accuracy':<12}")
    print("-" * 35)
    print(f"{'Random Forest':<20} {acc_rf:.1%}")
    print(f"{'Neural Network':<20} {acc_nn:.1%}")
    
    # Save results
    results = pd.DataFrame({
        'Cell_ID': X_test.index,
        'Actual': y_test.values,
        'Actual_Name': [CLASS_NAMES[i] for i in y_test.values],
        'RF_Predicted': y_pred_rf,
        'RF_Name': [CLASS_NAMES[i] for i in y_pred_rf],
        'NN_Predicted': y_pred_nn,
        'NN_Name': [CLASS_NAMES[i] for i in y_pred_nn]
    })
    results.to_csv(f'{OUTPUT_DIR}/4class_predictions.csv', index=False)
    print(f"\nSaved predictions to {OUTPUT_DIR}/4class_predictions.csv")

if __name__ == "__main__":
    main()
