"""
Collagen Classifier - Binary Classification: +Collagen vs No-Collagen
Analyzes whether the presence of collagen in the cell culture can be predicted
from cellular structure features (actin, mitochondria, nucleus).
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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Configuration
FEATURES_FILE = 'outputs/Advanced_Features_Raw.csv'
OUTPUT_DIR = 'classification_results/collagen'

def extract_collagen_status(filename):
    """
    Extract collagen status from filename.
    +coll or +Coll -> 1 (Collagen)
    -coll or no +coll -> 0 (No Collagen)
    """
    if pd.isna(filename):
        return None
    
    filename = str(filename).lower()
    
    # Positive Collagen
    if '+coll' in filename or '+col' in filename or 'plus coll' in filename:
        return 1  # Collagen
        
    # Negative Collagen (including DMSO control)
    if '-coll' in filename or '-col' in filename or 'nocoll' in filename or 'no coll' in filename or 'dmso' in filename:
        return 0  # No Collagen
        
    # Check if there's no mention of collagen - assume no collagen
    # But safer to return None for unclear cases
    return None

def normalize_filename(f):
    """Normalize filename to get unique cell ID."""
    s = str(f).replace('.nii.gz', '').replace('.tif', '')
    s = s.replace('_segmentation', '').replace('_visible', '')
    return s.strip()

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("COLLAGEN CLASSIFIER: +Collagen vs No-Collagen")
    print("=" * 60)
    
    # 1. Load Features
    if not os.path.exists(FEATURES_FILE):
        print(f"Error: {FEATURES_FILE} not found.")
        return
        
    df_features = pd.read_csv(FEATURES_FILE)
    print(f"Features loaded: {df_features.shape}")
    
    # 2. Extract Collagen Status from CellName
    df_features['Collagen'] = df_features['CellName'].apply(extract_collagen_status)
    
    # Drop rows where collagen status couldn't be determined
    initial_count = len(df_features)
    df_features = df_features.dropna(subset=['Collagen'])
    print(f"Collagen status determined for {len(df_features)}/{initial_count} rows.")
    
    if df_features.empty:
        print("No samples with determinable collagen status.")
        return
    
    # 3. Aggregate by Cell
    df_features['Cell_ID'] = df_features['CellName'].apply(normalize_filename)
    
    feature_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    if 'Collagen' in feature_cols:
        feature_cols.remove('Collagen')
    
    agg_dict = {col: 'max' for col in feature_cols}
    agg_dict['Collagen'] = 'first'
    
    df_agg = df_features.groupby('Cell_ID').agg(agg_dict)
    print(f"Aggregated to {len(df_agg)} unique cells.")
    
    # 4. Prepare ML Data
    X = df_agg.drop(columns=['Collagen'])
    y = df_agg['Collagen']
    
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    print(f"\nClass Balance:")
    print(f"  No Collagen (0): {sum(y == 0)}")
    print(f"  + Collagen (1):  {sum(y == 1)}")
    
    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 6. Feature Scaling
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
    print(classification_report(y_test, y_pred_rf, target_names=['No Coll (0)', '+Coll (1)']))
    
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
        hidden_layer_sizes=(50, 25),
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
    print(classification_report(y_test, y_pred_nn, target_names=['No Coll (0)', '+Coll (1)']))
    
    # ============================================================
    # VISUALIZATION
    # ============================================================
    
    # Confusion Matrix Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['No Coll', '+Coll'], yticklabels=['No Coll', '+Coll'])
    axes[0].set_title(f'Random Forest (Acc: {acc_rf:.1%})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['No Coll', '+Coll'], yticklabels=['No Coll', '+Coll'])
    axes[1].set_title(f'Neural Network (Acc: {acc_nn:.1%})', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.suptitle('Collagen Classification: Confusion Matrices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/collagen_confusion_matrices.png', dpi=150)
    print(f"\nSaved confusion matrices to {OUTPUT_DIR}/collagen_confusion_matrices.png")
    
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
    plt.title('Collagen Classification - Feature Importances', fontsize=14, fontweight='bold')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/collagen_feature_importance.png', dpi=150)
    print(f"Saved feature importance to {OUTPUT_DIR}/collagen_feature_importance.png")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY: Collagen Classification")
    print("=" * 50)
    print(f"\n{'Classifier':<20} {'Accuracy':<12}")
    print("-" * 35)
    print(f"{'Random Forest':<20} {acc_rf:.1%}")
    print(f"{'Neural Network':<20} {acc_nn:.1%}")
    
    # Save results
    results = pd.DataFrame({
        'Cell_ID': X_test.index,
        'Actual': y_test.values,
        'RF_Predicted': y_pred_rf,
        'NN_Predicted': y_pred_nn
    })
    results.to_csv(f'{OUTPUT_DIR}/collagen_predictions.csv', index=False)
    print(f"\nSaved predictions to {OUTPUT_DIR}/collagen_predictions.csv")

if __name__ == "__main__":
    main()
