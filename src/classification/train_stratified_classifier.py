"""
Stratified Classifier - Healthy vs TAA within Collagen Subgroups
Analyzes how well we can distinguish Healthy from TAA cells:
  1. Within the +Collagen group only
  2. Within the No-Collagen group only
  3. Comparison of performance between subgroups
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
METADATA_FILE = 'data/Book1.xlsx'
OUTPUT_DIR = 'classification_results/stratified'

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
    if '+coll' in filename:
        return 1  # Collagen
    elif '-coll' in filename or 'nocoll' in filename or 'no coll' in filename:
        return 0  # No Collagen
    else:
        return None

def load_metadata(filepath):
    """Load healthy/TAA labels from metadata file."""
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
        print(f"Error parsing metadata: {e}")
        return set(), set()

def normalize_filename(f):
    """Normalize filename to get unique cell ID."""
    s = str(f).replace('.nii.gz', '').replace('.tif', '')
    s = s.replace('_segmentation', '').replace('_visible', '')
    return s.strip()

def train_and_evaluate(X_train, X_test, y_train, y_test, subgroup_name):
    """Train RF and NN classifiers and return results."""
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    
    # Neural Network
    nn = MLPClassifier(
        hidden_layer_sizes=(50, 25),
        max_iter=1000,
        random_state=42,
        solver='adam',
        learning_rate_init=0.001,
        alpha=0.1
    )
    nn.fit(X_train_scaled, y_train)
    y_pred_nn = nn.predict(X_test_scaled)
    acc_nn = accuracy_score(y_test, y_pred_nn)
    cm_nn = confusion_matrix(y_test, y_pred_nn)
    
    # Get feature importances
    importances = rf.feature_importances_
    
    return {
        'subgroup': subgroup_name,
        'rf_accuracy': acc_rf,
        'nn_accuracy': acc_nn,
        'rf_cm': cm_rf,
        'nn_cm': cm_nn,
        'feature_importances': importances,
        'feature_names': X_train.columns.tolist(),
        'y_test': y_test,
        'y_pred_rf': y_pred_rf,
        'y_pred_nn': y_pred_nn,
        'X_test_index': X_test.index
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("STRATIFIED ANALYSIS: Healthy vs TAA within Collagen Subgroups")
    print("=" * 70)
    
    # 1. Load Metadata
    healthy_ids, taa_ids = load_metadata(METADATA_FILE)
    if not healthy_ids and not taa_ids:
        print("Failed to load metadata labels.")
        return
    print(f"Loaded {len(healthy_ids)} Healthy IDs, {len(taa_ids)} TAA IDs")
    
    # 2. Load Features
    if not os.path.exists(FEATURES_FILE):
        print(f"Error: {FEATURES_FILE} not found.")
        return
        
    df_features = pd.read_csv(FEATURES_FILE)
    print(f"Features loaded: {df_features.shape}")
    
    # 3. Extract IDs and Labels
    df_features['Numeric_ID'] = df_features['Filename'].apply(extract_numeric_id)
    df_features['Collagen'] = df_features['Filename'].apply(extract_collagen_status)
    
    def get_disease_label(nid):
        if nid in healthy_ids:
            return 0  # Healthy
        if nid in taa_ids:
            return 1  # TAA
        return None
    
    df_features['Disease'] = df_features['Numeric_ID'].apply(get_disease_label)
    
    # Drop unknown labels
    df_features = df_features.dropna(subset=['Disease', 'Collagen'])
    print(f"Samples with both labels: {len(df_features)}")
    
    # 4. Aggregate by Cell
    df_features['Cell_ID'] = df_features['Filename'].apply(normalize_filename)
    
    feature_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    for col in ['Numeric_ID', 'Disease', 'Collagen']:
        if col in feature_cols:
            feature_cols.remove(col)
    
    agg_dict = {col: 'max' for col in feature_cols}
    agg_dict['Disease'] = 'first'
    agg_dict['Collagen'] = 'first'
    
    df_agg = df_features.groupby('Cell_ID').agg(agg_dict)
    print(f"Aggregated to {len(df_agg)} unique cells.")
    
    # Clean data
    X_all = df_agg.drop(columns=['Disease', 'Collagen'])
    X_all = X_all.replace([np.inf, -np.inf], np.nan)
    X_all = X_all.fillna(0)
    
    # Split by collagen status
    mask_coll = df_agg['Collagen'] == 1
    mask_nocoll = df_agg['Collagen'] == 0
    
    print(f"\n+Collagen group: {sum(mask_coll)} cells")
    print(f"  Healthy: {sum((df_agg['Disease'] == 0) & mask_coll)}")
    print(f"  TAA:     {sum((df_agg['Disease'] == 1) & mask_coll)}")
    
    print(f"\nNo-Collagen group: {sum(mask_nocoll)} cells")
    print(f"  Healthy: {sum((df_agg['Disease'] == 0) & mask_nocoll)}")
    print(f"  TAA:     {sum((df_agg['Disease'] == 1) & mask_nocoll)}")
    
    results = []
    
    # ============================================================
    # ANALYSIS 1: +Collagen subgroup
    # ============================================================
    print("\n" + "=" * 60)
    print("SUBGROUP 1: +COLLAGEN")
    print("=" * 60)
    
    X_coll = X_all[mask_coll]
    y_coll = df_agg.loc[mask_coll, 'Disease'].astype(int)
    
    if len(X_coll) >= 10 and len(y_coll.unique()) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X_coll, y_coll, test_size=0.3, random_state=42, stratify=y_coll
        )
        result_coll = train_and_evaluate(X_train, X_test, y_train, y_test, "+Collagen")
        results.append(result_coll)
        
        print(f"\nRandom Forest Accuracy: {result_coll['rf_accuracy']:.1%}")
        print(f"Neural Network Accuracy: {result_coll['nn_accuracy']:.1%}")
        print("\nRF Confusion Matrix:")
        print(result_coll['rf_cm'])
    else:
        print("Insufficient samples for +Collagen analysis.")
        result_coll = None
    
    # ============================================================
    # ANALYSIS 2: No-Collagen subgroup
    # ============================================================
    print("\n" + "=" * 60)
    print("SUBGROUP 2: NO COLLAGEN")
    print("=" * 60)
    
    X_nocoll = X_all[mask_nocoll]
    y_nocoll = df_agg.loc[mask_nocoll, 'Disease'].astype(int)
    
    if len(X_nocoll) >= 10 and len(y_nocoll.unique()) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X_nocoll, y_nocoll, test_size=0.3, random_state=42, stratify=y_nocoll
        )
        result_nocoll = train_and_evaluate(X_train, X_test, y_train, y_test, "No-Collagen")
        results.append(result_nocoll)
        
        print(f"\nRandom Forest Accuracy: {result_nocoll['rf_accuracy']:.1%}")
        print(f"Neural Network Accuracy: {result_nocoll['nn_accuracy']:.1%}")
        print("\nRF Confusion Matrix:")
        print(result_nocoll['rf_cm'])
    else:
        print("Insufficient samples for No-Collagen analysis.")
        result_nocoll = None
    
    # ============================================================
    # VISUALIZATION
    # ============================================================
    
    if results:
        # Comparison Bar Plot
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Accuracy comparison
        subgroups = [r['subgroup'] for r in results]
        rf_accs = [r['rf_accuracy'] for r in results]
        nn_accs = [r['nn_accuracy'] for r in results]
        
        x = np.arange(len(subgroups))
        width = 0.35
        
        axes[0].bar(x - width/2, rf_accs, width, label='Random Forest', color='#3498DB')
        axes[0].bar(x + width/2, nn_accs, width, label='Neural Network', color='#27AE60')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Classification Accuracy by Subgroup', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(subgroups)
        axes[0].legend()
        axes[0].set_ylim(0, 1)
        for i, (rf, nn) in enumerate(zip(rf_accs, nn_accs)):
            axes[0].text(i - width/2, rf + 0.02, f'{rf:.0%}', ha='center', fontsize=9)
            axes[0].text(i + width/2, nn + 0.02, f'{nn:.0%}', ha='center', fontsize=9)
        
        # Confusion matrices
        for idx, result in enumerate(results[:2]):
            ax = axes[idx + 1]
            sns.heatmap(result['rf_cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Healthy', 'TAA'], yticklabels=['Healthy', 'TAA'])
            ax.set_title(f"{result['subgroup']} (RF: {result['rf_accuracy']:.0%})", fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.suptitle('Stratified Analysis: Healthy vs TAA within Collagen Subgroups', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/stratified_comparison.png', dpi=150)
        print(f"\nSaved comparison to {OUTPUT_DIR}/stratified_comparison.png")
        
        # Feature importance comparison
        if len(results) == 2:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            for idx, result in enumerate(results):
                importances = result['feature_importances']
                feature_names = result['feature_names']
                indices = np.argsort(importances)[::-1][:10]
                
                colors = []
                for i in indices:
                    feat = feature_names[i]
                    if 'Actin' in feat:
                        colors.append('#E74C3C')
                    elif 'Mito' in feat:
                        colors.append('#27AE60')
                    else:
                        colors.append('#3498DB')
                
                axes[idx].bar(range(10), importances[indices], color=colors)
                axes[idx].set_xticks(range(10))
                axes[idx].set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
                axes[idx].set_title(f"{result['subgroup']} - Top Features", fontweight='bold')
                axes[idx].set_ylabel('Importance')
            
            plt.suptitle('Feature Importance Comparison by Collagen Status', 
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/stratified_feature_importance.png', dpi=150)
            print(f"Saved feature importance to {OUTPUT_DIR}/stratified_feature_importance.png")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("STRATIFIED ANALYSIS SUMMARY")
    print("=" * 70)
    
    print("\nQuestion: Does collagen affect our ability to classify Healthy vs TAA?")
    print("-" * 70)
    
    summary_data = []
    for result in results:
        summary_data.append({
            'Subgroup': result['subgroup'],
            'RF_Accuracy': f"{result['rf_accuracy']:.1%}",
            'NN_Accuracy': f"{result['nn_accuracy']:.1%}",
            'N_Test': len(result['y_test'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    if len(results) == 2:
        diff_rf = abs(results[0]['rf_accuracy'] - results[1]['rf_accuracy'])
        diff_nn = abs(results[0]['nn_accuracy'] - results[1]['nn_accuracy'])
        print(f"\nAccuracy Difference (RF): {diff_rf:.1%}")
        print(f"Accuracy Difference (NN): {diff_nn:.1%}")
        
        if results[0]['rf_accuracy'] > results[1]['rf_accuracy']:
            better = results[0]['subgroup']
        else:
            better = results[1]['subgroup']
        print(f"\nClassification works better in: {better} subgroup")
    
    # Save summary
    summary_df.to_csv(f'{OUTPUT_DIR}/stratified_summary.csv', index=False)
    print(f"\nSaved summary to {OUTPUT_DIR}/stratified_summary.csv")

if __name__ == "__main__":
    main()
