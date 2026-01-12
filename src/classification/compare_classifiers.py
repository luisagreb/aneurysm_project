#!/usr/bin/env python3
"""
Compare multiple machine learning classifiers for Healthy vs TAA classification.
Outputs: comparison plot, detailed results CSV, and cross-validation scores.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

warnings.filterwarnings('ignore')

# Configuration
FEATURES_FILE = 'Advanced_Features_Raw.csv'
METADATA_FILE = 'data/Book1.xlsx'
OUTPUT_DIR = Path('classification_results')


def extract_subject_id(filename):
    """Extract numeric subject ID from filename."""
    if pd.isna(filename):
        return None
    text = str(filename).strip()
    first_token = text.split(' ')[0].strip('-')
    parts = first_token.split('-')
    if len(parts) > 1:
        last_part = parts[-1]
        if last_part.isdigit():
            return int(last_part)
        sub_digits = re.findall(r'\d+', last_part)
        if sub_digits:
            return int(sub_digits[0])
    digits = re.findall(r'\d+', first_token)
    if digits:
        return int(digits[-1])
    return None


def normalize_filename(f):
    """Normalize filename for cell-level grouping."""
    s = str(f).replace('.nii.gz', '').replace('.tif', '')
    s = s.replace('_segmentation', '').replace('_visible', '')
    return s.strip()


def load_data():
    """Load and prepare the dataset."""
    print("Loading data...")
    
    # Load metadata
    df_meta = pd.read_excel(METADATA_FILE, header=None, skiprows=18)
    healthy_ids = set()
    taa_ids = set()
    for x in df_meta[0].dropna():
        nid = extract_subject_id(str(x))
        if nid:
            healthy_ids.add(nid)
    for x in df_meta[1].dropna():
        nid = extract_subject_id(str(x))
        if nid:
            taa_ids.add(nid)
    
    print(f"  Healthy IDs: {sorted(healthy_ids)}")
    print(f"  TAA IDs: {sorted(taa_ids)}")
    
    # Load features
    df = pd.read_csv(FEATURES_FILE)
    
    # Extract IDs and Labels
    df['Numeric_ID'] = df['Filename'].apply(extract_subject_id)
    df = df.dropna(subset=['Numeric_ID'])
    
    def get_label(nid):
        if nid in healthy_ids:
            return 0
        if nid in taa_ids:
            return 1
        return None
    
    df['Label'] = df['Numeric_ID'].apply(get_label)
    df = df.dropna(subset=['Label'])
    
    # Cell-level aggregation
    df['Cell_ID'] = df['Filename'].apply(normalize_filename)
    
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c not in ['Label', 'Numeric_ID']]
    
    agg_dict = {col: 'max' for col in feature_cols}
    agg_dict['Label'] = 'first'
    
    df_agg = df.groupby('Cell_ID').agg(agg_dict)
    
    # Prepare X and y
    X = df_agg.drop(columns=['Label'])
    y = df_agg['Label']
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"  Dataset: {len(X)} cells ({int((y==0).sum())} Healthy, {int((y==1).sum())} TAA)")
    
    return X, y


def get_classifiers():
    """Return dictionary of classifiers to compare."""
    return {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'K-Nearest Neighbors': Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=5))
        ]),
        'Neural Network': Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, 
                                  solver='adam', learning_rate_init=0.001,
                                  alpha=0.1, early_stopping=False, random_state=42))
        ]),
    }


def evaluate_classifiers(X, y):
    """Evaluate all classifiers using cross-validation."""
    print("\n" + "=" * 60)
    print("CLASSIFIER COMPARISON (5-Fold Stratified Cross-Validation)")
    print("=" * 60)
    
    classifiers = get_classifiers()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    
    for name, clf in classifiers.items():
        # Cross-validation scores
        acc_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        f1_scores = cross_val_score(clf, X, y, cv=cv, scoring='f1')
        auc_scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
        
        results.append({
            'Classifier': name,
            'Accuracy_Mean': acc_scores.mean(),
            'Accuracy_Std': acc_scores.std(),
            'F1_Mean': f1_scores.mean(),
            'F1_Std': f1_scores.std(),
            'AUC_Mean': auc_scores.mean(),
            'AUC_Std': auc_scores.std(),
            'All_Acc_Scores': acc_scores
        })
        
        print(f"\n{name}:")
        print(f"  Accuracy: {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")
        print(f"  F1 Score: {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")
        print(f"  AUC-ROC:  {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")
    
    return pd.DataFrame(results)


def plot_comparison(results_df, output_path):
    """Create comparison visualization."""
    # Sort by accuracy
    results_df = results_df.sort_values('Accuracy_Mean', ascending=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    # Color palette
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results_df)))
    
    # Plot 1: Accuracy
    ax1 = axes[0]
    bars1 = ax1.barh(results_df['Classifier'], results_df['Accuracy_Mean'], 
                     xerr=results_df['Accuracy_Std'], color=colors, capsize=3)
    ax1.set_xlabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlim(0.4, 1.0)
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    
    # Add value labels
    for i, (acc, std) in enumerate(zip(results_df['Accuracy_Mean'], results_df['Accuracy_Std'])):
        ax1.text(acc + std + 0.02, i, f'{acc:.1%}', va='center', fontsize=10)
    
    # Plot 2: F1 Score
    ax2 = axes[1]
    ax2.barh(results_df['Classifier'], results_df['F1_Mean'],
             xerr=results_df['F1_Std'], color=colors, capsize=3)
    ax2.set_xlabel('F1 Score', fontsize=12)
    ax2.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlim(0.4, 1.0)
    ax2.set_yticklabels([])
    
    # Plot 3: AUC-ROC
    ax3 = axes[2]
    ax3.barh(results_df['Classifier'], results_df['AUC_Mean'],
             xerr=results_df['AUC_Std'], color=colors, capsize=3)
    ax3.set_xlabel('AUC-ROC', fontsize=12)
    ax3.set_title('AUC-ROC Comparison', fontsize=14, fontweight='bold')
    ax3.set_xlim(0.4, 1.0)
    ax3.set_yticklabels([])
    
    plt.suptitle('Machine Learning Classifier Comparison\n(5-Fold Cross-Validation)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved comparison plot: {output_path}")


def main():
    print("=" * 60)
    print("CLASSIFIER COMPARISON SCRIPT")
    print("Healthy vs TAA Cell Classification")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load data
    X, y = load_data()
    
    # Evaluate classifiers
    results_df = evaluate_classifiers(X, y)
    
    # Sort and display ranking
    results_sorted = results_df.sort_values('Accuracy_Mean', ascending=False)
    
    print("\n" + "=" * 60)
    print("FINAL RANKING (by Accuracy)")
    print("=" * 60)
    for i, row in enumerate(results_sorted.itertuples(), 1):
        print(f"{i}. {row.Classifier}: {row.Accuracy_Mean:.1%} (±{row.Accuracy_Std:.1%})")
    
    # Save results
    results_csv = OUTPUT_DIR / 'classifier_comparison.csv'
    results_df.drop(columns=['All_Acc_Scores']).to_csv(results_csv, index=False)
    print(f"\nSaved results: {results_csv}")
    
    # Create plot
    plot_path = OUTPUT_DIR / 'classifier_comparison.png'
    plot_comparison(results_df, plot_path)
    
    # Best classifier recommendation
    best = results_sorted.iloc[0]
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print(f"Best performer: {best.Classifier}")
    print(f"  Accuracy: {best.Accuracy_Mean:.1%}")
    print(f"  F1 Score: {best.F1_Mean:.1%}")
    print(f"  AUC-ROC:  {best.AUC_Mean:.1%}")


if __name__ == '__main__':
    main()
