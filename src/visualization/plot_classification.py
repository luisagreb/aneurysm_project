#!/usr/bin/env python3
"""
Generate publication-quality Random Forest classification plots for presentation.
Includes: Confusion Matrix, Feature Importance, Classification Summary.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                            roc_curve, auc, precision_recall_curve)
from pathlib import Path

# Configuration
FEATURES_FILE = 'Advanced_Features_Raw.csv'
METADATA_FILE = 'data/Book1.xlsx'
OUTPUT_DIR = Path('/home/luisa/aneurysm_project/classification_results')


def extract_subject_id(filename):
    """Robustly extract Subject ID from filename."""
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
    """Normalize filename for cell-level aggregation."""
    s = str(f).replace('.nii.gz', '').replace('.tif', '')
    s = s.replace('_segmentation', '').replace('_visible', '')
    return s.strip()


def load_metadata(filepath):
    """Load healthy/TAA IDs from Excel."""
    df = pd.read_excel(filepath, header=None, skiprows=18)
    healthy_ids = set()
    taa_ids = set()
    for x in df[0].dropna():
        nid = extract_subject_id(str(x))
        if nid:
            healthy_ids.add(nid)
    for x in df[1].dropna():
        nid = extract_subject_id(str(x))
        if nid:
            taa_ids.add(nid)
    return healthy_ids, taa_ids


def main():
    print("=== Random Forest Classification Analysis ===\n")
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 1. Load Data
    print("Loading data...")
    healthy_ids, taa_ids = load_metadata(METADATA_FILE)
    df = pd.read_csv(FEATURES_FILE)
    
    # 2. Extract IDs and Labels
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
    
    # 3. Cell-level Aggregation
    df['Cell_ID'] = df['Filename'].apply(normalize_filename)
    
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c not in ['Label', 'Numeric_ID']]
    
    agg_dict = {col: 'max' for col in feature_cols}
    agg_dict['Label'] = 'first'
    agg_dict['Numeric_ID'] = 'first'
    
    df_agg = df.groupby('Cell_ID').agg(agg_dict)
    
    # 4. Prepare ML Data
    X = df_agg.drop(columns=['Label', 'Numeric_ID'])
    y = df_agg['Label']
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"Dataset: {len(X)} cells ({int((y==0).sum())} Healthy, {int((y==1).sum())} TAA)")
    
    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 6. Train Model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # 7. Predictions
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nAccuracy: {acc:.2%}")
    print(f"Confusion Matrix:\n{cm}")
    
    # ========== PLOTTING ==========
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 12
    
    # Create multi-panel figure
    fig = plt.figure(figsize=(16, 10))
    
    # --- Panel 1: Confusion Matrix ---
    ax1 = fig.add_subplot(2, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'TAA'], 
                yticklabels=['Healthy', 'TAA'],
                annot_kws={'size': 20}, ax=ax1)
    ax1.set_xlabel('Predicted', fontsize=14)
    ax1.set_ylabel('Actual', fontsize=14)
    ax1.set_title(f'Confusion Matrix\nAccuracy: {acc:.1%}', fontsize=16, fontweight='bold')
    
    # --- Panel 2: Feature Importance (Top 10) ---
    ax2 = fig.add_subplot(2, 2, 2)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    top_features = X.columns[indices]
    top_importances = importances[indices]
    
    # Color by structure type
    colors = []
    for f in top_features:
        if f.startswith('Actin'):
            colors.append('#E74C3C')  # Red
        elif f.startswith('Mito'):
            colors.append('#27AE60')  # Green
        elif f.startswith('Nucleus'):
            colors.append('#3498DB')  # Blue
        else:
            colors.append('#7F8C8D')  # Gray
    
    bars = ax2.barh(range(len(top_features)), top_importances, color=colors)
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels([f.replace('_', ' ') for f in top_features], fontsize=11)
    ax2.invert_yaxis()
    ax2.set_xlabel('Importance', fontsize=14)
    ax2.set_title('Top 10 Predictive Features', fontsize=16, fontweight='bold')
    
    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E74C3C', label='Actin'),
        Patch(facecolor='#27AE60', label='Mitochondria'),
        Patch(facecolor='#3498DB', label='Nucleus')
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # --- Panel 3: ROC Curve ---
    ax3 = fig.add_subplot(2, 2, 3)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    ax3.plot(fpr, tpr, color='#2C3E50', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax3.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    ax3.fill_between(fpr, tpr, alpha=0.3, color='#3498DB')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate', fontsize=14)
    ax3.set_ylabel('True Positive Rate', fontsize=14)
    ax3.set_title('ROC Curve', fontsize=16, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=12)
    
    # --- Panel 4: Class Distribution & Summary ---
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    # Summary text
    summary_text = f"""
    CLASSIFICATION SUMMARY
    ══════════════════════════════════════
    
    Dataset:
      • Total Cells: {len(X)}
      • Healthy: {int((y==0).sum())} cells
      • TAA (Aneurysm): {int((y==1).sum())} cells
    
    Train/Test Split:
      • Training: {len(X_train)} cells (70%)
      • Testing: {len(X_test)} cells (30%)
    
    Model Performance:
      • Accuracy: {acc:.1%}
      • AUC-ROC: {roc_auc:.3f}
      • True Positives (TAA): {cm[1,1]}
      • True Negatives (Healthy): {cm[0,0]}
      • False Positives: {cm[0,1]}
      • False Negatives: {cm[1,0]}
    
    Top 3 Predictive Features:
      1. {top_features[0].replace('_', ' ')}
      2. {top_features[1].replace('_', ' ')}
      3. {top_features[2].replace('_', ' ')}
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#ECF0F1', alpha=0.8))
    
    # Main title
    fig.suptitle('Random Forest Classification: Healthy vs TAA Cells', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_path = OUTPUT_DIR / 'classification_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    
    # Also save individual plots
    # Confusion Matrix alone
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'TAA'],
                yticklabels=['Healthy', 'TAA'],
                annot_kws={'size': 24}, ax=ax_cm)
    ax_cm.set_xlabel('Predicted', fontsize=14)
    ax_cm.set_ylabel('Actual', fontsize=14)
    ax_cm.set_title(f'Confusion Matrix (Accuracy: {acc:.1%})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=150, facecolor='white')
    plt.close()
    
    # Feature Importance alone
    fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
    ax_fi.barh(range(len(top_features)), top_importances, color=colors)
    ax_fi.set_yticks(range(len(top_features)))
    ax_fi.set_yticklabels([f.replace('_', ' ') for f in top_features], fontsize=12)
    ax_fi.invert_yaxis()
    ax_fi.set_xlabel('Importance', fontsize=14)
    ax_fi.set_title('Top 10 Predictive Features by Structure', fontsize=14, fontweight='bold')
    ax_fi.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance_colored.png', dpi=150, facecolor='white')
    plt.close()
    
    print(f"Saved: {OUTPUT_DIR / 'confusion_matrix.png'}")
    print(f"Saved: {OUTPUT_DIR / 'feature_importance_colored.png'}")
    
    print("\n=== Done ===")


if __name__ == '__main__':
    main()
