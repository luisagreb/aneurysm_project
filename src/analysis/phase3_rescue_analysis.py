"""
Phase 3: Rescue Experiment - Does Collagen "Cure" TAA Cells?

Goal: Use the Phase 2 trained model (which has NEVER seen collagen data)
      to test if TAA+Collagen cells are classified as Healthy or TAA.

Hypothesis:
- If model predicts TAA → Collagen did NOT rescue the phenotype
- If model predicts Healthy → Collagen DID rescue the phenotype

This is more powerful than simple statistics because it tests whether
collagen induces a comprehensive "digital fingerprint" change.

Outputs:
- Rescue predictions
- Phenotype shift analysis
- Visualization of rescue vs. non-rescue

Author: Antigravity AI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import re

# Configuration
FEATURES_FILE = 'outputs/Advanced_Features_Raw.csv'
METADATA_FILE = 'data/Book1.xlsx'
MODEL_PATH = 'src/analysis/outputs/phase2/healthy_vs_taa_model.pkl'
SCALER_PATH = 'src/analysis/outputs/phase2/scaler.pkl'
OUTPUT_DIR = 'src/analysis/outputs/phase3'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_numeric_id(text):
    """
    Robustly extracts Subject ID from filename.
    From train_classifier.py
    """
    if pd.isna(text):
        return None
        
    text = str(text).strip()
    
    # Step 1: Get first token (Subject ID part)
    first_token = text.split(' ')[0]
    
    # Step 2: Remove trailing hyphens
    token = first_token.strip('-')
    
    # Step 3: Extract number
    # Priority A: Hyphen separator (common in 01C-XXX, 01ASC-XXX)
    parts = token.split('-')
    if len(parts) > 1:
        last_part = parts[-1]
        if last_part.isdigit():
            return int(last_part)
        sub_digits = re.findall(r'\d+', last_part)
        if sub_digits:
            return int(sub_digits[0])
            
    # Priority B: No hyphen (e.g. 03Asc46), just take the last number group
    digits = re.findall(r'\d+', token)
    if digits:
        return int(digits[-1])
        
    return None

def load_metadata(filepath):
    """Load metadata to classify Healthy vs TAA."""
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
    """
    Extract collagen status from filename.
    From train_collagen_classifier.py
    """
    if pd.isna(filename):
        return None
    
    filename = str(filename).lower()
    
    # Positive Collagen
    if '+coll' in filename or '+col' in filename or 'plus coll' in filename:
        return 'Collagen'
        
    # Negative Collagen (including DMSO control)
    if '-coll' in filename or '-col' in filename or 'nocoll' in filename or 'no coll' in filename or 'dmso' in filename:
        return 'NoCollagen'
        
    return None

def main():
    print("=" * 80)
    print("PHASE 3: RESCUE EXPERIMENT - Does Collagen Rescue TAA Phenotype?")
    print("=" * 80)
    
    # 1. Load trained model from Phase 2
    print(f"\nLoading Phase 2 model: {MODEL_PATH}")
    rf_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # 2. Load data
    healthy_ids, taa_ids = load_metadata(METADATA_FILE)
    df = pd.read_csv(FEATURES_FILE)
    
    # 3. Add labels
    df['Numeric_ID'] = df['CellName'].apply(extract_numeric_id)
    df['Collagen_Status'] = df['CellName'].apply(extract_collagen_status)
    
    def get_disease(nid):
        if nid in healthy_ids:
            return 'Healthy'
        if nid in taa_ids:
            return 'TAA'
        return None
    
    df['Disease'] = df['Numeric_ID'].apply(get_disease)
    df = df.dropna(subset=['Disease', 'Collagen_Status'])
    
    # 4. Filter to TAA + Collagen cells ONLY
    df_rescue = df[(df['Disease'] == 'TAA') & (df['Collagen_Status'] == 'Collagen')].copy()
    
    print(f"\nTAA + Collagen cells: {len(df_rescue)}")
    
    if len(df_rescue) == 0:
        print("ERROR: No TAA+Collagen cells found!")
        return
    
    # 5. Prepare features (same as Phase 2)
    feature_cols = [col for col in df.columns if 
                    col.endswith('_µm') or col.endswith('_ratio') or col.endswith('_µm³') or 
                    col.endswith('_µm²') or col.endswith('_n')]
    
    X_rescue = df_rescue[feature_cols].copy()
    X_rescue = X_rescue.replace([np.inf, -np.inf], np.nan)
    X_rescue = X_rescue.fillna(0)
    
    # 6. Predict using Phase 2 model
    print("\nRunning AI prediction on TAA+Collagen cells...")
    y_pred_rescue = rf_model.predict(X_rescue)
    y_proba_rescue = rf_model.predict_proba(X_rescue)
    
    # 7. Analyze results
    predicted_healthy = sum(y_pred_rescue == 0)
    predicted_taa = sum(y_pred_rescue == 1)
    
    rescue_rate = predicted_healthy / len(y_pred_rescue) * 100
    
    print(f"\n{'='*80}")
    print("RESCUE ANALYSIS RESULTS")
    print(f"{'='*80}")
    print(f"TAA+Collagen cells predicted as Healthy: {predicted_healthy} ({rescue_rate:.1f}%)")
    print(f"TAA+Collagen cells predicted as TAA: {predicted_taa} ({100-rescue_rate:.1f}%)")
    
    # 8. Interpretation
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")
    if rescue_rate > 70:
        print("✅ STRONG RESCUE: Collagen treatment induces comprehensive phenotypic reversion.")
        print("   The AI model classifies most TAA+Collagen cells as Healthy.")
    elif rescue_rate > 40:
        print("⚠️  PARTIAL RESCUE: Collagen treatment shows moderate phenotypic effects.")
        print("   Some TAA+Collagen cells appear Healthy, but many retain TAA features.")
    else:
        print("❌ NO RESCUE: Collagen treatment does not rescue the TAA phenotype.")
        print("   The AI model still classifies most TAA+Collagen cells as diseased.")
    
    # 9. Save predictions
    predictions_df = pd.DataFrame({
        'CellName': df_rescue['CellName'].values,
        'True_Disease': 'TAA',
        'Treatment': 'Collagen',
        'AI_Prediction': pd.Series(y_pred_rescue).map({0: 'Healthy', 1: 'TAA'}).values,
        'Probability_Healthy': y_proba_rescue[:, 0],
        'Probability_TAA': y_proba_rescue[:, 1],
        'Rescued': y_pred_rescue == 0
    })
    predictions_df.to_csv(f'{OUTPUT_DIR}/rescue_predictions.csv', index=False)
    print(f"\nSaved: {OUTPUT_DIR}/rescue_predictions.csv")
    
    # 10. Visualization: Distribution of predicted probabilities
    plt.figure(figsize=(10, 6))
    
    plt.hist(y_proba_rescue[:, 0], bins=20, alpha=0.7, color='#3498DB', edgecolor='black', label='Predicted Healthy')
    plt.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Boundary')
    
    plt.xlabel('Probability of Being Healthy', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Cells', fontsize=12, fontweight='bold')
    plt.title(f'Phase 3: AI Rescue Prediction\n{rescue_rate:.1f}% of TAA+Collagen cells classified as Healthy', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/rescue_probability_distribution.png', dpi=300)
    print(f"Saved: {OUTPUT_DIR}/rescue_probability_distribution.png")
    
    # 11. Comparison plot: TAA-NoCol vs TAA+Col predictions
    df_taa_nocol = df[(df['Disease'] == 'TAA') & (df['Collagen_Status'] == 'NoCollagen')].copy()
    X_taa_nocol = df_taa_nocol[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_proba_taa_nocol = rf_model.predict_proba(X_taa_nocol)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # TAA - No Collagen
    axes[0].hist(y_proba_taa_nocol[:, 0], bins=20, alpha=0.7, color='#E74C3C', edgecolor='black')
    axes[0].axvline(0.5, color='black', linestyle='--', linewidth=2)
    axes[0].set_title('TAA - No Collagen\n(Ground Truth)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Probability of Being Healthy', fontsize=11)
    axes[0].set_ylabel('Number of Cells', fontsize=11)
    axes[0].grid(alpha=0.3)
    
    # TAA + Collagen
    axes[1].hist(y_proba_rescue[:, 0], bins=20, alpha=0.7, color='#3498DB', edgecolor='black')
    axes[1].axvline(0.5, color='black', linestyle='--', linewidth=2)
    axes[1].set_title(f'TAA + Collagen\n({rescue_rate:.1f}% Rescued)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Probability of Being Healthy', fontsize=11)
    axes[1].set_ylabel('Number of Cells', fontsize=11)
    axes[1].grid(alpha=0.3)
    
    plt.suptitle('Phase 3: Collagen Rescue Effect - AI Prediction Comparison', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/rescue_comparison.png', dpi=300)
    print(f"Saved: {OUTPUT_DIR}/rescue_comparison.png")
    
    print(f"\n{'='*80}")
    print("THESIS SECTION 3.3: Collagen Rescue Analysis Complete")
    print(f"{'='*80}")
    
    # Thesis-ready summary
    print(f"\nTHESIS SUMMARY:")
    print(f"  Using a pre-trained classifier, {rescue_rate:.1f}% of TAA+Collagen cells")
    print(f"  were classified as phenotypically Healthy, suggesting collagen treatment")
    if rescue_rate > 40:
        print(f"  induces partial phenotypic reversion toward the healthy state.")
    else:
        print(f"  does not fully rescue the diseased morphological phenotype.")

if __name__ == '__main__':
    main()
