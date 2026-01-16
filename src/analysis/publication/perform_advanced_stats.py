"""
Advanced Statistical Analysis for Publication
===================================================
Implementation of publication-grade statistical analysis:
1. Two-way ANOVA with Interaction (Disease x Collagen)
2. Effect Size Calculation (Partial Eta Squared)
3. Multiple Comparison Correction (Benjamini-Hochberg FDR)
4. Quantitative Rescue Index Calculation
5. Feature Correlation Pruning
6. Organelle Ablation Study (Model Performance by Structure)

Outputs:
- Comprehensive statistical summary CSV
- Rescue Index per feature
- Ablation study comparison
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import os
import re

# Configuration
FEATURES_FILE = 'outputs/Advanced_Features_Raw.csv'
METADATA_FILE = 'data/Book1.xlsx'
OUTPUT_DIR = 'classification_results/publication_stats'

def extract_numeric_id(text):
    if pd.isna(text): return None
    text = str(text).strip()
    first_token = text.split(' ')[0].strip('-')
    parts = first_token.split('-')
    if len(parts) > 1 and parts[-1].isdigit(): return int(parts[-1])
    digits = re.findall(r'\d+', first_token)
    return int(digits[-1]) if digits else None

def load_metadata(filepath):
    try:
        df = pd.read_excel(filepath, header=None, skiprows=18)
        healthy = set()
        taa = set()
        for x in df[0].dropna():
            nid = extract_numeric_id(x)
            if nid: healthy.add(nid)
        for x in df[1].dropna():
            nid = extract_numeric_id(x)
            if nid: taa.add(nid)
        return healthy, taa
    except:
        return set(), set()

def get_collagen_status(filename):
    s = str(filename).lower()
    if '+coll' in s: return 1
    if '-coll' in s or 'nocoll' in s or 'no coll' in s: return 0
    return None

def normalize_filename(f):
    s = str(f).replace('.nii.gz', '').replace('.tif', '')
    s = s.replace('_segmentation', '').replace('_visible', '')
    return s.strip()

def calculate_partial_eta_squared(anova_table):
    """Calculate Partial Eta Squared (Effect Size) from ANOVA table."""
    ss_factor = anova_table['sum_sq']
    ss_error = anova_table.loc['Residual', 'sum_sq']
    return ss_factor / (ss_factor + ss_error)

def calculate_rescue_index(df_agg, feature):
    """
    Calculate Rescue Index: 
    (TAA+Coll - TAA-NoColl) / (Healthy-NoColl - TAA-NoColl)
    """
    means = df_agg.groupby(['Disease', 'Collagen_Label'])[feature].mean()
    try:
        taa_coll = means.loc[('TAA', '+Coll')]
        taa_nocoll = means.loc[('TAA', '-Coll')]
        healthy_nocoll = means.loc[('Healthy', '-Coll')]
        
        denominator = healthy_nocoll - taa_nocoll
        if abs(denominator) < 1e-6: return np.nan # Avoid division by zero
        
        return (taa_coll - taa_nocoll) / denominator
    except KeyError:
        return np.nan

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading data...")
    healthy_ids, taa_ids = load_metadata(METADATA_FILE)
    df = pd.read_csv(FEATURES_FILE)
    
    # Labeling
    df['Numeric_ID'] = df['CellName'].apply(extract_numeric_id)
    df['Disease'] = df['Numeric_ID'].apply(lambda x: 'Healthy' if x in healthy_ids else ('TAA' if x in taa_ids else None))
    df['Collagen'] = df['CellName'].apply(get_collagen_status)
    df['Collagen_Label'] = df['Collagen'].map({0: '-Coll', 1: '+Coll'})
    
    # Drop unlabeled
    df = df.dropna(subset=['Disease', 'Collagen'])
    
    # Aggregation
    df['Cell_ID'] = df['CellName'].apply(normalize_filename)
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['Numeric_ID', 'Collagen', 'Voxel_Z_µm', 'Voxel_Y_µm', 'Voxel_X_µm']
    feature_cols = [f for f in feature_cols if f not in exclude]
    
    agg_dict = {col: 'max' for col in feature_cols}
    agg_dict.update({'Disease': 'first', 'Collagen_Label': 'first'})
    
    df_agg = df.groupby('Cell_ID').agg(agg_dict)
    df_agg = df_agg.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # --- PART 1: Advanced Statistics (ANOVA + Effect Size + FDR) ---
    print("\n1. Running Advanced Feature Statistics...")
    stats_results = []
    
    for feat in feature_cols:
        # ANOVA
        model_data = df_agg[[feat, 'Disease', 'Collagen_Label']].rename(columns={feat: 'Value'})
        try:
            model = ols('Value ~ C(Disease) + C(Collagen_Label) + C(Disease):C(Collagen_Label)', data=model_data).fit()
            anova = sm.stats.anova_lm(model, typ=2)
            
            p_disease = anova.loc['C(Disease)', 'PR(>F)']
            p_collagen = anova.loc['C(Collagen_Label)', 'PR(>F)']
            p_interaction = anova.loc['C(Disease):C(Collagen_Label)', 'PR(>F)']
            
            eta_sq = calculate_partial_eta_squared(anova)
            eff_disease = eta_sq['C(Disease)']
            eff_interaction = eta_sq['C(Disease):C(Collagen_Label)']
            
            rescue_idx = calculate_rescue_index(df_agg, feat)
            
            stats_results.append({
                'Feature': feat,
                'p_Disease': p_disease,
                'p_Collagen': p_collagen,
                'p_Interaction': p_interaction,
                'EffectSize_Disease_eta2': eff_disease,
                'EffectSize_Interaction_eta2': eff_interaction,
                'Rescue_Index': rescue_idx
            })
        except Exception as e:
            print(f"Skipping {feat}: {e}")
            
    stats_df = pd.DataFrame(stats_results)
    
    # FDR Correction (Benjamini-Hochberg)
    _, stats_df['p_Disease_adj'], _, _ = multipletests(stats_df['p_Disease'], method='fdr_bh')
    _, stats_df['p_Interaction_adj'], _, _ = multipletests(stats_df['p_Interaction'], method='fdr_bh')
    
    stats_df.sort_values('p_Disease_adj').to_csv(f'{OUTPUT_DIR}/comprehensive_statistics.csv', index=False)
    print(f"Saved stats to {OUTPUT_DIR}/comprehensive_statistics.csv")
    
    # --- PART 2: Ablation Study (Organelle Importance) ---
    print("\n2. Running Ablation Study (Organelle Contributions)...")
    
    organelles = {
        'Actin Only': [c for c in feature_cols if 'Actin' in c],
        'Mito Only': [c for c in feature_cols if 'Mito' in c],
        'Nucleus Only': [c for c in feature_cols if 'Nucleus' in c],
        'All Combined': feature_cols
    }
    
    ablation_results = []
    y = df_agg['Disease'].map({'Healthy': 0, 'TAA': 1})
    
    for name, feats in organelles.items():
        if not feats: continue
        X = df_agg[feats]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
        
        ablation_results.append({
            'Model': name,
            'Feature_Count': len(feats),
            'Mean_Accuracy': scores.mean(),
            'Std_Accuracy': scores.std()
        })
        
    ablation_df = pd.DataFrame(ablation_results).sort_values('Mean_Accuracy', ascending=False)
    ablation_df.to_csv(f'{OUTPUT_DIR}/ablation_study.csv', index=False)
    print("Ablation Results:")
    print(ablation_df)

if __name__ == "__main__":
    main()
