"""
Collagen Rescue Analysis
Question: Does collagen make TAA cells phenotypically more similar to healthy cells?

Approach:
1. Compute centroid (mean feature values) for each group
2. Measure distance: TAA+Coll → Healthy vs TAA-NoColl → Healthy
3. Statistical comparison of feature values
4. Visualization of "rescue effect"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from scipy import stats
from scipy.spatial.distance import cdist, mahalanobis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configuration
FEATURES_FILE = 'outputs/Advanced_Features_Raw.csv'
METADATA_FILE = 'data/Book1.xlsx'
OUTPUT_DIR = 'classification_results/rescue_analysis'

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
    if '+coll' in filename:
        return 1
    elif '-coll' in filename or 'nocoll' in filename or 'no coll' in filename:
        return 0
    return None

def load_metadata(filepath):
    """Load healthy/TAA labels from metadata file."""
    try:
        df = pd.read_excel(filepath, header=None, skiprows=18)
        healthy_ids = set()
        taa_ids = set()
        
        for x in df[0].dropna():
            nid = extract_numeric_id(x)
            if nid: healthy_ids.add(nid)
        for x in df[1].dropna():
            nid = extract_numeric_id(x)
            if nid: taa_ids.add(nid)
        return healthy_ids, taa_ids
    except:
        return set(), set()

def normalize_filename(f):
    s = str(f).replace('.nii.gz', '').replace('.tif', '')
    s = s.replace('_segmentation', '').replace('_visible', '')
    return s.strip()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("COLLAGEN RESCUE ANALYSIS")
    print("Question: Does collagen make TAA cells more like healthy cells?")
    print("=" * 70)
    
    # Load data
    healthy_ids, taa_ids = load_metadata(METADATA_FILE)
    df = pd.read_csv(FEATURES_FILE)
    
    # Extract labels
    df['Numeric_ID'] = df['Filename'].apply(extract_numeric_id)
    df['Collagen'] = df['Filename'].apply(extract_collagen_status)
    df['Disease'] = df['Numeric_ID'].apply(
        lambda x: 0 if x in healthy_ids else (1 if x in taa_ids else None)
    )
    
    # Create 4-class label
    df['Label'] = df.apply(
        lambda r: int(r['Disease']) * 2 + int(r['Collagen']) 
        if pd.notna(r['Disease']) and pd.notna(r['Collagen']) else None, axis=1
    )
    
    df = df.dropna(subset=['Label'])
    df['Cell_ID'] = df['Filename'].apply(normalize_filename)
    
    # Aggregate by cell
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in ['Label', 'Numeric_ID', 'Disease', 'Collagen']:
        if col in feature_cols: feature_cols.remove(col)
    
    agg_dict = {col: 'max' for col in feature_cols}
    agg_dict.update({'Label': 'first', 'Disease': 'first', 'Collagen': 'first'})
    
    df_agg = df.groupby('Cell_ID').agg(agg_dict)
    
    # Prepare data
    X = df_agg[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_agg['Label'].astype(int)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)
    
    print(f"\nSamples per group:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {sum(y == i)}")
    
    # ========================================
    # ANALYSIS 1: Centroid Distances
    # ========================================
    print("\n" + "=" * 60)
    print("ANALYSIS 1: Distance to Healthy Centroids")
    print("=" * 60)
    
    # Compute centroids for each group
    centroids = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = y == i
        centroids[name] = X_scaled[mask].mean().values
    
    # Compute "healthy centroid" as average of both healthy groups
    healthy_centroid = (centroids['Healthy-NoColl'] + centroids['Healthy+Coll']) / 2
    
    # Distance from each TAA cell to healthy centroid
    taa_nocoll_mask = y == 2
    taa_coll_mask = y == 3
    
    taa_nocoll_distances = np.linalg.norm(X_scaled[taa_nocoll_mask].values - healthy_centroid, axis=1)
    taa_coll_distances = np.linalg.norm(X_scaled[taa_coll_mask].values - healthy_centroid, axis=1)
    
    print(f"\nMean distance to Healthy centroid:")
    print(f"  TAA-NoColl: {np.mean(taa_nocoll_distances):.3f} ± {np.std(taa_nocoll_distances):.3f}")
    print(f"  TAA+Coll:   {np.mean(taa_coll_distances):.3f} ± {np.std(taa_coll_distances):.3f}")
    
    # Statistical test
    t_stat, p_value = stats.mannwhitneyu(taa_nocoll_distances, taa_coll_distances, alternative='two-sided')
    print(f"\nMann-Whitney U test: p = {p_value:.4f}")
    
    if np.mean(taa_coll_distances) < np.mean(taa_nocoll_distances):
        rescue_direction = "YES - TAA+Coll is CLOSER to healthy"
        rescue_magnitude = (np.mean(taa_nocoll_distances) - np.mean(taa_coll_distances)) / np.mean(taa_nocoll_distances) * 100
    else:
        rescue_direction = "NO - TAA+Coll is FURTHER from healthy"
        rescue_magnitude = (np.mean(taa_coll_distances) - np.mean(taa_nocoll_distances)) / np.mean(taa_nocoll_distances) * 100
    
    print(f"\n>>> RESCUE EFFECT: {rescue_direction}")
    print(f">>> Magnitude: {abs(rescue_magnitude):.1f}% change in distance")
    if p_value < 0.05:
        print(f">>> Statistically significant (p < 0.05)")
    else:
        print(f">>> NOT statistically significant (p = {p_value:.3f})")
    
    # ========================================
    # ANALYSIS 2: Feature-by-Feature Comparison
    # ========================================
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Feature Normalization by Collagen")
    print("=" * 60)
    
    rescue_features = []
    
    print("\nFeatures where TAA+Coll moves toward healthy values:")
    print("-" * 60)
    
    for feat in feature_cols[:15]:  # Top features
        healthy_mean = X[y.isin([0, 1])][feat].mean()
        taa_nocoll_mean = X[y == 2][feat].mean()
        taa_coll_mean = X[y == 3][feat].mean()
        
        # Check if TAA+Coll is between TAA-NoColl and Healthy
        dist_nocoll_to_healthy = abs(taa_nocoll_mean - healthy_mean)
        dist_coll_to_healthy = abs(taa_coll_mean - healthy_mean)
        
        if dist_coll_to_healthy < dist_nocoll_to_healthy:
            rescue_pct = (1 - dist_coll_to_healthy / dist_nocoll_to_healthy) * 100 if dist_nocoll_to_healthy > 0 else 0
            rescue_features.append({
                'Feature': feat,
                'Healthy_Mean': healthy_mean,
                'TAA-NoColl': taa_nocoll_mean,
                'TAA+Coll': taa_coll_mean,
                'Rescue%': rescue_pct
            })
            if rescue_pct > 20:
                print(f"  {feat}: {rescue_pct:.0f}% rescue")
    
    rescue_df = pd.DataFrame(rescue_features).sort_values('Rescue%', ascending=False)
    
    # ========================================
    # VISUALIZATION
    # ========================================
    
    # 1. PCA Plot
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {'Healthy-NoColl': '#3498DB', 'Healthy+Coll': '#2980B9', 
              'TAA-NoColl': '#E74C3C', 'TAA+Coll': '#F39C12'}
    markers = {'Healthy-NoColl': 'o', 'Healthy+Coll': 's', 
               'TAA-NoColl': 'o', 'TAA+Coll': 's'}
    
    for i, name in enumerate(CLASS_NAMES):
        mask = y == i
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=colors[name], marker=markers[name],
                       label=name, alpha=0.7, s=60)
    
    # Plot arrows showing rescue direction
    pca_centroids = {name: pca.transform([centroids[name]])[0] for name in CLASS_NAMES}
    
    # Arrow from TAA-NoColl to TAA+Coll
    axes[0].annotate('', xy=pca_centroids['TAA+Coll'], xytext=pca_centroids['TAA-NoColl'],
                    arrowprops=dict(arrowstyle='->', color='purple', lw=2))
    axes[0].text((pca_centroids['TAA+Coll'][0] + pca_centroids['TAA-NoColl'][0])/2,
                (pca_centroids['TAA+Coll'][1] + pca_centroids['TAA-NoColl'][1])/2 + 0.3,
                'Collagen\nEffect', ha='center', fontsize=10, color='purple')
    
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0].set_title('PCA: Phenotypic Space', fontweight='bold')
    axes[0].legend(loc='best')
    
    # 2. Distance comparison boxplot
    distance_data = pd.DataFrame({
        'Group': ['TAA-NoColl'] * len(taa_nocoll_distances) + ['TAA+Coll'] * len(taa_coll_distances),
        'Distance to Healthy': list(taa_nocoll_distances) + list(taa_coll_distances)
    })
    
    sns.boxplot(data=distance_data, x='Group', y='Distance to Healthy', 
                palette=['#E74C3C', '#F39C12'], ax=axes[1])
    axes[1].set_title('Distance to Healthy Centroid', fontweight='bold')
    
    # Add significance annotation
    y_max = max(distance_data['Distance to Healthy']) * 1.1
    if p_value < 0.001:
        sig_text = '***'
    elif p_value < 0.01:
        sig_text = '**'
    elif p_value < 0.05:
        sig_text = '*'
    else:
        sig_text = 'ns'
    
    axes[1].plot([0, 0, 1, 1], [y_max*0.95, y_max, y_max, y_max*0.95], 'k-', lw=1)
    axes[1].text(0.5, y_max*1.02, sig_text, ha='center', fontsize=12)
    
    plt.suptitle('Collagen Rescue Analysis: Does Collagen Make TAA Cells More Healthy?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/rescue_analysis.png', dpi=150)
    print(f"\nSaved visualization to {OUTPUT_DIR}/rescue_analysis.png")
    
    # 3. Feature heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap data: normalized values for each group
    heatmap_data = pd.DataFrame()
    for i, name in enumerate(CLASS_NAMES):
        mask = y == i
        heatmap_data[name] = X_scaled[mask].mean()
    
    # Select most variable features
    top_features = heatmap_data.var(axis=1).nlargest(15).index
    heatmap_data = heatmap_data.loc[top_features]
    
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                linewidths=0.5, ax=ax)
    ax.set_title('Feature Values by Group (Standardized)\nLook for TAA+Coll ≈ Healthy values',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/rescue_feature_heatmap.png', dpi=150)
    print(f"Saved feature heatmap to {OUTPUT_DIR}/rescue_feature_heatmap.png")
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("COLLAGEN RESCUE ANALYSIS SUMMARY")
    print("=" * 70)
    
    print(f"""
QUESTION: Does collagen make TAA cells more like healthy cells?

FINDINGS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Distance Analysis:
   • TAA-NoColl mean distance to healthy: {np.mean(taa_nocoll_distances):.3f}
   • TAA+Coll mean distance to healthy:   {np.mean(taa_coll_distances):.3f}
   • Difference: {abs(rescue_magnitude):.1f}% {"closer" if rescue_magnitude > 0 else "further"}
   • Statistical significance: p = {p_value:.4f}

2. Rescue Direction: {rescue_direction}

3. Top Rescued Features (TAA+Coll moves toward healthy):
""")
    
    if not rescue_df.empty:
        for _, row in rescue_df.head(5).iterrows():
            print(f"   • {row['Feature']}: {row['Rescue%']:.0f}% normalized")
    
    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTERPRETATION:
""")
    
    if np.mean(taa_coll_distances) < np.mean(taa_nocoll_distances) and p_value < 0.05:
        print("  ✅ EVIDENCE FOR RESCUE: Collagen significantly moves TAA cells")
        print("     toward a healthy phenotype in feature space.")
    elif np.mean(taa_coll_distances) < np.mean(taa_nocoll_distances):
        print("  ⚠️  SUGGESTIVE OF RESCUE: TAA+Coll is closer to healthy,")
        print("     but the difference is not statistically significant.")
    else:
        print("  ❌ NO RESCUE EFFECT: Collagen does not move TAA cells")
        print("     toward healthy phenotype (or may even move them further).")
    
    # Save results
    rescue_df.to_csv(f'{OUTPUT_DIR}/rescued_features.csv', index=False)
    
    summary = pd.DataFrame({
        'Metric': ['TAA-NoColl to Healthy', 'TAA+Coll to Healthy', 'Rescue %', 'p-value'],
        'Value': [np.mean(taa_nocoll_distances), np.mean(taa_coll_distances), 
                  rescue_magnitude, p_value]
    })
    summary.to_csv(f'{OUTPUT_DIR}/rescue_summary.csv', index=False)
    print(f"\nSaved results to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
