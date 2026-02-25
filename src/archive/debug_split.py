import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split

# Configuration
FEATURES_FILE = 'Advanced_Features_Raw.csv'
METADATA_FILE = 'data/Book1.xlsx'

def extract_numeric_id(text):
    if pd.isna(text):
        return None
    text = str(text).strip()
    parts = text.split('-')
    if len(parts) > 1:
        last_part = parts[-1]
        if not last_part and len(parts) > 1:
             last_part = parts[-2]
        digits = re.findall(r'\d+', last_part)
        if digits:
            return int(digits[0])
    base = text.split(' ')[0]
    digits = re.findall(r'\d+', base)
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

def main():
    print("--- Debug Aggregation ---")
    healthy_ids, taa_ids = load_metadata(METADATA_FILE)
    
    if not os.path.exists(FEATURES_FILE):
        print(f"Error: {FEATURES_FILE} not found.")
        return
        
    df_features = pd.read_csv(FEATURES_FILE)
    print(f"Raw Features loaded: {df_features.shape}")
    
    # Labeling logic
    df_features['Numeric_ID'] = df_features['Filename'].apply(extract_numeric_id)
    df_features = df_features.dropna(subset=['Numeric_ID'])
    
    def get_label(nid):
        if nid in healthy_ids:
            return 0
        if nid in taa_ids:
            return 1
        return None
        
    df_features['Label'] = df_features['Numeric_ID'].apply(get_label)
    df_features = df_features.dropna(subset=['Label'])
    
    # Aggregation
    def normalize_filename(f):
        s = str(f).replace('.nii.gz', '').replace('.tif', '')
        s = s.replace('_segmentation', '').replace('_visible', '')
        # Remove +coll, -coll parts to be super safe if they vary slightly? 
        # Actually just stripping extensions and segmentation suffixes should match row-to-row if base name is same.
        return s.strip()
        
    df_features['Cell_ID'] = df_features['Filename'].apply(normalize_filename)
    print(f"Created Cell_IDs. Unique Cells: {df_features['Cell_ID'].nunique()}")
    
    feature_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    if 'Label' in feature_cols: feature_cols.remove('Label')
    if 'Numeric_ID' in feature_cols: feature_cols.remove('Numeric_ID')
    
    agg_dict = {col: 'max' for col in feature_cols}
    agg_dict['Label'] = 'first'
    agg_dict['Numeric_ID'] = 'first'
    
    df_agg = df_features.groupby('Cell_ID').agg(agg_dict)
    
    print(f"Aggregated Dataset shape: {df_agg.shape} (Cells)")
    
    X = df_agg.drop(columns=['Label'])
    y = df_agg['Label']
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    print(f"Final X: {X.shape}, y: {y.shape}")
    print("Class Balance:")
    print(y.value_counts())
    
    # Split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    except ValueError:
        print("Warning: Stratification failed. Splitting randomly.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
    print("-" * 20)
    print(f"Training Set Size: {X_train.shape[0]}")
    print(f"Test Set Size: {X_test.shape[0]}")
    print("-" * 20)

if __name__ == "__main__":
    main()
