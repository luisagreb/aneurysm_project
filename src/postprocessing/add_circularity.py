import pandas as pd
import os

FINAL_CSV = 'outputs/Advanced_Features_Raw_Final.csv'
RESTORED_CSV = 'outputs/Advanced_Features_Restored.csv'
OUTPUT_CSV = 'outputs/Advanced_Features_Raw_Final_WithCircularity.csv'

def normalize(s):
    # Standardize filename for matching
    s = str(s).lower().strip()
    s = s.replace('.nii.gz', '').replace('.tif', '')
    s = s.replace('_segmentation', '').replace('_visible', '')
    return s.strip()

print("Loading data...")
df_final = pd.read_csv(FINAL_CSV)
df_restored = pd.read_csv(RESTORED_CSV)

# Check if Circularity already exists
if 'Nucleus_Circularity_ratio' in df_final.columns:
    print("Circularity already exists in Final CSV!")
    df_merged = df_final
else:
    print("Merging Circularity...")
    
    # Prepare Restored for merging
    # We want 'Nucleus_Circularity' column.
    # In Restored, it might be named 'Nucleus_Circularity' or 'Nucleus_Circularity_ratio'
    # My header check showed 'Circularity' in the raw file, but 'Advanced_Features_Restored.csv' merged them with prefix.
    
    target_col = None
    if 'Nucleus_Circularity' in df_restored.columns: target_col = 'Nucleus_Circularity'
    elif 'Nucleus_Circularity_ratio' in df_restored.columns: target_col = 'Nucleus_Circularity_ratio'
    
    if not target_col:
        print("Error: Could not find Nucleus_Circularity in Restored CSV")
        print("Columns:", df_restored.columns)
        exit(1)
        
    print(f"Found source column: {target_col}")
    
    # Create lookup dictionary
    # Normalize keys
    circ_map = {}
    for idx, row in df_restored.iterrows():
        key = normalize(row['Filename'])
        val = row[target_col]
        circ_map[key] = val
        
    # Map to Final
    import difflib
    map_keys = list(circ_map.keys())

    def get_circ(name):
        key = normalize(name)
        val = circ_map.get(key, None)
        if val is not None:
            return val
            
        # Fallback: Fuzzy match
        matches = difflib.get_close_matches(key, map_keys, n=1, cutoff=0.95)
        if matches:
            # print(f"Fuzzy matched: '{key}' -> '{matches[0]}'")
            return circ_map[matches[0]]
            
        return None
        
    df_final['Nucleus_Circularity_ratio'] = df_final['CellName'].apply(get_circ)
    
    # Check coverage
    missing_rows = df_final[df_final['Nucleus_Circularity_ratio'].isna()]
    missing = len(missing_rows)
    print(f"Mapped Circularity to {len(df_final) - missing}/{len(df_final)} rows.")
    if missing > 0:
        print(f"Warning: {missing} rows missing Circularity.")
    
    df_merged = df_final

# Save
df_merged.to_csv(OUTPUT_CSV, index=False)
print(f"Saved merged file to {OUTPUT_CSV}")
