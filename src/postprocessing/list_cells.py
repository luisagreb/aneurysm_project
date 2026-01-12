import pandas as pd
import numpy as np
import re
import os

# Configuration
FEATURES_FILE = 'Advanced_Features_Raw.csv'
METADATA_FILE = 'data/Book1.xlsx'

def extract_subject_id(filename):
    """
    Robustly extracts Subject ID from filename.
    Assumes standard format: "<SubjectID> <Conditions>..."
    
    Logic:
    1. Isolate the first token (space-separated).
       - "01C-222 +Col..." -> "01C-222"
       - "03Asc46 +coll..." -> "03Asc46"
       - "03Asc24- DMSO..." -> "03Asc24-"
       
    2. Clean the token.
       - "03Asc24-" -> "03Asc24"
       
    3. Extract Numeric Component.
       - "01C-222" (Hyphenated) -> Take part after last hyphen ("222").
       - "03Asc46" (No Hyphen) -> Take last contiguous digit sequence ("46").
    """
    if pd.isna(filename):
        return None
        
    text = str(filename).strip()
    
    # Step 1: Get first token (Subject ID part)
    first_token = text.split(' ')[0]
    
    # Step 2: Remove trailing hyphens
    token = first_token.strip('-')
    
    # Step 3: Extract number
    # Priority A: Hyphen separator (common in 01C-XXX, 01ASC-XXX)
    parts = token.split('-')
    if len(parts) > 1:
        last_part = parts[-1]
        # Check if the part after hyphen is numeric
        if last_part.isdigit():
            return int(last_part)
        # If not purely digit (e.g. 01ASC-0180A?), try extracting digits
        sub_digits = re.findall(r'\d+', last_part)
        if sub_digits:
            return int(sub_digits[0])
            
    # Priority B: No hyphen (e.g. 03Asc46), just take the last number group
    # We take the *last* group to avoid prefixes like '03' in '03Asc46' if it's 'Asc'.
    # Note: '03' might be a group. '46' is another. Usually the ID is the suffix.
    digits = re.findall(r'\d+', token)
    if digits:
        return int(digits[-1])
        
    return None

def normalize_filename(f):
    s = str(f).replace('.nii.gz', '').replace('.tif', '')
    s = s.replace('_segmentation', '').replace('_visible', '')
    return s.strip()

def load_metadata(filepath):
    print(f"Loading metadata from {filepath}...")
    try:
        df = pd.read_excel(filepath, header=None, skiprows=18)
        healthy_col = df[0]
        taa_col = df[1]
        healthy_ids = set()
        taa_ids = set()
        
        # We extract IDs from metadata using the SAME logic to ensure consistency
        # Metadata Example: "01Asc-180" -> 180
        for x in healthy_col.dropna():
            nid = extract_subject_id(str(x))
            if nid is not None:
                healthy_ids.add(nid)
        for x in taa_col.dropna():
            nid = extract_subject_id(str(x))
            if nid is not None:
                taa_ids.add(nid)
                
        return healthy_ids, taa_ids
    except Exception as e:
        print(f"Error parse metadata: {e}")
        return set(), set()

def main():
    print("--- Listing Cells and Metadata Mapping (Improved Logic) ---")
    healthy_ids, taa_ids = load_metadata(METADATA_FILE)
    print(f"Loaded {len(healthy_ids)} Healthy IDs and {len(taa_ids)} TAA IDs.")
    print(f"Healthy IDs: {sorted(list(healthy_ids))}")
    print(f"TAA IDs: {sorted(list(taa_ids))}")
    
    if not os.path.exists(FEATURES_FILE):
        print(f"Error: {FEATURES_FILE} not found.")
        return
        
    df = pd.read_csv(FEATURES_FILE)
    df['Cell_ID'] = df['Filename'].apply(normalize_filename)
    
    # Apply robust extraction
    df['Numeric_ID'] = df['Filename'].apply(extract_subject_id)
    
    # Get unique cells for verification
    unique_cells = df[['Cell_ID', 'Numeric_ID']].drop_duplicates()
    
    print(f"\nTotal Unique Cells Found in Features: {len(unique_cells)}")
    
    print("\n--- Matching Results Preview ---")
    print(f"{'Cell ID (First 50 chars)':<50} | {'Extr. ID':<8} | {'Status'}")
    print("-" * 80)
    
    matched_count = 0
    
    # Sort for easier reading
    unique_cells = unique_cells.sort_values('Cell_ID')
    
    for _, row in unique_cells.iterrows():
        cell_id = row['Cell_ID']
        nid = row['Numeric_ID']
        
        status = "UNKNOWN"
        if nid in healthy_ids:
            status = "HEALTHY"
            matched_count += 1
        elif nid in taa_ids:
            status = "TAA"
            matched_count += 1
        else:
            status = "NO MATCH"
            
        print(f"{cell_id[:50]:<50} | {str(nid):<8} | {status}")
        
    print("-" * 80)
    print(f"Total Matched Cells: {matched_count} / {len(unique_cells)}")

if __name__ == "__main__":
    main()
