import pandas as pd
import re
from pathlib import Path

# Manual aliases: cell name prefix → correct metadata prefix
# 01C and 01Asc refer to the same centre
PREFIX_ALIASES = {
    '01c': '01asc',
}

# Manual ID overrides: cell file has wrong specimen ID → correct metadata ID
ID_OVERRIDES = {
    '02asc-0021': '03asc-0021',
    '02asc-21':   '03asc-21',
}

def normalize_id(spec_id):
    """Strip leading zeros from numeric suffix for robust matching.
    e.g. '03asc-0024' -> '03asc-24', '01c-0096' -> '01c-96'
    """
    return re.sub(r'-0*(\d+)$', r'-\1', spec_id)

def extract_specimen_id(cell_name):
    """Extract specimen ID from cell name, handling missing-dash variants like '03Asc46'."""
    cell_str = str(cell_name)
    # Standard pattern: 03Asc-0024, 01C-0096, etc.
    m = re.search(r'(0[123][A-Za-z]+-\d+)', cell_str)
    if not m:
        # Missing-dash fallback: 03Asc46 -> treat as 03Asc-46
        m = re.search(r'(0[123][A-Za-z]+)(\d+)', cell_str)
        if not m:
            return None
        spec_id = (m.group(1) + '-' + m.group(2)).lower()
    else:
        spec_id = m.group(1).lower()
    return spec_id

def resolve_id(spec_id):
    """Return list of candidate IDs to try, applying overrides, prefix aliases and zero normalization."""
    candidates = []

    # Check hard overrides first
    norm = normalize_id(spec_id)
    if spec_id in ID_OVERRIDES:
        ov = ID_OVERRIDES[spec_id]
        candidates += [ov, normalize_id(ov)]
    if norm in ID_OVERRIDES:
        ov = ID_OVERRIDES[norm]
        candidates += [ov, normalize_id(ov)]

    # Apply prefix alias if needed (e.g. 01c- -> 01asc-)
    prefix = re.match(r'(0[123][a-z]+)-', spec_id)
    if prefix:
        p = prefix.group(1)
        aliased = PREFIX_ALIASES.get(p)
        if aliased:
            aliased_id = re.sub(r'^' + p, aliased, spec_id)
            candidates.append(aliased_id)
            candidates.append(normalize_id(aliased_id))

    candidates.append(spec_id)
    candidates.append(normalize_id(spec_id))
    return candidates

# Load features
df = pd.read_csv('outputs/Advanced_Features_Raw_Final.csv')

# Load metadata
meta = pd.read_excel('data/Book1.xlsx', sheet_name='specimen information')

# Build lookup maps with both raw and normalized IDs
age_map = {}
gender_map = {}

for _, row in meta.iterrows():
    if pd.isna(row['Aortic Specimen ID']):
        continue
    raw_id = str(row['Aortic Specimen ID']).strip().lower()
    norm_id = normalize_id(raw_id)
    age = row['age']
    gender = row['gender']
    for key in [raw_id, norm_id]:
        age_map[key] = age
        gender_map[key] = gender

def lookup(cell_name, lookup_map, default):
    sp_id = extract_specimen_id(cell_name)
    if not sp_id:
        return default
    for candidate in resolve_id(sp_id):
        val = lookup_map.get(candidate)
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            return val
    return default

df['Age'] = df['CellName'].apply(lambda x: lookup(x, age_map, pd.NA))
df['Gender'] = df['CellName'].apply(lambda x: lookup(x, gender_map, 'Unknown'))

print('Age missing in', df['Age'].isna().sum(), 'rows.')
print('Gender missing in', (df['Gender'] == 'Unknown').sum(), 'rows.')

df.to_csv('outputs/Advanced_Features_Raw_Final.csv', index=False)
print("Saved features with Age and Gender.")
