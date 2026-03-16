import pandas as pd
import re
from pathlib import Path

def extract_specimen_id(cell_name):
    # Try fully finding 0[123]Asc-0*[0-9]+
    m = re.search(r'(0[123][A-Za-z]+-0*\d+)', str(cell_name))
    if not m:
        return None
    spec_id = m.group(1).lower()
    return spec_id

# Load features
df = pd.read_csv('outputs/Advanced_Features_Raw_Final.csv')

# Load metadata
meta = pd.read_excel('data/Book1.xlsx', sheet_name='specimen information')

# Map IDs to age and gender
age_map = {}
gender_map = {}

for _, row in meta.iterrows():
    if pd.isna(row['Aortic Specimen ID']):
        continue
    spec_id = str(row['Aortic Specimen ID']).strip().lower()
    age = row['age']
    gender = row['gender']
    age_map[spec_id] = age
    gender_map[spec_id] = gender

# We need to map the features cell names
def get_age(cell_name):
    sp_id = extract_specimen_id(cell_name)
    if not sp_id:
        return pd.NA
    return age_map.get(sp_id, pd.NA)

def get_gender(cell_name):
    sp_id = extract_specimen_id(cell_name)
    if not sp_id:
        return 'Unknown'
    return gender_map.get(sp_id, 'Unknown')

df['Age'] = df['CellName'].apply(get_age)
df['Gender'] = df['CellName'].apply(get_gender)

print('Age missing in', df['Age'].isna().sum(), 'rows.')
print('Gender missing in', (df['Gender'] == 'Unknown').sum(), 'rows.')

df.to_csv('outputs/Advanced_Features_Raw_Final.csv', index=False)
print("Saved features with Age and Gender.")
