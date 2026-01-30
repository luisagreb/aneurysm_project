import pandas as pd
import re

def extract_id(name):
    name = str(name).strip()
    token = name.split(' ')[0].strip('-')
    parts = token.split('-')
    if len(parts) > 1 and parts[-1].isdigit(): return int(parts[-1])
    digits = re.findall(r'\d+', token)
    return int(digits[-1]) if digits else None

taa_ids = [24, 43, 45, 46, 47, 51, 54, 55, 56, 57]
df = pd.read_csv('outputs/Advanced_Features_Raw.csv')
df['ID'] = df['CellName'].apply(extract_id)
taa_df = df[df['ID'].isin(taa_ids)]
counts = taa_df['ID'].value_counts().sort_index()
print(counts)

