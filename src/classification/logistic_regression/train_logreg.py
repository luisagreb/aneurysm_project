"""
Logistic Regression Classifier for Healthy vs TAA
Includes:
- Training and Evaluation
- Confusion Matrix Plot
- Feature Importance (Coefficients) Plot
- Rescue Effect Analysis (Predictions on TAA+Collagen)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configuration
FEATURES_FILE = 'outputs/Advanced_Features_Raw.csv'
METADATA_FILE = 'data/Book1.xlsx'
OUTPUT_DIR = 'classification_results/logistic_regression'

def extract_numeric_id(text):
    if pd.isna(text): return None
    text = str(text).strip()
    first_token = text.split(' ')[0].strip('-')
    parts = first_token.split('-')
    if len(parts) > 1 and parts[-1].isdigit(): return int(parts[-1])
    digits = re.findall(r'\d+', first_token)
    return int(digits[-1]) if digits else None

def extract_collagen_status(filename):
    if pd.isna(filename): return None
    filename = str(filename).lower()
    if '+coll' in filename or '+col' in filename or 'plus coll' in filename: return 'Collagen'
    if '-coll' in filename or '-col' in filename or 'nocoll' in filename: return 'NoCollagen'
    return None

def normalize_filename(f):
    s = str(f).replace('.nii.gz', '').replace('.tif', '')
    s = s.replace('_segmentation', '').replace('_visible', '')
    return s.strip()

def load_data():
    print("Loading metadata...")
    df_meta = pd.read_excel(METADATA_FILE, header=None, skiprows=18)
    healthy_ids = {extract_numeric_id(x) for x in df_meta[0].dropna() if extract_numeric_id(x)}
    taa_ids = {extract_numeric_id(x) for x in df_meta[1].dropna() if extract_numeric_id(x)}
    
    print("Loading features...")
    df = pd.read_csv(FEATURES_FILE)
    df['Numeric_ID'] = df['CellName'].apply(extract_numeric_id)
    df['Collagen_Status'] = df['CellName'].apply(extract_collagen_status)
    
    # Labeling
    def get_label(nid):
        if nid in healthy_ids: return 0 # Healthy
        if nid in taa_ids: return 1 # TAA
        return None
    
    df['Label'] = df['Numeric_ID'].apply(get_label)
    df = df.dropna(subset=['Label'])
    
    # Filter for NO COLLAGEN only (for training)
    # We want to train on "Pure Healthy" vs "Pure TAA"
    # Then TEST on Collagen cells to see if they look healthy
    df_train_pool = df[df['Collagen_Status'] == 'NoCollagen'].copy()
    
    # Aggregation
    df['Cell_ID'] = df['CellName'].apply(normalize_filename)
    df_train_pool['Cell_ID'] = df_train_pool['CellName'].apply(normalize_filename)
    
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c not in ['Label', 'Numeric_ID']]
    
    agg_dict = {col: 'max' for col in feature_cols}
    agg_dict['Label'] = 'first'
    agg_dict['Collagen_Status'] = 'first'
    
    # Full dataset (for rescue analysis later)
    df_all_agg = df.groupby('Cell_ID').agg(agg_dict)
    
    # Training dataset (No Collagen only)
    df_train_agg = df_train_pool.groupby('Cell_ID').agg(agg_dict)
    
    return df_train_agg, df_all_agg

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    df_train, df_all = load_data()
    print(f"Training Data (No Coll): {len(df_train)} cells")
    
    X = df_train.drop(columns=['Label', 'Collagen_Status'])
    y = df_train['Label']
    X = X.fillna(0)
    
    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 3. Scale Features (Crucial for Logistic Regression!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    
    # 5. Evaluate
    y_pred = lr.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.1%}")
    
    # 6. Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'TAA'], yticklabels=['Healthy', 'TAA'])
    plt.title(f'Logistic Regression (Acc: {acc:.1%})', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/lr_confusion_matrix.png', dpi=300)
    print(f"Saved: {OUTPUT_DIR}/lr_confusion_matrix.png")
    
    # 7. Feature Importance (Coefficients)
    # In LogReg, coefficients indicate direction and strength
    coefs = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lr.coef_[0]
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    top_10 = coefs.head(10)
    
    plt.figure(figsize=(10, 6))
    colors = ['#E74C3C' if c > 0 else '#3498DB' for c in top_10['Coefficient']] # Red for TAA-associated, Blue for Healthy
    sns.barplot(data=top_10, x='Coefficient', y='Feature', palette=colors)
    plt.axvline(0, color='black', linewidth=1)
    plt.title('Top 10 Features (Logistic Regression Coefficients)', fontsize=14, fontweight='bold')
    plt.xlabel('Coefficient (Positive = TAA, Negative = Healthy)')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/lr_feature_importance.png', dpi=300)
    print(f"Saved: {OUTPUT_DIR}/lr_feature_importance.png")
    
    # 8. Rescue Analysis
    # Apply the model to TAA + Collagen cells
    print("\nRunning Rescue Analysis...")
    df_rescue = df_all[(df_all['Label'] == 1) & (df_all['Collagen_Status'] == 'Collagen')].copy()
    
    if len(df_rescue) > 0:
        X_rescue = df_rescue.drop(columns=['Label', 'Collagen_Status'])
        X_rescue = X_rescue.fillna(0)
        X_rescue_scaled = scaler.transform(X_rescue) # Use same scaler!
        
        # Predict: Does the model think they are Healthy (0) or TAA (1)?
        rescue_preds = lr.predict(X_rescue_scaled)
        percent_rescued = sum(rescue_preds == 0) / len(rescue_preds)
        
        print(f"TAA + Collagen Cells: {len(df_rescue)}")
        print(f"Classified as Healthy (RESCUED): {sum(rescue_preds == 0)} ({percent_rescued:.1%})")
        print(f"Classified as TAA (NOT RESCUED): {sum(rescue_preds == 1)}")
        
        # Plot Rescue Pie Chart
        plt.figure(figsize=(6, 6))
        plt.pie([sum(rescue_preds == 0), sum(rescue_preds == 1)], 
                labels=['Rescued (Looks Healthy)', 'Not Rescued (Looks TAA)'],
                colors=['#3498DB', '#E74C3C'], autopct='%1.1f%%', startangle=90)
        plt.title('Rescue Effect: TAA + Collagen Classification', fontsize=14, fontweight='bold')
        plt.savefig(f'{OUTPUT_DIR}/lr_rescue_pie_chart.png', dpi=300)
        print(f"Saved: {OUTPUT_DIR}/lr_rescue_pie_chart.png")
    else:
        print("No TAA + Collagen cells found for rescue analysis.")

if __name__ == "__main__":
    main()
