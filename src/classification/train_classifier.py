import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configuration
FEATURES_FILE = 'outputs/Advanced_Features_Raw.csv'
METADATA_FILE = 'data/Book1.xlsx'
OUTPUT_DIR = 'classification_results/binary'

def extract_numeric_id(text):
    """
    Robustly extracts Subject ID from filename.
    Assumes standard format: "<SubjectID> <Conditions>..."
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
        # Check if the part after hyphen is numeric
        if last_part.isdigit():
            return int(last_part)
        # If not purely digit, try extracting digits
        sub_digits = re.findall(r'\d+', last_part)
        if sub_digits:
            return int(sub_digits[0])
            
    # Priority B: No hyphen (e.g. 03Asc46), just take the last number group
    digits = re.findall(r'\d+', token)
    if digits:
        return int(digits[-1])
        
    return None

def load_metadata(filepath):
    print(f"Loading metadata from {filepath}...")
    # Based on inspection: Header at row 17 (0-indexed -> 17 means row 18)
    # Cols: "Non-aneurysmal" (Index 0), "Aneurysmal" (Index 1)
    try:
        # Load the whole sheet efficiently from row 17
        df = pd.read_excel(filepath, header=None, skiprows=18)
        # Columns 0 and 1 are relevant
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
                
        print(f"Found {len(healthy_ids)} Healthy IDs: {sorted(list(healthy_ids))}")
        print(f"Found {len(taa_ids)} TAA IDs: {sorted(list(taa_ids))}")
        
        return healthy_ids, taa_ids
        
    except Exception as e:
        print(f"Error parse metadata: {e}")
        return set(), set()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading data...")
    # 1. Load Metadata
    healthy_ids, taa_ids = load_metadata(METADATA_FILE)
    if not healthy_ids and not taa_ids:
        print("Failed to load metadata labels.")
        return

    # 2. Load Features
    if not os.path.exists(FEATURES_FILE):
        print(f"Error: {FEATURES_FILE} not found.")
        return
        
    df_features = pd.read_csv(FEATURES_FILE)
    print(f"Features loaded: {df_features.shape}")
    
    # 3. Create Subject_ID and Label
    # Helper to map filename to Label
    labeled_data = []
    
    # Aggregation requires grouping by ID first
    # So we first extract ID for every row
    df_features['Numeric_ID'] = df_features['CellName'].apply(extract_numeric_id)
    
    # Filter out rows where ID extraction failed
    df_features = df_features.dropna(subset=['Numeric_ID'])
    
    # Assign Group based on Numeric_ID
    def get_label(nid):
        if nid in healthy_ids:
            return 0 # Healthy
        if nid in taa_ids:
            return 1 # TAA
        return None
        
    df_features['Label'] = df_features['Numeric_ID'].apply(get_label)
    
    # Drop unknown labels
    input_len = len(df_features)
    df_features = df_features.dropna(subset=['Label'])
    print(f"Assigned labels to {len(df_features)}/{input_len} rows.")
    
    if df_features.empty:
        print("No matches found between file IDs and Metadata IDs.")
        print("Sample Metadata IDs:", list(healthy_ids)[:5])
        print("Sample File IDs:", df_features['Numeric_ID'].head().tolist())
        return

    # 4. Aggregation by CELL (Normalization)
    # Goal: Merge Actin, Mito, Nucleus rows for the SAME CELL.
    # Logic: Normalize filename to remove suffixes like _segmentation, _visible.
    def normalize_filename(f):
        s = str(f).replace('.nii.gz', '').replace('.tif', '')
        s = s.replace('_segmentation', '').replace('_visible', '')
        return s.strip()
        
    df_features['Cell_ID'] = df_features['CellName'].apply(normalize_filename)
    print(f"Created Cell_IDs. Unique Cells: {df_features['Cell_ID'].nunique()}")
    
    # Feature columns + 'Label' + 'Numeric_ID'
    feature_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude Label and ID from "features to average" but keep them in groupby or re-add
    if 'Label' in feature_cols: feature_cols.remove('Label')
    if 'Numeric_ID' in feature_cols: feature_cols.remove('Numeric_ID')
    
    # Aggregation:
    # Features: max (to get the value from the specific structure row)
    # Label/ID: first
    agg_dict = {col: 'max' for col in feature_cols}
    agg_dict['Label'] = 'first'
    agg_dict['Numeric_ID'] = 'first'
    
    df_agg = df_features.groupby('Cell_ID').agg(agg_dict)
    
    print(f"Aggregated Dataset shape: {df_agg.shape} (Cells)")
    
    # 5. Prepare ML Data
    # Drop Label AND Numeric_ID (to prevent data leakage - we want to predict based on features, not patient ID)
    X = df_agg.drop(columns=['Label', 'Numeric_ID'])
    y = df_agg['Label']
    
    # Impute missing values (aggregated mean might still be NaN if a subject has ONLY Actin data, so Mito cols are NaN)
    # Also handle infinity (caused by Division by Zero in some metrics like Tortuosity)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    print(f"Final X: {X.shape}, y: {y.shape}")
    print("Class Balance:")
    print(y.value_counts())
    
    # 6. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 7. Feature Scaling (required for Neural Network)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ============================================================
    # 8. RANDOM FOREST CLASSIFIER
    # ============================================================
    print("\n" + "="*50)
    print("RANDOM FOREST CLASSIFIER")
    print("="*50)
    
    from sklearn.ensemble import RandomForestClassifier
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)  # RF doesn't need scaling
    
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    
    print(f"\nAccuracy Score: {acc_rf:.4f}")
    print("\nConfusion Matrix:")
    print(cm_rf)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=['Healthy (0)', 'TAA (1)']))
    
    # Feature Importance (Random Forest)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    top_n = 10
    print(f"\nTop {top_n} Features:")
    for f in range(min(top_n, X.shape[1])):
        print(f"{f + 1}. {X.columns[indices[f]]} ({importances[indices[f]]:.4f})")
    
    # Feature Importance Plot
    plt.figure(figsize=(10, 6))
    plt.title("Random Forest - Feature Importances", fontsize=14, fontweight='bold')
    colors = []
    for feat in X.columns[indices[:top_n]]:
        if feat.startswith('Actin'):
            colors.append('#E74C3C')
        elif feat.startswith('Mito'):
            colors.append('#27AE60')
        else:
            colors.append('#3498DB')
    plt.bar(range(min(top_n, X.shape[1])), importances[indices[:top_n]], align="center", color=colors)
    plt.xticks(range(min(top_n, X.shape[1])), X.columns[indices[:top_n]], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/feature_importance.png')
    print(f"\nSaved feature importance plot to {OUTPUT_DIR}/feature_importance.png")
    
    # ============================================================
    # 9. NEURAL NETWORK CLASSIFIER
    # ============================================================
    print("\n" + "="*50)
    print("NEURAL NETWORK CLASSIFIER")
    print("="*50)
    
    nn = MLPClassifier(
        hidden_layer_sizes=(50, 25),
        max_iter=1000,
        random_state=42,
        solver='adam',
        learning_rate_init=0.001,
        alpha=0.1,  # Weight decay (AdamW-like regularization)
        early_stopping=False
    )
    nn.fit(X_train_scaled, y_train)
    
    y_pred_nn = nn.predict(X_test_scaled)
    acc_nn = accuracy_score(y_test, y_pred_nn)
    cm_nn = confusion_matrix(y_test, y_pred_nn)
    
    print(f"\nAccuracy Score: {acc_nn:.4f}")
    print("\nConfusion Matrix:")
    print(cm_nn)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_nn, target_names=['Healthy (0)', 'TAA (1)']))
    
    # Neural Network Training Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(nn.loss_curve_, linewidth=2, color='#2980B9')
    plt.title("Neural Network Training Loss", fontsize=14, fontweight='bold')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/nn_training_loss.png')
    print(f"\nSaved training loss plot to {OUTPUT_DIR}/nn_training_loss.png")
    
    # ============================================================
    # 10. COMPARISON SUMMARY
    # ============================================================
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    print(f"\n{'Classifier':<20} {'Accuracy':<12} {'Correct/Total'}")
    print("-" * 45)
    print(f"{'Random Forest':<20} {acc_rf:.1%}        {int(acc_rf*60)}/60")
    print(f"{'Neural Network':<20} {acc_nn:.1%}        {int(acc_nn*60)}/60")
    
    # Save Predictions
    results = pd.DataFrame({
        'Actual': y_test,
        'RF_Predicted': y_pred_rf,
        'NN_Predicted': y_pred_nn
    })
    results.to_csv(f'{OUTPUT_DIR}/predictions.csv', index=True)
    print(f"\nSaved predictions to {OUTPUT_DIR}/predictions.csv")

if __name__ == "__main__":
    main()

